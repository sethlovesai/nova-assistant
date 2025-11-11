import os
import ssl
import urllib.request
import whisper
import sounddevice as sd
import soundfile as sf

import webrtcvad
import struct

import numpy as np
from scipy.io.wavfile import write
from openai import OpenAI
import subprocess
import tempfile
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

from memory.memory_manager import retrieve_memory, save_memory, get_relevant_context
from memory.session_logger import summarise_session_log, log_session_note

from memory.conversation_logger import initialise_db, log_message

import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, collect_chunks

# Fix SSL certificate verification for Whisper model download
ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVEN_LABS_KEY"))

# Global variables
whisper_model = None
conversation_history = []

from pydantic import BaseModel
from typing import Optional, Literal

class queryResult(BaseModel):
    type: Literal["store_personal_fact", "add_task_today", "get_todays_tasks", "search_memory", "chat", "exit"]
    fact: Optional[str] = None          # for STORE_FACT
    task: Optional[str] = None          # for ADD_TASK
    when_text: Optional[str] = None     # natural time, e.g. "tonight"
    query: Optional[str] = None  
    category: Optional[str] = None  # for STORE_FACT

class NovaAssistant: 
    def __init__(self):
        self.whisper_model = whisper.load_model("base")
        self.conversation_history = []
        self.store = {}
        self.webrtc_vad = webrtcvad.Vad(2)
        self.silero_vad = load_silero_vad()
        self.sample_rate = 16000
        

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def record_audio(self, duration=5, sample_rate=16000):
        """Record audio from microphone"""
        try:
            print(f"Recording for {duration} seconds... Speak now!")
            audio = sd.rec(int(duration * sample_rate), 
                        samplerate=sample_rate, 
                        channels=1, 
                        dtype=np.float32)
            sd.wait()
            print("⏹️  Recording finished")
            return audio.flatten(), sample_rate
        except Exception as e:
            print(f"Recording error: {e}")
            return None, None

    def record_until_silence(self, max_seconds=10, silence_duration=1.0):
        print("Nova is listening...")
        
        frame_duration_ms = 30
        frame_size = int(self.sample_rate * frame_duration_ms / 1000)
        
        frames = []
        num_silent_frames = 0
        max_silent_frames = int(silence_duration * 1000 / frame_duration_ms)
        max_frames = int(max_seconds * 1000 / frame_duration_ms)
        
        speech_started = False
        
        # Real-time recording with WebRTC
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.int16,
            blocksize=frame_size
        ) as stream:
            for _ in range(max_frames):
                frame, _ = stream.read(frame_size)
                frames.append(frame)
                
                # Convert frame to bytes for WebRTCVAD detection
                frame_bytes = struct.pack(f"{len(frame)}h", *frame.flatten())
                # Detect if its a speech frame
                is_speech = self.webrtc_vad.is_speech(frame_bytes, self.sample_rate)
                
                
                if is_speech:
                    num_silent_frames = 0
                    if not speech_started:
                        speech_started = True
                        print("🗣️  Speech detected")
                else:
                    num_silent_frames += 1
                
                # Make sure audio doesnt stop prematurely
                if speech_started and num_silent_frames >= max_silent_frames:
                    print("Silence detected")
                    break
        
        if not speech_started:
            print("No speech detected")
            # Return empty speech array
            return np.zeros(0, dtype=np.float32), self.sample_rate
        
        # Joins all frames into one audio clip between -1 and 1- whisper expects this format
        audio = np.concatenate(frames).flatten().astype(np.float32) / 32768.0
        
        return audio, self.sample_rate

    def transcribe_audio(self, audio, sample_rate):
        """Convert speech to text"""
        if audio is None:
            return ""
        temp_filename = "temp_audio.wav"
        write(temp_filename, sample_rate, audio)
        result = self.whisper_model.transcribe(temp_filename)
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return result["text"].strip()

    def classify_input(self, user_input) -> queryResult:
        user_text = user_input.strip().lower()
        """Classify input into a category"""

        if any(word in user_text for word in ["goodbye","bye","exit","quit","stop"]):
            return queryResult(type="exit")


    # 🔹 Time expressions that imply urgency or today-level relevance
        TIME_EXPRESSIONS = [
            "later today",
            "tonight",
            "this evening",
            "before bed",
            "after class",
            "after the gym",
            "in a few hours",
            "before i sleep"
        ]

        text = user_input.lower()

        classification_prompt = f"""
        "You are an intent (NLU) classifier. Extract user intent and slots.\n"
        "Allowed types: store_personal_fact, add_task_today, get_todays_tasks, search_memory, chat, exit.\n"
        "- store_personal_fact: put factual personal info into long-term memory. Fill 'fact'.\n"
                "- For store_personal_fact: also infer a category from the query e.g. 'work', 'health', 'social', or 'personal' and fill 'category'.\n"
        "- add_task_today: add a to-do for today. Fill 'task' and optional 'when_text'.\n"
        "- get_todays_tasks: User asks about their tasks for today, what they need to do today or a anything regarding a task today.\n"
        "- search_memory: user wants to recall info; fill 'query'.\n"
        "- chat: regular conversation.\n\n"
        f"User: {user_text}\n"
        "Return ONLY the JSON object with fields: intent, fact, task, when_text, query."
        """

        out = ChatOpenAI(model="gpt-4o-mini", temperature=0) \
            .with_structured_output(queryResult) \
            .invoke(classification_prompt)

        print(f"🤖 Nova classified input{user_input} as: {out}")
        return out

    def execute_memory_tool(self, query: queryResult):

        if query.type == "store_personal_fact":
            return save_memory(query.fact, metadata={"category": query.category})
        elif query.type == "add_task_today":
            note = query.task.strip().capitalize() if query.task else ""
            return log_session_note(note)
        elif query.type == "get_todays_tasks":
            result = summarise_session_log(query.query)
            return result if result != "No tasks or notes logged today." else "No tasks for today."
        elif query.type == "search_memory":
            result = retrieve_memory(query.query, k=5)
            return result if result else "No relevant memories found."

        return None
 
    def speak(self, text):
        """Convert text to speech"""
        if not text.strip():
            return False
            # Generate speech
        audio_stream = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id="HE9Vblt34asUwmFv9IWS",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        audio = b"".join(audio_stream)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f: 
            f.write(audio)
            temp_filename = f.name
        subprocess.run(['afplay', temp_filename], check=True)
        os.unlink(temp_filename)
        return True

    def speak_with_nova(self, input_text, session_id='default_session'):
        global conversation_history

        template = """
        you play the role of a Secretary, follow the following requirements:
        1/ Your name is Nova, 28 years old, you work for me as my personal assistant in a tech company. 
        2/ You are a bubbly yet shy person and love engaging in conversation
        3/ You speak as if you were an actual Human Secretary, oftening keep your responses concise. 
        
        WHEN ANSWERING:
        1) If the user asks about ME (identity, preferences, past statements, contacts, schedule), first search `memory_context` for relevant facts. If found, answer using those facts (paraphrase allowed). 
        2) If NOT found in `memory_context`, check `session_context` for today’s notes/tasks. 
        3) If still unknown, say you don’t have that info yet and OFFER to save it if the user provides it. Do NOT invent or guess personal facts.

        TASKS / TO-DO:
        - If the user asks “my tasks today / todo / what did I note?”, craft a creative and concise response from {session_context}.
        - If they add a new task or reminder, ACK briefly and ask for time if missing.

        Should I ask you questions about myself, My personal info is retrieved from: 
        {memory_context}

        Conversation history:
        {history}

        Seth: {input}
        Nova:
        """

        context = get_relevant_context(input_text)
        memory_context = context.get('facts', 'No personal facts stored.') 
        session_context = context.get('tasks', 'No session notes.')
        semantic_memories = context.get('semantic_memories', 'No relevant memories found.')

        if semantic_memories:
            memory_context += f"\n\nRelevant past context:\n{semantic_memories}"
        
        print(f"Memory context: {memory_context}")

        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        chain = prompt | llm
        # memory_context = retrieve_memory(input_text)

        # Create a runnable with history
        chain_with_history = RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        
        # memory_context = retrieve_memory(input_text) or "No personal facts stored."
        # session_context = summarise_session_log(input_text) or "No session notes."

        # Invoke the chain with session_id
        try: 
            response = chain_with_history.invoke(
                {
                    "input": input_text,
                    "memory_context": memory_context,
                    "session_context": session_context, 
                },
                config={"configurable": {"session_id": session_id}}
            )
            return response.content
        except Exception as e:
            print(f"Error: {e}")
            return "I'm sorry, I'm having trouble processing that right now."

    def run_nova(self):
        """Main voice assistant loop"""
        initialise_db()
        print("Nova Voice Assistant Starting...")

        # Welcome message
        self.speak("Hello boss! How can I help you today?")
    
        while True:
            print("\n" + "="*20)
            
            # Record and transcribe
            # audio, sr = self.record_audio(duration=5)
            # user_input = self.transcribe_audio(audio, sr)

                
            audio, sr = self.record_until_silence()
            user_input = self.transcribe_audio(audio, sr)
            
            if not user_input.strip():
                self.speak("I didn't catch that. Could you repeat?")
                continue
            
            # Get Nova's response
            print("🤖 Nova is thinking...")
            input_type = self.classify_input(user_input)

            if input_type.type == "exit":
                self.speak("Goodbye boss! Have a great day!")
                break
            elif input_type.type in {"store_personal_fact", "add_task_today", "search_memory", "get_todays_tasks"}:
                result = self.execute_memory_tool(input_type)
                # say = (
                #     "I've added that to long-term memory. Anything else?"
                #     if input_type.type == "store_personal_fact" else
                #     "I've added that to today's tasks. Anything else?"
                #     if input_type.type == "add_task_today" else
                #     result if isinstance(result, str) else "Done."
                # )
                # self.speak(say)
                # self.speak(result)

            # elif input_type == "long_term":
            #     self.speak("I've added that to my long term memory. Do you have anything else to add?")
            # elif input_type == "short_term":
            #     key_words = ['remember', 'remind me', 'take a note']
            #     reminder = user_input.lower()
            #     for trigger in key_words: 
            #         if trigger in reminder: 
            #             start = user_input.lower().find(trigger)
            #             reminder = user_input[start + len(trigger):].strip().capitalize()
            #     reminder = reminder.strip().capitalize()
            #     log_session_note(reminder)
            #     self.speak("I've added that to my short term memory. Do you have anything else to add?")
            response = self.speak_with_nova(user_input)
            session_id = "default_session"
            log_message(session_id, "user", user_input)
            log_message(session_id, "assistant", response)
            print(f"🤖 Nova: {response}")
            self.speak(response)

if __name__ == "__main__":
    nova = NovaAssistant()
    nova.run_nova()
