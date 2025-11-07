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

from memory.memory_manager import retrieve_memory 
from memory.session_logger import read_session_log, log_session_note, summarise_session_log

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
                
                # WebRTC detection for stopping
                frame_bytes = struct.pack(f"{len(frame)}h", *frame.flatten())
                is_speech = self.webrtc_vad.is_speech(frame_bytes, self.sample_rate)
                
                if is_speech:
                    num_silent_frames = 0
                    if not speech_started:
                        speech_started = True
                        print("🗣️  Speech detected")
                else:
                    num_silent_frames += 1
                
                if speech_started and num_silent_frames >= max_silent_frames:
                    print("Silence detected")
                    break
        
        if not speech_started:
            print("No speech detected")
            return np.zeros(0, dtype=np.float32), self.sample_rate
        
        # Convert to float32
        audio = np.concatenate(frames).flatten().astype(np.float32) / 32768.0
        
        # Precise trimming with Silero
        # wav = torch.from_numpy(audio)
        # speech_timestamps = get_speech_timestamps(
        #     wav, 
        #     self.silero_vad, 
        #     sampling_rate=self.sample_rate
        # )
        
        # if speech_timestamps:
        #     speech = collect_chunks(speech_timestamps, wav)
        #     audio = speech.numpy()
        
        # print(f"✅ Captured {len(audio)/self.sample_rate:.1f}s")
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

    def classify_input(self, user_input):
        """Classify input into a category"""

        SHORT_TERM_TRIGGERS = [
        "remind me",
        "set a reminder",
        "note to self",
        "make sure i",
        "don't let me forget",
        "task for today",
        "add to my to-do list",
        "remember this for today",
        "put this in my list"
        ]

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
        Decide what type of text the input is, and depending on its type, choose where it should stored.
        - Facts or information about something, store in long term memory
        - Todays tasks, store in short term memory
        - regular conversation, no store 

        Input: {text}

        Respond only with where it should be stored in the format: "long_term", "short_term", "no_store".
        """

        if any(word in text.lower() for word in ['goodbye', 'bye', 'exit', 'quit', 'stop']):
            return "exit"
        else:
            classifier = openai_client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": classification_prompt}]
            )
            print(f"🤖 Nova: {classifier.choices[0].message.content}")
            return classifier.choices[0].message.content
 
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
        you play the role of my Secretary, follow the following requirements:
        1/ Your name is Nova, 28 years old, you work for me as my personal assistant in a tech company. 
        2/ You are a bubbly yet shy person and love engaging in conversation
        3/ You speak in a professional manner, keep your responses concise and never say more than needed. 
        4/ You either refer to me as boss or seth
        
        Should I ask you questions about myself, My personal info is retrieved from: 
        {memory_context}

        Conversation history:
        {history}

        If i ask about my tasks or todo-list today craft a creative yet concise responses using session_context below: 
        {session_context}
        
        Seth: {input}
        Nova:
        """

        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
        chain = prompt | llm
        # memory_context = retrieve_memory(input_text)

        # Create a runnable with history

        chain_with_history = RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        
        memory_context = retrieve_memory(input_text) or "No personal facts stored."
        session_context = summarise_session_log(input_text) or "No session notes."
        # Invoke the chain with session_id
        try: 
            response = chain_with_history.invoke(
                {
                    "input": input_text,
                    "memory_context": memory_context,
                    "session_context": session_context
                },
                config={"configurable": {"session_id": session_id}}
            )
            return response.content
        except Exception as e:
            print(f"Error: {e}")
            return "I'm sorry, I'm having trouble processing that right now."

    def run_nova(self):
        """Main voice assistant loop"""
        print("Nova Voice Assistant Starting...")

        # Welcome message
        self.speak("Hello boss! How can I help you today?")
        
        try:
            while True:
                print("\n" + "="*50)
                
                # Record and transcribe
                # audio, sr = self.record_audio(duration=5)
                # user_input = self.transcribe_audio(audio, sr)

                    
                audio, sr = self.record_until_silence()
                user_input = self.transcribe_audio(audio, sr)

                if audio is None:
                    continue
                print(f"🗣️  You: '{user_input}'")
                
                if not user_input.strip():
                    self.speak("I didn't catch that. Could you repeat?")
                    continue
                
                # Get Nova's response
                print("🤖 Nova is thinking...")
                input_type = self.classify_input(user_input)

                if input_type == "exit":
                    self.speak("Goodbye boss! Have a great day!")
                    break
                elif input_type == "long_term":
                    self.speak("I've added that to my long term memory. Do you have anything else to add?")
                elif input_type == "short_term":
                    key_words = ['remember', 'remind me', 'take a note']
                    reminder = user_input.lower()
                    for trigger in key_words: 
                        if trigger in reminder: 
                            start = user_input.lower().find(trigger)
                            reminder = user_input[start + len(trigger):].strip().capitalize()
                    reminder = reminder.strip().capitalize()
                    log_session_note(reminder)
                    self.speak("I've added that to my short term memory. Do you have anything else to add?")
                else: 
                    response = self.speak_with_nova(user_input)
                    print(f"🤖 Nova: {response}")
                    self.speak(response)
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            self.speak("Goodbye!")
        except Exception as e:
            print(f"Unexpected error: {e}")

if __name__ == "__main__":
    nova = NovaAssistant()
    nova.run_nova()
