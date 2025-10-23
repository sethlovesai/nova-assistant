# main.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langchain.memory import ChatMessageHistory

from elevenlabs.client import ElevenLabs
from elevenlabs import play

from dotenv import find_dotenv, load_dotenv
import os
import requests

import tempfile
from pydub import AudioSegment
from pydub.playback import play as play_audio

from memory.memory_manager import retrieve_memory, save_memory
from memory.session_logger import log_session_note, read_session_log

import time

load_dotenv(find_dotenv())

ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_KEY")

# Dictionary to store chat histories
store = {}

# Function to get or create chat history for a session
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_response(input_text, session_id="default_session"):

    classify_input(input_text)

    key_words = ['remember', 'remind me', 'take a note']

    if any(word in input_text.lower() for word in key_words):
        reminder = input_text.lower()
        for trigger in key_words: 
            if trigger in reminder: 
                start = input_text.lower().find(trigger)
                reminder = input_text[start + len(trigger):].strip().capitalize()
        reminder = reminder.strip().capitalize()
        log_session_note(reminder)

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
    chat = ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True)
    chain = prompt | chat

    memory_context = retrieve_memory(input_text)

    # Create a runnable with history
    chain_with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    # Invoke the chain with session_id
    for chunk in chain_with_message_history.stream(
        {"input": input_text, "memory_context":memory_context, "session_context": summarise_session_log(input_text) },
        config={"configurable": {"session_id": session_id}}
    ):
        if chunk.content: 
            yield chunk.content

    
    
def summarise_session_log(query): 
    raw = read_session_log()

    prompt = f"""
        You are Nova. Read the following text and extract a bullet-point summary for each invidividual task for today:

        {raw}

        Return only the summary.
        """
    
    summary =  ChatOpenAI(model="gpt-3.5-turbo-1106").invoke(prompt)

    return summary.content
    
    # keywords = extract_key_words(query)

    # relevant_entries = []
    # for entry in raw: 
    #     item = entry.lower()
    #     if any(keyword in item for keyword in keywords): 
    #         relevant_entries.append(entry)
    #     else: 
    #         print('No relevant entries')

    # return relevant_entries

def extract_key_words(query): 
    query = query.lower()
    # print(query)

    keyword_mapping = {  
        "gym": ['exercise', 'work out', 'cardio', 'lift weights'],
        "study": ['read', 'revise', 'revision']
    }

    search_keywords = []
    for category, keywords in keyword_mapping.items():
        for keyword in keywords: 
            if keyword in query:
                # print('matching word', keyword)
                search_keywords.append(category)
            else: 
                print('No keywords')

    return search_keywords if search_keywords else query.split()

def get_voice_message(message):
    try :

        client = ElevenLabs(
        api_key=os.getenv("ELEVEN_LABS_KEY"),
        )
            # Generate speech using ElevenLabs
        audio_stream = client.text_to_speech.convert(
            text=message,
            voice_id="HE9Vblt34asUwmFv9IWS",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            )
        
        audio = b"".join(audio_stream)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f: 
            f.write(audio)
            temp_filename = f.name

        sound = AudioSegment.from_mp3(temp_filename)
        play_audio(sound)

        os.unlink(temp_filename)

        return True
    except Exception as e:
        print(f"Error generating speech: {e}")
        return False

def classify_input(input_text : str): 

    LONG_TERM_TRIGGERS = [
    # "i am",
    # "i have",
    # "i usually",
    # "i always",
    # "my name is",
    # "i prefer",
    # "i like",
    # "i don't like",
    # "i go to",
    # "i want to remember",
    # "note this for the future"
    ]
    
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

    text = input_text.lower()

    classification_prompt = f"""
    Decide what type of text the input is, and depending on its type, choose where it should stored.
    - Facts or information about something, store in long term memory
    - Todays tasks, store in short term memory
    - regular conversation, no store

    Input: {input_text}

    Respond with type of text and where it should be stored.
    """

    classifier = ChatOpenAI(model="gpt-3.5-turbo-1106").invoke(classification_prompt)
    return classifier.content

    # for trigger in SHORT_TERM_TRIGGERS: 
    #     if trigger in text: 
    #         print('Text is short term')
    #         return
        
    
    # for trigger in TIME_EXPRESSIONS: 
    #     if trigger in text: 
    #         print('Text is short term')
    #         return
        
    # for trigger in LONG_TERM_TRIGGERS: 
    #     if trigger in text: 
    #         save_memory(input_text)
    #         print('Text saved to long term memory')
    #         return

def yield_message(msg): 
    for word in msg.split(): 
        yield word


if __name__ == "__main__":
    # Test the function
    response = get_response('Say hi to my friend Jenny')
    print('Response', response)
    get_voice_message(response)


    
   
