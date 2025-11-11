from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import find_dotenv, load_dotenv
import os
from datetime import datetime
from memory.session_logger import summarise_session_log, read_session_log, log_session_note

# Load environment variables first
load_dotenv(find_dotenv())

CHROMA_DIR = "memory/chroma_db"

db = Chroma(persist_directory=CHROMA_DIR, embedding_function=OpenAIEmbeddings())


# def categorise_memory(text: str): 

def save_memory(text: str, metadata: dict=None):
    """
        Save to vector database for semantic search
        For example: Personal facts, preferences, context that needs semantic retrieval

        Args:
        text: The content to store
        metadata: Optional metadata dict with keys like:
            - category: preference, personal, work, health, social
            - importance: high, medium, low
            - source: user_told, inferred, imported
        
        Example metadata: {"category": "preference", "date": "2025-01-15"}
    """

    if metadata is None: 
        metadata = {}

    metadata['timestamp'] = datetime.now().isoformat()

    doc = Document(page_content=text, metadata=metadata)

    db.add_documents([doc])
    print(f"💾 Saved to vector memory: {text[:50]}...")
    return "Saved to memory."

def retrieve_memory(query: str, k: int=2): 

    results = db.similarity_search(query, k=k)
    if not results: 
        return "No memories found."

    memories = "\n".join([r.page_content for r in results])
    return memories

def read_profile(filepath): 
    with open(filepath, "r") as profile:
        for line in profile:
            line = line.strip()
            if line: 
                save_memory(line)
    print("Profile imported to long-term memory")

def display_memories(): 
    memories = db.get()
    print(memories['documents'])

def get_relevant_context(query: str): 
    """
    Smart context retrieval combining all memory sources
    
    Returns:
        dict with keys: 'facts', 'tasks', 'semantic_memories'
    """
    context = {}

    # Todays tasks/notes for short-term
    session_summary =  summarise_session_log(query)
    context['tasks'] = session_summary if session_summary != 'No logs for today.' else None

    # Semantic search in vector database
    semantic_memories = retrieve_memory(query, k=2)
    context['semantic_memories'] = semantic_memories

    # 3. Structured facts from SQL (long-term structured)
    context['facts'] = 'Retrieved structured facts from SQL database.'

    return context





