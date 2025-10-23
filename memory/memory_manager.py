from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os

CHROMA_DIR = "memory/chroma_db"

db = Chroma(persist_directory=CHROMA_DIR, embedding_function=OpenAIEmbeddings())


# def categorise_memory(text: str): 

def save_memory(text: str, metadata: dict={}):
    doc = Document(page_content=text, metadata=metadata)
    db.add_documents([doc])
    return "Saved to memory."

def retrieve_memory(query: str, k: int=2): 
    results = db.similarity_search(query, k=2)
    return "\n".join([r.page_content for r in results])

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

