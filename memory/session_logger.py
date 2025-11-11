import os
from datetime import datetime 
from langchain_openai import ChatOpenAI

LOG_DIRECTORY = "memory/session_logs"
os.makedirs(LOG_DIRECTORY, exist_ok=True)

def log_session_note(note: str): 
    date = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(LOG_DIRECTORY, f"{date}.txt")
    with open(path, "a") as f: 
        f.write(f"{note}\n")

def read_session_log(): 
    date = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(LOG_DIRECTORY, f"{date}.txt")
    if os.path.exists(path):
        with open(path, "r") as f: 
            lines = [line.rstrip() for line in f]
            if lines: 
                return lines
            else: 
                print('No logs for today.')
    else: 
        print('No logs exist')

def summarise_session_log(query): 
    raw = read_session_log()

    if not raw: 
        return "No logs for today."

    # if query: 
    #     [line for line in raw if any(word in line,)]

    prompt = f"""
        You are Nova. Read the following text and extract a bullet-point summary for each invidividual task for today:

        {raw}
        Your summary should address the query if provided.
        If no query is provided, summarise all tasks for today

        Return only the summary.
        """
    
    summary =  ChatOpenAI(model="gpt-3.5-turbo-1106").invoke(prompt)

    return summary.content