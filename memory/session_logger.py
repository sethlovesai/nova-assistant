import os
from datetime import datetime 

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

session_log = read_session_log()
print(session_log)
