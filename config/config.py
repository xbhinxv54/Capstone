import os
from dotenv import load_dotenv

def load_environment_variables():
    load_dotenv()
    return {
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY")
    }
