from config.config import load_environment_variables
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os

def config_llm():
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    llM=ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.7,
        convert_system_message_to_human=True)
    return llM