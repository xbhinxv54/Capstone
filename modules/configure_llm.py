from config.config import load_environment_variables
from langchain_groq import ChatGroq
import google.generativeai as genai
import os

def config_llm():
    api_key = os.getenv('GROQ_API_KEY')

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="mixtral-8x7b-32768",  # You can also use "llama2-70b-4096"
        temperature=0.7,
    )
    return llm