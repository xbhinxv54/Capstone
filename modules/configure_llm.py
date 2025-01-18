from config.config import load_environment_variables
from langchain_google_genai import ChatGoogleGenerativeAI


def config_llm():
    env_vars=load_environment_variables()
    GEMINI_API_KEY=env_vars["GEMINI_API_KEY"]
    llM=ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    return llM