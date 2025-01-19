from config.config import load_environment_variables
from modules.custom_wikipedia import get_wikipedia_query_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from modules.configure_llm import config_llm
from modules.chains import run_chat

def main():
    env_vars=load_environment_variables()
    GEMINI_API_KEY=env_vars["GEMINI_API_KEY"]
    

  
if __name__ == "__main__":
    llm=config_llm()
    run_chat(llm)