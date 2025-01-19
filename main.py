from config.config import load_environment_variables
from modules.custom_wikipedia import get_wikipedia_query_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from modules.custom_wikipedia import create_wiki_chain

def main(question):
    env_vars=load_environment_variables()
    GEMINI_API_KEY=env_vars["GEMINI_API_KEY"]


    chain,memory=create_wiki_chain()
    result=chain.invoke({"question":question})
    print(result)
if __name__ == "__main__":
    main("Tell me about vellore institute of Technology")