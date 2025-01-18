from config.config import load_environment_variables
from modules.custom_wikipedia import get_wikipedia_query_tool
from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    env_vars=load_environment_variables()
    GEMINI_API_KEY=env_vars["GEMINI_API_KEY"]


    llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    wiki=get_wikipedia_query_tool(top_k_results=1)
    query = "Albert Einstein"
    result=wiki.run(query)
    print(result)
if __name__ == "__main__":
    main()