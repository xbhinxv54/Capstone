from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from modules.custom_wikipedia import get_wikipedia_query_tool
from modules.configure_llm import config_llm




llm=config_llm()



def create_wiki_chain():
    wiki=get_wikipedia_query_tool(1)
    template="""
        Based on the following Wikipedia information, please answer the question.
            
            Wikipedia Information: {wiki_data}
            
            Question: {question}
            
            Answer: """
    prompt=PromptTemplate(input_variables=["wiki_data","question"],template=template)

#     chain=LLMChain(prompt=prompt,llm=llm)
#     return wiki,chain
# def answer_questions(question: str, wiki, chain):
#     wiki_data=wiki.run



    

