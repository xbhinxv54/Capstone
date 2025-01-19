from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import random
from langchain.prompts import PromptTemplate,MessagesPlaceholder,ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from modules.configure_llm import config_llm
from langchain_core.output_parsers import CommaSeparatedListOutputParser,StrOutputParser


class CustomWikipediaQueryRun(WikipediaQueryRun):
    def _run(self, query: str) -> str:
        random_n=random.randint(1,10)
        if random_n%2==0:
            print(f"Searching Wikipedia for: {query}")
        result = super()._run(query)
        #print(f"\nResults:\n{result}")
        return result

def get_wikipedia_query_tool(top_k_results:int=1):
    wiki_api = WikipediaAPIWrapper(top_k_results=top_k_results)
    wiki_tool=CustomWikipediaQueryRun(api_wrapper=wiki_api)
    return wiki_tool


def create_wiki_chain(llm):
    wiki=get_wikipedia_query_tool(top_k_results=1)
    memory=ConversationBufferWindowMemory(
        return_messages=True,
        memory_key="chat_history",k=5
    )

    extract_topic_template=PromptTemplate(input_variables=["question"],template="""
        Extract the main topic or subject to search for in Wikipedia from this question.
        Respond with ONLY the topic, no other text.
        Question: {question}
        Topic""")
    output_parser=CommaSeparatedListOutputParser()


    extract_chain=extract_topic_template|llm|output_parser

    return extract_chain,memory




    