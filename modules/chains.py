from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from modules.custom_wikipedia import get_wikipedia_query_tool
from modules.configure_llm import config_llm
from langchain.prompts import PromptTemplate,MessagesPlaceholder,ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from modules.configure_llm import config_llm
from langchain_core.output_parsers import CommaSeparatedListOutputParser,StrOutputParser




llm=config_llm()



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


def chat_chain(llm):
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant engaging in a conversation about a topic. 
        Use the Wikipedia information provided as context for your responses, but also draw on your general knowledge.
        Keep your responses conversational and engaging."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", """Context from Wikipedia: {wiki_data}
        
        Question: {question}""")
    ])
    chat_chain=chat_prompt|llm|StrOutputParser()

    return chat_chain



    

