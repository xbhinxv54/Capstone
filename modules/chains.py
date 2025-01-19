from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from modules.custom_wikipedia import get_wikipedia_query_tool
from modules.configure_llm import config_llm
from langchain.prompts import PromptTemplate,MessagesPlaceholder,ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from modules.configure_llm import config_llm
from langchain_core.output_parsers import CommaSeparatedListOutputParser,StrOutputParser







def create_wiki_chain(llm):
    wiki=get_wikipedia_query_tool(top_k_results=1)


    extract_topic_template=PromptTemplate(input_variables=["question"],template="""
        Extract the main topic or subject to search for in Wikipedia from this question.
        Respond with ONLY the topic, no other text.
        Question: {question}
        Topic""")
    output_parser=StrOutputParser()


    extract_chain=extract_topic_template|llm|output_parser

    return extract_chain,wiki


def chat_chain(llm):
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        input_key="question",
        output_key="output"
    )
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant engaging in a conversation about a topic. 
        Use the Wikipedia information provided as context for your responses, but also draw on your general knowledge.
        Keep your responses conversational and engaging. Dont include any confidence percentage or anything like that"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", """Context from Wikipedia: {wiki_data}
        
        Question: {question}""")
    ])
    chat_chain=chat_prompt|llm|StrOutputParser()

    return chat_chain,memory

def run_chat(llm):
    """For command line interface only"""
    
    extract_chain,wiki=create_wiki_chain(llm)
    chats_chain,memory=chat_chain(llm)

    wiki_data=None
    print("Welcome! Ask me a question about any topic. Type 'FINISH' to end the conversation.")

    while True:
        question=input("\nYou: ").strip()
        if question.upper() == 'FINISH':
            print("\nGoodbye! Thanks for chatting!")
            break


        if wiki_data is None: 
            topic=extract_chain.invoke({"question":question})
            wiki_data=wiki.invoke({"query":topic})
        

        chat_history = memory.load_memory_variables({})["chat_history"]
        response=chats_chain.invoke({"wiki_data":wiki_data,"question":question,"chat_history":chat_history})
        
        memory.save_context(
            {"question": question},
            {"output": response}
        )


        print(f"\nAssistant: {response}")
        if question.lower() in ["new topic", "change topic", "different topic"]:
            change = input("\nWould you like to switch to a new topic? (yes/no): ").lower()
            if change.startswith('y'):
                wiki_data = None
                memory.clear()
                print("\nOK, let's start fresh! Ask me about any topic.")
