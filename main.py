from config.config import load_environment_variables
from modules.custom_wikipedia import get_wikipedia_query_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from modules.configure_llm import config_llm
from modules.chains import run_chat
from modules.interface import initialize_state,handle_topic_change,process_user_input,display_context_sidebar
import streamlit as st

def main():
    st.title("AI Chat Assistant 🤖")

    initialize_state()

    display_context_sidebar()
    
    chat_container=st.container()


    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_input:= st.chat_input("Ask me anything..."):
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        if user_input.lower() in ["new topic", "change topic", "different topic"]:
            handle_topic_change()
            return
        
        
        response = process_user_input(user_input)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        

  
if __name__ == "__main__":
    main() 