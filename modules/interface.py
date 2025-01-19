import streamlit as st
from langchain.memory import ConversationBufferMemory
from modules.chains import create_wiki_chain,chat_chain
from modules.configure_llm import config_llm



def initialize_state():
    llm=config_llm()
    """Initialize session state variables"""
    if "wiki_data" not in st.session_state:
        st.session_state.wiki_data = None
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            input_key="question",
            output_key="output"
        )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_full_context" not in st.session_state:
        st.session_state.show_full_context = False
    if "chains_initialized" not in st.session_state:
        # Properly unpack the tuples returned from chain creation
        extract_chain_tuple, wiki = create_wiki_chain(llm)
        chat_chain_tuple, _ = chat_chain(llm)  # We already have memory in session state
        
        # Store the actual chain objects, not the tuples
        st.session_state.extract_chain = extract_chain_tuple
        st.session_state.wiki = wiki
        st.session_state.chats_chain = chat_chain_tuple
        st.session_state.chains_initialized = True

def process_user_input(user_input):
    """Process user input and generate response"""
    # Get or update Wikipedia data
    if st.session_state.wiki_data is None:
        with st.spinner("Searching for information..."):
            topic = st.session_state.extract_chain.invoke({"question": user_input})
            st.session_state.wiki_data = st.session_state.wiki.invoke({"query": topic})

    # Generate response
    with st.spinner("Thinking..."):
        chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]
        response = st.session_state.chats_chain.invoke({
            "wiki_data": st.session_state.wiki_data,
            "question": user_input,
            "chat_history": chat_history
        })

        # Save to memory
        st.session_state.memory.save_context(
            {"question": user_input},
            {"output": response}
        )

        return response


def handle_topic_change():
    """Handle topic change in the sidebar"""
    st.session_state.wiki_data=None
    st.session_state.memory.clear()
    st.session_state.chat_history=[]
    st.rerun()

def clean_wiki_text(text):
    if text.startswith("Page:"):
        text=text.split("Page:",1)[1].strip()
    return text

def display_context_sidebar():
    with st.sidebar:
        st.header("Topic management")
        if st.button("Start New Topic"):
            handle_topic_change()
        if st.session_state.wiki_data:
            st.sidebar.subheader("Current Context")

            clean_text=clean_wiki_text(st.session_state.wiki_data)

            preview = clean_text[:300] + "..."
            st.markdown(preview)

            if st.button("Show full context", key="show_context"):
                # Set a session state variable to show the full context
                st.session_state.show_full_context = True
            
            # Show full context in an expander if button was clicked
            if st.session_state.get('show_full_context', False):
                with st.expander("Full Context", expanded=True):
                    st.markdown(clean_text)
                    if st.button("Hide full context"):
                        st.session_state.show_full_context = False
                        st.rerun()


        
                