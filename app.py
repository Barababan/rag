import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import torch
from pathlib import Path

# Load environment variables
load_dotenv()

# Check for required environment variables
if not os.getenv("OPENAI_API_KEY"):
    st.error("Please set the OPENAI_API_KEY environment variable")
    st.stop()

if not os.getenv("HUGGINGFACE_API_KEY"):
    st.error("Please set the HUGGINGFACE_API_KEY environment variable")
    st.stop()

# Create model cache directory
cache_dir = Path("./model_cache")
cache_dir.mkdir(exist_ok=True)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Page config
st.set_page_config(
    page_title="Physiotherapy Assistant",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("Physiotherapy Assistant")
st.markdown("""
This assistant can help answer questions about physiotherapy based on the provided documentation.
The RAG index is pre-created and loaded from the repository.
""")

# Check for index
if not os.path.exists("chroma_index"):
    st.error("""
    Chroma index not found. Please create the index first by running the create_index.py script.
    
    Instructions:
    1. Create a 'pdfs' directory if it doesn't exist
    2. Add PDF files to the 'pdfs' directory
    3. Run the create_index.py script to create the index
    4. After successful creation of the index, update this page
    """)
    st.stop()

# Initialize the language model
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize embeddings with Hugging Face token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACE_API_KEY")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': False},
    cache_folder=str(cache_dir)
)

# Initialize the vector store
try:
    vectorstore = Chroma(
        persist_directory="chroma_index",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    st.error(f"Error loading the vector store: {str(e)}")
    st.stop()

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Initialize the conversation chain
try:
    with st.spinner("Loading model..."):
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=True
        )
        st.session_state.conversation = qa_chain
        st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error initializing the conversation chain: {str(e)}")
    st.stop()

# Chat interface
if st.session_state.conversation is not None:
    # Chat input
    user_question = st.text_input("Ask your question about physiotherapy:")
    
    if user_question:
        with st.spinner("Searching for answer..."):
            try:
                # Get the response
                response = st.session_state.conversation({"question": user_question})
                
                # Display the chat history
                for i, message in enumerate(st.session_state.chat_history):
                    if i % 2 == 0:
                        st.write(f"👤 You: {message}")
                    else:
                        st.write(f"🤖 Assistant: {message}")
                
                # Add the new messages to chat history
                st.session_state.chat_history.append(user_question)
                st.session_state.chat_history.append(response["answer"])
                
                # Display the latest response
                st.write(f"👤 You: {user_question}")
                st.write(f"🤖 Assistant: {response['answer']}")
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")

# Display chat history
if st.session_state.chat_history:
    st.subheader("Chat History")
    for question, answer in zip(st.session_state.chat_history[::2], st.session_state.chat_history[1::2]):
        st.write(f"Q: {question}")
        st.write(f"A: {answer}")
        st.write("---")

# FAQ Section
st.sidebar.header("Frequently Asked Questions")
faqs = {
    "What is physiotherapy?": "Physiotherapy is a healthcare profession that focuses on improving physical function, mobility, and quality of life through physical interventions, exercise, and education.",
    "How can physiotherapy help?": "Physiotherapy can help with pain management, injury recovery, improving mobility, preventing future injuries, and enhancing overall physical function.",
    "What conditions can physiotherapy treat?": "Physiotherapy can treat various conditions including sports injuries, back pain, arthritis, stroke recovery, respiratory problems, and post-surgery rehabilitation.",
    "How long does physiotherapy treatment take?": "The duration of physiotherapy treatment varies depending on the condition, severity, and individual progress. It can range from a few sessions to several months.",
    "Is physiotherapy painful?": "Physiotherapy should not be painful, though some exercises or treatments might cause mild discomfort. Your physiotherapist will work within your comfort level."
}

for question, answer in faqs.items():
    with st.sidebar.expander(question):
        st.write(answer)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This application uses RAG (Retrieval-Augmented Generation) to provide accurate answers based on physiotherapy documentation.
The index is pre-created and updated periodically.
""") 