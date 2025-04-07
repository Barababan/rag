import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os
from pdf_processor import PDFProcessor
import tempfile

load_dotenv()

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processor" not in st.session_state:
    st.session_state.pdf_processor = PDFProcessor()

# FAQ data
FAQ_DATA = {
    "Что такое физиотерапия?": "Физиотерапия - это область медицины, которая использует физические методы лечения для восстановления, поддержания и улучшения физического состояния пациента.",
    "Какие основные методы физиотерапии существуют?": "Основные методы включают электротерапию, магнитотерапию, ультразвуковую терапию, лазерную терапию, массаж и лечебную физкультуру.",
    "В чем разница между детской и взрослой физиотерапией?": "Детская физиотерапия учитывает особенности развития детского организма, использует более щадящие методы и часто включает элементы игры в процесс лечения.",
    "Как часто нужно посещать физиотерапевта?": "Частота посещений зависит от диагноза, состояния пациента и назначенного курса лечения. Обычно это 2-3 раза в неделю.",
    "Есть ли противопоказания к физиотерапии?": "Да, противопоказания включают острые воспалительные процессы, онкологические заболевания, тяжелые сердечно-сосудистые заболевания и некоторые другие состояния."
}

def initialize_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    if os.path.exists("faiss_index"):
        vectorstore = FAISS.load_local("faiss_index", embeddings)
        llm = ChatOpenAI(temperature=0.7)
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        return chain
    return None

def main():
    st.title("Ассистент по физиотерапии")
    
    # Sidebar with FAQ and PDF upload
    with st.sidebar:
        st.header("Часто задаваемые вопросы")
        for question, answer in FAQ_DATA.items():
            with st.expander(question):
                st.write(answer)
        st.header("Загрузка PDF")
        pdf_files = st.file_uploader(
            "Загрузите PDF файлы", type=["pdf"], accept_multiple_files=True
        )
        if st.button("Обработать PDF"):
            if pdf_files:
                with st.spinner("Обработка PDF..."):
                    # Create a temporary directory to store uploaded PDFs
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Save uploaded files to temporary directory
                        for pdf_file in pdf_files:
                            file_path = os.path.join(temp_dir, pdf_file.name)
                            with open(file_path, "wb") as f:
                                f.write(pdf_file.getvalue())
                        
                        # Process PDFs using the PDFProcessor
                        processor = PDFProcessor(pdf_dir=temp_dir)
                        vectorstore = processor.process_pdfs()
                        
                        if vectorstore:
                            # Initialize the conversation chain with the new vectorstore
                            llm = ChatOpenAI(temperature=0.7)
                            st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                                llm=llm,
                                retriever=vectorstore.as_retriever(),
                                return_source_documents=True
                            )
                            st.success("PDF файлы успешно обработаны! Теперь вы можете задавать вопросы.")
                        else:
                            st.error("Произошла ошибка при обработке PDF файлов.")
            else:
                st.warning("Пожалуйста, загрузите PDF файлы.")
    
    # Main chat interface
    st.header("Задайте вопрос")
    
    # Initialize the conversation chain
    if st.session_state.conversation is None:
        st.session_state.conversation = initialize_chain()
    
    # Chat input
    user_question = st.text_input("Ваш вопрос:")
    
    if user_question:
        if st.session_state.conversation:
            response = st.session_state.conversation({
                "question": user_question,
                "chat_history": st.session_state.chat_history
            })
            
            st.session_state.chat_history.append((user_question, response["answer"]))
            
            # Display chat history
            for question, answer in st.session_state.chat_history:
                st.write(f"Q: {question}")
                st.write(f"A: {answer}")
                st.write("---")
        else:
            st.error("Пожалуйста, сначала добавьте PDF файлы и обработайте их.")

if __name__ == "__main__":
    main()