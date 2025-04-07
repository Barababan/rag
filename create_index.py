import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

def create_index():
    """
    Create RAG index from PDF documents in the specified directory
    """
    print("Starting RAG index creation...")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key")
        return
    
    # Create directory if it doesn't exist
    if not os.path.exists("pdfs"):
        os.makedirs("pdfs")
        print("Создана директория 'pdfs'. Пожалуйста, добавьте PDF файлы и запустите скрипт снова.")
        return

    # Check if there are PDF files in the directory
    pdf_files = [f for f in os.listdir("pdfs") if f.endswith(".pdf")]
    if not pdf_files:
        print("В директории 'pdfs' нет PDF файлов. Пожалуйста, добавьте файлы и запустите скрипт снова.")
        return

    # Load and process PDFs
    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join("pdfs", pdf_file))
        documents.extend(loader.load())
        print(f"Обработан файл: {pdf_file}")

    if not documents:
        print("No documents were successfully loaded")
        return

    print(f"\nSuccessfully loaded {len(documents)} pages from PDF files")

    # Split documents into chunks
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Создано {len(chunks)} чанков из {len(documents)} страниц")

    # Create embeddings and store in Chroma
    print("\nCreating embeddings and storing in Chroma...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="chroma_index"
        )
        
        # Save the index
        vectorstore.persist()
        print("\nIndex created and saved successfully!")
        print("You can now run the Streamlit app using: streamlit run app.py")
    except Exception as e:
        print(f"\nError creating embeddings: {str(e)}")
        print("Please make sure all dependencies are installed correctly")
        return

if __name__ == "__main__":
    create_index() 