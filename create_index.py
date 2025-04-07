import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Check for required environment variables
if not os.getenv("HUGGINGFACE_API_KEY"):
    print("Error: HUGGINGFACE_API_KEY not found in environment variables.")
    exit(1)

def create_index():
    """
    Create RAG index from PDF documents in the specified directory
    """
    print("Starting RAG index creation...")
    
    # Create directory if it doesn't exist
    if not os.path.exists("pdfs"):
        os.makedirs("pdfs")
        print("Created 'pdfs' directory. Please add your PDF files there.")
        return

    # Check if there are PDF files in the directory
    pdf_files = [f for f in os.listdir("pdfs") if f.endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in the 'pdfs' directory.")
        return

    # Load and process PDFs
    documents = []
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        loader = PyPDFLoader(os.path.join("pdfs", pdf_file))
        documents.extend(loader.load())
        print(f"Loaded {len(documents)} pages from {pdf_file}")

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
    print(f"Created {len(chunks)} chunks from the documents")

    # Initialize embeddings with Hugging Face token
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
    )

    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_index"
    )
    vectorstore.persist()
    print("Vector store created and persisted successfully!")
    print("You can now run the Streamlit app using: streamlit run app.py")

if __name__ == "__main__":
    create_index() 