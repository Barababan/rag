import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

class PDFProcessor:
    def __init__(self, pdf_dir: str = "pdfs"):
        self.pdf_dir = pdf_dir
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def load_pdfs(self) -> List[str]:
        """Load all PDF files from the specified directory."""
        if not os.path.exists(self.pdf_dir):
            os.makedirs(self.pdf_dir)
            print(f"Created directory {self.pdf_dir}. Please add your PDF files there.")
            return []
            
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
        return pdf_files
        
    def process_pdfs(self):
        """Process all PDFs and create a FAISS index."""
        pdf_files = self.load_pdfs()
        if not pdf_files:
            return None
            
        documents = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(os.path.join(self.pdf_dir, pdf_file))
            documents.extend(loader.load())
            
        texts = self.text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        vectorstore.save_local("faiss_index")
        return vectorstore

if __name__ == "__main__":
    processor = PDFProcessor()
    processor.process_pdfs() 