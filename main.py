from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import torch
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check for required environment variables
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY environment variable is not set")
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

if not os.getenv("HUGGINGFACE_API_KEY"):
    logger.error("HUGGINGFACE_API_KEY environment variable is not set")
    raise ValueError("Please set the HUGGINGFACE_API_KEY environment variable")

# Create necessary directories
cache_dir = Path("./model_cache")
cache_dir.mkdir(exist_ok=True)

templates_dir = Path("./templates")
templates_dir.mkdir(exist_ok=True)

static_dir = Path("./static")
static_dir.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="RAG Chat Assistant")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Initialize the language model
try:
    logger.info("Initializing language model...")
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    logger.info("Language model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing language model: {str(e)}")
    raise

# Initialize embeddings with Hugging Face token
try:
    logger.info("Initializing embeddings...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True},
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
        cache_folder=str(cache_dir)
    )
    logger.info("Embeddings initialized successfully")
except Exception as e:
    logger.error(f"Error initializing embeddings: {str(e)}")
    raise

# Initialize the vector store
try:
    logger.info("Initializing vector store...")
    if not os.path.exists("chroma_index"):
        logger.warning("Chroma index not found. Creating a new one...")
        # Create a simple document for initial index
        from langchain.docstore.document import Document
        docs = [Document(page_content="This is a placeholder document for the initial index.")]
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="chroma_index"
        )
        vectorstore.persist()
        logger.info("Created new Chroma index")
    else:
        vectorstore = Chroma(
            persist_directory="chroma_index",
            embedding_function=embeddings
        )
        logger.info("Loaded existing Chroma index")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    logger.info("Vector store initialized successfully")
except Exception as e:
    logger.error(f"Error initializing vector store: {str(e)}")
    raise

# Initialize memory
try:
    logger.info("Initializing memory...")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    logger.info("Memory initialized successfully")
except Exception as e:
    logger.error(f"Error initializing memory: {str(e)}")
    raise

# Initialize the conversation chain
try:
    logger.info("Initializing conversation chain...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )
    logger.info("Conversation chain initialized successfully")
except Exception as e:
    logger.error(f"Error initializing conversation chain: {str(e)}")
    raise

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: Request, message: str = Form(...)):
    try:
        logger.info(f"Received message: {message}")
        response = qa_chain({"question": message})
        logger.info("Generated response successfully")
        return {"response": response["answer"]}
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 