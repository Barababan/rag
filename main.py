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

# Load environment variables
load_dotenv()

# Check for required environment variables
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

if not os.getenv("HUGGINGFACE_API_KEY"):
    raise ValueError("Please set the HUGGINGFACE_API_KEY environment variable")

# Create model cache directory
cache_dir = Path("./model_cache")
cache_dir.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="RAG Chat Assistant")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Initialize the language model
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize embeddings with Hugging Face token
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
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
    print(f"Error loading the vector store: {str(e)}")
    raise

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Initialize the conversation chain
try:
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )
except Exception as e:
    print(f"Error initializing the conversation chain: {str(e)}")
    raise

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: Request, message: str = Form(...)):
    try:
        response = qa_chain({"question": message})
        return {"response": response["answer"]}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 