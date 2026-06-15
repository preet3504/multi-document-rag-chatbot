import os
import tempfile
import uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from rag_retriever import RAGRetriever

app = FastAPI(title="Multi-Document RAG API", version="1.0.0")

# Initialize the retriever singleton
try:
    retriever = RAGRetriever()
except Exception as e:
    print(f"Warning: Failed to initialize RAGRetriever on startup: {e}")
    retriever = None

# In-memory storage for chat history keyed by a session_id
# Structure: { "session_id_1": [HumanMessage, AIMessage, ...], ... }
# Note: For production use Redis or a database.
session_store = {}

class ChatRequest(BaseModel):
    session_id: str
    query: str

class WebRequest(BaseModel):
    url: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Multi-Document RAG API!"}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to process an uploaded PDF, splitting it into vectors
    and storing them in ChromaDB.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    global retriever
    if retriever is None:
        try:
            retriever = RAGRetriever()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed setting up Retriever. Did you provide GROQ_API_KEY? {e}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
            
        # Ingest it into ChromaDB
        retriever.ingest_pdf(temp_file_path)
        
        # Cleanup temp file
        os.remove(temp_file_path)
        
        return {"status": "success", "message": f"Successfully ingested {file.filename}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_url")
async def upload_url(request: WebRequest):
    """
    Endpoint to process a web URL, extracting content 
    and storing it in ChromaDB.
    """
    global retriever
    if retriever is None:
        try:
            retriever = RAGRetriever()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed setting up Retriever. {e}")

    try:
        retriever.ingest_url(request.url)
        return {"status": "success", "message": f"Successfully ingested {request.url}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    """
    Endpoint to ask a question to the system. Requires a session_id 
    in order to maintain consistent context and chat history.
    """
    global retriever
    if retriever is None:
        raise HTTPException(status_code=500, detail="RAG system is not initialized. Provide GROQ_API_KEY.")
    
    # Retrieve or create new chat history for the given session_id
    if request.session_id not in session_store:
        session_store[request.session_id] = []
        
    chat_history = session_store[request.session_id]
    
    try:
        result = retriever.ask_question(query=request.query, chat_history=chat_history)
        
        # Append the new interaction to the session history
        chat_history.append(HumanMessage(content=request.query))
        chat_history.append(AIMessage(content=result["answer"]))
        
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
