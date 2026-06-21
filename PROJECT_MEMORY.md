# Multi-Document RAG Chatbot - Project Memory

## Project Overview
A full-stack Retrieval-Augmented Generation (RAG) chatbot that allows users to interact with multiple data sources including PDF documents, web pages, and SQL databases through a natural language interface. The application uses a FastAPI backend for processing and a Streamlit frontend for the user interface.

## Tech Stack
- **Backend**: FastAPI, Python 3.8+
- **Frontend**: Streamlit
- **LLM & AI Framework**: LangChain, Groq API (Llama 3)
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Database**: ChromaDB
- **Databases Supported**: SQLite, MySQL (via SQLAlchemy and pymysql)
- **Document Loaders**: PDFMuxLoader (for PDFs), WebBaseLoader (for URLs)
- **Text Splitters**: RecursiveCharacterTextSplitter
- **Other**: python-dotenv, beautifulsoup4, requests, sentence-transformers, uvicorn, python-multipart, cryptography, langchain-classic

## Folder Structure
```
multi-document-rag-chatbot/
│
├── .claude/                  # Claude Code settings
├ .venv/                      # Python virtual environment (ignored)
├── chroma_db/                # ChromaDB vector storage
│   ├── 24e3f2f6-e972-497d-ac1a-ae812b87a394/  # Collection data
│   │   ├── data_level0.bin
│   │   ├── header.bin
│   │   ├── length.bin
│   │   └── link_lists.bin
│   └── chroma.sqlite3        # ChromaDB metadata
├── docs/                     # Documentation
│   └── project_details.md    # Detailed project setup and features
├── __pycache__/              # Python cache (ignored)
├── .env                      # Environment variables (ignored)
├── .env.example              # Example environment variables
├── .gitignore                # Git ignore rules
├── main.py                   # FastAPI backend application
├── rag_retriever.py          # Core RAG logic (PDF/URL ingestion, SQL connection, querying)
├── requirements.txt          # Python dependencies
├── streamlit_app.py          # Streamlit frontend application
└── test.db                   # Sample SQLite database
```

## Features & Status

| Feature | Description | Status |
|---------|-------------|--------|
| PDF Document Analysis | Upload and index PDF files to extract insights and ask direct questions. Uses PDFMuxLoader for high-quality text extraction. | Implemented |
| Website Ingestion | Paste any URL to scrape, index, and chat with the website's content. Uses WebBaseLoader. | Implemented |
| SQL Database Querying (Text-to-SQL) | Connect to SQLite or MySQL databases and query structured data conversationally. Uses LangChain's SQLDatabase and create_sql_query_chain. | Implemented |
| Chat History | Maintains conversation context per session using in-memory storage (session_store). | Implemented |
| Source Citations | Provides citations with page numbers for PDFs and source information for URLs and database queries. | Implemented |
| Multi-Mode Querying | Switch between vector search (for documents/urls) and SQL mode (for databases) via API parameter. | Implemented |
| Asynchronous Backend | FastAPI endpoints handle file uploads, URL processing, and database connections asynchronously. | Implemented |
| Responsive UI | Streamlit interface with sidebar for data source selection and main chat area. | Implemented |
| Strict Grounding | Ensures answers are based solely on ingested sources; returns a canned message when relevant information is not found. | Implemented |

## API Documentation

### Base URL
`http://127.0.0.1:8000`

### Endpoints

#### GET `/`
- **Description**: Welcome message.
- **Response**: 
  ```json
  {
    "message": "Welcome to the Multi-Document RAG API!"
  }
  ```

#### POST `/upload_pdf`
- **Description**: Upload and ingest a PDF file.
- **Parameters**: 
  - `file` (UploadFile): PDF file to ingest.
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Successfully ingested {filename}."
  }
  ```
- **Errors**: 
  - 400: Only PDF files are supported.
  - 500: Internal error during ingestion or missing GROQ_API_KEY.

#### POST `/upload_url`
- **Description**: Ingest content from a web URL.
- **Parameters**: 
  - `url` (string): URL to ingest.
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Successfully ingested {url}."
  }
  ```
- **Errors**: 
  - 500: Internal error during ingestion or missing GROQ_API_KEY.

#### POST `/connect_db`
- **Description**: Connect to a SQL database (SQLite or MySQL).
- **Parameters**: 
  - `db_type` (string): "sqlite" or "mysql".
  - For SQLite: `db_url` (string, required) - e.g., "sqlite:///test.db".
  - For MySQL: `host`, `user`, `password`, `database` (strings, required), `port` (integer, optional, default 3306).
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Successfully connected to {db_label}."
  }
  ```
- **Errors**: 
  - 400: Missing required parameters or invalid db_type.
  - 500: Connection failed or missing GROQ_API_KEY.

#### POST `/ask`
- **Description**: Ask a question to the system (requires session_id for chat history).
- **Parameters**: 
  - `session_id` (string): Unique session identifier.
  - `query` (string): The user's question.
  - `mode` (string, optional): "vector" (default) for document/URL queries, "sql" for database queries.
- **Response**:
  ```json
  {
    "answer": "The generated answer.",
    "sources": [
      "Page 1 of source_file.pdf",
      "Source: https://example.com",
      "Database: sqlite:///test.db"
    ]
  }
  ```
- **Errors**: 
  - 400: Bad request (e.g., no database connected for SQL mode, no documents for vector mode).
  - 500: Internal error or RAG system not initialized.

## Database Schema
The application does not define a fixed database schema; it introspects the connected database at runtime.

- **SQLite**: The sample `test.db` file is provided. Its schema is unknown without inspection.
- **MySQL**: schema depends on the connected database.

The `SQLDatabase` utility from LangChain is used to reflect the database schema and generate appropriate SQL queries.

## Authentication Flow
The application does not implement user authentication. It relies on:
- **GROQ_API_KEY**: Stored in `.env` for accessing the Groq LLM service.
- **Session Management**: Chat history is stored in-memory keyed by a `session_id` (UUID) generated per user session in Streamlit. No authentication is required to start a session; anyone with access to the API can send requests with any session_id.

## AI/RAG Components
- **LLM**: `ChatGroq` with model `openai/gpt-oss-120b` (note: this appears to be a placeholder; Groq actually provides Llama 3 models, but the code uses this string).
- **Embeddings**: `HuggingFaceEmbeddings` with model `all-MiniLM-L6-v2`.
- **Vector Store**: `Chroma` with persistence directory `./chroma_db` and collection name `pdf_docs`.
- **Text Splitter**: `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=100`.
- **Retrieval Chain**: 
  - For vector mode: Uses `create_history_aware_retriever` and `create_stuff_documents_chain` to form a `create_retrieval_chain`.
  - For SQL mode: Uses `create_sql_query_chain` and `QuerySQLDataBaseTool` to generate and execute SQL, then answers based on results.
- **Document Loaders**:
  - PDF: `PDFMuxLoader` (from `langchain-pdfmux`) with quality="high".
  - Web: `WebBaseLoader` (from `langchain-community`).

## Environment Variables
| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `GROQ_API_KEY` | API key for Groq LLM service | Yes | `gsk_...` |
| `LANGCHAIN_API_KEY` | API key for LangChain tracing (optional) | No | `lsv2_pt_...` |
| `LANGCHAIN_PROJECT` | Project name for LangChain tracing | No | `multidocument-rag-chatbot` |
| `LANGCHAIN_TRACING_V2` | Enable LangChain tracing | No | `true` |
| `RAG_SCORE_THRESHOLD` | Relevance score threshold for strict grounding (lower distance = more similar). Adjust based on embedding model and desired precision/recall tradeoff. | No | `0.5` |

Note: The `.env` file in the repository contains actual keys (for demonstration purposes). In production, these should be kept secret.

## Known Issues
1. **PDFMuxLoader Dependency**: The `langchain-pdfmux` package may not be readily available on PyPI; installation might fail or require specific handling.
2. **In-Memory Session Store**: Chat history is stored in a global dictionary (`session_store`) which is lost when the backend restarts and does not scale to multiple workers.
3. **ChromaDB Persistence**: While ChromaDB is persisted to disk, the application does not handle multiple collections or provide a way to list/clear ingested data.
4. **SQL Connection Reuse**: The `RAGRetriever` instance holds a single SQL connection; switching databases requires re-instantiation or updating the connection.
5. **LLM Model String**: The model string `openai/gpt-oss-120b` is not a valid Groq model; Groq provides models like `llama3-70b-8192` or `mixtral-8x7b-32768`. This may cause errors if not corrected.
6. **Missing Error Handling for Ingestion**: If a PDF or URL ingestion fails partway through, there is no cleanup of partial vectors in ChromaDB.
7. **Source Deduplication**: Sources are deduplicated by converting to a set, but the display may still show similar sources (e.g., multiple pages from same document) as separate entries.
8. **Windows Compatibility**: The conditional dependency `pysqlite3-binary; sys_platform == 'linux'` in requirements.txt may cause issues on Windows if the default sqlite3 is insufficient for ChromaDB.

## Pending Tasks
1. **Fix LLM Model String**: Replace `openai/gpt-oss-120b` with a valid Groq model (e.g., `llama3-70b-8192`).
2. **Implement Persistent Session Store**: Replace in-memory `session_store` with Redis or a database for persistence and scalability.
3. **Add Data Management Endpoints**: Provide API endpoints to list ingested documents/URLs, delete specific sources, or clear the vector store.
4. **Enhanced PDF Metadata**: Ensure PDF ingestion captures metadata like author, title, etc., and includes it in sources.
5. **URL Ingestion Improvements**: Handle JavaScript-heavy sites (maybe use Selenium or Playwright as fallback) and respect robots.txt.
6. **SQL Connection Pooling**: Implement proper connection pooling for MySQL connections.
7. **Authentication Layer**: Add optional user authentication (e.g., API keys or OAuth) for multi-tenant scenarios.
8. **Deployment Configuration**: Add Dockerfile and docker-compose.yml for easy deployment.
9. **Logging**: Implement structured logging instead of print statements.
10. **Unit/Integration Tests**: Write tests for the core RAG retriever and API endpoints.
11. **Frontend Enhancements**: 
    - Add ability to clear chat history.
    - Show ingested sources in the sidebar.
    - Improve error messages and loading states.
12. **Documentation**: 
    - Expand API docs with examples.
    - Create user manual.
    - Document environment variables and configuration.

## Development & Deployment Instructions

### Prerequisites
- Python 3.8 or higher
- Git
- Groq API key (obtain from https://groq.com)

### Local Development
1. **Clone the repository**:
   ```bash
   git clone https://github.com/preet3504/multi-document-rag-chatbot.git
   cd multi-document-rag-chatbot
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Unix/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   - Copy `.env.example` to `.env` (if not already present).
   - Edit `.env` and add your Groq API key:
     ```env
     GROQ_API_KEY=your_groq_api_key_here
     ```
   - Optional: Add LangChain tracing variables if desired.

5. **Start the FastAPI backend**:
   ```bash
   uvicorn main:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000` with Swagger UI at `http://127.0.0.1:8000/docs`.

6. **Start the Streamlit frontend** (in a new terminal):
   ```bash
   streamlit run streamlit_app.py
   ```
   The UI will open at `http://localhost:8501`.

### Deployment
#### Option 1: Streamlit Cloud (for frontend) + Render/Fly.io (for backend)
1. Push the code to a GitHub repository.
2. For the backend: Deploy the `main.py` file using a service that supports FastAPI (e.g., Render, Fly.io, Railway) with the environment variables set.
3. For the frontend: Deploy the `streamlit_app.py` file to Streamlit Cloud, pointing to the deployed backend URL.

#### Option 2: Docker (Recommended for production)
1. Create a `Dockerfile` (example):
   ```dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   EXPOSE 8000
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```
   Note: This only runs the backend. A separate service is needed for Streamlit, or you can combine them with a process manager.

2. Build and run:
   ```bash
   docker build -t multi-doc-rag .
   docker run -p 8000:8000 --env-file .env multi-doc-rag
   ```

3. For Streamlit, create another Dockerfile or use a docker-compose.yml to run both services.

### Notes
- The `chroma_db` directory should be persisted if using Docker (mount a volume).
- The `test.db` file is included for demonstration; in production, point to your own database.
- Ensure the backend is accessible from the frontend (adjust `API_URL` in `streamlit_app.py` if deploying to different hosts).