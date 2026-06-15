# Multi-Document RAG Chatbot Project Setup

## Overview
The goal of this project is to build a full-stack chatbot capable of performing Retrieval-Augmented Generation (RAG) across multiple knowledge sources. The application will empower users to interactively query information from uploaded documents, public web pages, and internal databases, all through a unified and intuitive conversational interface.

## Core Sources of Intelligence
The chatbot will connect to three primary data sources:
1. **Uploaded PDFs & Documents**: Users can upload PDF documents and extract insights directly from them.
2. **Web Pages**: By simply pasting a URL, the chatbot will crawl the webpage and use its content as context for answering questions.
3. **SQL Databases**: The system will connect to relational databases (e.g., storing company sales, customer data, or internal metrics) to answer analytical or structured queries using natural language.

## Key Features
- **Conversation History & Message Trimming**: The chatbot will retain memory of the current session to enable context-aware follow-up questions. It will also implement message trimming to avoid exceeding token limits during lengthy conversations.
- **Source Citations with Page Numbers**: To build trust and allow verification, the chatbot guarantees transparent responses by citing specifically where information was found (e.g., giving page numbers for PDFs or link sections for web pages).
- **Multiple Retrievers (Vector + SQL)**: 
  - *Vector Search* for unstructured data (Documents, URLs)
  - *Text-to-SQL* capabilities for dynamic data retrieval from relational databases.
- **Hybrid Search**: Combining keyword-based search with semantic vector search to improve retrieval accuracy and robustness, especially for domain-specific terminology.
- **Interactive UI & Robust Backend**: A frontend built with Streamlit for an immediate, responsive, and easy-to-use interface, backed by a FastAPI backend for robust REST API points and scaling.

## Proposed Technology Stack
### Frameworks & Logic
- **LangChain / LangGraph**: The core orchestration framework for bridging LLMs with prompt templates, document loaders, vector stores, and memory management. LangGraph will be leveraged for complex agentic workflows (e.g., deciding whether to query the SQL DB or the Vector DB).
- **Streamlit**: For rapidly building the interactive frontend conversational UI.
- **FastAPI**: Backend REST API framework separating the business logic and inference from the frontend, ensuring scalability.

### LLMs & AI Inference
- **Groq (Llama 3)**: Primary LLM provider for fast, low-latency text generation.
- **Embeddings**: OpenAI or HuggingFace embeddings for generating text vectors for similarity search.

### Vector Databases & Storage
- **ChromaDB**: Local, lightweight vector database for semantic storage and quick retrieval of embeddings from documents and web content.

### Deployment & Hosting
- **Streamlit Cloud / Render / Hugging Face Spaces**: Platform options for simple, fast, and accessible deployment of the application to share with users and stakeholders.

## Next Steps
1. Initialize the Python environment and prepare dependencies.
2. Develop the ingestion pipelines for PDFs, URLs, and SQL databases.
3. Implement LangChain logic, chains, and agents for the multi-retriever system.
4. Construct the Streamlit User Interface.
5. Provide testing, bug fixes, and deploy.
