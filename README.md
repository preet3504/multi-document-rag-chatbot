# Multi-Document RAG Chatbot

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-Integration-blue.svg)

A powerful, versatile Retrieval-Augmented Generation (RAG) system that allows you to chat intelligently with multiple data sources. Whether it's a PDF document, a web page, or a structured SQL database, this application seamlessly ingests the information and empowers you to query it using natural language.

## 🚀 Features

* **📄 PDF Document Analysis:** Upload and index your PDF files to extract insights and ask direct questions.
* **🌐 Website Ingestion:** Paste any URL to scrape, index, and chat with the website's content.
* **🗄️ SQL Database Querying (Text-to-SQL):** Connect to your **SQLite** or **MySQL** databases and query your structured data conversationally.
* **⚡ High Performance & Accurate:** Uses **Groq's** lightning-fast Llama-3 models (`llama-3.3-70b-versatile`), combined with **HuggingFace** embeddings (`all-MiniLM-L6-v2`) and **ChromaDB** for efficient vector storage.
* **🎨 User-Friendly Interface:** Built with **Streamlit** for a smooth, intuitive, and interactive chat experience.
* **🔗 Robust Backend:** Powered by **FastAPI**, handling data ingestion and querying with asynchronous endpoints.

## 🛠️ Tech Stack

* **Backend:** FastAPI, Python
* **Frontend:** Streamlit
* **LLM & AI Framework:** LangChain, Groq API (Llama 3)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Vector Database:** ChromaDB
* **Databases Supported:** SQLite, MySQL (`pymysql`)
* **Document Loaders:** PyPDF, BeautifulSoup4

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/preet3504/multi-document-rag-chatbot.git
cd multi-document-rag-chatbot
```

### 2. Set up a Virtual Environment (Recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory (you can use `.env.example` as a template) and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## 🏃‍♂️ Running the Application

This project requires both the FastAPI backend and the Streamlit frontend to be running simultaneously.

### Start the FastAPI Backend
Open a terminal and run:
```bash
uvicorn main:app --reload
```
The backend API will be available at `http://127.0.0.1:8000`. You can view the Swagger UI documentation at `http://127.0.0.1:8000/docs`.

### Start the Streamlit Frontend
Open a **new** terminal window and run:
```bash
streamlit run streamlit_app.py
```
The UI will automatically open in your default browser at `http://localhost:8501`.

## 💡 How to Use

1. **Open the App:** Navigate to the Streamlit UI in your browser.
2. **Select Data Source:** Use the sidebar to choose your preferred data source:
   * **Upload PDF:** Select a PDF file from your computer and click "Ingest PDF".
   * **Website Link:** Enter a valid URL and click "Ingest Website".
   * **From Database:** Choose SQLite or MySQL, provide the connection credentials, and click "Connect".
3. **Start Chatting:** Once the data is ingested or connected, use the chat input at the bottom of the screen to ask questions. The AI will retrieve relevant context or generate SQL queries to provide accurate answers along with source citations.
