import streamlit as st
import requests
import uuid
from typing import List, Dict

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Multi-Document RAG Chatbot", layout="wide")

# Initialize session state for session ID and chat messages
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "pdf_ingested_name" not in st.session_state:
    st.session_state["pdf_ingested_name"] = None
if "url_ingested" not in st.session_state:
    st.session_state["url_ingested"] = False
if "db_connected" not in st.session_state:
    st.session_state["db_connected"] = False

def show_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.caption(f"- {source}")

def handle_chat_input(placeholder_text: str, mode: str = "vector"):
    if prompt := st.chat_input(placeholder_text):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    payload = {
                        "session_id": st.session_state["session_id"],
                        "query": prompt,
                        "mode": mode
                    }
                    
                    ans_response = requests.post(f"{API_URL}/ask", json=payload)
                    if ans_response.status_code == 200:
                        data = ans_response.json()
                        answer = data.get("answer", "No answer received.")
                        sources = data.get("sources", [])
                        
                        st.markdown(answer)
                        if sources:
                            with st.expander("Sources"):
                                for source in sources:
                                    st.caption(f"- {source}")
                                    
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        st.error(f"API Error: {ans_response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to the backend server: {e}")

st.title("📚 RAG Chatbot System")
st.markdown("Interact with your PDF documents, Websites, and Databases seamlessly.")

with st.sidebar:
    st.header("1. Select Data Source")
    
    # Radio options for functionalities
    data_source_mode = st.radio(
        "Choose where your knowledge comes from:",
        options=["Upload PDF", "Website Link", "From Database"],
        key="data_source"
    )
    
    st.divider()
    
    if data_source_mode == "From Database":
        st.markdown("### 🗄️ SQL Database Connection")
        db_type = st.radio("Database Type", ["SQLite", "MySQL"], index=0)
        
        if db_type == "SQLite":
            db_url = st.text_input("Database URI", "sqlite:///test.db", help="Format: sqlite:///path_to_db")
            if st.button("Connect to SQLite"):
                with st.spinner("Connecting..."):
                    try:
                        response = requests.post(f"{API_URL}/connect_db", json={"db_type": "sqlite", "db_url": db_url})
                        if response.status_code == 200:
                            st.session_state.db_connected = True
                            st.success("Connected to SQLite!")
                        else:
                            st.error(f"Failed to connect: {response.json().get('detail')}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else: # MySQL
            col1, col2 = st.columns(2)
            with col1:
                host = st.text_input("Host", "localhost")
                user = st.text_input("Username", "root")
            with col2:
                port = st.number_input("Port", value=3306)
                db_name = st.text_input("Database Name")
            password = st.text_input("Password", type="password")
            
            if st.button("Connect to MySQL"):
                with st.spinner("Connecting..."):
                    try:
                        payload = {
                            "db_type": "mysql",
                            "host": host,
                            "user": user,
                            "password": password,
                            "database": db_name,
                            "port": port
                        }
                        response = requests.post(f"{API_URL}/connect_db", json=payload)
                        if response.status_code == 200:
                            st.session_state.db_connected = True
                            st.success("Connected to MySQL!")
                        else:
                            st.error(f"Failed to connect: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
    elif data_source_mode == "Website Link":
        st.header("2. Enter Website URL")
        url_input = st.text_input("Paste URL here:", placeholder="https://example.com")
        
        if url_input and not st.session_state["url_ingested"]:
            if st.button("Ingest Website"):
                with st.spinner("Scraping and indexing the website..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/upload_url", 
                            json={"url": url_input}
                        )
                        
                        if response.status_code == 200:
                            st.success(response.json().get("message", "Website ingested!"))
                            st.session_state["url_ingested"] = True
                        else:
                            st.error(f"Error indexing URL: {response.text}")
                    except Exception as e:
                        st.error(f"Failed to connect to the backend server: {e}")
        
    elif data_source_mode == "Upload PDF":
        st.header("2. Upload Document")
        uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
        
        if uploaded_file is not None and st.session_state["pdf_ingested_name"] != uploaded_file.name:
            if st.button("Ingest PDF"):
                with st.spinner("Uploading and indexing your PDF. This might take a moment..."):
                    try:
                        # Send the file to FastAPI
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        response = requests.post(f"{API_URL}/upload_pdf", files=files)
                        
                        if response.status_code == 200:
                            st.success(response.json().get("message", "Ingestion completed!"))
                            st.session_state["pdf_ingested_name"] = uploaded_file.name
                        else:
                            st.error(f"Error indexing PDF: {response.text}")
                    except Exception as e:
                        st.error(f"Failed to connect to the backend server: {e}")

# Main Chat Interface
if data_source_mode == "Upload PDF":
    st.divider()
    st.subheader("Chat with your Document")
    
    if not st.session_state["pdf_ingested_name"]:
        st.info("👈 Please upload and ingest a PDF file from the sidebar to start chatting.")
    else:
        # Display chat messages from history on app rerun
        show_chat_history()
        handle_chat_input("Ask a question about the PDF...")

elif data_source_mode == "Website Link":
    st.divider()
    st.subheader("Chat with the Website")
    
    if not st.session_state["url_ingested"]:
        st.info("👈 Please enter a URL and ingest the website from the sidebar to start chatting.")
    else:
        # Display chat messages from history on app rerun
        show_chat_history()
        handle_chat_input("Ask a question about the website content...")

elif data_source_mode == "From Database":
    st.divider()
    st.subheader("Chat with your Database")
    
    if not st.session_state["db_connected"]:
        st.info("👈 Please enter a DB URL and connect from the sidebar to start chatting.")
    else:
        # Display chat messages from history on app rerun
        show_chat_history()
        handle_chat_input("Ask a question about your database...", mode="sql")
