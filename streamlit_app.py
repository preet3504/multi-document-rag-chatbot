import streamlit as st
import uuid
import tempfile
import os
from typing import List, Dict

# Import the RAGRetriever from our module
from rag_retriever import RAGRetriever

# Set page configuration
st.set_page_config(page_title="Multi-Document RAG Chatbot", layout="wide")

def initialize_resources():
    """Initialize the RAG retriever and other resources in session state."""
    if 'resources_initialized' not in st.session_state:
        st.session_state.resources_initialized = True
        # We'll initialize the retriever lazily when needed to avoid early errors
        st.session_state.retriever = None
        # Chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        # Track what has been ingested
        if "ingested_pdfs" not in st.session_state:
            st.session_state.ingested_pdfs = []  # list of filenames
        if "ingested_urls" not in st.session_state:
            st.session_state.ingested_urls = []  # list of URLs
        # Database connection status
        if "db_connected" not in st.session_state:
            st.session_state.db_connected = False
        # Current data source mode (will be set by sidebar)
        if "current_mode" not in st.session_state:
            st.session_state.current_mode = "upload_pdf"  # default

# Initialize resources
initialize_resources()

def get_retriever():
    """Get or create the RAGRetriever instance."""
    if st.session_state.retriever is None:
        try:
            st.session_state.retriever = RAGRetriever()
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {e}")
            st.session_state.retriever = None
    return st.session_state.retriever

def ingest_pdf_file(uploaded_file):
    """Handle PDF ingestion."""
    retriever = get_retriever()
    if retriever is None:
        st.error("RAG system not initialized. Please check GROQ_API_KEY.")
        return False

    # Save uploaded file to a temporary location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Ingest the PDF
        retriever.ingest_pdf(tmp_path)

        # Clean up
        os.unlink(tmp_path)

        # Update state
        st.session_state.ingested_pdfs.append(uploaded_file.name)
        return True
    except Exception as e:
        st.error(f"Failed to ingest PDF: {e}")
        # Clean up temp file if it exists
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        return False

def ingest_url(url):
    """Handle URL ingestion."""
    retriever = get_retriever()
    if retriever is None:
        st.error("RAG system not initialized. Please check GROQ_API_KEY.")
        return False

    try:
        retriever.ingest_url(url)
        st.session_state.ingested_urls.append(url)
        return True
    except Exception as e:
        st.error(f"Failed to ingest URL: {e}")
        return False

def connect_database(db_type, db_url=None, host=None, user=None, password=None, database=None, port=3306):
    """Handle database connection."""
    retriever = get_retriever()
    if retriever is None:
        st.error("RAG system not initialized. Please check GROQ_API_KEY.")
        return False

    try:
        if db_type == "sqlite":
            if not db_url:
                st.error("Database URL is required for SQLite.")
                return False
            success = retriever.connect_db(db_url)
        elif db_type == "mysql":
            if not all([host, user, password, database]):
                st.error("Host, user, password, and database are required for MySQL.")
                return False
            mysql_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            success = retriever.connect_db(mysql_url)
        else:
            st.error("Invalid database type. Use 'sqlite' or 'mysql'.")
            return False

        if success:
            st.session_state.db_connected = True
            return True
        else:
            st.error("Failed to connect to the database.")
            return False
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return False

def ask_question(query, chat_history=None, mode="vector"):
    """Ask a question using the RAG system."""
    retriever = get_retriever()
    if retriever is None:
        st.error("RAG system not initialized. Please check GROQ_API_KEY.")
        return {"answer": "System error: RAG not initialized.", "sources": []}

    if chat_history is None:
        chat_history = []

    try:
        if mode == "sql":
            if not st.session_state.db_connected:
                return {"answer": "Please connect to a database first.", "sources": []}
            result = retriever.ask_sql(query=query)
        else:  # vector mode
            # Check if we have any documents
            if retriever.vector_store is None:
                return {"answer": "No documents loaded. Please ingest a PDF or URL first.", "sources": []}
            result = retriever.ask_question(query=query, chat_history=chat_history)
        return result
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return {"answer": f"An error occurred: {e}", "sources": []}

def display_chat_history():
    """Display the chat history from session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.caption(f"- {source}")

def handle_chat_input(placeholder_text: str, mode: str = "vector"):
    """Handle chat input and get response from the RAG system."""
    if prompt := st.chat_input(placeholder_text):
        # Add user message to chat history
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get response from assistant
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Pass the current chat history (excluding the current user message?
                # The RAGRetriever expects chat_history as list of messages)
                # We'll pass the current st.session_state.messages which includes the user message we just added?
                # Actually, we should pass the history without the current query.
                # Let's create a history list from the session state, excluding the last user message we just added.
                # But note: we already added the user message to the session state.
                # So we'll pass all messages except the last one (which is the current user message).
                history_for_chain = st.session_state.messages[:-1]  # Exclude the current user message

                result = ask_question(query=prompt, chat_history=history_for_chain, mode=mode)
                answer = result.get("answer", "No answer received.")
                sources = result.get("sources", [])

                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.caption(f"- {source}")

                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

# Initialize resources at the start
initialize_resources()

# App title and description
st.title("📚 RAG Chatbot System")
st.markdown("Interact with your PDF documents, Websites, and Databases seamlessly.")

# Sidebar for data source selection and controls
with st.sidebar:
    st.header("1. Select Data Source")

    # Radio options for functionalities
    data_source_mode = st.radio(
        "Choose where your knowledge comes from:",
        options=["Upload PDF", "Website Link", "From Database"],
        key="data_source"
    )

    # Store the current mode in session state for use in chat
    st.session_state.current_mode = data_source_mode.lower().replace(" ", "_")

    st.divider()

    if data_source_mode == "From Database":
        st.markdown("### 🗄️ SQL Database Connection")
        db_type = st.radio("Database Type", ["SQLite", "MySQL"], index=0, key="db_type")

        if db_type == "SQLite":
            db_url = st.text_input("Database URI", "sqlite:///test.db", help="Format: sqlite:///path_to_db", key="db_url")
            if st.button("Connect to SQLite", key="connect_sqlite"):
                with st.spinner("Connecting..."):
                    success = connect_database(db_type="sqlite", db_url=db_url)
                    if success:
                        st.success("Connected to SQLite!")
                    else:
                        st.error("Failed to connect to SQLite.")
        else:  # MySQL
            col1, col2 = st.columns(2)
            with col1:
                host = st.text_input("Host", "localhost", key="mysql_host")
                user = st.text_input("Username", "root", key="mysql_user")
            with col2:
                port = st.number_input("Port", value=3306, min=1, max=65535, key="mysql_port")
                db_name = st.text_input("Database Name", key="mysql_db_name")
            password = st.text_input("Password", type="password", key="mysql_password")

            if st.button("Connect to MySQL", key="connect_mysql"):
                with st.spinner("Connecting..."):
                    success = connect_database(
                        db_type="mysql",
                        host=host,
                        user=user,
                        password=password,
                        database=db_name,
                        port=port
                    )
                    if success:
                        st.success("Connected to MySQL!")
                    else:
                        st.error("Failed to connect to MySQL.")

    elif data_source_mode == "Website Link":
        st.header("2. Enter Website URL")
        url_input = st.text_input("Paste URL here:", placeholder="https://example.com", key="url_input")

        if url_input and not st.session_state.get("url_ingested_for_this_url", False):
            # We'll track per URL? For simplicity, we'll just check if this URL is in ingested_urls
            # But we want to avoid re-ingesting the same URL every time.
            # Let's use a simple approach: if the URL is in ingested_urls, we consider it ingested.
            # We'll disable the button if already ingested.
            if url_input in st.session_state.ingested_urls:
                st.info("This URL has already been ingested.")
                if st.button("Re-ingest Website", key="reingest_url"):
                    with st.spinner("Re-ingesting the website..."):
                        success = ingest_url(url_input)
                        if success:
                            st.success("Website re-ingested!")
                        else:
                            st.error("Failed to re-ingest website.")
            else:
                if st.button("Ingest Website", key="ingest_url"):
                    with st.spinner("Scraping and indexing the website..."):
                        success = ingest_url(url_input)
                        if success:
                            st.success("Website ingested!")
                            # Optionally, we can set a flag to avoid re-ingesting on every rerun
                            # But we rely on the session state list.
                        else:
                            st.error("Failed to ingest website.")

    elif data_source_mode == "Upload PDF":
        st.header("2. Upload Document")
        uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"], key="pdf_uploader")

        if uploaded_file is not None:
            # Check if this file has already been ingested (by name)
            if uploaded_file.name in st.session_state.ingested_pdfs:
                st.info(f"PDF '{uploaded_file.name}' has already been ingested.")
                if st.button("Re-ingest PDF", key="reingest_pdf"):
                    with st.spinner("Re-importing and indexing your PDF..."):
                        success = ingest_pdf_file(uploaded_file)
                        if success:
                            st.success(f"PDF '{uploaded_file.name}' re-ingested!")
                        else:
                            st.error(f"Failed to re-ingest PDF '{uploaded_file.name}'.")
            else:
                if st.button("Ingest PDF", key="ingest_pdf"):
                    with st.spinner("Uploading and indexing your PDF. This might take a moment..."):
                        success = ingest_pdf_file(uploaded_file)
                        if success:
                            st.success(f"PDF '{uploaded_file.name}' ingested!")
                        else:
                            st.error(f"Failed to ingest PDF '{uploaded_file.name}'.")

# Main Chat Interface
if data_source_mode == "Upload PDF":
    st.divider()
    st.subheader("Chat with your Document")

    if not st.session_state.ingested_pdfs:
        st.info("👈 Please upload and ingest a PDF file from the sidebar to start chatting.")
    else:
        # Show which PDFs are loaded
        if len(st.session_state.ingested_pdfs) == 1:
            st.caption(f"Currently chatting with: {st.session_state.ingested_pdfs[0]}")
        else:
            st.caption(f"Currently chatting with {len(st.session_state.ingested_pdfs)} PDFs: {', '.join(st.session_state.ingested_pdfs)}")

        # Display chat messages from history on app rerun
        display_chat_history()
        handle_chat_input("Ask a question about the PDF...", mode="vector")

elif data_source_mode == "Website Link":
    st.divider()
    st.subheader("Chat with the Website")

    if not st.session_state.ingested_urls:
        st.info("👈 Please enter a URL and ingest the website from the sidebar to start chatting.")
    else:
        # Show which URLs are loaded
        if len(st.session_state.ingested_urls) == 1:
            st.caption(f"Currently chatting with: {st.session_state.ingested_urls[0]}")
        else:
            st.caption(f"Currently chatting with {len(st.session_state.ingested_urls)} websites: {', '.join(st.session_state.ingested_urls)}")

        # Display chat messages from history on app rerun
        display_chat_history()
        handle_chat_input("Ask a question about the website content...", mode="vector")

elif data_source_mode == "From Database":
    st.divider()
    st.subheader("Chat with your Database")

    if not st.session_state.db_connected:
        st.info("👈 Please enter a DB URL and connect from the sidebar to start chatting.")
    else:
        # Display chat messages from history on app rerun
        display_chat_history()
        handle_chat_input("Ask a question about your database...", mode="sql")

# Optional: Add a sidebar section to show status and clear data
with st.sidebar:
    st.divider()
    st.header("3. System Status")

    # Show ingested files count
    if st.session_state.ingested_pdfs:
        st.write(f"📄 PDFs loaded: {len(st.session_state.ingested_pdfs)}")
    if st.session_state.ingested_urls:
        st.write(f"🌐 URLs loaded: {len(st.session_state.ingested_urls)}")
    if st.session_state.db_connected:
        st.write("🔌 Database: Connected")
    else:
        st.write("🔌 Database: Not connected")

    # Clear data button
    if st.button("Clear All Data", type="secondary"):
        # Clear the vector store and reset the retriever
        if st.session_state.retriever is not None:
            # We can't easily clear the ChromaDB without deleting the directory,
            # but we can create a new retriever instance which will start fresh.
            # However, note that the old ChromaDB data will still be on disk.
            # For simplicity, we'll just reset the in-memory state and note that persistence is limited.
            st.session_state.retriever = None
            # Note: The actual ChromaDB persistence directory will still have old data.
            # In a production scenario, we might want to delete the directory or use a new collection.
            st.warning("Note: Due to Streamlit Cloud limitations, vector data persistence is limited. Clearing here resets the in-memory reference, but old data may remain on disk until the app is redeployed.")

        # Reset session state
        st.session_state.messages = []
        st.session_state.ingested_pdfs = []
        st.session_state.ingested_urls = []
        st.session_state.db_connected = False
        st.success("All data cleared. Please refresh the page to start fresh.")