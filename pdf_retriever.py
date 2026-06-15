__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

class PDFRetriever:
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "pdf_docs"):
        """
        Initialize the PDF Retriever with a Groq LLM, HuggingFace embeddings,
        and a Chroma vector store.
        """
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY environment variable is not set.")
            
        self.llm = ChatGroq(model="llama3-8b-8192", temperature=0.2)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.db_path = db_path
        self.collection_name = collection_name
        self.vector_store = None
        
        if os.path.exists(self.db_path):
            self.vector_store = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )

    def ingest_pdf(self, pdf_path: str):
        """
        Load a PDF, split it into chunks, and store the embeddings in ChromaDB.
        Includes page metadata automatically via PyPDFLoader.
        """
        print(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        print(f"Loaded {len(documents)} pages. Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.db_path,
                collection_name=self.collection_name
            )
        else:
            self.vector_store.add_documents(documents=chunks)
            
        print("PDF ingestion complete!")

    def ask_question(self, query: str, chat_history: list = None) -> dict:
        """
        Retrieve context from the vector store and answer the user's question using Groq.
        Considers chat_history (list of LangChain message objects) if provided.
        Returns the answer and the sources with page numbers.
        """
        if self.vector_store is None:
            raise ValueError("No documents loaded. Please ingest a PDF first.")
        if chat_history is None:
            chat_history = []

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        
        # 1. Contextualize the question using chat history
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        # 2. Answer question using retrieved documents
        system_prompt = (
            "You are an intelligent assistant. Use the following retrieved context "
            "to answer the user's question. If you don't know the answer, just say "
            "that you don't know. Keep the answer concise, but thoroughly answer "
            "the question based on the context provided.\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # 3. Execute
        response = rag_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })
        
        # 4. Process sources
        sources = []
        for doc in response.get("context", []):
            page_num = doc.metadata.get("page", "Unknown")
            source_file = doc.metadata.get("source", "Unknown file")
            sources.append(f"Page {page_num} of {source_file}")
            
        return {
            "answer": response["answer"],
            "sources": list(set(sources)) 
        }
