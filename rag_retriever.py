__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_classic.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class RAGRetriever:
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "pdf_docs"):
        """
        Initialize the PDF Retriever with a Groq LLM, HuggingFace embeddings,
        and a Chroma vector store.
        """
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY environment variable is not set.")
            
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.db_path = db_path
        self.collection_name = collection_name
        self.vector_store = None
        self.sql_db = None
        self.sql_db_url = None
        
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

    def ingest_url(self, url: str):
        """
        Load content from a URL, split it into chunks, and store the embeddings in ChromaDB.
        """
        print(f"Loading URL: {url}")
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        print(f"Loaded {len(documents)} document(s). Splitting text...")
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
            
        print("URL ingestion complete!")

    def connect_db(self, db_url: str):
        """
        Connect to a SQL database using the provided URL.
        """
        print(f"Connecting to database: {db_url}")
        try:
            self.sql_db = SQLDatabase.from_uri(db_url)
            self.sql_db_url = db_url
            print("Database connection successful!")
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            raise e

    def ask_sql(self, query: str) -> dict:
        """
        Answer a question by generating and executing SQL queries.
        """
        if self.sql_db is None:
            raise ValueError("No database connected. Please connect to a database first.")

        # 1. Chain to generate SQL query
        dialect = self.sql_db.dialect
        sql_prompt = ChatPromptTemplate.from_template(
            """You are a {dialect} expert. Given an input question, create a syntactically correct {dialect} query to run.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause if applicable for {dialect}. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") or backticks (`) as appropriate for {dialect} to denote them as identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Only return the SQL query and nothing else. No markdown, no comments, no preamble.

Table Info:
{table_info}

Question: {input}"""
        )
        write_query = create_sql_query_chain(self.llm, self.sql_db, prompt=sql_prompt.partial(dialect=dialect))
        
        # 2. Tool to execute SQL query
        execute_query = QuerySQLDataBaseTool(db=self.sql_db)
        
        # 3. Chain to answer based on query result
        answer_prompt = ChatPromptTemplate.from_template(
            """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
        )
        
        answer_chain = answer_prompt | self.llm | StrOutputParser()
        
        # 4. Combine into a full chain
        chain = (
            RunnablePassthrough.assign(query=write_query).assign(
                result=itemgetter("query") | execute_query
            )
            | answer_chain
        )
        
        response = chain.invoke({"question": query})
        
        return {
            "answer": response,
            "sources": [f"Database: {self.sql_db_url}"]
        }

    def ask_question(self, query: str, chat_history: list = None) -> dict:
        """
        Retrieve context from the vector store and answer the user's question using Groq.
        Considers chat_history (list of LangChain message objects) if provided.
        Returns the answer and the sources with page numbers.
        """
        if self.vector_store is None:
            raise ValueError("No documents loaded. Please ingest a PDF or URL first.")
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
            source_file = doc.metadata.get("source", "Unknown file")
            page_num = doc.metadata.get("page")
            
            if page_num is not None:
                sources.append(f"Page {page_num} of {source_file}")
            else:
                sources.append(f"Source: {source_file}")
            
        return {
            "answer": response["answer"],
            "sources": list(set(sources)) 
        }
