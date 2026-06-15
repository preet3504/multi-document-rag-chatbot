import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# Using HuggingFace for embeddings (free and efficient)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()

class PDFRetriever:
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "pdf_docs"):
        """
        Initialize the PDF Retriever with a Groq LLM, HuggingFace embeddings,
        and a Chroma vector store.
        """
        # Ensure the GROQ_API_KEY is available
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY environment variable is not set.")
            
        # Initialize Groq LLM (using the fast llama3-8b-8192 model)
        self.llm = ChatGroq(model="llama3-8b-8192", temperature=0.2)
        
        # Initialize lightweight embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize ChromaDB configuration
        self.db_path = db_path
        self.collection_name = collection_name
        self.vector_store = None
        
        # Load existing DB if it exists
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
        # Text splitting to ensure context fits within LLM token limits
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Generated {len(chunks)} text chunks.")
        
        # Create or update Chroma Vector Store
        print("Saving embeddings to vector store...")
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

    def ask_question(self, query: str) -> dict:
        """
        Retrieve context from the vector store and answer the user's question using Groq.
        Returns the answer and the sources with page numbers.
        """
        if self.vector_store is None:
            raise ValueError("No documents loaded. Please ingest a PDF first.")

        # Set up the retriever
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        
        # Create a systemic prompt for the RAG chain
        system_prompt = (
            "You are an intelligent assistant. Use the following retrieved context "
            "to answer the user's question. If you don't know the answer, just say "
            "that you don't know. Keep the answer concise, but thoroughly answer "
            "the question based on the context provided.\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Create chains
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        # Execute the query
        print(f"Retrieving answers for: '{query}'...")
        response = rag_chain.invoke({"input": query})
        
        # Extract sources and format them
        sources = []
        for doc in response.get("context", []):
            page_num = doc.metadata.get("page", "Unknown")
            source_file = doc.metadata.get("source", "Unknown file")
            sources.append(f"Page {page_num} of {source_file}")
            
        return {
            "answer": response["answer"],
            "sources": list(set(sources)) # Remove exact duplicate source pages
        }

# ==============================================================================
# Example Usage:
# ==============================================================================
if __name__ == "__main__":
    # Ensure you have your keys exported: export GROQ_API_KEY="your-key"
    
    rag = PDFRetriever()
    
    # Example to ingest a file:
    # rag.ingest_pdf("sample.pdf")
    
    # Example to ask a question:
    # result = rag.ask_question("What is the main topic of the document?")
    # print("\nAnswer:", result["answer"])
    # print("\nSources:", result["sources"])
