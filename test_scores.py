import os
os.environ["GROQ_API_KEY"] = "dummy"  # not used for this test
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="pdf_docs"
)

# Test query
query = "What is the capital of France?"
docs_and_scores = vector_store.similarity_search_with_score(query, k=4)
print(f"Query: {query}")
for i, (doc, score) in enumerate(docs_and_scores):
    print(f"Doc {i}: score={score}, content preview: {doc.page_content[:100]}...")