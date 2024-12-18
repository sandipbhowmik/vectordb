from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient

qdrant_client = QdrantClient("http://localhost", port=6333)
collection_name = "document_embeddings"
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def query_qdrant(question, k=5):
    
    qdrant = Qdrant(client=qdrant_client, collection_name=collection_name, embeddings=embedding_model)
    
    retriever = qdrant.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    results = retriever.get_relevant_documents(question)
    return results

if __name__ == "__main__":
    question = "How to create a variable?"
    print("\nFetching top 5 responses...\n")
    
    results = query_qdrant(question, k=5)

    for i, doc in enumerate(results, 1):
        print(f"Result {i}:\n{doc.page_content}\n")
