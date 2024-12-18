from langchain_community.vectorstores import Redis
from langchain_community.embeddings import SentenceTransformerEmbeddings

redis_url = "redis://localhost:6379"
index_name = "document_embeddings"
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def query_redis(question, k=5):

    redis = Redis(redis_url=redis_url, index_name=index_name, embedding_model=embedding_model)

 
    retriever = redis.as_retriever(search_type="similarity", search_kwargs={"k": k})

 
    results = retriever.get_relevant_documents(question)
    return results

if __name__ == "__main__":
    question = "How to create a variable?"

    print("\nFetching top 5 responses...\n")
    results = query_redis(question, k=5)

    for i, doc in enumerate(results, 1):
        print(f"Result {i}:\n{doc.page_content}\n")
