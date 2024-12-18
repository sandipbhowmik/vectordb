from langchain_community.vectorstores import ElasticVectorSearch
from langchain_community.embeddings import SentenceTransformerEmbeddings

elastic_host = "http://localhost:9200"
index_name = "document_embeddings"
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def query_elastic(question, k=5):
    elastic = ElasticVectorSearch(elasticsearch_url=elastic_host, index_name=index_name, embedding_model=embedding_model)

    retriever = elastic.as_retriever(search_type="similarity", search_kwargs={"k": k})

    results = retriever.get_relevant_documents(question)
    return results

if __name__ == "__main__":
    question = "How to create a variable?"

    print("\nFetching top 5 responses...\n")
    results = query_elastic(question, k=5)

    for i, doc in enumerate(results, 1):
        print(f"Result {i}:\n{doc.page_content}\n")
