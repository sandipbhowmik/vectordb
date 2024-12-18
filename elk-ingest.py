import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import ElasticVectorSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter

elastic_host = "http://localhost:9200"
index_name = "document_embeddings"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def process_files(data_directory):
    documents = []

    for file_name in os.listdir(data_directory):
        file_path = os.path.join(data_directory, file_name)

        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_name.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            print(f"Skipping unsupported file: {file_name}")
            continue

        docs = loader.load()
        documents.extend(docs)

    return documents

def chunk_documents(documents):
    chunked_texts = text_splitter.split_documents(documents)
    return chunked_texts

def embed_and_load_to_elastic(chunked_texts):
    texts = [doc.page_content for doc in chunked_texts]
    embeddings = embedding_model.embed_documents(texts)

    elastic = ElasticVectorSearch.from_texts(
        texts,
        embedding_model,
        elasticsearch_url=elastic_host,
        index_name=index_name,
    )

    return len(embeddings)

if __name__ == "__main__":
    data_directory = "./data"

    print("Processing files...")
    documents = process_files(data_directory)

    print("Chunking documents...")
    chunked_texts = chunk_documents(documents)

    print("Generating embeddings and loading into ELK...")
    vector_count = embed_and_load_to_elastic(chunked_texts)

    print(f"Total vectors generated and loaded into ELK: {vector_count}")
