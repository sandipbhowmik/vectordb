import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams

# Directory containing files
data_directory = "./data"

# Qdrant client configuration
qdrant_client = QdrantClient("http://localhost", port=6333)
collection_name = "document_embeddings"

# Initialize text splitter and embedding model
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

def embed_and_load_to_qdrant(chunked_texts):
    # Generate embeddings
    texts = [doc.page_content for doc in chunked_texts]
    embeddings = embedding_model.embed_documents(texts)

    # Create or update the Qdrant collection with cosine similarity metric
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=len(embeddings[0]), distance="Cosine")
    )

    # Load data into the Qdrant collection
    qdrant = Qdrant(client=qdrant_client, collection_name=collection_name, embeddings=embedding_model)
    qdrant.add_texts(texts)

    return len(embeddings)

if __name__ == "__main__":
    # Step 1: Process files
    print("Processing files...")
    documents = process_files(data_directory)

    # Step 2: Chunk documents
    print("Chunking documents...")
    chunked_texts = chunk_documents(documents)

    # Step 3: Embed and load to Qdrant
    print("Generating embeddings and loading into Qdrant...")
    vector_count = embed_and_load_to_qdrant(chunked_texts)

    # Step 4: Print results
    print(f"Total vectors generated and loaded into Qdrant: {vector_count}")
