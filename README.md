This repository provides Python scripts to ingest data into and query data from the Qdrant vector database.

# Prerequisites

Python 3.x: Ensure Python 3.x is installed on your system.

Vector DBs: Set up Qdrant, Redis and ELK instance using docker. Refer to the respective products documentation for installation instructions.

# Installation

Clone the repository:

git clone https://github.com/sandipbhowmik/vectordb.git

cd vectordb

Install dependencies:

Put all the python modules in requirements.txt and run: pip install -r requirements.txt

# Usage
Data Ingestion

Use the <qdrant|elk|redis>-ingest.py script to ingest data into the respective vector database instance.

Data Querying

Use the <qdrant|elk|redis>-query.py script to query data from the respective vector database instance.

# Contributing

Contributions are welcome! Please fork the repository and submit a pull request.
