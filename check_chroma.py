# check_chroma.py

import chromadb

client = chromadb.PersistentClient(path="./chroma_store")

print(client.list_collections())