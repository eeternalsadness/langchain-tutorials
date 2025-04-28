import os
import asyncio
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import ObsidianLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

# from qdrant_client import QdrantClient
from langchain_core.runnables import chain

# constants
CHUNK_SIZE = 300
CHUNK_OVERLAP = 40
QDRANT_URL = "http://127.0.0.1:53366"  # minikube
QDRANT_GRPC_PORT = 53366
OBSIDIAN_FOLDERS = [
    "0-inbox",
    "00-zet",
    "1-projects",
    "2-areas",
    "3-resources",
    "Periodic Notes",
]


# load only necessary folders in Obsidian Vault
async def load_docs(folders: List[str]) -> List[Document]:
    # create a list of tasks for asyncio
    tasks = [
        ObsidianLoader(f"{os.getenv('OBSIDIAN', '')}/{folder}").aload()
        for folder in folders
    ]

    # run all the aload() tasks in parallel
    results = await asyncio.gather(*tasks)

    # results is a list of lists, so flatten it
    all_docs = [doc for docs in results for doc in docs]

    return all_docs


docs = asyncio.run(load_docs(OBSIDIAN_FOLDERS))
print(len(docs))
