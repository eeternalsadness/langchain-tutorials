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
MODEL = "llama3.2:3b"
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


def split_docs(
    docs: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    headers_to_split = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split,
        strip_headers=False,  # preserve headers in the chunks
    )

    md_splits = []
    for doc in docs:
        md_splits.extend(md_splitter.split_text(doc.page_content))

    # text splitter for each markdown chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # split all markdown chunks
    splits = text_splitter.split_documents(md_splits)

    return splits


def get_embeddings(model: str) -> OllamaEmbeddings:
    embeddings = OllamaEmbeddings(model=model)
    return embeddings


def init_qdrant(
    splits: List[Document],
    model: str,
    qdrant_url: str,
    qdrant_grpc_port: int,
    collection_name: str,
) -> QdrantVectorStore:
    # init from documents
    qdrant = QdrantVectorStore.from_documents(
        splits,
        get_embeddings(model),
        url=qdrant_url,
        prefer_grpc=True,
        grpc_port=qdrant_grpc_port,  # need to specify this, otherwise it connects to the default grpc port
        collection_name=collection_name,
    )

    return qdrant


docs = asyncio.run(load_docs(OBSIDIAN_FOLDERS))
print(f"Docs: {len(docs)}")
splits = split_docs(docs, 500, 100)
print(f"Splits: {len(splits)}")
print("Initializing Qdrant")
qdrant = init_qdrant(splits, MODEL, QDRANT_URL, QDRANT_GRPC_PORT, "obsidian")
