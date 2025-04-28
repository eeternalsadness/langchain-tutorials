import os
import asyncio
import time
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_community.document_loaders import ObsidianLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph

# constants
MODEL = "llama3.2:3b"
TEMPERATURE = 0.2
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
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


# state (things to keep track of) of the application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# langgraph nodes
def retrieve(state: State):
    retrieved_docs = qdrant.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt_template.invoke(
        {"question": state["question"], "context": docs_content}
    )
    response = model.invoke(messages)

    return {"answer": response.content}


# compile graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


# prepare data
# docs = asyncio.run(load_docs(OBSIDIAN_FOLDERS))
# print(f"Docs: {len(docs)}")
# splits = split_docs(docs, CHUNK_SIZE, CHUNK_OVERLAP)
# print(f"Splits: {len(splits)}")
# print("Initializing Qdrant")
# start_time = time.time()
# qdrant = init_qdrant(splits, MODEL, QDRANT_URL, QDRANT_GRPC_PORT, "obsidian")
# end_time = time.time()
# print(f"Time elapsed: {end_time - start_time:.2f} seconds")


# proompting
prompt = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
Question: {question}
Context: {context}
Answer:
"""
prompt_template = ChatPromptTemplate.from_template(prompt)
model = ChatOllama(
    model=MODEL,
    temperature=TEMPERATURE,
)
