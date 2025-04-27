import os
from langchain_core.documents import Document
from langchain_community.document_loaders import ObsidianLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient

# constants
CHUNK_SIZE = 300
CHUNK_OVERLAP = 40
QDRANT_URL = "http://127.0.0.1:53366"  # minikube
QDRANT_GRPC_PORT = 53366

# generate sample documents
# a Document represents a chunk of a larger document
documents = [
    Document(
        page_content="Arch Linux is the best operating system.",
        metadata={"source": "me"},
    ),
    Document(
        page_content="I use Arch Linux btw.",
        metadata={"source": "me"},
    ),
]

# use loader to load documents
loader = ObsidianLoader(os.getenv("OBSIDIAN", ""))
docs = loader.load()

print(len(docs))

# print out document content & metadata
print(f"{docs[0].page_content[:200]}\n")  # print out first 200 chars
print(docs[0].metadata)

# text splitting
# split markdown by headers (https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/)
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
    # this returns a list of Documents, which can be split on again
    md_splits.extend(md_splitter.split_text(doc.page_content))
print(len(md_splits))

# text splitter for each markdown chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# split all markdown chunks
splits = text_splitter.split_documents(md_splits)
print(len(splits))

# embed chunks
embeddings = OllamaEmbeddings(
    model="llama3.2:3b"  # model has the format of {name}:{tag}
)
vector_1 = embeddings.embed_query(splits[0].page_content)
vector_2 = embeddings.embed_query(splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])

# store embeddings in qdrant (https://python.langchain.com/docs/integrations/vectorstores/qdrant/)
# init from documents
# qdrant = QdrantVectorStore.from_documents(
#    splits,
#    embeddings,
#    url=QDRANT_URL,
#    prefer_grpc=True,
#    grpc_port=QDRANT_GRPC_PORT,  # need to specify this, otherwise it connects to the default grpc port
#    collection_name="obsidian",
# )

# use existing qdrant collection
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="obsidian",
    url=QDRANT_URL,
    prefer_grpc=True,
    grpc_port=QDRANT_GRPC_PORT,
)
