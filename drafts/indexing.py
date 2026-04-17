from langchain_openai import OpenAIEmbeddings
import tiktoken
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import pprint
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

load_dotenv()

embd = OpenAIEmbeddings(model = "text-embedding-3-small")
# Indexing
knowledge_base_dir = Path("./knowledge_base")
pdf_paths = list(knowledge_base_dir.glob("*.pdf"))

all_documents = []
for path in pdf_paths:
    all_documents.extend(PyPDFLoader(path).load())

print(len(all_documents))
pprint.pp(all_documents[0].metadata)

# Splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    add_start_index = True,
)

all_splits = text_splitter.split_documents(all_documents)
print(f"Split {len(all_splits)} chunks")

# Vector Store
vector_store = InMemoryVectorStore(embd)
document_ids = vector_store.add_documents(documents = all_splits)
print(document_ids)

#Retrieval

retriever = vector_store.as_retriever(search_kwargs={"k": 7})
docs = retriever.invoke("Планета Обезьян")
print(len(docs))
