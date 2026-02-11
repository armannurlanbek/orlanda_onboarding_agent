from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")

vector_store = InMemoryVectorStore(embeddings)

file_path = "./knowledge_base/PLANS.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    add_start_index = True,
)
all_splits = text_splitter.split_documents(docs)

print(f"Split {len(all_splits)} chunks")

document_ids = vector_store.add_documents(documents = all_splits)
print(document_ids)