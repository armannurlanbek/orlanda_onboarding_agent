from langchain_openai import OpenAIEmbeddings
import tiktoken
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import pprint
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

question = "Когда фасад должен быть откорректирован?"
document = """
В случае если при проектировании были предоставлены замеры той или иной
части фасада в недопустимой области при сканировании, замеры должны быть
нанесены на ту или иную часть фасада, фасад должен быть откорректирован в
соответствие предоставленным замерам.
"""

def num_of_tokens(string: str, encoding_name: str) -> int:
    "Returns the number of tokens in a text string."
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

print(f"Number of tokens in question: {question} is {num_of_tokens(question, 'gpt-4o-mini')}")

embd = OpenAIEmbeddings(model = "text-embedding-3-small")
query_result = embd.embed_query(question)
document_result = embd.embed_query(document)
print(len(query_result))

# Cosine similarity

import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

similarity = cosine_similarity(query_result, document_result)
print(f"Cosine similarity between query and document: {similarity}")


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
