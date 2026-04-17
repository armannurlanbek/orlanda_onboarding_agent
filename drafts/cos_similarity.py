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

