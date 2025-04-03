from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

faiss_index_path = "./faiss_index"
vector_db = FAISS.load_local(faiss_index_path, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)

docs = vector_db.similarity_search("안녕하세요", k=3)

for doc in docs:
    print(doc.page_content)