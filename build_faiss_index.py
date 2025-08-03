from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# 1. Load all .txt files from docs folder
loader = DirectoryLoader('docs', glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()

print(f"Loaded {len(docs)} documents")

# 2. Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Build FAISS index
vector_store = FAISS.from_documents(docs, embeddings)

# 4. Save locally
vector_store.save_local("faiss_index")

print("✅ FAISS index built and saved to faiss_index/")
