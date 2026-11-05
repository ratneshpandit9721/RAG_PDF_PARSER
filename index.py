from dotenv import load_dotenv
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader


load_dotenv()

pdf_path = Path(__file__).parent / "nodejs.pdf"
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

# Chunkings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
split_docs = text_splitter.split_documents(documents=docs)


embedding_models = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url="http://vector-db:6333",
    collection_name="RAG_Project",
    embedding=embedding_models,
)

print("Indexing of documents done...")
