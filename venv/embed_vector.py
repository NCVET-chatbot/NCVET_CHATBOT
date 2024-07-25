import pickle
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

# Initialize components
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
loader = PyPDFDirectoryLoader("data")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(docs)

# Generate embeddings and create FAISS index
vectors = FAISS.from_documents(final_documents, embeddings)

# Save embeddings and FAISS index
vectors.save_local("faiss_index")
with open('documents.pkl', 'wb') as f:
    pickle.dump(final_documents, f)
