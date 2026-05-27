from langchain_community.document_loaders import PyPDFLoader  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain.embeddings import HuggingFaceEmbeddings 
# load pdf 
loader = PyPDFLoader("FAQ_materiality.pdf") 
documents = loader.load() 
# split text 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)    
docs = text_splitter.split_documents(documents) 
# embedding model 
embedding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2" 
)   
# store in ChromaDB 
vectorstore = Chroma.from_documents(
    documents = PyPDFLoader, 
    embedding = embedding_model, 
    persist_directory="./chroma_db"
)  
vectorstore.persist() 
print("PDF processed successfully!")  