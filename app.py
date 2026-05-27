
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
import os

# Load .env
load_dotenv()

# Load PDF
loader = PyPDFLoader("AIAgents.pdf")
documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store in ChromaDB
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

# Save database
vectorstore.persist()

print("PDF processed successfully!")

# LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="qwen/qwen3-32b"
)

# Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

query = "what is AI Agents?"

# Retrieve top chunks
docs = retriever.invoke(query)

print("Retrieved Chunks:\n")

for i, doc in enumerate(docs):
    print(f"\nChunk {i+1}")
    print(doc.page_content)

# RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# Final Answer
response = qa_chain.invoke({"query": query})

print("\nFinal Answer:\n")
print(response["result"])