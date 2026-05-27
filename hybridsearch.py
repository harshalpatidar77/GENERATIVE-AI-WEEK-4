from dotenv import load_dotenv 
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
import os  
# load env 
load_dotenv() 
# LOAD PDF
loader = PyPDFLoader("AIAgents.pdf")
documents = loader.load()  
# SPLIT DOCUMENTS
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents) 
# EMBEDDING MODEL
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
) 
# VECTOR DATABASE
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./chroma_db"
) 
# VECTOR RETRIEVER
vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
) 
# BM25 RETRIEVER
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3
# HYBRID SEARCH
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]
) 
# LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="qwen/qwen3-32b"
) 
# QUERY
query = "What are  AI agents?" 
# RETRIEVE DOCUMENTS
retrieved_docs = hybrid_retriever.invoke(query)
print("Retrieved Chunks:\n") 
for i, doc in enumerate(retrieved_docs):
    print(f"\nChunk {i+1}")
    print(doc.page_content)
# RAG CHAIN
from langchain_core.prompts import PromptTemplate

# CUSTOM PROMPT
prompt_template = """
Use only the provided context to answer the question.

Give a concise and exam-oriented answer.
Keep the answer short and avoid unnecessary technical details.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# RAG CHAIN
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=hybrid_retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT}
)
# FINAL ANSWER
response = qa_chain.invoke({"query": query})
print("\nFinal Answer:\n")
print(response["result"])    