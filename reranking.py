from dotenv import load_dotenv
import os
# DOCUMENT LOADER
from langchain_community.document_loaders import PyPDFLoader
# TEXT SPLITTER
from langchain_text_splitters import RecursiveCharacterTextSplitter
# VECTOR DB + EMBEDDINGS
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# RETRIEVERS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever
)
# RERANKER
from langchain_classic.retrievers.document_compressors import (
    CrossEncoderReranker
)
from langchain_community.cross_encoders import (
    HuggingFaceCrossEncoder
)
# LLM
from langchain_groq import ChatGroq
# CHAIN
from langchain_classic.chains import RetrievalQA
# PROMPT
from langchain_core.prompts import PromptTemplate
load_dotenv()
loader = PyPDFLoader("PYTHON PROGRAMMING NOTES.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents) 
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)
vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)
cross_encoder_model = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
compressor = CrossEncoderReranker(
    model=cross_encoder_model,
    top_n=3
)
rerank_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=hybrid_retriever
)

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="qwen/qwen3-32b"
)
prompt_template = """
Answer using only the provided context.

Rules:
- Give short and concise answer.
- Avoid unnecessary technical details.
- Keep answer exam-oriented.

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
query = "What is python?"
retrieved_docs = rerank_retriever.invoke(query)
print("\nRetrieved Chunks:\n")
for i, doc in enumerate(retrieved_docs):
    print(f"\nChunk {i+1}\n")
    print(doc.page_content)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=rerank_retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT}
)
response = qa_chain.invoke({"query": query})
print("\nFinal Answer:\n")
print(response["result"])   