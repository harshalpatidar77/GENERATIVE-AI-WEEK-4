# Call Summarization Project
# STEP 1 : Load PDF + Chunking
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
loader = PyPDFLoader("callsummarization.pdf")   
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=500 
)
chunks = splitter.split_documents(docs)
print("Total Chunks:", len(chunks))
# STEP 2 : LLM Setup
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import json
load_dotenv()
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="qwen/qwen3-32b"
)
prompt = ChatPromptTemplate.from_template("""
You are a call summarizer AI.
Return ONLY pure valid JSON.
Do not add explanation.
Do not add markdown.
Do not add ```json.
Format:
{{
    "short_summary": "",
    "customer_issue": "",
    "sentiment": "",
    "category": "",
    "important_keywords": []
}}
Transcript:
{transcript}
""")
# STEP 3 : Create Chain
chain = prompt | llm
all_summaries = []
# STEP 4 : Generate JSON Summaries
for i, chunk in enumerate(chunks):
    print(f"\nProcessing Chunk {i+1}...\n")
    response = chain.invoke({
        "transcript": chunk.page_content
    })
    try:
        cleaned_response = response.content.strip()

        # Extract JSON only
        start = cleaned_response.find("{")
        end = cleaned_response.rfind("}") + 1

        json_string = cleaned_response[start:end]

        json_output = json.loads(json_string)

        all_summaries.append(json_output)

        print(json.dumps(json_output, indent=4))
    except Exception as e:
        print("JSON Parsing Error")
        print(e)
        print("\nRAW RESPONSE:\n")
        print(response.content)
# FINAL JSON
print("\n========== FINAL JSON ==========\n")
print(json.dumps(all_summaries, indent=4))
# SAVE JSON FILE
with open("call_summaries.json", "w", encoding="utf-8") as json_file:
    json.dump(all_summaries, json_file, indent=4, ensure_ascii=False)
print("\nJSON file saved successfully")    
# STEP 5 : Create Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# STEP 6 : Prepare Documents
from langchain_core.documents import Document
documents = []
for item in all_summaries:
    doc = Document(
        page_content=item["short_summary"],
        metadata={
            "customer_issue": item["customer_issue"],
            "sentiment": item["sentiment"],
            "category": item["category"],
            "keywords": ",".join(item["important_keywords"])
        }
    )
documents.append(doc)
# STEP 7 : Store in ChromaDB
from langchain_chroma import Chroma
vector_db = Chroma.from_documents(

    documents=documents,
    embedding=embedding_model,
    persist_directory="chroma_db"
)
print("\nVector Database Stored Successfully")   
# STEP 8 : Retrieval Query
query = "customer complaining about product failure"
results = vector_db.similarity_search(
    query,
    k=2
)
print("\n========== RETRIEVAL RESULTS ==========\n")
for i, result in enumerate(results):
    print(f"\nRESULT {i+1}\n")
    print("SUMMARY:")
    print(result.page_content)
    print("\nMETADATA:")
    print(result.metadata)        
