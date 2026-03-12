import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
INDEX_NAME = "meetings"  # single index, namespaces per meeting

def _get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def _get_or_create_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(INDEX_NAME)

def _build_chain(retriever):
    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
    prompt = ChatPromptTemplate.from_template(
        "Use the meeting transcript below to answer the question.\n\n"
        "Transcript:\n{context}\n\n"
        "Question: {question}"
    )
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def get_all_meetings() -> list[str]:
    try:
        index = _get_or_create_index()
        stats = index.describe_index_stats()
        return sorted([ns for ns in stats.namespaces.keys() if ns.startswith("meeting-")])
    except Exception:
        return []

def store_meeting(transcript: list[dict]) -> str:
    _get_or_create_index()
    embeddings = _get_embeddings()
    namespace = f"meeting-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    docs = [
        f"[{seg['speaker']} | {seg['start']}s - {seg['end']}s]: {seg['text']}"
        for seg in transcript
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents(["\n".join(docs)])

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=namespace
    )
    return namespace

def query_meeting(namespace: str, question: str) -> str:
    embeddings = _get_embeddings()
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=namespace
    )
    chain = _build_chain(vectorstore.as_retriever(search_kwargs={"k": 4}))
    return chain.invoke(question)

def query_all_meetings(question: str) -> str:
    meetings = get_all_meetings()
    if not meetings:
        return "No meetings found."

    embeddings = _get_embeddings()
    all_answers = []
    for namespace in meetings:
        vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            namespace=namespace
        )
        chain = _build_chain(vectorstore.as_retriever(search_kwargs={"k": 3}))
        result = chain.invoke(question)
        all_answers.append(f"**{namespace}:**\n{result}")

    return "\n\n".join(all_answers)