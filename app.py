import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
LLM_PROVIDER = "gemini"
LLM_MODEL = "openai/gpt-oss-20b"
CORPUS_PATH = "./knowledge_base/"   
load_dotenv()  # local dev: reads a .env file if present

try:
    for key in ("GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "LANGCHAIN_API_KEY"):
        if key in st.secrets and not os.environ.get(key):
            os.environ[key] = st.secrets[key]
except Exception:
    pass  # no secrets.toml present — fine when running purely from .env

st.set_page_config(page_title="HR RAG Assistant", page_icon="\U0001F4BC")

if not os.getenv("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY not found.\n\n"
        "- Local run: add a `.env` file next to `app.py` with `GROQ_API_KEY=your_key_here`.\n"
        "- Streamlit Cloud: add `GROQ_API_KEY` under your app's Settings -> Secrets."
    )
    st.stop()

# Pipeline setup (cached — runs once per session, not on every question)
@st.cache_resource(show_spinner="Loading knowledge base and building the vector store...")
def load_pipeline():
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(model=LLM_MODEL, temperature=0.1, max_tokens=512)

    return retriever, llm


retriever, llm = load_pipeline()

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an HR assistant. Answer the question using ONLY
the context below. If the answer isn't in the context,
say you don't have that information.

Context: {context}
Question: {question}""")

GUARDRAIL_PROMPT = ChatPromptTemplate.from_template("""
You are a scope classifier for an HR assistant. Decide whether the question
below is something an HR assistant should answer (company leave policy,
reimbursement, code of conduct, or other internal HR topics) or something
out of scope (general knowledge, coding help, unrelated small talk, etc).

Respond with exactly one word: IN_SCOPE or OUT_OF_SCOPE.

Question: {question}
""")

REFUSAL_MESSAGE = (
    "I'm an HR assistant and can only help with questions about company HR "
    "policies (leave, reimbursement, code of conduct, etc.). I don't have "
    "information to answer that question."
)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def rag_chain(question: str):
    docs = retriever.invoke(question)
    context = format_docs(docs)
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return {"answer": answer, "sources": docs}


def ask_bot(question: str):
    guardrail_chain = GUARDRAIL_PROMPT | llm | StrOutputParser()
    verdict = guardrail_chain.invoke({"question": question}).strip().upper()

    if "OUT_OF_SCOPE" in verdict:
        return {"answer": REFUSAL_MESSAGE, "sources": []}

    return rag_chain(question)

# UI

st.title("HR RAG Assistant")
st.caption("Ask a question about company leave, reimbursement, or conduct policies.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_bot(question)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources"):
                for i, doc in enumerate(result["sources"], 1):
                    src = doc.metadata.get("source", "unknown")
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**{i}.** `{src}` (page {page})")
                    st.caption(doc.page_content[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
