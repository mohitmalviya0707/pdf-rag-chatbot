import os
import hashlib
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langdetect import detect
from deep_translator import GoogleTranslator
from gtts import gTTS

load_dotenv()

st.title("📄🤖 Advanced RAG Chatbot (Mohit AI)")

# ─── Cached resources ─────────────────────────────────────────────────────────

@st.cache_resource
def load_embedding():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return ChatMistralAI(model="mistral-small-latest")

# ─── Build / cache vectorstore per unique PDF ─────────────────────────────────

@st.cache_resource(show_spinner="Building vector index…")
def build_vectorstore(file_hash: str, file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200
    )
    docs = splitter.split_documents(documents)
    docs = [d for d in docs if d.page_content.strip()]

    if not docs:
        return None

    embeddings = load_embedding()
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10},
    )

# ─── Session state ────────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

# ─── Prompt template ──────────────────────────────────────────────────────────

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer ONLY from the given context.\n"
     "If the answer is not in the context, say: I could not find the answer."),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
])

# ─── PDF Upload ───────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

retriever = None

if uploaded_file:
    raw_bytes = uploaded_file.read()
    file_hash = hashlib.md5(raw_bytes).hexdigest()

    # Only re-process if a new/different PDF is uploaded
    if st.session_state.file_hash != file_hash:
        if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
            os.unlink(st.session_state.pdf_path)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(raw_bytes)
        tmp.close()
        st.session_state.pdf_path = tmp.name
        st.session_state.file_hash = file_hash

    retriever = build_vectorstore(file_hash, st.session_state.pdf_path)

    if retriever is None:
        st.error("❌ No readable text found in the PDF.")
        st.stop()
    else:
        st.success("✅ PDF ready!")

# ─── Chat history display ─────────────────────────────────────────────────────

for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["user"])
    with st.chat_message("assistant"):
        st.write(chat["ai"])

# ─── Query input ──────────────────────────────────────────────────────────────

query = st.text_input("Ask your question:")

if query and retriever:
    original_query = query
    llm = load_llm()

    # Detect language & translate to English if Hindi
    try:
        detected_lang = detect(query)
    except Exception:
        detected_lang = "en"

    if detected_lang == "hi":
        try:
            query = GoogleTranslator(source="hi", target="en").translate(query)
        except Exception:
            pass  # If translation fails, use original query

    with st.spinner("🔍 Searching document…"):
        retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        answer = "❌ No relevant content found in the document."
        answer_hi = answer
    else:
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        with st.spinner("🤖 Generating answer…"):
            final_prompt = prompt.invoke({"context": context, "question": query})
            response = llm.invoke(final_prompt)
            answer = response.content

        # Translate answer to Hindi
        try:
            answer_hi = GoogleTranslator(source="en", target="hi").translate(answer)
        except Exception:
            answer_hi = answer

        with st.expander("📄 Retrieved source chunks"):
            st.write(context[:800])

    # Save to history
    st.session_state.chat_history.append({"user": original_query, "ai": answer})

    # Display answers
    st.write("### 🤖 Answer (English):")
    st.write(answer)

    st.write("### 🇮🇳 Answer (Hindi):")
    st.write(answer_hi)

    # Voice output
    try:
        tts = gTTS(text=answer_hi, lang="hi", slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
            tts.save(audio_file.name)
            st.audio(audio_file.name)
    except Exception as e:
        st.warning(f"🔊 Audio generation failed: {e}")

elif query and not retriever:
    st.warning("⚠️ Please upload a PDF first.")


