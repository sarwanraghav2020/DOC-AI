"""
=========================================================
DOC AI
---------------------------------------------------------
Intelligent Medical Report Assistant

Features
---------
• Upload Medical Reports
• Google Gemini & Ollama Support
• FAISS & ChromaDB
• Multiple Embedding Models
• Medical Report Summary
• Medical Q&A
• Explain Medical Terms
• Questions for Doctor
• Retrieved Context Viewer

Author : Himanshu Rajak
=========================================================
"""

# =========================================================
# Imports
# =========================================================

import os
import time
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils import (
    PDFLoader,
    DocumentSplitter,
    EmbeddingModel,
    VectorStore,
    LLMManager,
    PromptManager,
    RAGChain,
)

# =========================================================
# Load Environment Variables
# =========================================================

load_dotenv()

# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="DOC AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# Custom CSS
# =========================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* Main App - Premium Blue Night */
.stApp {
    background: radial-gradient(circle at top left, rgba(69, 128, 255, 0.18), transparent 28%),
                linear-gradient(180deg, #050b26 0%, #0d1742 45%, #0a1437 100%);
    color: #f5f7ff;
}

/* Smooth page container */
.block-container,
element-container,
.stMarkdown,
.css-1outpf7,
.css-1lcbmhc {
    background: transparent !important;
}

/* Sidebar styling */
[data-testid='stSidebar'] {
    background: rgba(8, 20, 61, 0.96) !important;
    border-right: 1px solid rgba(99, 132, 255, 0.20);
    color: #eef2ff !important;
}

[data-testid='stSidebar'] h2,
[data-testid='stSidebar'] h3,
[data-testid='stSidebar'] h4,
[data-testid='stSidebar'] strong,
[data-testid='stSidebar'] b {
    color: #eef2ff !important;
}

[data-testid='stSidebar'] .stTextInput>label,
[data-testid='stSidebar'] .stTextArea>label,
[data-testid='stSidebar'] .stSelectbox>label,
[data-testid='stSidebar'] .stSlider>label,
[data-testid='stSidebar'] .stFileUploader>label,
[data-testid='stSidebar'] .stNumberInput>label,
[data-testid='stSidebar'] .stMultiSelect>label,
[data-testid='stSidebar'] .stCheckbox>label,
[data-testid='stSidebar'] .stRadio>label {
    color: #0f213d !important;
}

[data-testid='stSidebar'] .stSelectbox>div>div>div>div,
[data-testid='stSidebar'] .stSelectbox>div>div>div>div span,
[data-testid='stSidebar'] .stSelectbox>div>div>div>div div,
[data-testid='stSidebar'] .stMultiSelect>div>div,
[data-testid='stSidebar'] .stFileUploader {
    color: #0f213d !important;
}

[data-testid='stSidebar'] .stMarkdown,
[data-testid='stSidebar'] .stMarkdown span,
[data-testid='stSidebar'] .stMarkdown p,
[data-testid='stSidebar'] .stMarkdown strong {
    color: #eef2ff !important;
}

[data-testid='stSidebar'] .stButton>button,
[data-testid='stSidebar'] .stButton>button:hover {
    background: linear-gradient(135deg, #4d7dff 0%, #2f5bde 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(140, 170, 255, 0.35) !important;
}

[data-testid='stSidebar'] .stTextInput>div>input,
[data-testid='stSidebar'] .stTextArea>div>textarea,
[data-testid='stSidebar'] .stSelectbox>div>div>div>div,
[data-testid='stSidebar'] .stFileUploader,
[data-testid='stSidebar'] .stNumberInput>div>input,
[data-testid='stSidebar'] .stMultiSelect>div>div,
[data-testid='stSidebar'] .stCheckbox>div,
[data-testid='stSidebar'] .stRadio>div {
    background: rgba(255,255,255,0.12) !important;
    color: #0f213d !important;
    border: 1px solid rgba(120, 143, 255, 0.24) !important;
    border-radius: 16px !important;
}

[data-testid='stSidebar'] .stTextInput>label,
[data-testid='stSidebar'] .stTextArea>label,
[data-testid='stSidebar'] .stSelectbox>label,
[data-testid='stSidebar'] .stSlider>label,
[data-testid='stSidebar'] .stFileUploader>label {
    color: #0f213d !important;
}

[data-testid='stSidebar'] .stSelectbox>div>div>div>div,
[data-testid='stSidebar'] .stSelectbox>div>div>div>div span,
[data-testid='stSidebar'] .stSelectbox>div>div>div>div div,
[data-testid='stSidebar'] .stMultiSelect>div>div,
[data-testid='stSidebar'] .stFileUploader {
    color: #0f213d !important;
}

[data-testid='stSidebar'] .stTextInput>div>input::placeholder,
[data-testid='stSidebar'] .stTextArea>div>textarea::placeholder,
[data-testid='stSidebar'] textarea::placeholder,
[data-testid='stSidebar'] input::placeholder {
    color: rgba(15, 33, 61, 0.55) !important;
}

[data-testid='stSidebar'] button,
[data-testid='stSidebar'] input,
[data-testid='stSidebar'] textarea,
[data-testid='stSidebar'] select {
    color: #eef2ff !important;
}

.stButton>button {
    background: linear-gradient(135deg, #4f7bff 0%, #2f58db 100%) !important;
    border: 1px solid rgba(132, 162, 255, 0.32) !important;
    border-radius: 16px !important;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #6079ff 0%, #3a5fe3 100%) !important;
}

.stTextInput>div>input,
.stTextArea>div>textarea,
.stSelectbox>div>div>div>div,
.stFileUploader,
.stNumberInput>div>input,
.stMultiSelect>div>div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(120, 143, 255, 0.22) !important;
    border-radius: 16px !important;
    color: #eef2ff !important;
}

.stTextInput>div>input::placeholder,
.stTextArea>div>textarea::placeholder,
textarea::placeholder,
input::placeholder {
    color: rgba(235, 242, 255, 0.55) !important;
}

.stTextInput>div>input:focus,
.stTextArea>div>textarea:focus,
.stSelectbox>div>div>div>div:focus,
.stNumberInput>div>input:focus,
input:focus,
textarea:focus,
select:focus {
    outline: 2px solid rgba(116, 157, 255, 0.55) !important;
    outline-offset: 2px;
}

/* Hero block */
.hero-block {
    background: linear-gradient(135deg, #11216b 0%, #0f1854 42%, #08112e 100%);
    border: 1px solid rgba(57, 96, 255, 0.28);
    border-radius: 32px;
    padding: 44px 36px;
    margin: 0 auto 34px;
    max-width: 1100px;
    box-shadow: 0 40px 120px rgba(0, 0, 0, 0.24);
    position: relative;
    overflow: hidden;
}

.hero-block::before {
    content: '';
    position: absolute;
    top: -12%;
    right: -10%;
    width: 260px;
    height: 260px;
    background: radial-gradient(circle, rgba(0, 128, 255, 0.20), transparent 58%);
    filter: blur(28px);
}

.hero-block::after {
    content: '';
    position: absolute;
    bottom: -14%;
    left: -8%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(65, 160, 255, 0.12), transparent 60%);
    filter: blur(34px);
}

.main-title {
    text-align: center;
    font-size: 44px;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.05;
    margin-bottom: 14px;
    background: linear-gradient(135deg, #8bcdff 0%, #d1e8ff 48%, #b3d5ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #c8d5f2;
    font-size: 16px;
    font-weight: 400;
    line-height: 1.8;
    letter-spacing: 0.15px;
    max-width: 860px;
    margin: 0 auto;
}

.subtitle strong,
.subtitle b {
    color: #e5edff;
}

/* Card - glassmorphism effect */
.card {
    background: rgba(16, 26, 70, 0.72);
    padding: 28px;
    border-radius: 22px;
    border: 1px solid rgba(97, 131, 255, 0.18);
    backdrop-filter: blur(18px);
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18);
    color: #eef2ff;
    transition: transform 0.35s ease, border-color 0.35s ease;
}

.card:hover {
    transform: translateY(-2px);
    border-color: rgba(97, 131, 255, 0.30);
}

.card h2,
.card h3 {
    color: #ffffff;
}

/* Headings and text */
h1, h2, h3 {
    font-weight: 700;
    letter-spacing: -0.01em;
}

p, span, a, li {
    color: #d8e0ff;
}

.stTextInput>div>input,
.stTextArea>div>textarea,
.stSelectbox>div>div>div>div,
.stFileUploader,
.stNumberInput>div>input,
.stMultiSelect>div>div,
button {
    font-size: 15px !important;
}

/* Divider style */
.stDivider {
    border-color: rgba(110, 145, 255, 0.18) !important;
}

</style>
""",
    unsafe_allow_html=True
)

# =========================================================
# Header
# =========================================================

st.markdown(
    """
<div class="hero-block">
  <div class="main-title">🩺 DOC AI</div>
  <div class="subtitle">
    Intelligent Medical Report Assistant for medical professionals, powered by
    <strong>LangChain • Google Gemini • Ollama • FAISS • ChromaDB</strong>
  </div>
</div>
""",
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# Sidebar
# =========================================================

st.sidebar.title("⚙️ Configuration")

st.sidebar.divider()

# ---------------------------------------------------------
# Upload Medical Report
# ---------------------------------------------------------

uploaded_pdf = st.sidebar.file_uploader(
    "📄 Upload Medical Report",
    type=["pdf"]
)

# ---------------------------------------------------------
# LLM Provider
# ---------------------------------------------------------

provider = st.sidebar.selectbox(
    "🤖 LLM Provider",
    LLMManager.available_providers()
)

# ---------------------------------------------------------
# LLM Model
# ---------------------------------------------------------

model_name = st.sidebar.selectbox(
    "🧠 Model",
    LLMManager.available_models(provider)
)

# ---------------------------------------------------------
# Vector Database
# ---------------------------------------------------------

vector_db = st.sidebar.selectbox(
    "🗄️ Vector Database",
    [
        "FAISS",
        "ChromaDB"
    ]
)

# ---------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------

embedding_model = st.sidebar.selectbox(
    "🧠 Embedding Model",
    EmbeddingModel.available_models()
)

# ---------------------------------------------------------
# Chunk Size
# ---------------------------------------------------------

chunk_size = st.sidebar.slider(
    "Chunk Size",
    min_value=500,
    max_value=2000,
    value=1000,
    step=100
)

# ---------------------------------------------------------
# Chunk Overlap
# ---------------------------------------------------------

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    min_value=0,
    max_value=500,
    value=200,
    step=50
)

# ---------------------------------------------------------
# Top K
# ---------------------------------------------------------

top_k = st.sidebar.slider(
    "Top K Retrieval",
    min_value=1,
    max_value=10,
    value=3
)

st.sidebar.divider()

# ---------------------------------------------------------
# Buttons
# ---------------------------------------------------------

build_database = st.sidebar.button(
    "🚀 Build Database",
    use_container_width=True
)

clear_chat = st.sidebar.button(
    "🧹 Clear Chat",
    use_container_width=True
)

delete_database = st.sidebar.button(
    "🗑 Delete Database",
    use_container_width=True
)

st.sidebar.divider()

# ---------------------------------------------------------
# Project Information
# ---------------------------------------------------------

st.sidebar.info(
    """
### 🩺 DOC AI

**Version :** 1.0

Supports:

- Google Gemini
- Ollama
- FAISS
- ChromaDB
- Multiple Embedding Models

Educational Use Only
"""
)


# =========================================================
# Session State Initialization
# =========================================================

# ---------------------------------------------------------
# RAG Chain
# ---------------------------------------------------------

if "rag" not in st.session_state:
    st.session_state.rag = None

# ---------------------------------------------------------
# Uploaded Documents
# ---------------------------------------------------------

if "documents" not in st.session_state:
    st.session_state.documents = []

# ---------------------------------------------------------
# Split Chunks
# ---------------------------------------------------------

if "chunks" not in st.session_state:
    st.session_state.chunks = []

# ---------------------------------------------------------
# Retrieved Documents
# ---------------------------------------------------------

if "retrieved_docs" not in st.session_state:
    st.session_state.retrieved_docs = []

# ---------------------------------------------------------
# Chat History
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------
# Database Ready Flag
# ---------------------------------------------------------

if "database_ready" not in st.session_state:
    st.session_state.database_ready = False

# ---------------------------------------------------------
# Current PDF Name
# ---------------------------------------------------------

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

# ---------------------------------------------------------
# Current Vector Database
# ---------------------------------------------------------

if "vector_db" not in st.session_state:
    st.session_state.vector_db = ""

# ---------------------------------------------------------
# Current Embedding Model
# ---------------------------------------------------------

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = ""

# ---------------------------------------------------------
# Current LLM Provider
# ---------------------------------------------------------

if "provider" not in st.session_state:
    st.session_state.provider = ""

# ---------------------------------------------------------
# Current LLM Model
# ---------------------------------------------------------

if "llm_model" not in st.session_state:
    st.session_state.llm_model = ""

# ---------------------------------------------------------
# Report Summary
# ---------------------------------------------------------

if "report_summary" not in st.session_state:
    st.session_state.report_summary = ""

# ---------------------------------------------------------
# Questions for Doctor
# ---------------------------------------------------------

if "doctor_questions" not in st.session_state:
    st.session_state.doctor_questions = ""


# =========================================================
# Build Vector Database
# =========================================================

if build_database:

    # -----------------------------------------------------
    # Check PDF Upload
    # -----------------------------------------------------

    if uploaded_pdf is None:

        st.warning("Please upload a medical report first.")

    else:

        # -------------------------------------------------
        # Save Uploaded PDF
        # -------------------------------------------------

        data_folder = Path("data")
        data_folder.mkdir(exist_ok=True)

        # Remove previous reports
        for file in data_folder.glob("*.pdf"):
            file.unlink()

        pdf_path = data_folder / uploaded_pdf.name

        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        # -------------------------------------------------
        # Progress Bar
        # -------------------------------------------------

        progress = st.progress(0)

        status = st.empty()

        try:

            # =============================================
            # Step 1 : Load PDF
            # =============================================

            status.info("📄 Loading Medical Report...")

            loader = PDFLoader(str(pdf_path))

            documents = loader.load()

            progress.progress(15)

            # =============================================
            # Step 2 : Split Report
            # =============================================

            status.info("✂ Splitting Report into Chunks...")

            splitter = DocumentSplitter(

                chunk_size=chunk_size,

                chunk_overlap=chunk_overlap

            )

            chunks = splitter.split(documents)

            progress.progress(35)

            # =============================================
            # Step 3 : Embeddings
            # =============================================

            status.info(" Loading Embedding Model...")

            embedding = EmbeddingModel(

                embedding_model

            ).get_embedding_model()

            progress.progress(55)

            # =============================================
            # Step 4 : Vector Database
            # =============================================

            status.info(
                f"🗄 Creating {vector_db} Database..."
            )

            vectorstore = VectorStore(

                embedding

            )

            db = vectorstore.create(

                chunks,

                db_type=vector_db

            )

            progress.progress(75)

            # =============================================
            # Step 5 : Initialize RAG
            # =============================================

            status.info("🤖 Initializing AI Assistant...")

            rag = RAGChain(

                vector_db=db,

                provider=provider,

                model_name=model_name

            )

            progress.progress(100)

            # =============================================
            # Save Session
            # =============================================

            st.session_state.rag = rag

            st.session_state.documents = documents

            st.session_state.chunks = chunks

            st.session_state.database_ready = True

            st.session_state.pdf_name = uploaded_pdf.name

            st.session_state.vector_db = vector_db

            st.session_state.embedding_model = embedding_model

            st.session_state.provider = provider

            st.session_state.llm_model = model_name

            status.success(
                "DOC AI is Ready!"
            )

            st.success(
                f"Successfully processed {uploaded_pdf.name}"
            )

            st.balloons()

            progress.empty()

        except Exception as e:

            st.error(f"Error : {e}")

            progress.empty()

# =========================================================
# Document Statistics
# =========================================================

if st.session_state.database_ready:

    st.divider()

    st.subheader("📊 Document Statistics")

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(

            label="📄 Pages",

            value=len(st.session_state.documents)

        )

    with col2:

        st.metric(

            label="✂️ Chunks",

            value=len(st.session_state.chunks)

        )

    with col3:

        st.metric(

            label="🗄️ Vector DB",

            value=st.session_state.vector_db

        )

    with col4:

        st.metric(

            label="🤖 LLM",

            value=st.session_state.provider

        )

    st.divider()

    # -----------------------------------------------------
    # Additional Information
    # -----------------------------------------------------

    col5, col6 = st.columns(2)

    with col5:

        st.info(
f"""
### 📄 Medical Report

**File Name**

{st.session_state.pdf_name}

"""
        )

    with col6:

        st.info(
f"""
### ⚙️ Configuration

**Embedding Model**

{st.session_state.embedding_model}

**LLM Model**

{st.session_state.llm_model}

"""
        )


# =========================================================
# Medical Report Summary
# =========================================================

st.divider()

st.subheader("📄 Medical Report Summary")

st.caption(
    "Generate a simple summary of the uploaded medical report."
)

generate_summary = st.button(
    "📋 Generate Summary",
    use_container_width=True
)

if generate_summary:

    if not st.session_state.database_ready:

        st.warning(
            "Please build the database first."
        )

    else:

        with st.spinner(
            "Generating report summary..."
        ):

            try:

                summary = st.session_state.rag.summarize()

                st.session_state.report_summary = summary

            except Exception as e:

                st.error(e)

# ---------------------------------------------------------
# Display Summary
# ---------------------------------------------------------

if st.session_state.report_summary:

    st.success("Summary Generated Successfully")

    st.write(st.session_state.report_summary)


# =========================================================
# Medical Term Explanation
# =========================================================

st.divider()

st.subheader("🩺 Medical Term Explanation")

st.caption(
    "Enter a medical term to understand its meaning in simple language."
)

# ---------------------------------------------------------
# Medical Term Input
# ---------------------------------------------------------

medical_term = st.text_input(
    "Medical Term",
    placeholder="Example: Hemoglobin, RBC, Creatinine..."
)

# ---------------------------------------------------------
# Explain Button
# ---------------------------------------------------------

explain_term = st.button(
    "🔍 Explain Medical Term",
    use_container_width=True
)

# ---------------------------------------------------------
# Generate Explanation
# ---------------------------------------------------------

if explain_term:

    if not st.session_state.database_ready:

        st.warning(
            "Please build the database first."
        )

    elif medical_term.strip() == "":

        st.warning(
            "Please enter a medical term."
        )

    else:

        with st.spinner(
            "Searching medical report..."
        ):

            try:

                answer, docs = st.session_state.rag.explain_term(
                    medical_term,
                    k=top_k
                )

                st.session_state.retrieved_docs = docs

                st.success("Explanation Generated")

                st.write(answer)

            except Exception as e:

                st.error(e)

# =========================================================
# Medical Q&A
# =========================================================

st.divider()

st.subheader("💬 Chat with Medical Report")

st.caption(
    "Ask any question related to the uploaded medical report."
)

# ---------------------------------------------------------
# Question Input
# ---------------------------------------------------------

question = st.text_input(
    "Ask your question",
    placeholder="Example: What is Hemoglobin? Explain my cholesterol level."
)

# ---------------------------------------------------------
# Ask Button
# ---------------------------------------------------------

ask_question = st.button(
    "🤖 Ask AI",
    use_container_width=True
)

# ---------------------------------------------------------
# Generate Answer
# ---------------------------------------------------------

if ask_question:

    if not st.session_state.database_ready:

        st.warning(
            "Please build the database first."
        )

    elif question.strip() == "":

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Analyzing medical report..."
        ):

            try:

                start = time.time()

                answer, docs = st.session_state.rag.ask(

                    question,

                    k=top_k

                )

                end = time.time()

                response_time = end - start

                # Save Retrieved Documents

                st.session_state.retrieved_docs = docs

                # Save Chat History

                st.session_state.messages.append(

                    {

                        "role": "user",

                        "content": question

                    }

                )

                st.session_state.messages.append(

                    {

                        "role": "assistant",

                        "content": answer

                    }

                )

                st.success("Answer")

                st.write(answer)

                st.caption(

                    f"⚡ Response Time : {response_time:.2f} sec"

                )

            except Exception as e:

                st.error(e)

# ---------------------------------------------------------
# Chat History
# ---------------------------------------------------------

if len(st.session_state.messages) > 0:

    st.divider()

    st.subheader("💬 Conversation")

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])


# =========================================================
# Retrieved Chunks & Source Pages
# =========================================================

if len(st.session_state.retrieved_docs) > 0:

    st.divider()

    st.subheader("📚 Retrieved Context")

    st.caption(
        "These document chunks were retrieved from the vector database and provided to the AI model."
    )

    # -----------------------------------------------------
    # Display Retrieved Chunks
    # -----------------------------------------------------

    for index, doc in enumerate(
        st.session_state.retrieved_docs,
        start=1
    ):

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        with st.expander(
            f"📄 Chunk {index} | Page {page}"
        ):

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Page Number",
                    page
                )

            with col2:

                st.metric(
                    "Characters",
                    len(doc.page_content)
                )

            st.markdown("### 📖 Retrieved Text")

            st.write(doc.page_content)

            st.markdown("---")

            st.markdown("### 📌 Metadata")

            st.json(doc.metadata)

    # -----------------------------------------------------
    # Source Pages
    # -----------------------------------------------------

    st.divider()

    st.subheader("📄 Source Pages")

    pages = sorted(
        {
            str(
                doc.metadata.get(
                    "page",
                    "Unknown"
                )
            )
            for doc in st.session_state.retrieved_docs
        }
    )

    st.success(
        "Answer Generated Using Page(s): "
        + ", ".join(pages)
    )

# =========================================================
# Medical Disclaimer
# =========================================================

st.divider()

st.warning(
    """
⚠ **Medical Disclaimer**

DOC AI is intended for **educational and informational purposes only**.

• It does **not** diagnose diseases.
• It does **not** prescribe medications.
• It does **not** replace professional medical advice.
• Always consult a qualified healthcare professional for medical diagnosis and treatment.

The generated responses are based only on the uploaded medical report and the selected Large Language Model.
"""
)

# =========================================================
# Footer
# =========================================================

st.divider()

col1, col2, col3 = st.columns(3)

# ---------------------------------------------------------
# Technology Stack
# ---------------------------------------------------------

with col1:

    st.markdown(
        """
### 🚀 Technology Stack

- Streamlit
- LangChain
- Python
- HuggingFace
"""
    )

# ---------------------------------------------------------
# AI Components
# ---------------------------------------------------------

with col2:

    st.markdown(
        f"""
### 🤖 AI Components

**LLM Provider**

{st.session_state.provider}

**LLM Model**

{st.session_state.llm_model}

**Embedding**

{st.session_state.embedding_model}
"""
    )

# ---------------------------------------------------------
# Vector Database
# ---------------------------------------------------------

with col3:

    st.markdown(
        f"""
### 🗄️ Vector Database

**Database**

{st.session_state.vector_db}

**Top K Retrieval**

{top_k}
"""
    )

st.divider()

# =========================================================
# Bottom Footer
# =========================================================

st.markdown(
"""
<div style="text-align:center; color:gray;">

<h3>🩺  DOC AI</h3>

<p>
Intelligent Medical Report Assistant using
<b>Retrieval-Augmented Generation (RAG)</b>
</p>

<p>

🤖 Google Gemini &nbsp; | &nbsp;
🦙 Ollama &nbsp; | &nbsp;
🗄️ FAISS & ChromaDB &nbsp; | &nbsp;
🧠 HuggingFace Embeddings

</p>

<hr>

<p>

Built using

<b>Streamlit</b> •
<b>LangChain</b> •
<b>Python</b>

</p>

<p>

Developed by <b>RAGHAV SARWAN</b>

</p>

<p>

Version 1.0

</p>

</div>
""",
unsafe_allow_html=True
)