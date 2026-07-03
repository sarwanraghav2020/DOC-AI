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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* Main App - Premium Dark Background */
.stApp{
    background: linear-gradient(135deg, #0a0e27 0%, #16213e 50%, #0f1b3c 100%);
    color:#F5F7FA;
    background-attachment: fixed;
}

/* Main Title - Premium Typography */
.main-title{
    text-align:center;
    font-size:48px;
    font-weight:800;
    letter-spacing:-0.02em;
    background: linear-gradient(135deg, #E8D4F8 0%, #FFE082 50%, #E8D4F8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom:8px;
    animation: fadeInDown 0.8s ease-out;
}

/* Subtitle - Refined & Elegant */
.subtitle{
    text-align:center;
    color:#C5CCDC;
    font-size:16px;
    font-weight:300;
    letter-spacing:0.5px;
    margin-bottom:32px;
    line-height:1.6;
    opacity:0.85;
}

/* Card - Glassmorphism Premium Effect */
.card{
    background: linear-gradient(135deg, rgba(45, 27, 78, 0.6) 0%, rgba(30, 20, 50, 0.4) 100%);
    padding:28px;
    border-radius:16px;
    border:1px solid rgba(232, 212, 248, 0.15);
    backdrop-filter: blur(20px);
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.3),
        0 0 1px rgba(232, 212, 248, 0.2) inset;
    color:#F5F7FA;
    transition: all 0.4s cubic-bezier(0.23, 1, 0.320, 1);
    position: relative;
    overflow: hidden;
}

.card::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 100px;
    height: 100px;
    background: radial-gradient(circle, rgba(232, 212, 248, 0.1) 0%, transparent 70%);
    border-radius: 50%;
    opacity: 0;
    transition: opacity 0.4s ease;
}

.card:hover {
    border-color: rgba(232, 212, 248, 0.4);
    box-shadow: 
        0 16px 48px rgba(232, 212, 248, 0.15),
        0 0 1px rgba(232, 212, 248, 0.3) inset;
    transform: translateY(-2px);
}

.card:hover::before {
    opacity: 1;
}

/* Footer - Premium */
.footer{
    text-align:center;
    color:#8FA3BE;
    font-size:13px;
    font-weight:300;
    letter-spacing:0.5px;
    margin-top:48px;
    padding-top:24px;
    border-top:1px solid rgba(232, 212, 248, 0.1);
}

/* Animations */
@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Additional Premium Styling */
h1, h2, h3 {
    font-weight:700;
    letter-spacing:-0.01em;
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
<div class="main-title">

🩺 DOC AI

</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="subtitle">

Intelligent Medical Report Assistant powered by

<b>LangChain • Google Gemini • Ollama • FAISS • ChromaDB</b>

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