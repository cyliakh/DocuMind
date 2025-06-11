import streamlit as st
import requests
import json
import tempfile
import hashlib
import os
import numpy as np
import faiss

from langchain.vectorstores.faiss import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

# === Style ===
st.markdown("""
    <style>
        body, .stApp {
            background-color: #ffffff !important;
        }
        .message-user {
            background-color: #fff0f5;
            padding: 10px;
            border-radius: 12px;
            margin: 8px 0;
            display: flex;
            align-items: center;
        }
        .message-assistant {
            background-color: #f0e6ff;
            padding: 10px;
            border-radius: 12px;
            margin: 8px 0;
            display: flex;
            align-items: center;
        }
        .avatar {
            width: 32px;
            height: 32px;
            margin-right: 10px;
        }
        .message-content {
            flex: 1;
        }
        .stTextInput > div > div > input {
            background-color: #fff0f5;
        }
        .stButton button {
            background-color: #ff99cc;
            color: white;
        }
        section[data-testid="stSidebar"] {
            background-color: #ffe6f0 !important;
            color: #d63384;
        }
        .stFileUploader {
            background-color: #fff0f5;
            border: 1px dashed #ff66a3;
            padding: 10px;
            border-radius: 12px;
            margin-top: 10px;
        }
        h1 {
            text-align: center;
            color: #d63384;
            font-size: 2.5rem;
            padding: 20px;
        }
    </style>
""", unsafe_allow_html=True)

OLLAMA_HOST = "http://34.68.20.35:11434"
MODEL_NAME = "mixtral"
EMBEDDING_MODEL = "mxbai-embed-large"

CACHE_DIR = "embedding_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

st.markdown("<h1 style='color: #d94484;'>DocuMind: AI Assistant</h1>", unsafe_allow_html=True)

@st.cache_resource
def warm_up_models():
    try:
        # Warm-up embedding model
        embed_resp = requests.post(
            f"{OLLAMA_HOST}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": ["warmup"]},
            timeout=1000,
        )
        embed_resp.raise_for_status()

        # Warm-up chat model
        chat_resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hello."}
                ]
            },
            timeout=1000,
        )
        chat_resp.raise_for_status()
    except Exception as e:
        st.warning(f"Warm-up failed: {e}")

warm_up_models()

st.sidebar.header("Settings")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_user_input" not in st.session_state:
    st.session_state.pending_user_input = None

if "retrievers" not in st.session_state:
    st.session_state.retrievers = {}

if "selected_pdfs" not in st.session_state:
    st.session_state.selected_pdfs = {}

def submit_input():
    st.session_state.pending_user_input = st.session_state["chat_input"]

with st.sidebar:
    st.markdown("<h3 style='color: #d63384;'>Upload your PDFs</h3>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(" ", accept_multiple_files=True, type=["pdf"])

def file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def get_embedding_vectors(documents, batch_size=10):
    texts = [doc.page_content for doc in documents]
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        response = requests.post(
            f"{OLLAMA_HOST}/api/embed",
            json={
                "model": EMBEDDING_MODEL,
                "input": batch_texts,
            },
            timeout=1000,
        )
        response.raise_for_status()
        data = response.json()
        batch_embeddings = data.get("embeddings", [])
        if not batch_embeddings:
            raise ValueError("No embeddings returned from the server")

        embeddings.extend(batch_embeddings)

    return embeddings

class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        raise NotImplementedError("Embedding done remotely, not supported here.")

    def embed_query(self, text):
        raise NotImplementedError("Embedding done remotely, not supported here.")

dummy_embed = DummyEmbeddings()

# Process uploaded PDFs and checkbox to include/exclude
if uploaded_files:
    for pdf_file in uploaded_files:
        pdf_bytes = pdf_file.read()
        h = file_hash(pdf_bytes)
        # Use directory cache path for FAISS index, not a single file
        cache_path = os.path.join(CACHE_DIR, h)

        # Checkbox inside sidebar for PDF selection, default True if first time
        is_selected = st.sidebar.checkbox(f"Include: {pdf_file.name}", key=f"select_{pdf_file.name}",
                                          value=st.session_state.selected_pdfs.get(pdf_file.name, True))
        st.session_state.selected_pdfs[pdf_file.name] = is_selected

        if pdf_file.name not in st.session_state.retrievers:
            if os.path.exists(cache_path):
                st.sidebar.write(f"Loading cached FAISS index for {pdf_file.name}")
                faiss_index = FAISS.load_local(cache_path, dummy_embed)
                st.session_state.retrievers[pdf_file.name] = faiss_index
            else:
                st.sidebar.write(f"Processing and embedding {pdf_file.name}...")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_bytes)
                    tmp_path = tmp_file.name

                from langchain.document_loaders import PyPDFLoader
                loader = PyPDFLoader(tmp_path)
                documents = loader.load_and_split()

                embeddings = get_embedding_vectors(documents)

                embedding_vectors = np.array(embeddings).astype(np.float32)
                dim = embedding_vectors.shape[1]

                index = faiss.IndexFlatL2(dim)
                index.add(embedding_vectors)

                # Add filename metadata for source
                docs = [
                    Document(
                        page_content=doc.page_content,
                        metadata={**doc.metadata, "source_pdf": pdf_file.name}
                    )
                    for doc in documents
                ]
                docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})
                index_to_docstore_id = {i: str(i) for i in range(len(docs))}

                faiss_index = FAISS(
                    embedding_function=dummy_embed,
                    index=index,
                    docstore=docstore,
                    index_to_docstore_id=index_to_docstore_id,
                )
                faiss_index.save_local(cache_path)
                st.session_state.retrievers[pdf_file.name] = faiss_index

                os.remove(tmp_path)

# Display chat messages with avatars and style
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]

    if role == "user":
        avatar_url = "https://cdn-icons-png.flaticon.com/512/6997/6997662.png"  # cute user icon
        css_class = "message-user"
    else:
        avatar_url = "https://cdn-icons-png.flaticon.com/512/4712/4712037.png"  # cute robot icon
        css_class = "message-assistant"

    st.markdown(f"""
        <div class="{css_class}">
            <img src="{avatar_url}" class="avatar" />
            <div class="message-content">{content}</div>
        </div>
    """, unsafe_allow_html=True)

st.text_input("You:", key="chat_input", on_change=submit_input)

if st.session_state.pending_user_input:
    user_msg = st.session_state.pending_user_input
    st.session_state.messages.append({"role": "user", "content": user_msg})
    st.markdown(f"""
        <div class="message-user">
            <img src="https://cdn-icons-png.flaticon.com/512/6997/6997662.png" class="avatar" />
            <div class="message-content">{user_msg}</div>
        </div>
        <div class="message-assistant">
            <img src="https://cdn-icons-png.flaticon.com/512/4712/4712037.png" class="avatar" />
            <div class="message-content"><em>.......</em> 🤖</div>
        </div>
    """, unsafe_allow_html=True)

    reply = ""
    placeholder = st.empty()

    selected_retrievers = [
        st.session_state.retrievers[name]
        for name, selected in st.session_state.selected_pdfs.items()
        if selected and name in st.session_state.retrievers
    ]

    try:
        if selected_retrievers:
            # Embed query remotely
            query_embed_resp = requests.post(
                f"{OLLAMA_HOST}/api/embed",
                json={"model": EMBEDDING_MODEL, "input": [user_msg]},
                timeout=1000,
            )
            query_embed_resp.raise_for_status()
            query_embedding = np.array(query_embed_resp.json()["embeddings"][0]).astype(np.float32)

            # Search all selected retrievers and collect docs + distances
            all_docs = []
            all_distances = []

            for faiss_index in selected_retrievers:
                D, I = faiss_index.index.search(np.array([query_embedding]), k=3)
                for dist, idx in zip(D[0], I[0]):
                    if idx == -1:
                        continue
                    doc = faiss_index.docstore.search(faiss_index.index_to_docstore_id[idx])
                    all_docs.append(doc)
                    all_distances.append(dist)

            # Sort all docs by distance (closest first)
            sorted_docs = [doc for _, doc in sorted(zip(all_distances, all_docs))]

            # Take top 5 results (you can increase/decrease)
            top_docs = sorted_docs[:5]

            # Include source PDF in context so model knows which document text comes from
            context_text = "\n\n---\n\n".join(
                [f"[Source: {doc.metadata.get('source_pdf','Unknown')}]\n{doc.page_content}" for doc in top_docs]
            )
            prompt = f"Use the following context to answer:\n{context_text}\n\nUser: {user_msg}\nAssistant:"

            with requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": True,
                },
                stream=True,
                timeout=1000,
            ) as response:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode("utf-8"))
                            delta = data.get("message", {}).get("content", "")
                            reply += delta
                            placeholder.markdown(f"**Assistant:** {reply}")
                        except Exception:
                            # Ignore malformed line
                            continue

        else:
            # No PDFs selected, normal chat
            with requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": MODEL_NAME,
                    "messages": st.session_state.messages,
                    "stream": True,
                },
                stream=True,
                timeout=1000,
            ) as response:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode("utf-8"))
                            delta = data.get("message", {}).get("content", "")
                            reply += delta
                            placeholder.markdown(f"**Assistant:** {reply}")
                        except Exception:
                            continue

    except Exception as e:
        reply = f"Error: {e}"
        placeholder.markdown(f"**Assistant:** {reply}")

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.pending_user_input = None
    st.rerun()
