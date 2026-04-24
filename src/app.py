import os

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import streamlit as st
import json
import faiss
import numpy as np
from google import genai
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load data (cached)
# -----------------------------
@st.cache_data
def load_chunks():
    with open("data/processed/multimodal_chunks.json", "r", encoding="utf-8") as f:
        return json.load(f)

chunks = load_chunks()

@st.cache_resource
def load_index():
    return faiss.read_index("vectorstores/multimodal.index")

index = load_index()

# -----------------------------
# Load embedding model (LOCAL)
# -----------------------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()

# -----------------------------
# Gemini setup
# -----------------------------
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY not set. Please add it to your environment.")
    st.stop()

client = genai.Client(api_key=api_key)

# -----------------------------
# Session state
# -----------------------------
if "cache" not in st.session_state:
    st.session_state.cache = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# UI
# -----------------------------
st.title("IFC Annual Report RAG Chatbot")

if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.session_state.cache = {}
    st.rerun()

st.divider()

# show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# user input
query = st.chat_input("Ask a question about the IFC report")

# -----------------------------
# Chat logic
# -----------------------------
if query:
    key = " ".join(query.lower().split())

    # show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # -----------------------------
    # Cache check
    # -----------------------------
    if key in st.session_state.cache:
        answer = st.session_state.cache[key]

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                # -----------------------------
                # 1. Embedding (LOCAL FIX)
                # -----------------------------
                query_embedding = np.array(
                    embed_model.encode([query]),
                    dtype="float32"
                )

                # -----------------------------
                # 2. Retrieval
                # -----------------------------
                distances, indices = index.search(query_embedding, 5)

                context = ""

                with st.expander("🔍 Retrieved Chunks"):
                    for idx, i in enumerate(indices[0]):
                        st.write("Score:", round(float(distances[0][idx]), 4))
                        st.write("Type:", chunks[i]["type"])
                        st.write(chunks[i]["content"])
                        st.write("---")

                        context += f"[Chunk {idx+1}]: {chunks[i]['content'][:500]}\n\n"

                if not context.strip():
                    st.write("I couldn’t find relevant information in the document.")
                    st.stop()

                # -----------------------------
                # 3. Prompt
                # -----------------------------
                prompt = f"""You are a helpful assistant answering questions using ONLY the provided context.

Rules:
- Use ONLY the given context.
- You can summarize and combine information.
- If the answer is not in the context, say:
"I don’t have enough information in the document to answer that."

Question: {query}

Context:
{context}

Answer clearly in 2–3 sentences.
"""

                # -----------------------------
                # 4. Generation
                # -----------------------------
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )

                answer = response.text or "I couldn’t generate an answer."

                # store in cache
                st.session_state.cache[key] = answer

                # display answer
                st.write(answer)

                # save chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"API error: {str(e)}")
                st.write("Retrieved context above can still be used.")