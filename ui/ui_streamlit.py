import streamlit as st
import requests

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="HR Policy RAG Assistant", layout="wide")

st.title("📘 HR Policy Chat Assistant")

# -------- Session Memory --------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------- Sidebar Upload --------

st.sidebar.header("📄 Upload Document")

file = st.sidebar.file_uploader(
    "Upload PDF / MD / TXT",
    type=["pdf", "md", "txt"]
)

if file:

    with st.sidebar.spinner("Uploading & indexing..."):
        res = requests.post(
            f"{API}/upload-doc",
            files={"file": file.getvalue()}
        )

    st.sidebar.success("Indexed ✅")

# -------- Chat Display --------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------- Chat Input --------

prompt = st.chat_input("Ask HR policy question...")

if prompt:

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            res = requests.post(
                f"{API}/ask",
                json={"question": prompt, "k": 4}
            )

            data = res.json()

            answer = data["answer"]
            conf = data.get("confidence", 0)

            st.markdown(answer)
            st.caption(f"Confidence: {conf:.3f}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
