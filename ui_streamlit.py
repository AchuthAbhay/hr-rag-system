import streamlit as st
import requests
import pandas as pd

API = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="HR Policy RAG Assistant",
    layout="wide"
)

# =====================================
# PAGE NAVIGATION
# =====================================

page = st.sidebar.radio(
    "Navigation",
    ["💬 Chat Assistant", "📊 Analytics Dashboard"]
)

# =====================================
# SESSION STATE INIT
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_file" not in st.session_state:
    st.session_state.current_file = None


# =====================================
# CHAT PAGE
# =====================================

if page == "💬 Chat Assistant":

    st.title("📘 HR Policy Chat Assistant")

    # -------- Upload --------

    st.sidebar.header("📄 Upload Document")

    file = st.sidebar.file_uploader(
        "Upload PDF / MD / TXT",
        type=["pdf", "md", "txt"]
    )

    if file:

        if st.session_state.current_file != file.name:

            with st.sidebar.spinner("Uploading & indexing..."):

                files = {
                    "file": (
                        file.name,
                        file.getvalue(),
                        file.type
                    )
                }

                res = requests.post(
                    f"{API}/upload-doc",
                    files=files
                )

                if res.status_code == 200:

                    st.session_state.current_file = file.name
                    st.session_state.messages = []  # clear old chat

                    st.sidebar.success(
                        f"Indexed: {file.name} ✅"
                    )

                else:
                    st.sidebar.error("Upload failed ❌")

    # Show current file
    if st.session_state.current_file:

        st.sidebar.write("Current indexed file:")
        st.sidebar.write(
            f"✅ {st.session_state.current_file}"
        )


    # -------- Chat display --------

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


    # -------- Chat input --------

    prompt = st.chat_input(
        "Ask HR policy question..."
    )

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
                    json={
                        "question": prompt,
                        "k": 6
                    }
                )

                data = res.json()

                answer = data["answer"]
                confidence = data.get(
                    "confidence", 0
                )

                st.markdown(answer)

                st.caption(
                    f"Confidence: {confidence:.3f}"
                )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


# =====================================
# ANALYTICS PAGE
# =====================================

elif page == "📊 Analytics Dashboard":

    st.title("📊 HR RAG Analytics Dashboard")

    with st.spinner("Loading analytics..."):

        res = requests.get(
            f"{API}/analytics"
        )

        if res.status_code != 200:

            st.error(
                "Failed to load analytics"
            )

        else:

            analytics = res.json()["analytics"]

            # Metrics Row
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Queries",
                analytics["total_queries"]
            )

            col2.metric(
                "Average Confidence",
                f'{analytics["avg_confidence"]:.3f}'
            )

            col3.metric(
                "Documents Used",
                len(analytics["top_sources"])
            )


            st.divider()


            # Top Questions
            st.subheader("Top Questions")

            if analytics["top_questions"]:

                df_q = pd.DataFrame(
                    analytics["top_questions"],
                    columns=[
                        "Question",
                        "Count"
                    ]
                )

                st.dataframe(
                    df_q,
                    width="stretch"
                )

                st.bar_chart(
                    df_q.set_index("Question")
                )


            st.divider()


            # Top Documents
            st.subheader("Top Documents")

            if analytics["top_sources"]:

                df_s = pd.DataFrame(
                    analytics["top_sources"],
                    columns=[
                        "Document",
                        "Count"
                    ]
                )

                st.dataframe(
                    df_s,
                    width="stretch"
                )

                st.bar_chart(
                    df_s.set_index("Document")
                )