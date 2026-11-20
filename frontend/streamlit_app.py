
import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Modular RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Modular RAG Chatbot")

if "token" not in st.session_state:
    st.session_state.token = None

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def get_headers():
    return {
        "Authorization": f"Bearer {st.session_state.token}"
    }


def get_documents():
    response = requests.get(
        f"{API_URL}/documents",
        headers=get_headers(),
        timeout=30
    )
    response.raise_for_status()
    return response.json()["documents"]


# Authentication

if not st.session_state.token:

    st.subheader("Demo Login")

    user_id = st.text_input("Enter your User ID")

    if st.button("Login"):
        if not user_id.strip():
            st.warning("Enter a User ID")
        else:
            try:
                response = requests.post(
                    f"{API_URL}/auth/demo-login",
                    json={"user_id": user_id.strip()},
                    timeout=10
                )
                response.raise_for_status()

                data = response.json()

                st.session_state.token = data["access_token"]
                st.session_state.user_id = data["user_id"]

                st.rerun()

            except requests.RequestException:
                st.error("Unable to authenticate with backend.")

else:

    st.sidebar.success(f"User: {st.session_state.user_id}")

    if st.sidebar.button("Logout"):
        st.session_state.token = None
        st.session_state.user_id = None
        st.session_state.messages = []
        st.rerun()

    # PDF Upload

    st.header("Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    if st.button("Upload Document", disabled=uploaded_file is None):
        try:
            response = requests.post(
                f"{API_URL}/documents/upload",
                headers=get_headers(),
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf"
                    )
                },
                timeout=300
            )
            response.raise_for_status()

            st.success("Document uploaded successfully!")
            st.json(response.json())

        except requests.RequestException as e:
            st.error(f"Upload failed: {e}")

    st.divider()

    # Document Selection

    st.header("Your Documents")

    try:
        documents = get_documents()

        if documents:
            document_options = {
                f"{doc['file_name']} ({doc['chunks']} chunks)": doc["document_id"]
                for doc in documents
            }

            selected_label = st.selectbox(
                "Select a document",
                options=list(document_options.keys())
            )

            selected_document_id = document_options[selected_label]

            if st.button("Delete Selected Document"):
                response = requests.delete(
                    f"{API_URL}/documents/{selected_document_id}",
                    headers=get_headers(),
                    timeout=30
                )
                response.raise_for_status()

                st.session_state.messages = []
                st.success("Document deleted successfully.")
                st.rerun()

        else:
            st.info("No documents uploaded yet.")
            selected_document_id = None

    except requests.RequestException as e:
        st.error(f"Unable to load documents: {e}")
        selected_document_id = None

    st.divider()

    # RAG Chat

    st.header("Chat with Your Document")

    if selected_document_id:

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message.get("sources"):
                    with st.expander("View Sources"):
                        for source in message["sources"]:
                            st.markdown(
                                f"**{source['file_name']} — Page {source['page_number']}**"
                            )
                            st.caption(source["content"])

        question = st.chat_input("Ask a question about your document")

        if question:

            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):

                with st.spinner("Searching and generating answer..."):

                    try:
                        response = requests.post(
                            f"{API_URL}/chat",
                            headers=get_headers(),
                            json={
                                "question": question,
                                "document_id": selected_document_id
                            },
                            timeout=180
                        )
                        response.raise_for_status()

                        result = response.json()

                        st.markdown(result["answer"])

                        if result["sources"]:
                            with st.expander("View Sources"):
                                for source in result["sources"]:
                                    st.markdown(
                                        f"**{source['file_name']} — Page {source['page_number']}**"
                                    )
                                    st.caption(source["content"])

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"]
                        })

                    except requests.RequestException as e:
                        st.error(f"Chat request failed: {e}")

    else:
        st.info("Upload and select a document to start chatting.")
