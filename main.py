# main.py

import os
import streamlit as st
import tempfile
from pdf_utils import load_pdf, chunk_text
from vector_store import create_vector_store
from agent import route_to_tool

st.set_page_config(page_title="PDF AI Agent", layout="wide")
st.title("📄 PDF Agent with DeepSeek API")



# Upload section
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])


if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success("✅ PDF uploaded successfully!")


    with st.spinner("🔍 Processing PDF..."):
        text = load_pdf(pdf_path)
        chunks = chunk_text(text)
        vector_store = create_vector_store(chunks)

    st.success("🧠 PDF processed and embedded. Ready for interaction!")



    # Chat interface
    user_query = st.text_input("Ask a question or type 'summarize' to get a summary", placeholder="e.g., What is the invoice date?")

    if st.button("Submit") and user_query.strip():
        with st.spinner("🤖 Thinking..."):
            response = route_to_tool(user_query, vector_store)


        st.markdown("### 💬 Response")
        st.write(response)
