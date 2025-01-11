import os
import base64
import gc
import random
import tempfile
import time
import uuid
from typing import Optional

from IPython.display import Markdown, display
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from rag_code import EmbedData, QdrantVDB_QB, Retriever, RAG
import streamlit as st
from qdrant_client import QdrantClient

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.selected_collection = None
    st.session_state.uploaded_file = None
    st.session_state.new_collection_name = None


session_id = st.session_state.id
batch_size = 32

load_dotenv()

def get_existing_collections() -> list:
    """Retrieve list of existing collections from Qdrant."""
    try:
        client = QdrantClient("localhost", port=6333)
        collections = client.get_collections().collections
        return [collection.name for collection in collections]
    except Exception as e:
        st.error(f"Error connecting to Qdrant: {e}")
        return []

def initialize_existing_collection(collection_name: str) -> Optional[RAG]:
    """Initialize RAG system with existing collection."""
    try:
        client = QdrantClient("localhost", port=6333)
        collection_info = client.get_collection(collection_name)
        
        # Check if collection has any points
        collection_stats = client.get_collection(collection_name)
        if collection_stats.points_count == 0:
            st.error(f"Collection '{collection_name}' is empty. Please create a new collection with documents.")
            return None
            
        embeddata = EmbedData(embed_model_name="BAAI/bge-m3", batch_size=batch_size)
        
        qdrant_vdb = QdrantVDB_QB(
            collection_name=collection_name,
            batch_size=batch_size,
            vector_dim=1024
        )
        qdrant_vdb.define_client()
        
        retriever = Retriever(vector_db=qdrant_vdb, embeddata=embeddata)
        query_engine = RAG(retriever=retriever, llm_name="Meta-Llama-3.3-70B-Instruct")
        
        return query_engine
    except Exception as e:
        st.error(f"Error initializing collection: {e}")
        return None

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%">
                    </iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)

def index_documents(uploaded_file, new_collection_name):
    """Index documents into a new collection."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            file_key = f"{session_id}-{new_collection_name}"
            
            if os.path.exists(temp_dir):
                loader = SimpleDirectoryReader(
                    input_dir=temp_dir,
                    required_exts=[".pdf"],
                    recursive=True
                )
            else:    
                st.error('Could not find the file you uploaded, please check again...')
                return None
            
            docs = loader.load_data()
            documents = [doc.text for doc in docs]

            embeddata = EmbedData(embed_model_name="BAAI/bge-m3", batch_size=batch_size)
            embeddata.embed(documents)

            qdrant_vdb = QdrantVDB_QB(
                collection_name=new_collection_name,
                batch_size=batch_size,
                vector_dim=1024
            )
            qdrant_vdb.define_client()
            qdrant_vdb.create_collection()
            qdrant_vdb.ingest_data(embeddata=embeddata)

            retriever = Retriever(vector_db=qdrant_vdb, embeddata=embeddata)
            query_engine = RAG(retriever=retriever, llm_name="Meta-Llama-3.3-70B-Instruct")

            return query_engine
    except Exception as e:
        st.error(f"An error occurred during indexing: {e}")
        return None

with st.sidebar:
    st.header("Collection Management")
    
    # Get existing collections
    existing_collections = get_existing_collections()
    
    # Collection selection/creation
    collection_mode = st.radio(
        "Choose collection mode:",
        ["Use Existing Collection", "Create New Collection"],
        index=1 if not existing_collections else 0
    )
    
    if collection_mode == "Use Existing Collection":
        if existing_collections:
            selected_collection = st.selectbox(
                "Select existing collection",
                existing_collections
            )

            if selected_collection and selected_collection != st.session_state.get('selected_collection'):
                st.session_state.selected_collection = selected_collection
                query_engine = initialize_existing_collection(selected_collection)
                if query_engine:
                    st.session_state.file_cache[f"{session_id}-{selected_collection}"] = query_engine
                    st.success(f"Connected to collection: {selected_collection} and ready to chat!")
                    reset_chat()

            # Allow uploading files to the existing collection
            uploaded_file = st.file_uploader("Upload additional `.pdf` file to this collection", type="pdf")

            if uploaded_file:
                with st.spinner("Adding document to the existing collection..."):
                    updated_engine = index_documents(uploaded_file, selected_collection)
                    if updated_engine:
                        st.session_state.file_cache[f"{session_id}-{selected_collection}"] = updated_engine
                        st.success(f"Successfully added the document to collection: {selected_collection}!")
        else:
            st.warning("No collections found. Please create a new collection.")

    if collection_mode == "Create New Collection":
        new_collection_name = st.text_input("Enter new collection name")
        
        if new_collection_name in existing_collections:
            st.error("Collection name already exists!")

        uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            # display_pdf(uploaded_file)
        
        if new_collection_name and new_collection_name not in existing_collections and uploaded_file:
            if st.button("Start Uploading"):
                with st.spinner("Uploading documents..."):
                    query_engine = index_documents(uploaded_file, new_collection_name)
                    
                    if query_engine:
                        file_key = f"{session_id}-{new_collection_name}"
                        st.session_state.file_cache[file_key] = query_engine
                        st.session_state.selected_collection = new_collection_name
                        st.success("Successfully uploaded your documents and ready to chat!")
                        reset_chat()


# Rest of the chat interface code
col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Tanya jawab dengan dokumen Anda!")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

if "messages" not in st.session_state:
    reset_chat()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Masukkan pertanyaan Anda..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        if st.session_state.selected_collection:
            file_key = f"{session_id}-{st.session_state.selected_collection}"
            query_engine = st.session_state.file_cache.get(file_key)
            
            if query_engine:
                streaming_response = query_engine.query(prompt)
                
                for chunk in streaming_response:
                    try:
                        new_text = chunk.raw["choices"][0]["delta"]["content"]
                        full_response += new_text
                        message_placeholder.markdown(full_response + "▌")
                    except:
                        pass

                message_placeholder.markdown(full_response)
            else:
                message_placeholder.error("Please select or create a collection first.")
        else:
            message_placeholder.error("Please select or create a collection first.")

    st.session_state.messages.append({"role": "assistant", "content": full_response})