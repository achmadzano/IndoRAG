import streamlit as st
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import os
import json
from langchain_sambanova import ChatSambaNovaCloud
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import getpass

st.set_page_config(
    page_title="Document QnA System",
    page_icon="🤖",
    layout="wide"
)

# Milvus connection details
milvus_host = "localhost"
milvus_port = "19530"
collection_name = "zanobab123"

# Initialize models in session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

@st.cache_resource
def load_models(dense_model_id='intfloat/multilingual-e5-large-instruct',
                sparse_model_id='naver/splade-cocondenser-ensembledistil'):
    """Initialize models once and cache them using Streamlit's cache_resource"""
    dense_model = SentenceTransformer(dense_model_id)
    sparse_model = AutoModelForMaskedLM.from_pretrained(sparse_model_id)
    tokenizer = AutoTokenizer.from_pretrained(sparse_model_id)
    return dense_model, sparse_model, tokenizer

# Hybrid search function
def hybrid_search(user_question: str, dense_model, sparse_model, tokenizer, alpha: float = 0.6, limit: int = 3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sparse_model.to(device)
    
    dense_embedding = dense_model.encode(
        user_question,
        show_progress_bar=False,
        convert_to_tensor=True,
        device=device
    ).cpu().tolist()

    inputs = tokenizer(user_question, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = sparse_model(**inputs).logits
        relu_log = torch.log1p(torch.relu(logits))
        sparse_embedding = relu_log.max(dim=1).values.squeeze().cpu()

    indices = torch.nonzero(sparse_embedding > 0).squeeze().tolist()
    values = sparse_embedding[sparse_embedding > 0].tolist()
    sparse_dict = dict(zip(indices, values))

    collection = Collection(name=collection_name, using="default")
    collection.load()

    try:
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        results = collection.search(
            data=[dense_embedding],
            anns_field="dense_vector",
            param=search_params,
            limit=limit,
            output_fields=["id", "document_id", "metadata_json", "embedding_raw"],
            consistency_level="Eventually"
        )

        hits = results[0]
        final_results = collection.query(
            expr=f"id in {[hit.id for hit in hits]}",
            output_fields=["id", "document_id", "metadata_json", "embedding_raw"],
            consistency_level="Eventually"
        )
        
        for result in final_results:
            for hit in hits:
                if hit.id == result["id"]:
                    result["search_score"] = hit.score
                    break
        
        return final_results

    finally:
        collection.release()

# Process question with SambaNova
def process_question(user_question: str, data):
    if not os.getenv("SAMBANOVA_API_KEY"):
        os.environ["SAMBANOVA_API_KEY"] = getpass.getpass("Enter your SambaNova Cloud API key: ")

    llm = ChatSambaNovaCloud(
        model="Meta-Llama-3.1-8B-Instruct",
        max_tokens=1024,
        temperature=0.7,
        top_p=0.01,
    )

    template = """<|im_start|>system
    Anda adalah asisten profesional yang membantu menjawab pertanyaan berdasarkan dokumen atau informasi yang diberikan. Analisis Anda harus logis, akurat, dan sesuai dengan konteks yang relevan. Pastikan jawaban Anda didukung oleh referensi yang jelas dari dokumen yang tersedia.

    **Petunjuk:**
    1. Gunakan informasi yang relevan dari dokumen yang diberikan untuk menjawab pertanyaan.
    2. Jawab dengan bahasa sehari-hari yang mudah dipahami.
    3. Output hanya dalam bentuk JSON dengan format berikut:
       {{"answer": "Jawaban langsung berdasarkan dokumen"}} jangan sertakan apapun
    4. Jika tidak ada informasi yang relevan, jawab dengan "Tidak ada informasi yang relevan".

    **Konteks:**
    {data}

    **Pertanyaan:**
    {user_question}
    <|im_start|>assistant
    <|im_end|>
    """

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "data": format_context(data),
        "user_question": user_question
    })

# Helper functions
def format_context(data):
    return "\n".join([str(item) for item in data])

def extract_answer_from_json(text: str) -> str:
    """Extract the answer field from JSON response"""
    try:
        # Accumulate JSON text until we have a valid JSON object
        json_text = ""
        for char in text:
            json_text += char
            try:
                data = json.loads(json_text)
                if "answer" in data:
                    return data["answer"]
            except json.JSONDecodeError:
                continue
        return text  # Return original text if no valid JSON found
    except Exception as e:
        print(f"Error extracting answer from JSON: {e}")
        return text

# Streamlit Chatbot App
def main():
    st.title("QnA Retrieval Chatbot")
    st.write("Ask me anything, and I'll retrieve the best answer for you!")

    # Load models at startup using Streamlit's caching
    with st.spinner("Loading models..."):
        dense_model, sparse_model, tokenizer = load_models()
        # Connect to Milvus
        connections.connect("default", host=milvus_host, port=milvus_port)
        st.session_state.models_loaded = True

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if user_question := st.chat_input("Ask your question:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Perform hybrid search with pre-loaded models
        with st.spinner("Searching for the best answer..."):
            results = hybrid_search(user_question, dense_model, sparse_model, tokenizer)
            data = [item['embedding_raw'] for item in results]

        # Stream the final answer
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            json_accumulator = ""
            
            for chunk in process_question(user_question, data):
                json_accumulator += chunk
                # Try to extract answer from accumulated JSON
                answer = extract_answer_from_json(json_accumulator)
                response_placeholder.markdown(answer + "▌")
            
            # Final extraction attempt
            final_answer = extract_answer_from_json(json_accumulator)
            response_placeholder.markdown(final_answer)
            
            # Add assistant message to chat history with extracted answer
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

if __name__ == "__main__":
    main()