# Tanya jawab dengan dokumen Anda!

## Installation and Setup

### **Setup SambaNova**

Get an API key from [SambaNova](https://sambanova.ai/) and set it in the `.env` file as follows:

```bash
SAMBANOVA_API_KEY=<YOUR_SAMBANOVA_API_KEY>
```

### **Setup Qdrant VectorDB**

Run the following command to set up Qdrant VectorDB:

```bash
docker run -p 6333:6333 -p 6334:6334 \
-v $(pwd)/qdrant_storage:/qdrant/storage:z \
qdrant/qdrant
```

### **Set Up Virtual Environment**

#### On Windows

```bash
py -3 -m venv .venv
.venv\Scripts\activate
```

#### On Linux or macOS

```bash
python3 -m venv .venv
. .venv/bin/activate
```

### **Install Dependencies**

Ensure you have Python 3.11 or later installed. Then, install the required dependencies:

```bash
pip install streamlit llama-index-vector-stores-qdrant llama-index-llms-sambanovasystems sseclient-py llama-index-embeddings-huggingface
```

### **Run the App**

Run the app by executing the following command:

```bash
streamlit run app.py
```