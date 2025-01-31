# Tanya jawab dengan dokumen Anda!

## Installation and Setup

### **Setup SambaNova**

Get an API key from [SambaNova](https://sambanova.ai/) and set it in the `.env` file as follows:

```bash
SAMBANOVA_API_KEY=<YOUR_SAMBANOVA_API_KEY>
```

### **Setup Milvus VectorDB**

Run the following command to set up Milvus VectorDB:

```bash
# Pull Milvus standalone docker image
docker pull milvusdb/milvus:v2.3.3

# Create a docker network
docker network create milvus-network

# Start Milvus
docker run -d --name milvus-standalone \
-p 19530:19530 \
-p 9301:9301 \
--network milvus-network \
-v $(pwd)/milvus_storage:/var/lib/milvus \
milvusdb/milvus:v2.3.3
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
pip install streamlit pymilvus sentence-transformers transformers langchain langchain-community langchain-sambanova torch fitz
```

### **Run the App**

Run the app by executing the following command:

```bash
streamlit run app.py
```