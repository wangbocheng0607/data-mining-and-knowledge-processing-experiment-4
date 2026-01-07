# Milvus Lite Configuration
MILVUS_LITE_DATA_PATH = "./milvus_lite_data.db" # Path to store Milvus Lite data

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY = "./chroma_data" # Path to store ChromaDB data
COLLECTION_NAME = "medical_rag_lite" # Shared collection name for both databases

# Vector Store Type: "milvus" or "chromadb"
VECTOR_STORE_TYPE = "chromadb"

# Data Configuration
DATA_FILE = "./data/processed_data_cleaned.json"

# Model Configuration with multilingual support
# 使用支持中文的多语言模型替代英文模型
EMBEDDING_MODEL_NAME = './hf_cache/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf'  # Fallback to local English model with improved retrieval logic
GENERATION_MODEL_NAME = './hf_cache/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987'
EMBEDDING_DIM = 384 # Must match EMBEDDING_MODEL_NAME

# Indexing and Search Parameters
MAX_ARTICLES_TO_INDEX = 500
TOP_K = 3
# Milvus index parameters (adjust based on data size and needs)
INDEX_METRIC_TYPE = "L2" # Or "IP"
INDEX_TYPE = "IVF_FLAT"  # Milvus Lite 支持的索引类型
# HNSW index params (adjust as needed)
INDEX_PARAMS = {"nlist": 128}
# HNSW search params (adjust as needed)
SEARCH_PARAMS = {"nprobe": 16}

# Generation Parameters
MAX_NEW_TOKENS_GEN = 512
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# Global map to store document content (populated during indexing)
# Key: document ID (int), Value: dict {'title': str, 'abstract': str, 'content': str}
id_to_doc_map = {} 