import streamlit as st
import time
import os
os.environ['HF_HOME'] = './hf_cache'  # 设置模型缓存目录
# 不设置HF_ENDPOINT，确保使用本地模型 


# Import functions and config from other modules
from config import (
    DATA_FILE, EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME, TOP_K,
    MAX_ARTICLES_TO_INDEX, MILVUS_LITE_DATA_PATH, COLLECTION_NAME,
    id_to_doc_map, VECTOR_STORE_TYPE, CHROMA_PERSIST_DIRECTORY # Import the global map and vector store config
)
from data_utils import load_data
from models import load_embedding_model, load_generation_model
from rag_core import generate_answer

# Dynamically import vector store utilities based on configuration
if VECTOR_STORE_TYPE == "chromadb":
    from chromadb_utils import get_chroma_client, get_or_create_collection, index_data_if_needed, search_similar_documents
else:
    from milvus_utils import get_milvus_client, setup_milvus_collection, index_data_if_needed, search_similar_documents

# --- Streamlit UI 设置 ---
st.set_page_config(layout="wide")

# Dynamic UI based on vector store type
if VECTOR_STORE_TYPE == "chromadb":
    st.title("📄 医疗 RAG 系统 (ChromaDB)")
    st.markdown(f"使用 ChromaDB, `{EMBEDDING_MODEL_NAME}`, 和 `{GENERATION_MODEL_NAME}`。")
else:
    st.title("📄 医疗 RAG 系统 (Milvus Lite)")
    st.markdown(f"使用 Milvus Lite, `{EMBEDDING_MODEL_NAME}`, 和 `{GENERATION_MODEL_NAME}`。")

# --- 初始化与缓存 ---
vector_store_client = None
collection_is_ready = False

# Initialize vector store client based on configuration
if VECTOR_STORE_TYPE == "chromadb":
    vector_store_client = get_chroma_client()
    if vector_store_client:
        collection_is_ready = get_or_create_collection(vector_store_client)
else:
    vector_store_client = get_milvus_client()
    if vector_store_client:
        collection_is_ready = setup_milvus_collection(vector_store_client)

# 加载模型 (缓存) only if vector store client is available
if vector_store_client:
    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    generation_model, tokenizer = load_generation_model(GENERATION_MODEL_NAME)

    # 检查所有组件是否成功加载
    models_loaded = embedding_model and generation_model and tokenizer

    if collection_is_ready and models_loaded:
        # 加载数据 (未缓存)
        pubmed_data = load_data(DATA_FILE)

        # 如果需要则索引数据 (这会填充 id_to_doc_map)
        if pubmed_data:
            indexing_successful = index_data_if_needed(vector_store_client, pubmed_data, embedding_model)
        else:
            st.warning(f"无法从 {DATA_FILE} 加载数据。跳过索引。")
            indexing_successful = False # 如果没有数据，则视为不成功

        st.divider()

        # --- RAG 交互部分 ---
        if not indexing_successful and not id_to_doc_map:
             st.error("数据索引失败或不完整，且没有文档映射。RAG 功能已禁用。")
        else:
            query = st.text_input("请提出关于已索引医疗文章的问题:", key="query_input")

            if st.button("获取答案", key="submit_button") and query:
                start_time = time.time()

                # 1. 搜索向量存储
                with st.spinner("正在搜索相关文档..."):
                    retrieved_ids, distances = search_similar_documents(vector_store_client, query, embedding_model)

                if not retrieved_ids:
                    st.warning("在数据库中找不到相关文档。")
                else:
                    # 2. 从映射中检索上下文
                    retrieved_docs = [id_to_doc_map[id] for id in retrieved_ids if id in id_to_doc_map]

                    if not retrieved_docs:
                         st.error("检索到的 ID 无法映射到加载的文档。请检查映射逻辑。")
                    else:
                        st.subheader("检索到的上下文文档:")
                        for i, doc in enumerate(retrieved_docs):
                            # 如果距离可用则显示，否则只显示 ID
                            dist_str = f", 距离: {distances[i]:.4f}" if distances else ""
                            with st.expander(f"文档 {i+1} (ID: {retrieved_ids[i]}{dist_str}) - {doc['title'][:60]}"):
                                st.write(f"**标题:** {doc['title']}")
                                st.write(f"**摘要:** {doc['abstract']}") # 假设 'abstract' 存储的是文本块

                        st.divider()

                        # 3. 生成答案
                        st.subheader("生成的答案:")
                        with st.spinner("正在根据上下文生成答案..."):
                            answer = generate_answer(query, retrieved_docs, generation_model, tokenizer)
                            st.write(answer)

                end_time = time.time()
                st.info(f"总耗时: {end_time - start_time:.2f} 秒")

    else:
        if VECTOR_STORE_TYPE == "chromadb":
            st.error("加载模型或设置 ChromaDB collection 失败。请检查日志和配置。")
        else:
            st.error("加载模型或设置 Milvus Lite collection 失败。请检查日志和配置。")
else:
    if VECTOR_STORE_TYPE == "chromadb":
        st.error("初始化 ChromaDB 客户端失败。请检查日志。")
    else:
        st.error("初始化 Milvus Lite 客户端失败。请检查日志。")


# --- 页脚/信息侧边栏 ---
st.sidebar.header("系统配置")
if VECTOR_STORE_TYPE == "chromadb":
    st.sidebar.markdown(f"**向量存储:** ChromaDB")
    st.sidebar.markdown(f"**数据路径:** `{CHROMA_PERSIST_DIRECTORY}`")
else:
    st.sidebar.markdown(f"**向量存储:** Milvus Lite")
    st.sidebar.markdown(f"**数据路径:** `{MILVUS_LITE_DATA_PATH}`")
st.sidebar.markdown(f"**Collection:** `{COLLECTION_NAME}`")
st.sidebar.markdown(f"**数据文件:** `{DATA_FILE}`")
st.sidebar.markdown(f"**嵌入模型:** `{EMBEDDING_MODEL_NAME}`")
st.sidebar.markdown(f"**生成模型:** `{GENERATION_MODEL_NAME}`")
st.sidebar.markdown(f"**最大索引数:** `{MAX_ARTICLES_TO_INDEX}`")
st.sidebar.markdown(f"**检索 Top K:** `{TOP_K}`")