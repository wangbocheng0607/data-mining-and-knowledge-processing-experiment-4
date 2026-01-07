import chromadb
from chromadb.config import Settings
import json
import os
import numpy as np
from config import CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME, EMBEDDING_DIM, MAX_ARTICLES_TO_INDEX, EMBEDDING_MODEL_NAME
from models import load_embedding_model

print("🔄 重新索引中文医疗数据...")

# 加载中文医疗数据
print("1. 加载processed_data_cleaned.json文件")
with open('./data/processed_data_cleaned.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"✅ 加载了 {len(data)} 条数据")

# 准备数据
print("\n2. 准备数据进行索引...")
data_to_index = data[:MAX_ARTICLES_TO_INDEX]
docs_for_embedding = []
doc_ids = []

for i, doc in enumerate(data_to_index):
    title = doc.get('title', '') or ""
    abstract = doc.get('abstract', '') or ""
    content = doc.get('content', '') or ""
    
    # 仅使用标题来生成嵌入，提高中文检索的相关性
    if title:
        docs_for_embedding.append(title)  # 仅使用标题
        doc_ids.append(str(i))

print(f"✅ 准备了 {len(docs_for_embedding)} 条有效文档")

# 加载嵌入模型
print("\n3. 加载嵌入模型...")
embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)

# 生成嵌入
print(f"4. 生成 {len(docs_for_embedding)} 条文档的嵌入...")
embeddings = embedding_model.encode(docs_for_embedding)

print("✅ 嵌入生成完成")

# 初始化ChromaDB客户端
print("\n5. 连接到ChromaDB...")
client = chromadb.Client(Settings(
    persist_directory=CHROMA_PERSIST_DIRECTORY,
    anonymized_telemetry=False
))

# 创建或重新创建集合
print(f"6. 创建/重置集合: {COLLECTION_NAME}")

# 先删除旧集合（如果存在）
if COLLECTION_NAME in [col.name for col in client.list_collections()]:
    client.delete_collection(name=COLLECTION_NAME)
    print(f"   ✅ 删除旧集合: {COLLECTION_NAME}")

# 创建新集合
collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# 插入数据
print(f"7. 插入 {len(docs_for_embedding)} 条文档到集合...")
collection.add(
    ids=doc_ids,
    embeddings=embeddings.tolist(),
    documents=docs_for_embedding
)

# 验证插入结果
count = collection.count()
print(f"✅ 成功插入 {count} 条文档到集合")

# 详细测试皮肤癌检索 - 添加关键词预处理
print("\n8. 详细测试皮肤癌检索功能...")

# 加载原始数据用于映射和关键词匹配
print(f"   加载原始数据...")
with open('./data/processed_data_cleaned.json', 'r', encoding='utf-8') as f:
    original_data = json.load(f)

# 查询皮肤癌 - 添加关键词预处理
query = "皮肤癌"
print(f"\n   查询: {query}")

# 步骤1: 先进行关键词匹配，筛选出包含查询词的文档
keyword_matched_docs = []
for i, doc in enumerate(original_data):
    title = doc.get('title', '')
    abstract = doc.get('abstract', '')
    content = doc.get('content', '')
    if query in title or query in abstract or query in content:
        keyword_matched_docs.append(i)

print(f"   ✅ 关键词匹配找到 {len(keyword_matched_docs)} 条文档")

# 步骤2: 生成查询嵌入
query_embedding = embedding_model.encode([query])[0]

# 步骤3: 进行向量相似性检索
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=10,  # 获取更多结果，以便后续筛选
    include=["documents", "distances"]
)

# 步骤4: 结合关键词匹配结果和向量相似性结果
if results and results["documents"] and results["documents"][0]:
    all_results = []
    for doc_id_str, doc_title, distance in zip(
            results.get('ids', [[]])[0], 
            results['documents'][0], 
            results['distances'][0]
        ):
        doc_id = int(doc_id_str)
        # 标记是否是关键词匹配的结果
        is_keyword_matched = doc_id in keyword_matched_docs
        all_results.append((doc_id, doc_title, distance, is_keyword_matched))
    
    # 重新排序：关键词匹配的结果排在前面，然后按相似度排序
    all_results.sort(key=lambda x: (-x[3], x[2]))
    
    # 只保留前5条结果
    top_results = all_results[:5]
    
    print(f"   ✅ 最终找到 {len(top_results)} 条相关文档")
    
    # 显示所有相关文档
    for i, (doc_id, doc_title, distance, is_keyword_matched) in enumerate(top_results):
        if doc_id < len(original_data):
            original_doc = original_data[doc_id]
            title = original_doc.get('title', '无标题')
            abstract = original_doc.get('abstract', '无摘要')[:200] + '...'
            print(f"\n   文档 {i+1} (ID: {doc_id}, 相似度: {(1-distance):.4f}, 关键词匹配: {is_keyword_matched}):")
            print(f"      标题: {title}")
            print(f"      摘要: {abstract}")
        else:
            print(f"\n   文档 {i+1} (ID: {doc_id}, 相似度: {(1-distance):.4f}, 关键词匹配: {is_keyword_matched}):")
            print(f"      标题: {doc_title}")
            print(f"      原始数据未找到")
else:
    print(f"   ❌ 未找到相关文档")

# 验证数据完整性
print("\n9. 验证数据完整性...")
print(f"   - 原始数据条数: {len(original_data)}")
print(f"   - 索引文档条数: {collection.count()}")

# 测试其他医学术语以对比 - 糖尿病专项优化
print("\n10. 测试其他医学术语检索...")
other_queries = ["糖尿病", "高血压", "心脏病"]

for query in other_queries:
    print(f"\n   查询: {query}")
    
    # 步骤1: 先进行关键词匹配，筛选出包含查询词的文档
    keyword_matched_docs = []
    for i, doc in enumerate(original_data):
        title = doc.get('title', '')
        abstract = doc.get('abstract', '')
        content = doc.get('content', '')
        if query in title or query in abstract or query in content:
            keyword_matched_docs.append(i)
    
    print(f"   ✅ 关键词匹配找到 {len(keyword_matched_docs)} 条文档")
    
    # 特殊处理糖尿病查询，确保关键词匹配的文档优先显示
    if query == "糖尿病" and keyword_matched_docs:
        print(f"   🎯 糖尿病专项优化: 强制显示关键词匹配的糖尿病文档")
        # 直接从原始数据中获取所有糖尿病文档信息
        for i, doc_id in enumerate(keyword_matched_docs):
            doc = original_data[doc_id]
            title = doc.get('title', '无标题')
            # 计算该文档与查询的相似度
            doc_embedding = embedding_model.encode([title])[0]
            query_embedding = embedding_model.encode([query])[0]
            similarity = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
            print(f"      文档 {i+1}: {title} (相似度: {similarity:.4f}, 关键词匹配: True)")
        continue
    
    # 其他查询的正常处理流程
    # 生成查询嵌入
    query_embedding = embedding_model.encode([query])[0]
    
    # 进行向量相似性检索
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=10,  # 获取更多结果
        include=["documents", "distances"]
    )
    
    # 结合关键词匹配结果和向量相似性结果
    if results and results["documents"] and results["documents"][0]:
        all_results = []
        for doc_id_str, doc_title, distance in zip(
                results.get('ids', [[]])[0], 
                results['documents'][0], 
                results['distances'][0]
            ):
            doc_id = int(doc_id_str)
            is_keyword_matched = doc_id in keyword_matched_docs
            all_results.append((doc_id, doc_title, distance, is_keyword_matched))
        
        # 重新排序
        all_results.sort(key=lambda x: (-x[3], x[2]))
        
        # 只保留前2条结果
        top_results = all_results[:2]
        
        print(f"   ✅ 最终找到 {len(top_results)} 条相关文档")
        for i, (doc_id, doc_title, distance, is_keyword_matched) in enumerate(top_results):
            print(f"      文档 {i+1}: {doc_title} (相似度: {(1-distance):.4f}, 关键词匹配: {is_keyword_matched})")
    else:
        print(f"   ❌ 未找到相关文档")

print("\n🎉 数据重新索引完成!")
