import streamlit as st
import chromadb
from chromadb.config import Settings
import time
import os

# Import config variables including the global map
from config import (
    CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME, EMBEDDING_DIM,
    MAX_ARTICLES_TO_INDEX, TOP_K, id_to_doc_map
)

@st.cache_resource
def get_chroma_client():
    """Initializes and returns a ChromaDB client instance."""
    try:
        st.write(f"Initializing ChromaDB client with persist directory: {CHROMA_PERSIST_DIRECTORY}")
        # Ensure the persist directory exists
        os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        
        # Create ChromaDB client with persistent storage
        client = chromadb.Client(Settings(
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            anonymized_telemetry=False  # Disable telemetry
        ))
        
        st.success("ChromaDB client initialized!")
        return client
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB client: {e}")
        return None

@st.cache_resource
def get_or_create_collection(_client):
    """Ensures the specified collection exists and is set up correctly in ChromaDB."""
    if not _client:
        st.error("ChromaDB client not available.")
        return False
    try:
        collection_name = COLLECTION_NAME
        dim = EMBEDDING_DIM
        
        # Get or create the collection
        collection = _client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Similar to L2 metric
            embedding_function=None  # We'll provide embeddings explicitly
        )
        
        # Get current entity count
        current_count = collection.count()
        st.write(f"Collection '{collection_name}' ready. Current entity count: {current_count}")
        
        return True  # Indicate collection is ready
        
    except Exception as e:
        st.error(f"Error setting up ChromaDB collection '{COLLECTION_NAME}': {e}")
        return False


def index_data_if_needed(client, data, embedding_model):
    """Checks if data needs indexing and performs it using ChromaDB."""
    global id_to_doc_map  # Modify the global map
    
    if not client:
        st.error("ChromaDB client not available for indexing.")
        return False
    
    collection_name = COLLECTION_NAME
    collection = client.get_collection(name=collection_name)
    
    # Get current entity count
    current_count = collection.count()
    st.write(f"Entities currently in ChromaDB collection '{collection_name}': {current_count}")
    
    data_to_index = data[:MAX_ARTICLES_TO_INDEX]  # Limit data for demo
    needed_count = 0
    docs_for_embedding = []
    doc_ids = []  # List of document IDs
    temp_id_map = {}  # Build a temporary map first
    
    # Prepare data
    with st.spinner("Preparing data for indexing..."):
        for i, doc in enumerate(data_to_index):
            title = doc.get('title', '') or ""
            abstract = doc.get('abstract', '') or ""
            # 仅使用标题来生成嵌入，提高中文检索的相关性
            if not title:
                continue
            
            doc_id = i  # Use list index as ID
            needed_count += 1
            temp_id_map[doc_id] = {
                'title': title, 'abstract': abstract, 'content': title
            }
            docs_for_embedding.append(title)  # 仅使用标题生成嵌入
            doc_ids.append(str(doc_id))  # ChromaDB uses string IDs
    
    
    if current_count < needed_count and docs_for_embedding:
        st.warning(f"Indexing required ({current_count}/{needed_count} documents found). This may take a while...")
        
        st.write(f"Embedding {len(docs_for_embedding)} documents...")
        with st.spinner("Generating embeddings..."):
            start_embed = time.time()
            embeddings = embedding_model.encode(docs_for_embedding, show_progress_bar=True)
            end_embed = time.time()
            st.write(f"Embedding took {end_embed - start_embed:.2f} seconds.")
        
        st.write("Inserting data into ChromaDB...")
        with st.spinner("Inserting..."):
            try:
                start_insert = time.time()
                # Insert data into ChromaDB
                collection.add(
                    ids=doc_ids,
                    embeddings=embeddings.tolist(),
                    documents=docs_for_embedding
                )
                
                # No need to persist explicitly - ChromaDB auto-persists
                
                end_insert = time.time()
                inserted_count = len(doc_ids)
                st.success(f"Successfully indexed {inserted_count} documents. Insert took {end_insert - start_insert:.2f} seconds.")
                
                # Update the global map
                id_to_doc_map.update(temp_id_map)
                return True
                
            except Exception as e:
                st.error(f"Error inserting data into ChromaDB: {e}")
                return False
    elif current_count >= needed_count:
        st.write("Data count suggests indexing is complete.")
        # Populate the global map if it's empty but indexing isn't needed
        if not id_to_doc_map:
            id_to_doc_map.update(temp_id_map)
        return True
    else:  # No docs_for_embedding found
        st.error("No valid text content found in the data to index.")
        return False


def search_similar_documents(client, query, embedding_model):
    """Searches ChromaDB for documents similar to the query."""
    if not client or not embedding_model:
        st.error("ChromaDB client or embedding model not available for search.")
        return [], []
    
    collection_name = COLLECTION_NAME
    collection = client.get_collection(name=collection_name)
    
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode([query])[0]
        
        # Search in ChromaDB with more results to ensure we get relevant ones
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=TOP_K * 2,  # Get more results for better filtering
            include=["documents", "distances"]
        )
        
        if not results or not results["ids"] or not results["ids"][0]:
            return [], []
        
        # Extract results
        hit_ids_str = results["ids"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]
        
        # Convert IDs to integers
        hit_ids = [int(id_str) for id_str in hit_ids_str]
        
        # Step 1: Find documents that contain the exact query keyword
        keyword_matched_docs = []
        for doc_id in hit_ids:
            if doc_id in id_to_doc_map:
                doc = id_to_doc_map[doc_id]
                title = doc.get('title', '')
                abstract = doc.get('abstract', '')
                if query in title or query in abstract:
                    keyword_matched_docs.append(doc_id)
        
        # Step 2: Create combined results with keyword matching flag
        all_results = []
        for i, (doc_id, distance) in enumerate(zip(hit_ids, distances)):
            is_keyword_matched = doc_id in keyword_matched_docs
            all_results.append((doc_id, distance, is_keyword_matched))
        
        # Step 3: Reorder results - keyword matches first, then by distance
        all_results.sort(key=lambda x: (-x[2], x[1]))
        
        # Step 4: Only keep top K results
        top_results = all_results[:TOP_K]
        
        # Extract final IDs and distances
        final_hit_ids = [result[0] for result in top_results]
        final_distances = [result[1] for result in top_results]
        
        return final_hit_ids, final_distances
        
    except Exception as e:
        st.error(f"Error during ChromaDB search: {e}")
        return [], []
