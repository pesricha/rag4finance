import os
from parse_pdfs import PDFChunker
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
load_dotenv('./.env') 

# === 1. Paths & PDF parsing ===
root_dir = "/home/biomedialab/Desktop/Sandeep/Placements/Projects/rag4finance"
chunker = PDFChunker()
data_chunks = chunker.parse_all_pdfs_by_month(f"{root_dir}/data")

print(f"📄 Parsed {len(data_chunks)} chunks from PDFs")


# === 2. Load embedding model ===
print("📦 Loading embedding model...")
model = SentenceTransformer("all-distilroberta-v1")

# === 3. Connect to Elasticsearch ===
print("🔌 Connecting to Elasticsearch...")
es = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "z-3n8-w7USNszckT7af*"),
    verify_certs=False  # for local dev only
)


print(es.info())


index_name = "pdf_chunks"

# === 4. Create index if not exists ===
if not es.indices.exists(index=index_name):
    print(f"📂 Creating index '{index_name}'...")
    es.indices.create(
        index=index_name,
        mappings={
            "properties": {
                "month": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "text": {"type": "text"},
                "token_count": {"type": "integer"},
                "embedding": {"type": "dense_vector", "dims": 768}
            }
        }
    )
else:
    print(f"✅ Index '{index_name}' already exists.")

# === 5. Function to index chunks ===
def index_chunks(chunks):
    for i, chunk in enumerate(chunks, start=1):
        try:
            # Compute embedding
            embedding = model.encode(chunk["text"], convert_to_numpy=True).tolist()

            # Document to index
            doc = {
                "month": chunk["month"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "token_count": chunk["token_count"],
                "embedding": embedding
            }

            # Index in Elasticsearch
            es.index(index=index_name, id=chunk["chunk_id"], document=doc)
            print(f"✅ Indexed chunk {i}/{len(chunks)} → {chunk['chunk_id']}")

        except Exception as e:
            print(f"❌ Error indexing chunk {chunk.get('chunk_id', 'UNKNOWN')}: {e}")

# === 6. Index the parsed chunks ===
if data_chunks:
    index_chunks(data_chunks)
    print("🎯 All chunks indexed successfully!")
else:
    print("⚠️ No chunks to index!")


# === 7. Semantic search function ===
def semantic_search(query, top_k=5):
    # Step 1: Embed the query
    query_embedding = model.encode(query, convert_to_numpy=True).tolist()

    # Step 2: KNN search in Elasticsearch
    response = es.search(
        index=index_name,
        knn={
            "field": "embedding",
            "query_vector": query_embedding,
            "k": top_k,
            "num_candidates": top_k * 2
        },
        _source=["month", "chunk_id", "text", "token_count"]
    )

    # Step 3: Print results
    hits = response["hits"]["hits"]
    print(f"\n🔍 Top {len(hits)} results for query: '{query}'\n")
    for i, hit in enumerate(hits, start=1):
        score = hit["_score"]
        src = hit["_source"]
        print(f"{i}. [Score: {score:.2f}] {src['month']} - {src['chunk_id']}")
        print(f"   {src['text'][:150]}...\n")

    return hits

if __name__ == "__main__":
    # Example queries
    semantic_search("movie ticket transactions")
    semantic_search("biggest expense in August 2024")