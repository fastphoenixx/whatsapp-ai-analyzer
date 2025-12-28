import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
import sys

# Configurações
COLLECTION_NAME = "whatsapp_chat"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_DB_PATH = "./data/qdrant_db"
BATCH_SIZE = 64

def build_vector_store(parquet_path):
    print("🚀 Iniciando Pipeline de Vetorização...")
    
    if not os.path.exists(parquet_path):
        print(f"❌ Arquivo não encontrado: {parquet_path}")
        return
    
    df = pd.read_parquet(parquet_path)
    print(f"📂 Dados carregados: {len(df)} mensagens.")

    # Prepara texto
    df['text_to_embed'] = df['author'] + ": " + df['content']
    documents = df['text_to_embed'].tolist()
    metadata = df[['date', 'time', 'author', 'content']].to_dict('records')

    # Carrega Modelo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 Carregando modelo '{MODEL_NAME}' no dispositivo: {device.upper()}")
    
    encoder = SentenceTransformer(MODEL_NAME, device=device)

    # Inicializa Qdrant
    client = QdrantClient(path=VECTOR_DB_PATH)
    
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE
        )
    )

    print("⚡ Gerando embeddings e indexando (Isso usa a GPU)...")
    
    total_batches = len(documents) // BATCH_SIZE + 1
    
    for i in tqdm(range(0, len(documents), BATCH_SIZE), total=total_batches):
        batch_docs = documents[i : i + BATCH_SIZE]
        batch_meta = metadata[i : i + BATCH_SIZE]
        
        embeddings = encoder.encode(batch_docs, show_progress_bar=False)
        
        points = [
            models.PointStruct(
                id=i + idx,
                vector=emb.tolist(),
                payload=meta
            )
            for idx, (emb, meta) in enumerate(zip(embeddings, batch_meta))
        ]
        
        client.upload_points(
            collection_name=COLLECTION_NAME,
            points=points
        )

    print(f"✅ Sucesso! Banco vetorial salvo em '{VECTOR_DB_PATH}'")
    # REMOVIDO: Bloco de teste de busca que causava crash no Streamlit
