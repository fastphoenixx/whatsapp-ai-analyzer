import os
import sys
import torch
import pandas as pd
from pathlib import Path
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from termcolor import colored  # pip install termcolor se não tiver, ou removemos

# Configurações
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "whatsapp_chat"
VECTOR_DB_PATH = "./data/qdrant_db"

def print_status(step, status, message):
    icon = "✅" if status else "❌"
    color = "green" if status else "red"
    print(f"{icon} [{step}] {message}")
    if not status:
        sys.exit(1)

def check_structure():
    print("\n📁 1. Verificando Estrutura de Pastas...")
    required = [
        "data/raw", "data/processed", "data/qdrant_db", 
        "src/ingestion", "src/embeddings", "src/llm"
    ]
    for path in required:
        if os.path.exists(path):
            print_status("DIR", True, f"Encontrado: {path}")
        else:
            print_status("DIR", False, f"Faltando: {path}")

def check_hardware():
    print("\n⚙️ 2. Verificando Hardware (ROCm/GPU)...")
    try:
        is_cuda = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if is_cuda else "Unknown"
        
        # Aceitamos "AMD Radeon RX 6600 XT" ou similar
        status = is_cuda and "AMD" in device_name or "Radeon" in device_name
        
        print_status("GPU", status, f"Detectado: {device_name}")
        
        # Teste de tensor na VRAM
        if status:
            t = torch.ones(1).cuda()
            print_status("VRAM", True, "Alocação de memória na GPU funcionou.")
            
    except Exception as e:
        print_status("GPU", False, f"Erro crítico no PyTorch: {e}")

def check_data():
    print("\n💾 3. Verificando Integridade dos Dados...")
    parquet_path = "data/processed/chat_history.parquet"
    
    if os.path.exists(parquet_path):
        try:
            df = pd.read_parquet(parquet_path)
            status = len(df) > 0
            print_status("DATA", status, f"Parquet legível com {len(df)} mensagens.")
        except:
            print_status("DATA", False, "Arquivo Parquet corrompido.")
    else:
        print_status("DATA", False, "Arquivo Parquet não encontrado.")

def check_vector_db():
    print("\n🧠 4. Verificando Busca Semântica (Vector DB)...")
    
    # 1. Conexão
    try:
        client = QdrantClient(path=VECTOR_DB_PATH)
        collections = [c.name for c in client.get_collections().collections]
        
        if COLLECTION_NAME in collections:
            count = client.count(COLLECTION_NAME).count
            print_status("DB", True, f"Coleção '{COLLECTION_NAME}' ativa com {count} vetores.")
        else:
            print_status("DB", False, f"Coleção '{COLLECTION_NAME}' não encontrada.")
            return
            
    except Exception as e:
        print_status("DB", False, f"Erro ao conectar no Qdrant: {e}")
        return

    # 2. Teste Real de Busca (Load Model + Inference)
    print("   ⏳ Carregando modelo para teste de inferência...")
    try:
        encoder = SentenceTransformer(MODEL_NAME, device="cuda")
        query_text = "bom dia ou cumprimento"
        query_vector = encoder.encode(query_text).tolist()
        
        # USANDO A SINTAXE NOVA QUE DESCOBRIMOS NO DEBUG
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3
        ).points
        
        if len(results) > 0:
            top_msg = results[0].payload['content']
            print_status("SEARCH", True, f"Busca funcionou! Top result para '{query_text}': \"{top_msg}\"")
        else:
            print_status("SEARCH", False, "Busca retornou 0 resultados (banco vazio?).")
            
    except Exception as e:
        print_status("SEARCH", False, f"Erro na inferência ou busca: {e}")

if __name__ == "__main__":
    print("🔍 INICIANDO AUDITORIA DO SISTEMA WHATSAPP AI...")
    check_structure()
    check_hardware()
    check_data()
    check_vector_db()
    print("\n🎉 SISTEMA 100% OPERACIONAL. PRONTO PARA SPRINT 3 (LLM).")
