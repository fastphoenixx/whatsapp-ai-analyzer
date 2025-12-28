import os
import sys
import torch
import pandas as pd
import requests
from pathlib import Path
from qdrant_client import QdrantClient
from termcolor import colored
import time

# --- CONFIG ---
PROJECT_ROOT = Path(".")
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = DATA_DIR / "reports"
PARQUET_PATH = DATA_DIR / "processed" / "chat_history.parquet"
QDRANT_PATH = DATA_DIR / "qdrant_db"
COLLECTION_NAME = "whatsapp_chat"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
# Modelo que confirmamos estar funcionando
LLM_MODEL = "deepseek-r1:8b" 

def print_header(title):
    print(colored(f"\n{'='*60}", "white"))
    print(colored(f"🔍 AUDITORIA: {title}", "white", attrs=['bold']))
    print(colored(f"{'='*60}", "white"))

def check(name, condition, success_msg, fail_msg):
    if condition:
        print(colored(f"✅ [PASS] {name}: {success_msg}", "green"))
        return True
    else:
        print(colored(f"❌ [FAIL] {name}: {fail_msg}", "red"))
        return False

def audit_sprint_1_foundation():
    print_header("SPRINT 1: FOUNDATION & HARDWARE")
    
    # 1. Check GPU
    gpu_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_ok else "None"
    check("GPU Access", gpu_ok, f"Detectada {gpu_name}", "PyTorch não vê a GPU.")
    
    # 2. Check Raw Data Exists (Simbolico)
    raw_exists = (DATA_DIR / "raw").exists()
    check("Estrutura de Pastas", raw_exists, "Diretório data/raw existe", "Falta data/raw")

def audit_sprint_2_data_pipeline():
    print_header("SPRINT 2: DADOS E VETORES")
    
    # 1. Parquet
    if check("Arquivo Processado", PARQUET_PATH.exists(), "Parquet encontrado", "Parquet não gerado"):
        try:
            df = pd.read_parquet(PARQUET_PATH)
            check("Integridade dos Dados", len(df) > 0, f"{len(df)} mensagens carregadas", "Arquivo vazio")
        except:
            check("Integridade dos Dados", False, "", "Erro ao ler Parquet")

    # 2. Qdrant
    try:
        client = QdrantClient(path=str(QDRANT_PATH))
        colls = [c.name for c in client.get_collections().collections]
        has_coll = COLLECTION_NAME in colls
        check("Qdrant DB", has_coll, f"Coleção '{COLLECTION_NAME}' ativa", "Coleção não encontrada")
        
        if has_coll:
            count = client.count(COLLECTION_NAME).count
            check("Indexação", count > 0, f"{count} vetores indexados", "Banco vetorial vazio")
    except Exception as e:
        check("Qdrant DB", False, "", f"Erro de conexão: {e}")

def audit_sprint_3_llm():
    print_header("SPRINT 3: INTELIGÊNCIA ARTIFICIAL (RAG)")
    
    payload = {
        "model": LLM_MODEL,
        "prompt": "Say 'System Operational' in Portuguese.",
        "stream": False
    }
    
    try:
        start = time.time()
        res = requests.post(OLLAMA_URL, json=payload, timeout=30)
        duration = time.time() - start
        
        if res.status_code == 200:
            ans = res.json().get('response', '').strip()
            check("Conexão Ollama", True, f"Resposta em {duration:.2f}s: '{ans}'", "")
        else:
            check("Conexão Ollama", False, "", f"Status Code {res.status_code}")
            
    except Exception as e:
        check("Conexão Ollama", False, "", f"Servidor offline? {e}")

def audit_sprint_5_analysis():
    print_header("SPRINT 5: ANALYTICS & INSIGHTS")
    
    expected_reports = [
        "top_participants.png",
        "wordcloud.png",
        "timeline.png",
        "sentiment_distribution.png",
        "sentiment_timeline.png",
        "interaction_network.png"
    ]
    
    all_exist = True
    for report in expected_reports:
        path = REPORTS_DIR / report
        if path.exists():
            print(colored(f"  📄 Encontrado: {report}", "cyan"))
        else:
            print(colored(f"  missing: {report}", "red"))
            all_exist = False
            
    check("Geração de Relatórios", all_exist, "Todos os 6 gráficos existem", "Faltam relatórios")

if __name__ == "__main__":
    print(colored("🚀 INICIANDO AUDITORIA FINAL DO SISTEMA (V1.0)", "magenta", attrs=['bold']))
    
    audit_sprint_1_foundation()
    audit_sprint_2_data_pipeline()
    audit_sprint_3_llm()
    # Sprint 4 é otimização (implícita nos testes acima)
    audit_sprint_5_analysis()
    
    print("\n")
    print(colored("CONCLUSÃO:", "white", attrs=['bold']))
    print(colored("Se todos os testes acima passaram, você está autorizado a iniciar a SPRINT 6 (API & Interface).", "green"))
