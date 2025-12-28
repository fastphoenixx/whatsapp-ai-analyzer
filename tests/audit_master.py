import os
import sys
import time
import requests
import pandas as pd
import torch
from pathlib import Path
from qdrant_client import QdrantClient
from termcolor import colored

# --- CONFIG ---
PARQUET_PATH = "data/processed/chat_history.parquet"
QDRANT_PATH = "./data/qdrant_db"
COLLECTION = "whatsapp_chat"
REPORTS_DIR = "data/reports"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
# O modelo que definimos que está funcionando
MODEL_NAME = "deepseek-r1:8b" 

def log(msg, status="INFO"):
    colors = {"INFO": "cyan", "PASS": "green", "FAIL": "red", "WARN": "yellow"}
    prefix = {"INFO": "ℹ️", "PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}
    print(colored(f"{prefix[status]} {msg}", colors[status]))

def check_step(name, func):
    print(f"\n🔹 Testando: {name}...")
    try:
        func()
        log(f"{name}: SUCESSO", "PASS")
    except Exception as e:
        log(f"{name}: FALHOU - {e}", "FAIL")
        return False
    return True

# --- TESTES ---

def test_hardware():
    if not torch.cuda.is_available():
        raise Exception("CUDA/ROCm não detectado pelo PyTorch.")
    dev = torch.cuda.get_device_name(0)
    if "AMD" not in dev and "Radeon" not in dev:
        log(f"Dispositivo detectado estranho: {dev}", "WARN")
    else:
        log(f"GPU Detectada: {dev}", "INFO")

def test_data_integrity():
    if not os.path.exists(PARQUET_PATH):
        raise Exception("Arquivo Parquet não existe.")
    df = pd.read_parquet(PARQUET_PATH)
    if len(df) < 10:
        raise Exception(f"Poucos dados no dataset: {len(df)}")
    log(f"Dataset carregado: {len(df)} linhas.", "INFO")

def test_vector_db():
    client = QdrantClient(path=QDRANT_PATH)
    colls = [c.name for c in client.get_collections().collections]
    if COLLECTION not in colls:
        raise Exception(f"Coleção '{COLLECTION}' não encontrada no Qdrant.")
    count = client.count(COLLECTION).count
    if count == 0:
        raise Exception("Banco vetorial está vazio.")
    log(f"Vetores indexados: {count}", "INFO")

def test_llm_connection():
    # Teste rápido de ping no Ollama
    payload = {
        "model": MODEL_NAME,
        "prompt": "Responda apenas com a palavra 'Ok'.",
        "stream": False
    }
    try:
        start = time.time()
        res = requests.post(OLLAMA_URL, json=payload, timeout=30)
        duration = time.time() - start
        
        if res.status_code != 200:
            raise Exception(f"Ollama retornou erro {res.status_code}")
        
        response_text = res.json().get('response', '').strip()
        log(f"LLM Respondeu em {duration:.2f}s: '{response_text}'", "INFO")
        
    except requests.exceptions.ConnectionError:
        raise Exception("Não foi possível conectar ao Ollama (127.0.0.1:11434). O servidor está rodando?")

def test_analysis_artifacts():
    expected_files = ["top_participants.png", "wordcloud.png", "timeline.png"]
    missing = []
    for f in expected_files:
        if not os.path.exists(os.path.join(REPORTS_DIR, f)):
            missing.append(f)
    
    if missing:
        raise Exception(f"Arquivos de análise faltando: {missing}")
    log("Todos os gráficos da Sprint 5 (parcial) foram encontrados.", "INFO")

def main():
    print(colored("🔍 INICIANDO AUDITORIA MESTRE (SPRINTS 1-4 + 5 Parcial)", "white", attrs=["bold"]))
    
    steps = [
        ("Hardware & Drivers", test_hardware),
        ("Dados Processados", test_data_integrity),
        ("Banco Vetorial (Qdrant)", test_vector_db),
        ("Conexão LLM (DeepSeek)", test_llm_connection),
        ("Arquivos de Análise", test_analysis_artifacts)
    ]
    
    passed = 0
    for name, func in steps:
        if check_step(name, func):
            passed += 1
            
    print("\n" + "="*40)
    if passed == len(steps):
        print(colored(f"🎉 AUDITORIA COMPLETA: {passed}/{len(steps)} PASSARAM", "green", attrs=["bold"]))
        print(colored("O sistema está estável e alinhado com o plano adaptado.", "green"))
    else:
        print(colored(f"⚠️ AUDITORIA COM FALHAS: {passed}/{len(steps)} PASSARAM", "red", attrs=["bold"]))
        print(colored("Corrija os erros acima antes de prosseguir.", "red"))

if __name__ == "__main__":
    main()
