import streamlit as st
import matplotlib
matplotlib.use('Agg') # <--- CORREÇÃO CRÍTICA: Impede o travamento gráfico
import matplotlib.pyplot as plt

import os
import sys
import shutil
import pandas as pd
from pathlib import Path
import ollama
import time
import torch
import gc
import subprocess
import json
import psutil
from contextlib import redirect_stdout
import io

# Adiciona raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ingestion.processor import WhatsAppProcessor
from src.embeddings.vector_store import build_vector_store
from src.llm.chat_engine import WhatsAppChat
from src.analysis.trends import generate_trends
from src.analysis.network_graph import generate_network_graph

# --- CONFIGURAÇÃO ---
st.set_page_config(
    page_title="WhatsApp AI Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicialização de Estado
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
REPORTS_DIR = Path("data/reports")
INTERNAL_CHAT_PATH = DATA_RAW / "current_chat_import.txt" 
PARQUET_PATH = DATA_PROCESSED / "chat_history.parquet"

# --- MONITORAMENTO ---
def get_hw_metrics():
    metrics = {'cpu': psutil.cpu_percent(), 'ram': psutil.virtual_memory().percent, 'gpu_load': 0, 'gpu_temp': 0}
    try:
        res = subprocess.run(['rocm-smi', '--showuse', '--showtemp', '--json'], capture_output=True, text=True)
        if res.returncode == 0:
            d = json.loads(res.stdout)
            card = list(d.keys())[0]
            metrics['gpu_load'] = float(d[card].get('GPU use (%)', 0))
            metrics['gpu_temp'] = float(d[card].get('Temperature (Sensor edge) (C)', 0))
    except: pass
    return metrics

# --- PIPELINE ---
def run_pipeline(hw_placeholder):
    status = st.status("🚀 Iniciando Motor...", expanded=True)
    log_area = status.empty()
    logs = []

    def log(msg):
        print(msg) # Terminal
        logs.append(msg)
        log_area.code("\n".join(logs[-8:]), language="bash") # UI
        
        # ATUALIZA HARDWARE A CADA LOG (Truque para ser "Live" sem travar)
        hw = get_hw_metrics()
        with hw_placeholder.container():
             c1, c2 = st.columns(2)
             c1.metric("GPU Load", f"{hw['gpu_load']}%")
             c1.metric("Temp", f"{hw['gpu_temp']}°C")
             c2.metric("CPU", f"{hw['cpu']}%")
             c2.metric("RAM", f"{hw['ram']}%")

    try:
        # 1. Ingestão
        log("📂 [1/5] Lendo arquivo...")
        proc = WhatsAppProcessor()
        with redirect_stdout(io.StringIO()):
            df = proc.parse_file(str(INTERNAL_CHAT_PATH))
            proc.save_processed(df, str(PARQUET_PATH))
        
        if df.empty:
            status.update(label="❌ Erro: Arquivo inválido", state="error")
            st.stop()
        log(f"✅ Ingestão: {len(df)} msgs.")

        # 2. Vetores
        log("🧠 [2/5] Vetorização (GPU)...")
        with redirect_stdout(io.StringIO()):
            build_vector_store(str(PARQUET_PATH))
        log("✅ Vetores criados.")

        # 3. Gráficos (AQUI QUE TRAVAVA)
        log("📊 [3/5] Gerando Gráficos...")
        # O backend 'Agg' configurado no topo impede o travamento
        with redirect_stdout(io.StringIO()):
            generate_trends()
            generate_network_graph()
        log("✅ Gráficos gerados.")

        # 4. Sentimento
        log("💔 [4/5] Analisando Sentimentos...")
        proc = subprocess.Popen(["python", "src/analysis/sentiment.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            # Atualiza log e hardware a cada linha do sentimento
            log(f"  | {line.strip()}")
        proc.wait()
        log("✅ Sentimento concluído.")

        # 5. Engine
        log("🤖 [5/5] Carregando Chat Engine...")
        st.session_state.chat_engine = WhatsAppChat()
        
        status.update(label="✨ Processamento Completo!", state="complete", expanded=False)
        st.session_state.processing_complete = True
        st.rerun()

    except Exception as e:
        status.update(label="❌ Falha", state="error")
        st.error(f"{e}")
        import traceback
        traceback.print_exc()

def reset_session():
    if os.path.exists(DATA_RAW): shutil.rmtree(DATA_RAW)
    if os.path.exists(DATA_PROCESSED): shutil.rmtree(DATA_PROCESSED)
    if os.path.exists(REPORTS_DIR): shutil.rmtree(REPORTS_DIR)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    st.session_state.messages = []
    st.session_state.chat_engine = None
    st.session_state.processing_complete = False
    torch.cuda.empty_cache()
    gc.collect()

def get_stats(raw, df):
    stats = {'total': 0, 'media': 0, 'valid': len(df), 'period': '-'}
    if os.path.exists(raw):
        with open(raw, 'r', errors='ignore') as f:
            for l in f:
                stats['total'] += 1
                if "omitida" in l or "omitted" in l: stats['media'] += 1
    
    if 'date' in df.columns and not df.empty:
        df['dt'] = pd.to_datetime(df['date'], format='%m/%d/%y', errors='coerce')
        if not df['dt'].isnull().all():
            stats['period'] = df['dt'].dt.strftime('%m/%Y').value_counts().idxmax()
    return stats, df['author'].unique().tolist() if not df.empty else []

def get_models():
    try:
        m = ollama.list()
        if 'models' in m: return [x['name'] for x in m['models']]
    except: pass
    return ["deepseek-r1:8b"]

# --- UI LAYOUT ---
# Sidebar
with st.sidebar:
    st.header("🎛️ Painel")
    
    # Placeholder para Hardware (Será atualizado durante o processamento)
    hw_placeholder = st.empty()
    
    # Exibe hardware estático se não estiver processando
    if not st.session_state.get("is_processing", False):
        hw = get_hw_metrics()
        with hw_placeholder.container():
             c1, c2 = st.columns(2)
             c1.metric("GPU", f"{hw['gpu_load']}%", help="Carga Atual")
             c1.metric("Temp", f"{hw['gpu_temp']}°C")
             c2.metric("CPU", f"{hw['cpu']}%")
             c2.metric("RAM", f"{hw['ram']}%")
    
    st.divider()
    model = st.selectbox("Modelo IA", get_models())
    
    uploaded = st.file_uploader("Arquivo .txt", type="txt")
    if uploaded:
        if st.button("🔄 Iniciar Análise", type="primary", use_container_width=True):
            reset_session()
            with open(INTERNAL_CHAT_PATH, "wb") as f:
                f.write(uploaded.getbuffer())
            st.session_state.is_processing = True
            run_pipeline(hw_placeholder) # Passa o placeholder pra atualizar live
            st.session_state.is_processing = False

# Main
if st.session_state.processing_complete:
    df = pd.read_parquet(PARQUET_PATH)
    stats, participants = get_stats(INTERNAL_CHAT_PATH, df)
    
    tab1, tab2 = st.tabs(["📊 Dashboard", "💬 Chat"])
    
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Linhas", f"{stats['total']:,}")
        c2.metric("Msg Texto", f"{stats['valid']:,}")
        c3.metric("Mídias", f"{stats['media']:,}")
        c4.metric("Pico", stats['period'])
        st.divider()
        
        l, r = st.columns(2)
        with l:
            if (REPORTS_DIR / "interaction_network.png").exists():
                st.image(str(REPORTS_DIR / "interaction_network.png"), caption="Rede")
            if (REPORTS_DIR / "wordcloud.png").exists():
                st.image(str(REPORTS_DIR / "wordcloud.png"), caption="Termos")
        with r:
            if (REPORTS_DIR / "sentiment_timeline.png").exists():
                st.image(str(REPORTS_DIR / "sentiment_timeline.png"), caption="Humor")
            if (REPORTS_DIR / "top_participants.png").exists():
                st.image(str(REPORTS_DIR / "top_participants.png"), caption="Ativos")

    with tab2:
        msgs_container = st.container()
        with msgs_container:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]): st.markdown(m["content"])
            st.write("")
            
        if prompt := st.chat_input("Pergunte..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with msgs_container:
                st.chat_message("user").markdown(prompt)
                
                with st.chat_message("assistant"):
                    status = st.empty()
                    resp = st.empty()
                    
                    ctx = st.session_state.chat_engine.get_context(prompt)
                    sys_p = f"Analista de WhatsApp.\nParticipantes: {', '.join(participants)}\nContexto: {ctx}"
                    
                    try:
                        stream = ollama.chat(model=model, messages=[{'role':'system','content':sys_p}, {'role':'user','content':prompt}], stream=True)
                        full, buf, thinking = "", "", False
                        expander = status.status("🧠...", expanded=False)
                        
                        for chunk in stream:
                            txt = chunk['message']['content']
                            if "<think>" in txt: thinking=True; txt=txt.replace("<think>",""); expander.update(expanded=True)
                            if "</think>" in txt: thinking=False; txt=txt.replace("</think>",""); expander.update(label="💡 Ok", state="complete", expanded=False)
                            
                            if thinking: expander.write(txt)
                            else: full+=txt; resp.markdown(full + "▌")
                        
                        resp.markdown(full)
                        st.session_state.messages.append({"role": "assistant", "content": full})
                    except Exception as e: st.error(f"Erro: {e}")

else:
    st.markdown("<h1 style='text-align: center;'>WhatsApp AI Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Carregue seu arquivo para começar 👈</p>", unsafe_allow_html=True)
