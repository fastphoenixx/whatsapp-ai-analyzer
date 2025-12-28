import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
from termcolor import colored
import gc

# --- CONFIG ---
INPUT_FILE = "data/processed/chat_history.parquet"
OUTPUT_DIR = "data/reports"
MODEL_NAME = "pysentimiento/robertuito-sentiment-analysis"

SYSTEM_STOPWORDS = [
    "mídia omitida", "media omitted", "missed voice call", "chamada de voz perdida",
    "chamada de vídeo perdida", "location:", "localização:", "file attached",
    "arquivo anexado", "vcard", "contact card", "contato:", "null",
    "mudou o nome do grupo", "adicionou", "removeu", "saiu do grupo",
    "código de segurança", "waiting for this message", "aguardando esta mensagem"
]

def is_valid_message(text):
    if not isinstance(text, str) or len(text) < 2: return False
    text_lower = text.lower()
    if any(term in text_lower for term in SYSTEM_STOPWORDS): return False
    return True

def analyze_sentiment():
    print(colored("🚀 Iniciando Análise de Sentimento (MODO TURBO)...", "cyan"))
    
    # Limpeza prévia
    torch.cuda.empty_cache()
    gc.collect()

    if not Path(INPUT_FILE).exists():
        print(colored("❌ Arquivo de dados não encontrado.", "red"))
        return

    # 1. Carregar e Filtrar
    df = pd.read_parquet(INPUT_FILE)
    df = df[df['content'].apply(is_valid_message)].copy()
    msgs = df['content'].tolist()
    print(colored(f"📂 Processando {len(msgs)} mensagens na GPU...", "cyan"))

    # 2. Setup OTIMIZADO
    device = 0 if torch.cuda.is_available() else -1
    
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=device,
        top_k=None,
        truncation=True,
        max_length=128,
        torch_dtype=torch.float16,
        batch_size=256  # <--- AQUI ESTÁ A MÁGICA (Aumentado 8x)
    )

    # 3. Inferência Contínua (Sem loop manual lento)
    print(colored("⚡ Classificando em alta velocidade...", "yellow"))
    
    results = []
    # O pipeline iterável é muito mais rápido pois faz pre-fetch dos dados
    for output in tqdm(sentiment_pipeline(msgs), total=len(msgs)):
        # output é lista de scores [{'label': 'POS', 'score': 0.9}, ...]
        best = max(output, key=lambda x: x['score'])
        
        label = best['label']
        if label == 'POS': val = 1
        elif label == 'NEG': val = -1
        else: val = 0 
        
        results.append({'sentiment_label': label, 'sentiment_val': val})

    # 4. Consolidação
    sentiment_df = pd.DataFrame(results, index=df.index)
    df = pd.concat([df, sentiment_df], axis=1)

    # 5. Relatórios Visuais
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    plt.style.use('dark_background')

    # A) Pizza
    plt.figure(figsize=(10, 10))
    colors = {'POS': '#00ff00', 'NEU': '#888888', 'NEG': '#ff0000'}
    counts = df['sentiment_label'].value_counts()
    pie_colors = [colors.get(l, '#ffffff') for l in counts.index]
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=pie_colors, startangle=140)
    plt.title(f'Distribuição de Sentimento ({len(df)} msgs)')
    plt.savefig(f"{OUTPUT_DIR}/sentiment_distribution.png")
    plt.close()

    # B) Timeline
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y', errors='coerce')
    daily_sentiment = df.groupby('date')['sentiment_val'].mean()
    rolling_sentiment = daily_sentiment.rolling(window=7).mean()

    plt.figure(figsize=(16, 8))
    plt.axhline(y=0, color='white', linestyle='--', alpha=0.3)
    plt.plot(rolling_sentiment.index, rolling_sentiment.values, color='#00ffff', linewidth=2, label="Média Móvel (7 dias)")
    
    # Áreas coloridas
    plt.fill_between(rolling_sentiment.index, rolling_sentiment.values, 0, 
                     where=(rolling_sentiment.values >= 0), color='green', alpha=0.3, interpolate=True)
    plt.fill_between(rolling_sentiment.index, rolling_sentiment.values, 0, 
                     where=(rolling_sentiment.values < 0), color='red', alpha=0.3, interpolate=True)
    
    plt.title('Evolução do Humor do Grupo')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sentiment_timeline.png")
    plt.close()

    print(colored(f"\n✅ Concluído! Gráficos gerados em: {OUTPUT_DIR}", "green"))

if __name__ == "__main__":
    analyze_sentiment()
