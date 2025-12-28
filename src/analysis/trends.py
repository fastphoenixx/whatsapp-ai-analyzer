import matplotlib
matplotlib.use('Agg') # <--- OBRIGATÓRIO PARA NÃO TRAVAR O SERVIDOR
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
from pathlib import Path
from termcolor import colored

plt.style.use('dark_background')
sns.set_palette("husl")

INPUT_FILE = "data/processed/chat_history.parquet"
OUTPUT_DIR = "data/reports"

def generate_trends():
    print(colored("📊 Iniciando Trends...", "cyan"))
    
    if not Path(INPUT_FILE).exists(): return

    df = pd.read_parquet(INPUT_FILE)
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y', errors='coerce')
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Top Participants
    plt.figure(figsize=(10, 6))
    plt.clf()
    top = df['author'].value_counts().head(10)
    if not top.empty:
        sns.barplot(x=top.values, y=top.index)
        plt.title('Top Participantes')
        plt.xlabel('Msgs')
        plt.savefig(f"{OUTPUT_DIR}/top_participants.png")
    plt.close()

    # Wordcloud
    text = " ".join(str(msg) for msg in df['content'].dropna())
    # Lista básica de stopwords para limpar o visual
    stopwords = ["de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com", "não", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "foi", "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser", "quando", "muito", "nos", "já", "está", "eu", "também", "só", "pelo", "pela", "até", "isso", "ela", "entre", "era", "depois", "sem", "mesmo", "aos", "ter", "seus", "quem", "nas", "me", "esse", "eles", "estão", "você", "tinha", "foram", "essa", "num", "nem", "suas", "meu", "às", "minha", "têm", "numa", "pelos", "elas", "havia", "seja", "qual", "será", "nós", "tenho", "lhe", "deles", "essas", "esses", "pelas", "este", "fosse", "dele", "tu", "te", "vocês", "vos", "lhes", "meus", "minhas", "teu", "tua", "teus", "tuas", "nosso", "nossa", "nossos", "nossas", "dela", "delas", "esta", "estes", "estas", "aquele", "aquela", "aqueles", "aquelas", "isto", "aquilo", "estou", "está", "estamos", "estão", "estive", "esteve", "estivemos", "estiveram", "estava", "estávamos", "estavam", "estivera", "estivéramos", "esteja", "stejamos", "estejam", "estivesse", "estivéssemos", "estivessem", "estiver", "estivermos", "estiverem", "hei", "há", "havemos", "hão", "houve", "houvemos", "houveram", "houvera", "houvéramos", "haja", "hajamos", "hajam", "houvesse", "houvéssemos", "houvessem", "houver", "houvermos", "houverem", "houverei", "houverá", "houveremos", "houverão", "houveria", "houveríamos", "houveriam", "sou", "somos", "são", "era", "éramos", "eram", "fui", "foi", "fomos", "foram", "fora", "fôramos", "seja", "sejamos", "sejam", "fosse", "fôssemos", "fossem", "for", "formos", "forem", "serei", "será", "seremos", "serão", "seria", "seríamos", "seriam", "tenho", "tem", "temos", "tém", "tinha", "tínhamos", "tinham", "tive", "teve", "tivemos", "tiveram", "tivera", "tivéramos", "tenha", "tenhamos", "tenham", "tivesse", "tivéssemos", "tivessem", "tiver", "tivermos", "tiverem", "terei", "terá", "teremos", "terão", "teria", "teríamos", "teriam"]
    
    if len(text) > 0:
        wc = WordCloud(width=1600, height=800, background_color='black', stopwords=stopwords).generate(text)
        plt.figure(figsize=(20,10))
        plt.clf()
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(f"{OUTPUT_DIR}/wordcloud.png")
    plt.close()

    # Timeline (Agrupado por mês para ficar mais limpo)
    daily = df.groupby(df['date'].dt.to_period('M')).size()
    if not daily.empty:
        plt.figure(figsize=(15, 5))
        plt.clf()
        daily.plot(kind='line', color='#00ff00', marker='o')
        plt.title('Mensagens por Mês')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{OUTPUT_DIR}/timeline.png")
    plt.close()

    print(colored(f"✅ Trends geradas em: {OUTPUT_DIR}", "green"))

if __name__ == "__main__":
    generate_trends()
