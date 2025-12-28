import click
import os
import sys
from termcolor import colored

# Adiciona raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@click.group()
def cli():
    """WhatsApp AI Analyzer - Ferramenta CLI V1.0"""
    pass

@cli.command()
@click.option('--file', default='data/raw/_chat.txt', help='Caminho do arquivo de chat exportado')
def ingest(file):
    """1. Processar arquivo de texto bruto"""
    from src.ingestion.processor import WhatsAppProcessor
    print(colored(f"🚀 Iniciando ingestão de: {file}", "cyan"))
    proc = WhatsAppProcessor()
    df = proc.parse_file(file)
    proc.save_processed(df, "data/processed/chat_history.parquet")

@cli.command()
def vector():
    """2. Criar/Atualizar Banco Vetorial (Embeddings)"""
    from src.embeddings.vector_store import build_vector_store
    print(colored("🧠 Gerando Embeddings...", "cyan"))
    build_vector_store("data/processed/chat_history.parquet")

@cli.command()
def analyze():
    """3. Gerar Todos os Relatórios (Sentimento, Rede, Trends)"""
    print(colored("📊 Rodando Suíte de Análise Completa...", "magenta"))
    
    # Trends
    from src.analysis.trends import generate_trends
    generate_trends()
    
    # Sentimento
    # Importante: O script de sentimento limpa a memória, então rodamos ele isolado ou com cuidado
    print(colored("\n💔 Iniciando Análise de Sentimento...", "magenta"))
    os.system("python src/analysis/sentiment.py") # Rodamos via system para garantir gestão de memória limpa
    
    # Rede
    print(colored("\n🕸️  Iniciando Análise de Rede...", "magenta"))
    from src.analysis.network_graph import generate_network_graph
    generate_network_graph()

@cli.command()
def serve():
    """4. Iniciar Servidor API (Backend)"""
    print(colored("🌐 Iniciando API Server...", "green"))
    os.system("python src/interface/api.py")

@cli.command()
def chat():
    """5. Conversar no Terminal (Modo CLI)"""
    from src.llm.chat_engine import WhatsAppChat
    app = WhatsAppChat()
    app.chat_loop()

if __name__ == '__main__':
    cli()
