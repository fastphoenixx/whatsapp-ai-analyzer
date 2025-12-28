🤖 WhatsApp AI Analyzer

Uma ferramenta poderosa de RAG (Retrieval-Augmented Generation) local para análise profunda de conversas de WhatsApp. Transforme arquivos de texto em insights visuais e converse com seus dados usando IA, tudo rodando localmente para garantir privacidade total.

✨ Funcionalidades

🕵️ Ingestão Inteligente: Processa arquivos _chat.txt exportados do WhatsApp (Android/iOS), limpando logs de sistema e formatando dados.

🧠 RAG Local & Privado: Usa Qdrant para vetorização semântica e DeepSeek R1 (via Ollama) para raciocínio complexo sobre as conversas.

📊 Dashboard Interativo:

Timeline de Sentimento: Analisa o humor do grupo ao longo do tempo.

Rede de Interações: Grafo visual mostrando quem responde a quem.

Nuvem de Palavras: Termos mais utilizados.

Monitoramento de Hardware: Acompanhe o uso de CPU, RAM e GPU (Suporte a AMD ROCm) em tempo real.

⚡ Otimizado para GPU: Configurado para rodar eficientemente em GPUs com 8GB VRAM (Testado em AMD Radeon RX 6600 XT).

🛠️ Pré-requisitos

Antes de começar, certifique-se de ter instalado:

Python 3.10+

Ollama (Para rodar o modelo de IA).

Drivers de GPU (Recomendado para performance, mas funciona em CPU).

Linux (AMD): Drivers ROCm instalados.

Windows/Linux (NVIDIA): Drivers CUDA.

🚀 Instalação

Clone o repositório:

git clone [https://github.com/fastphoenixx/whatsapp-ai-analyzer.git](https://github.com/fastphoenixx/whatsapp-ai-analyzer.git)
cd whatsapp-ai-analyzer


Crie e ative um ambiente virtual:

python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows


Instale as dependências:

pip install -r requirements.txt


> Nota para usuários AMD: Certifique-se de instalar a versão do PyTorch compatível com ROCm se quiser aceleração de GPU.

Baixe o modelo no Ollama:

ollama pull deepseek-r1:8b


🎮 Como Usar

1. Inicie o Servidor Ollama

Abra um terminal separado para rodar o backend da IA.

Para usuários AMD (Linux/ROCm):

OLLAMA_FLASH_ATTENTION=0 HSA_OVERRIDE_GFX_VERSION=10.3.0 ollama serve


Para usuários NVIDIA ou CPU:

ollama serve


2. Inicie o Dashboard

No terminal do projeto (com o venv ativo):

streamlit run src/interface/app.py


3. Acesse e Analise

O navegador abrirá automaticamente em http://localhost:8501.

Na barra lateral, faça o upload do seu arquivo exportado do WhatsApp (_chat.txt).

No WhatsApp: Abra a conversa -> Três pontinhos -> Mais -> Exportar conversa -> Sem Mídia.

Clique em "Iniciar Análise" e acompanhe o progresso no terminal embutido.

📂 Estrutura do Projeto

whatsapp-ai-analyzer/
├── data/                  # Armazenamento local (ignorado pelo Git)
│   ├── raw/               # Chats brutos
│   ├── processed/         # Parquet estruturado
│   └── qdrant_db/         # Banco vetorial
├── src/
│   ├── ingestion/         # Parsers e limpeza de texto
│   ├── embeddings/        # Geração de vetores e Qdrant
│   ├── analysis/          # Scripts de Sentimento, Grafos e Trends
│   ├── llm/               # Integração com Ollama
│   └── interface/         # Frontend Streamlit
├── requirements.txt       # Dependências do projeto
└── README.md              # Este arquivo


🛡️ Privacidade

Este projeto foi desenhado para ser 100% Local.

Nenhum dado das suas conversas sai da sua máquina.

Nenhum dado é enviado para APIs de terceiros (como OpenAI ou Google).

Tudo é processado na sua RAM/GPU e armazenado na pasta data/ localmente.

🤝 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir Issues ou Pull Requests para melhorar a análise de sentimentos, adicionar novos gráficos ou suportar novos modelos.

Desenvolvido com ❤️ e muita cafeína.
