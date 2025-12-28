# 🤖 WhatsApp AI Analyzer

Uma ferramenta completa de **RAG (Retrieval-Augmented Generation)** local para análise de grupos de WhatsApp. Transforma históricos de conversa em insights, gráficos e um chat interativo com IA.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/Interface-Streamlit-red)
![AI](https://img.shields.io/badge/Model-DeepSeek%20R1-purple)
![Privacy](https://img.shields.io/badge/Privacy-100%25%20Local-green)

## ✨ Funcionalidades

* **🕵️ Ingestão Inteligente:** Processa arquivos `.txt` exportados do WhatsApp (Android/iOS).
* **🧠 RAG Local:** Usa **Qdrant** para busca semântica e **DeepSeek R1** (via Ollama) para raciocínio.
* **📊 Dashboard Visual:**
    * Rede de Interações (Quem fala com quem).
    * Timeline de Sentimento (Humor do grupo ao longo do tempo).
    * Nuvem de Palavras e Estatísticas de hardware em tempo real.
* **⚡ Otimizado:** Configurado para rodar liso em GPUs com 8GB VRAM (testado em AMD RX 6600 XT).

## 🛠️ Pré-requisitos

1.  **Python 3.10+**
2.  **Ollama** instalado e rodando.
3.  **GPU** recomendada (funciona em CPU, mas é lento).

## 🚀 Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/fastphoenixx/whatsapp-ai-analyzer](https://github.com/fastphoenixx/whatsapp-ai-analyzer)
    cd whatsapp-ai-analyzer
    ```

2.  **Crie o ambiente virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
    *Nota para usuários AMD:* Certifique-se de instalar o PyTorch com suporte a ROCm.

## 🎮 Como Usar

### 1. Inicie o Servidor Ollama
Em um terminal separado, inicie o backend da IA.
*Exemplo para AMD (Linux):*
```bash
OLLAMA_FLASH_ATTENTION=0 HSA_OVERRIDE_GFX_VERSION=10.3.0 ollama serve

2. Inicie o Dashboard

No terminal do projeto:
Bash

streamlit run src/interface/app.py

3. Use a Ferramenta

    Abra o navegador em http://localhost:8501.

    Na barra lateral, faça upload do arquivo _chat.txt (exportado do WhatsApp -> Exportar Conversa -> Sem Mídia).

    Clique em "Iniciar Análise".

📂 Estrutura

    src/ingestion: Parsers de texto e limpeza de dados.

    src/embeddings: Geração de vetores e banco Qdrant.

    src/analysis: Scripts de ciência de dados (Sentimento, Grafos).

    src/interface: Aplicação Web (Streamlit).
