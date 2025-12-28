import sys
import ollama
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from termcolor import colored

# --- CONFIGURAÇÃO ---
OLLAMA_MODEL = "deepseek-r1:8b" 
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "whatsapp_chat"
VECTOR_DB_PATH = "./data/qdrant_db"

class WhatsAppChat:
    def __init__(self):
        print(colored("⏳ Inicializando componentes...", "yellow"))
        self.client = QdrantClient(path=VECTOR_DB_PATH)
        print(colored("🧠 Carregando modelo de embedding...", "yellow"))
        self.encoder = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
        print(colored(f"✅ Sistema pronto! Usando: {OLLAMA_MODEL}", "green"))

    def get_context(self, query_text, limit=15):
        query_vector = self.encoder.encode(query_text).tolist()
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit
        ).points
        
        context_str = ""
        for hit in results:
            msg = hit.payload
            context_str += f"[{msg['date']} {msg['author']}]: {msg['content']}\n"
        return context_str

    def chat_loop(self):
        print("\n" + "="*50)
        print("🤖 WHATSAPP AI - DEEPSEEK R1 (Digite 'sair')")
        print("="*50 + "\n")

        while True:
            try:
                user_input = input(colored("\nVocê: ", "cyan"))
                if user_input.lower() in ['sair', 'exit']: break
                if not user_input.strip(): continue

                print(colored("🔍 Recuperando contexto...", "grey"))
                context = self.get_context(user_input)
                
                system_prompt = f"""
                Você é um analista de conversas.
                Responda em Português.
                Analise o contexto abaixo para responder.
                
                CONTEXTO:
                {context}
                """

                print(colored("🤖 Gerando resposta...", "grey"))
                
                # --- Lógica de Streaming com Cores ---
                stream = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_input},
                    ],
                    stream=True,
                )

                full_response = ""
                thinking_mode = False
                
                # Buffer para detectar tags que chegam quebradas
                buffer = ""

                for chunk in stream:
                    content = chunk['message']['content']
                    
                    # Imprime pensamento em AMARELO e resposta em VERDE
                    if "<think>" in content:
                        thinking_mode = True
                        print(colored("\n[Raciocínio Iniciado]\n", "yellow"), end="")
                        content = content.replace("<think>", "")
                    
                    if "</think>" in content:
                        thinking_mode = False
                        content = content.replace("</think>", "")
                        print(colored("\n\n[Resposta Final]: ", "green"), end="")
                    
                    if thinking_mode:
                        print(colored(content, "yellow"), end="", flush=True)
                    else:
                        print(colored(content, "green"), end="", flush=True)

                    full_response += content
                print("\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(colored(f"\n❌ Erro: {e}", "red"))

if __name__ == "__main__":
    app = WhatsAppChat()
    app.chat_loop()
