from qdrant_client import QdrantClient

# Conecta no banco que acabamos de criar
client = QdrantClient(path="./data/qdrant_db")

print(f"📂 Coleções encontradas: {[c.name for c in client.get_collections().collections]}")

# Verifica quantos pontos temos na coleção
info = client.count(collection_name="whatsapp_chat")
print(f"📊 Total de mensagens indexadas: {info.count}")

# Vamos tentar buscar usando o método mais básico possível (Scroll) para ver se lemos algo
print("\n🔍 Lendo 1 mensagem de teste (Scroll):")
res = client.scroll(
    collection_name="whatsapp_chat",
    limit=1,
    with_payload=True,
    with_vectors=False
)
if res[0]:
    msg = res[0][0].payload
    print(f"   Autor: {msg['author']}")
    print(f"   Texto: {msg['content']}")
else:
    print("   ❌ Nenhuma mensagem encontrada.")

# Check de métodos disponíveis (para entendermos o erro anterior)
print("\n🛠️ Métodos de busca disponíveis no cliente:")
methods = [m for m in dir(client) if "search" in m or "query" in m]
print(methods)
