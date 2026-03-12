import re
import pandas as pd
from pathlib import Path
import sys

class WhatsAppProcessor:
    def __init__(self):
        # Matches date and time (with or without seconds), captures the rest after " - "
        # Author and message are then split on the first ": " so any character (/, :, etc.)
        # in the author name is handled correctly.
        self.log_pattern = re.compile(
            r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2}(?::\d{2})?)\s+-\s+(.+)$'
        )

    def parse_file(self, file_path):
        print(f"📂 Lendo arquivo: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            print("⚠️ Erro de encoding utf-8, tentando latin-1...")
            with open(file_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()

        data = []
        buffer_date = ""
        buffer_time = ""
        buffer_author = ""
        buffer_message = []

        matches_found = 0

        for line in lines:
            line = line.strip()
            # Remove caracteres de controle estranhos do WhatsApp
            line = line.replace('\u200e', '').replace('\u200f', '')

            if not line: continue

            match = self.log_pattern.match(line)

            if match:
                date, time_val, rest = match.groups()

                # Skip system messages that have no "author: message" structure
                if ': ' not in rest:
                    continue

                author, msg_content = rest.split(': ', 1)

                matches_found += 1
                if buffer_author:
                    full_msg = " ".join(buffer_message)
                    if not any(x in full_msg for x in ["<Media omitted>", "<Mídia omitida>", "null"]):
                        data.append({
                            'date': buffer_date,
                            'time': buffer_time,
                            'author': buffer_author,
                            'content': full_msg
                        })

                buffer_date, buffer_time, buffer_author = date, time_val, author
                buffer_message = [msg_content]

            else:
                if buffer_author:
                    buffer_message.append(line)

        if buffer_author and buffer_message:
            full_msg = " ".join(buffer_message)
            if not any(x in full_msg for x in ["<Media omitted>", "<Mídia omitida>"]):
                data.append({
                    'date': buffer_date,
                    'time': buffer_time,
                    'author': buffer_author,
                    'content': full_msg
                })

        print(f"📊 Diagnóstico: {len(lines)} linhas lidas, {matches_found} padrões encontrados.")
        
        df = pd.DataFrame(data)
        if len(df) > 0:
            print(f"✅ Sucesso: {len(df)} mensagens válidas extraídas.")
        else:
            print("❌ Erro: Nenhuma mensagem extraída. Verifique o Regex.")
            
        return df

    def save_processed(self, df, output_path):
        if df.empty: return
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path)
        print(f"💾 Salvo em: {output_path}")

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/_chat.txt"
    output_file = "data/processed/chat_history.parquet"
    
    if not Path(input_file).exists():
        print(f"❌ Arquivo não encontrado: {input_file}")
        sys.exit(1)

    processor = WhatsAppProcessor()
    df = processor.parse_file(input_file)
    
    if not df.empty:
        processor.save_processed(df, output_file)
        print("\n🔍 Amostra dos dados:")
        print(df[['date', 'author', 'content']].head())
    else:
        sys.exit(1)
