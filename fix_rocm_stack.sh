#!/bin/bash

echo "🐧 Mudando estratégia para NATIVA do Arch/CachyOS..."

# 1. Limpar a bagunça anterior
echo "🗑️ Removendo venv quebrado..."
rm -rf venv

# 2. Instalar PyTorch ROCm via Pacman (O jeito CachyOS)
# Isso garante compatibilidade binária perfeita e zero erros de 'execstack'
echo "📦 Instalando python-pytorch-rocm nativo..."
sudo pacman -S --noconfirm python-pytorch-rocm python-torchvision-rocm python-torchaudio-rocm

# 3. Criar venv com acesso aos pacotes do sistema
# A flag --system-site-packages permite que o venv "enxergue" o torch instalado pelo pacman
echo "✨ Criando venv com --system-site-packages..."
python3 -m venv venv --system-site-packages
source venv/bin/activate

# 4. Limpar requirements.txt
# Precisamos remover torch/torchvision do txt para o pip não tentar baixar por cima e quebrar tudo
echo "📝 Ajustando requirements.txt..."
sed -i '/torch/d' requirements.txt
sed -i '/torchvision/d' requirements.txt
sed -i '/torchaudio/d' requirements.txt

# 5. Instalar o resto das dependências
echo "📚 Instalando o resto via pip..."
pip install -r requirements.txt

# 6. Configurar Variáveis de Ambiente (CRUCIAL para RX 6600 XT)
echo "⚙️ Configurando variáveis para RX 6600 XT..."
ACTIVATE_SCRIPT="venv/bin/activate"
# Evitar duplicatas se rodar o script 2x
if ! grep -q "HSA_OVERRIDE_GFX_VERSION" "$ACTIVATE_SCRIPT"; then
    echo "" >> "$ACTIVATE_SCRIPT"
    echo "# Configuração AMD RX 6600 XT" >> "$ACTIVATE_SCRIPT"
    echo "export HSA_OVERRIDE_GFX_VERSION=10.3.0" >> "$ACTIVATE_SCRIPT"
    echo "export ROCM_PATH=/opt/rocm" >> "$ACTIVATE_SCRIPT"
fi

echo "✅ Tudo pronto!"
echo "Teste final:"
echo "source venv/bin/activate"
echo "python -c \"import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')\""
