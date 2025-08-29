# Transcrição de Áudio com Diarização de Locutores

Este projeto realiza a **transcrição de áudio** e a **diarização de locutores** utilizando os modelos **Whisper** e **Pyannote**. O resultado é exportado em um arquivo Word (`.docx`) contendo uma tabela com as falas, os tempos e os locutores identificados.

## Funcionalidades

- **🎧 Transcrição de Áudio**: Utiliza o modelo Whisper para transcrever o áudio.
- **🗣️ Diarização de Locutores**: Identifica os diferentes locutores no áudio usando Pyannote.
- **📄 Exportação para Word**: Gera um arquivo `.docx` com uma tabela contendo:
  - Tempo de início e fim de cada fala no formato mm:ss.
  - Locutor identificado.
  - Texto transcrito.

---

## Configuração do Ambiente
### ⚙️ Criar ambiente virtual
  ```bash
  python -m venv venv
  ```

### Ativar o ambiente virtual
  ```bash
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
  ```

---

## Instalação do pacote **FFmpeg**
### 🔹 Instalação no Linux (Debian/Ubuntu)

```bash
sudo apt update
sudo apt install ffmpeg -y
```

### 🔹 Instalação no Windows
- Acesse o site oficial do FFmpeg: https://ffmpeg.org/download.html
- Baixe a versão mais recente para Windows.
- Extraia os arquivos em uma pasta, por exemplo: C:\ffmpeg\.
  - Adicione o caminho C:\ffmpeg\bin à variável de ambiente PATH.
  - Painel de Controle → Sistema → Configurações avançadas do sistema → Variáveis de Ambiente.

---

##  Instalação
### 📥 Clonar o repositório
  ```bash
git clone https://github.com/issjunior/trancricao-audio-diarizacao-streamlit.git
```

## Instalação de dependências
  ```bash
  pip install -r requirements.txt
  ```

## Criando TOKEN Hugging Face
### 🔑 Configuração do Hugging Face
- Este projeto depende de modelos hospedados no Hugging Face.
- Será necessário criar uma conta e gerar um token de acesso.
- Criar conta gratuita no Hugging Face: https://huggingface.co/join
- Após login, gerar token em: https://huggingface.co/settings/tokens
  - Clique em New Token, dê um nome (ex: spav-token) e copie o valor.
  - Escolha o tipo "Read".
  - Crie um arquivo `.env` na raiz do projeto com o conteúdo:
```python
HUGGINGFACE_TOKEN=seu_token_aqui
```
##### ⚠️ O pipeline pyannote/speaker-diarization precisa de acesso autenticado ao Hugging Face.

---

## ▶️ Execute o script:
```bash
streamlit run transcrever.py
```
### 📌 Saída esperada

```csharp
[00:00 - 00:12] Locutor 1 Bom dia, tudo bem?
[00:12 - 00:20] Locutor 2 Tudo sim, e você?
[00:20 - 00:25] Locutor 1 Também, obrigado.
```


