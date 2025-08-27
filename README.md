# Transcrição de Áudio com Diarização de Locutores

Este projeto realiza a **transcrição de áudio** e a **diarização de locutores** utilizando os modelos **Whisper** e **Pyannote**. O resultado é exportado em um arquivo Word (`.docx`) contendo uma tabela com as falas, os tempos e os locutores identificados.

---

## Funcionalidades

- **Transcrição de Áudio**: Utiliza o modelo Whisper para transcrever o áudio.
- **Diarização de Locutores**: Identifica os diferentes locutores no áudio usando Pyannote.
- **Exportação para Word**: Gera um arquivo `.docx` com uma tabela contendo:
  - Tempo de início e fim de cada fala.
  - Locutor identificado.
  - Texto transcrito.

---

## Requisitos

- **Python 3.10+**
- **Bibliotecas Python** (instale com `pip`):
  ```bash
  pip install -r requirements.txt
  ```

---

## Como usar
### 1. Coloque o áudio que deseja transcrever na pasta do projeto (ex: exemplo.mp3).
### 2. Edite o arquivo Python com seu token Hugging Face:
```python
HUGGINGFACE_TOKEN = "seu_token_aqui"
```
#### O pipeline pyannote/speaker-diarization precisa de acesso autenticado ao Hugging Face.

### Configuração do TOKEN
O projeto utiliza a biblioteca **python-dotenv** para carregar variáveis de ambiente. Crie um arquivo .env na raiz do projeto com o seguinte conteúdo:
```python
HUGGINGFACE_TOKEN=seu_token_aqui
```
#### Substitua "seu_token_aqui" pelo seu token de autenticação do Hugging Face. Esse token é necessário para acessar o pipeline de diarização de locutores.

### 3.Execute o script:
```bash
python transcrever.py
```
### 4. O resultado será exportado para transcricao_diarizada.docx.
---

## Saída esperada

```csharp
[00:00 - 00:12] Locutor 1: Bom dia, tudo bem?
[00:12 - 00:20] Locutor 2: Tudo sim, e você?
[00:20 - 00:25] Locutor 1: Também, obrigado.
```


