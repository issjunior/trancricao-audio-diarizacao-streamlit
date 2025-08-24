import os
import streamlit as st
from dotenv import load_dotenv
import whisper
from pyannote.audio import Pipeline
from docx import Document
import pandas as pd  # Adicionado para manipulação de tabelas

# Configuração da página
st.set_page_config(layout="wide", page_title="SPAV - Transcrição", page_icon="🎙️")

# Função para formatar o tempo
def formatar_tempo(tempo_em_segundos):
    """Converte tempo em segundos para o formato mm:ss."""
    minutos = int(tempo_em_segundos // 60)
    segundos = int(tempo_em_segundos % 60)
    return f"{minutos:02}:{segundos:02}"

# Função para processar o áudio
@st.cache_data
def processar_audio(audio_path, huggingface_token):
    # Transcrição com Whisper
    modelo = whisper.load_model("base")  # tiny, base, small, medium, large
    resultado = modelo.transcribe(audio_path)

    # Diarização com Pyannote
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=huggingface_token
    )
    diarization = pipeline(audio_path)

    # Mesclar transcrição e locutores
    falas = []
    mapa_locutores = {}
    contador_locutor = 1

    for segmento in resultado["segments"]:
        start = segmento["start"]
        end = segmento["end"]
        texto = f'"{segmento["text"].strip()}"'

        speaker = "Desconhecido"
        for turno in diarization.itertracks(yield_label=True):
            seg_dia = turno[0]
            locutor_original = turno[-1]
            s = seg_dia.start
            e = seg_dia.end
            if s <= start <= e:
                speaker = locutor_original
                break

        if speaker not in mapa_locutores and speaker != "Desconhecido":
            mapa_locutores[speaker] = f"Locutor {contador_locutor}"
            contador_locutor += 1

        nome_final = mapa_locutores.get(speaker, speaker)

        falas.append({
            "tempo": f"{formatar_tempo(start)} - {formatar_tempo(end)}",
            "locutor": nome_final,
            "texto": texto
        })

    # Gerar documento Word
    doc = Document()
    doc.add_heading("Tabela 1 - transcrição de áudio", level=1)

    tabela = doc.add_table(rows=1, cols=3)
    hdr_cells = tabela.rows[0].cells
    hdr_cells[0].text = 'Tempo'
    hdr_cells[1].text = 'Locutor'
    hdr_cells[2].text = 'Transcrição'

    for fala in falas:
        row_cells = tabela.add_row().cells
        row_cells[0].text = fala["tempo"]
        row_cells[1].text = fala["locutor"]
        row_cells[2].text = fala["texto"]

    doc_path = "transcricao_diarizada.docx"
    doc.save(doc_path)

    return falas, doc_path

# -------------------------------
# 1. Configuração inicial
# -------------------------------
st.title("SPAV - Transcrição")
st.write("Carregue um arquivo de áudio para transcrição e identificação de locutores.")

# Carregar variáveis de ambiente
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if not HUGGINGFACE_TOKEN:
    st.error("❌ Token do HuggingFace não encontrado. Defina no arquivo `.env`.")
    st.stop()

# -------------------------------
# 2. Upload do arquivo de áudio
# -------------------------------
audio_file = st.file_uploader("Carregue um arquivo de áudio", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    with open("temp_audio_file", "wb") as f:
        f.write(audio_file.read())
    audio_path = "temp_audio_file"

    # Processar o áudio
    progresso = st.progress(0)
    status = st.empty()
    status.text("Processando áudio...")
    progresso.progress(50)

    falas, doc_path = processar_audio(audio_path, HUGGINGFACE_TOKEN)

    progresso.progress(100)
    status.text("Processamento concluído!")

    # Botão para download do arquivo Word
    with open(doc_path, "rb") as file:
        st.download_button(
            label="📥 Baixar Arquivo Word",
            data=file,
            file_name="transcricao_diarizada.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    # Exibir resultados como tabela sem índice
    st.write("### Resultados")
    tabela_falas = pd.DataFrame([{"Tempo": fala["tempo"], "Locutor": fala["locutor"], "Transcrição": fala["texto"]} for fala in falas])
    tabela_falas.index += 1  # Ajusta o índice para começar em 1
    st.dataframe(tabela_falas, use_container_width=True)  # Exibe a tabela com índice ajustado