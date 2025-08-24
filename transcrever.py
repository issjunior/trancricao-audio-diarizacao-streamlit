import os
import streamlit as st
from dotenv import load_dotenv
import whisper
from pyannote.audio import Pipeline
from docx import Document
from docx.oxml import OxmlElement
from streamlit.runtime.scriptrunner import RerunException
from streamlit.runtime.state.session_state import SessionState

# Função para formatar o tempo
def formatar_tempo(tempo_em_segundos):
    """Converte tempo em segundos para o formato mm:ss."""
    minutos = int(tempo_em_segundos // 60)
    segundos = int(tempo_em_segundos % 60)
    return f"{minutos:02}:{segundos:02}"

# -------------------------------
# 1. Configuração inicial
# -------------------------------
st.title("Transcrição e Diarização de Áudio")
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

    # Barra de progresso e status
    progresso = st.progress(0)
    status = st.empty()

    # -------------------------------
    # 3. Transcrição com Whisper
    # -------------------------------
    status.text("Rodando Whisper...")
    progresso.progress(25)
    modelo = whisper.load_model("tiny")  # tiny, base, small, medium, large
    resultado = modelo.transcribe(audio_path)

    # -------------------------------
    # 4. Diarização com Pyannote
    # -------------------------------
    status.text("Rodando Diarização...")
    progresso.progress(50)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=HUGGINGFACE_TOKEN
    )
    diarization = pipeline(audio_path)

    # -------------------------------
    # 5. Mesclar transcrição + locutores numerados
    # -------------------------------
    status.text("Mesclando transcrição e locutores...")
    progresso.progress(75)
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

    # -------------------------------
    # 6. Exibir resultados no Streamlit
    # -------------------------------
    status.text("Exibindo resultados...")
    progresso.progress(90)
    st.write("### Resultados")
    for fala in falas:
        st.write(f"**{fala['tempo']}** | **{fala['locutor']}**: {fala['texto']}")

    # -------------------------------
    # 7. Exportar para Word
    # -------------------------------
    status.text("Salvando resultados no Word...")
    progresso.progress(100)
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

    doc.save("transcricao_diarizada.docx")
    with open("transcricao_diarizada.docx", "rb") as file:
        if st.download_button(
            label="📥 Baixar Arquivo Word",
            data=file,
            file_name="transcricao_diarizada.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            st.success("Arquivo baixado com sucesso! Você pode carregar outro áudio agora.")
            st.experimental_rerun()  # Reinicia a aplicação para permitir novo upload