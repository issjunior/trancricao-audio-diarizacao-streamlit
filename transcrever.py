import os
import streamlit as st
from dotenv import load_dotenv
import whisper
from pyannote.audio import Pipeline
from docx import Document
import pandas as pd

# -------------------------------
# 0. Configuração da página
# -------------------------------
st.set_page_config(layout="wide", page_title="SPAV - Transcrição", page_icon="🎙️")
os.environ["SPEECHBRAIN_LOCAL_CACHE_STRATEGY"] = "copy"

# -------------------------------
# Funções auxiliares
# -------------------------------
def formatar_tempo(tempo_em_segundos):
    minutos = int(tempo_em_segundos // 60)
    segundos = int(tempo_em_segundos % 60)
    return f"{minutos:02}:{segundos:02}"

def atualizar_progresso(progresso_bar, status_text, etapa, valor):
    status_text.text(etapa)
    progresso_bar.progress(valor)

# -------------------------------
# 1. Configuração inicial
# -------------------------------
st.title("SPAV - Transcrição de Áudio")
st.write("Carregue um arquivo de áudio para transcrição e identificação de locutores.")

# Carregar variáveis de ambiente
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    st.error("❌ Token do HuggingFace não encontrado. Defina no arquivo `.env` ou chame o suporte izaias.junior@policiacivil.am.gov.br")
    st.stop()

# Seleção do modelo Whisper e idioma
col1, col2 = st.columns(2)
with col1:
    opcoes_modelos = ["tiny", "base", "small", "medium", "large"]
    modelo_escolhido = st.selectbox("Escolha o modelo Whisper:", opcoes_modelos, index=4)
with col2:
    opcoes_idiomas = {
        "pt": "Português (Brasil)",
        "en": "Inglês",
        "es": "Espanhol",
        "fr": "Francês",
        "de": "Alemão",
        "auto": "Detectar automaticamente"
    }
    idioma_escolhido_label = st.selectbox("Escolha o idioma do áudio:", list(opcoes_idiomas.values()), index=0)
    idioma_escolhido = list(opcoes_idiomas.keys())[list(opcoes_idiomas.values()).index(idioma_escolhido_label)]

explicacoes_modelos = {
    "tiny": "⚡ Muito rápido e consome pouca memória. Útil apenas para pré-análises.",
    "base": "⚡ Rápido e mais estável que o tiny. Indicado para triagem inicial.",
    "small": "⚖️ Bom equilíbrio entre velocidade e precisão. Adequado para análises preliminares.",
    "medium": "🧐 Mais lento e exige mais recursos, mas alcança boa precisão.",
    "large": "🔍 Mais demorado, porém oferece a maior qualidade e fidelidade na transcrição."
}

st.markdown(f"""
📢 **Modelo escolhido:** `{modelo_escolhido}`  
🌍 **Idioma escolhido:** {idioma_escolhido_label}  
💡 **Trade-off do modelo:** {explicacoes_modelos[modelo_escolhido]}
""")

# -------------------------------
# 2. Upload do arquivo de áudio
# -------------------------------
audio_file = st.file_uploader("Carregue um arquivo de áudio", type=["mp3", "wav", "m4a"])

# Botão para iniciar processamento
if audio_file is not None and st.button("▶️ Iniciar Transcrição"):
    # Salvar arquivo de áudio temporariamente
    audio_path = os.path.join(os.getcwd(), audio_file.name)
    with open(audio_path, "wb") as f:
        f.write(audio_file.read())

    # Criar barra de progresso e status
    progresso = st.progress(0)
    status = st.empty()

    # -------------------------------
    # 3. Processamento do áudio
    # -------------------------------
    atualizar_progresso(progresso, status, "🎧 Carregando modelo Whisper... Etapa 1 de 5", 0)
    modelo = whisper.load_model(modelo_escolhido)
    atualizar_progresso(progresso, status, "🎧 Transcrevendo áudio... Etapa 2 de 5", 10)
    
    resultado = modelo.transcribe(audio_path, language=idioma_escolhido)
    
    atualizar_progresso(progresso, status, "🗣️ Inicializando diarização... Etapa 3 de 5", 30)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=HUGGINGFACE_TOKEN
    )
    
    atualizar_progresso(progresso, status, "🗣️ Realizando diarização... Etapa 3 de 5", 40)
    diarization = pipeline(audio_path)
    
    # Mesclar transcrição e locutores
    atualizar_progresso(progresso, status, "✍️ Mesclando transcrição com locutores... Etapa 4 de 5", 50)
    falas = []
    mapa_locutores = {}
    contador_locutor = 1
    total_segmentos = len(resultado["segments"])
    
    for i, segmento in enumerate(resultado["segments"]):
        start = segmento["start"]
        end = segmento["end"]
        texto = f'"{segmento["text"].strip()}"'

        speaker = "Desconhecido"
        for turno in diarization.itertracks(yield_label=True):
            seg_dia = turno[0]
            locutor_original = turno[-1]
            if seg_dia.start <= start <= seg_dia.end:
                speaker = locutor_original
                break

        if speaker not in mapa_locutores and speaker != "Desconhecido":
            mapa_locutores[speaker] = f"Locutor {contador_locutor}"
            contador_locutor += 1

        nome_final = mapa_locutores.get(speaker, speaker)
        falas.append({"tempo": f"{formatar_tempo(start)} - {formatar_tempo(end)}",
                      "locutor": nome_final,
                      "texto": texto})

        # Atualizar barra proporcional
        progresso_value = 50 + int(40 * (i + 1) / total_segmentos)  # 50->90%
        progresso.progress(progresso_value)

    # Criar pasta tabela_transcricao se não existir
    pasta_saida = os.path.join(os.getcwd(), "tabela_transcricao")
    os.makedirs(pasta_saida, exist_ok=True)

    # Nome do arquivo Word baseado no nome do áudio e modelo Whisper
    nome_base = os.path.splitext(audio_file.name)[0]  # remove extensão
    doc_path = os.path.join(pasta_saida, f"{nome_base}-{modelo_escolhido}.docx")

    # Criar documento Word
    atualizar_progresso(progresso, status, "📄 Gerando documento Word... Etapa 5 de 5", 90)
    doc = Document()
    doc.add_heading("Tabela 1 - Transcrição de Áudio", level=1)
    tabela = doc.add_table(rows=1, cols=3)
    hdr_cells = tabela.rows[0].cells
    hdr_cells[0].text = 'Tempo'
    hdr_cells[1].text = 'Locutor'
    hdr_cells[2].text = 'Transcrição'

    for i, fala in enumerate(falas):
        row_cells = tabela.add_row().cells
        row_cells[0].text = fala["tempo"]
        row_cells[1].text = fala["locutor"]
        row_cells[2].text = fala["texto"]
        progresso_value = 90 + int(10 * (i + 1) / len(falas))  # 90->100%
        progresso.progress(progresso_value)

    # Salvar documento Word
    doc.save(doc_path)
    atualizar_progresso(progresso, status, "✅ Processamento concluído!", 100)

    # -------------------------------
    # 4. Download e exibição
    # -------------------------------
    with open(doc_path, "rb") as file:
        st.download_button(
            label="📥 Baixar Arquivo Word",
            data=file,
            file_name=f"{nome_base}-{modelo_escolhido}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    st.write("### Transcrição")
    tabela_falas = pd.DataFrame(
        [{"Tempo": f["tempo"], "Locutor": f["locutor"], "Transcrição": f["texto"]} for f in falas]
    )
    tabela_falas.index += 1
    st.dataframe(tabela_falas, use_container_width=True)
