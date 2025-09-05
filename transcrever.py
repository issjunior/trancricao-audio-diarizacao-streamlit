import os
import traceback
import streamlit as st
from dotenv import load_dotenv
import whisper
from pyannote.audio import Pipeline
from docx import Document
import pandas as pd
from io import BytesIO
import queue

# -------------------------------
# 0. Configuração da página
# -------------------------------
st.set_page_config(layout="wide", page_title="SPAV - Transcrição", page_icon="🎙️")
os.environ["SPEECHBRAIN_LOCAL_CACHE_STRATEGY"] = "copy"

# -------------------------------
# Configurações de otimização
# -------------------------------
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

# -------------------------------
# Funções auxiliares
# -------------------------------
def formatar_tempo(tempo_em_segundos):
    minutos = int(tempo_em_segundos // 60)
    segundos = int(tempo_em_segundos % 60)
    return f"{minutos:02}:{segundos:02}"

def atualizar_progresso(progresso_bar, status_text, etapa, valor, detalhes=""):
    status_msg = f"{etapa}"
    if detalhes:
        status_msg += f" - {detalhes}"
    status_text.text(status_msg)
    progresso_bar.progress(int(min(valor, 100)))

def processar_audio_chunk(modelo, audio_path, language, progress_queue):
    try:
        lang_param = None if language == "auto" else language
        resultado = modelo.transcribe(
            audio_path,
            language=lang_param,
            verbose=False,
            fp16=False,
            temperature=0.0
        )
        progress_queue.put(("transcricao_completa", resultado))
    except Exception as e:
        progress_queue.put(("erro", str(e)))

def processar_diarizacao(audio_path, token, progress_queue):
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=token
        )
        diarization = pipeline(audio_path)
        progress_queue.put(("diarizacao_completa", diarization))
    except Exception as e:
        progress_queue.put(("erro", str(e)))

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("Setor de Áudio e Vídeo - SPAV")

# Modelo Whisper
st.sidebar.subheader("🎯 Modelo Whisper")
opcoes_modelos = {
    "tiny": {"nome": "Tiny", "descricao": "⚡ Ultra rápido, baixa precisão", "tamanho": "39MB"},
    "base": {"nome": "Base", "descricao": "⚡ Rápido, precisão moderada", "tamanho": "74MB"},
    "small": {"nome": "Small", "descricao": "⚖️ Balanceado", "tamanho": "244MB"},
    "medium": {"nome": "Medium", "descricao": "🧐 Mais lento, boa precisão", "tamanho": "769MB"},
    "large": {"nome": "Large", "descricao": "🔍 Muito lento, alta precisão", "tamanho": "1.5GB"}
}
modelo_escolhido = st.sidebar.selectbox(
    "Modelo:",
    options=list(opcoes_modelos.keys()),
    index=2,
    format_func=lambda x: opcoes_modelos[x]["nome"]
)
if modelo_escolhido == "medium":
    st.sidebar.warning(f"⚠️ '{modelo_escolhido}' será lento na máquina atual.")
elif modelo_escolhido == "large":
    st.sidebar.error(f"❌ '{modelo_escolhido}' será MUITO lento na máquina atual.")
st.sidebar.info(f"Tamanho: {opcoes_modelos[modelo_escolhido]['tamanho']}")

# Idioma
st.sidebar.subheader("🗣️ Idioma do áudio")
opcoes_idiomas = {
    "pt": "Português (Brasil)",
    "en": "Inglês",
    "es": "Espanhol",
    "fr": "Francês",
    "de": "Alemão",
    "auto": "🌐 Detectar automaticamente"
}
idioma_escolhido_label = st.sidebar.selectbox(
    "Idioma:", list(opcoes_idiomas.values()), index=0
)
idioma_escolhido = list(opcoes_idiomas.keys())[list(opcoes_idiomas.values()).index(idioma_escolhido_label)]

# Avançado
st.sidebar.subheader("🔧 Avançado")
chunk_processing = st.sidebar.checkbox("📦 Processamento em chunks", value=True)
auto_cleanup = st.sidebar.checkbox("🧹 Limpeza automática de memória", value=True)

# -------------------------------
# Página principal
# -------------------------------
st.title("🎙️ SPAV - Transcrição de Áudio")
st.write("Transcrição e reconhecimento de voz.")

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    st.error("❌ Token do HuggingFace não encontrado. Defina no arquivo `.env`")
    st.stop()

# Upload do áudio
st.header("📁 Upload do Arquivo")
audio_file = st.file_uploader(
    "Selecione um arquivo de áudio", type=["mp3", "wav", "m4a", "flac"],
    help="Formatos suportados: MP3, WAV, M4A, FLAC"
)

if audio_file:
    try:
        file_size_mb = len(audio_file.read()) / (1024 * 1024)
        audio_file.seek(0)
        if file_size_mb > 100:
            st.warning("⚠️ Arquivo é grande. O processamento será lento.")
    except Exception:
        st.info("📏 Arquivo carregado com sucesso")
    
    if "audio_processado" not in st.session_state or st.session_state["audio_processado"] != audio_file.name:
        for key in ["tabela_falas", "nome_base", "modelo_escolhido", "doc_word", "csv_data"]:
            st.session_state.pop(key, None)
        st.session_state["audio_processado"] = audio_file.name

# Processamento do áudio
if audio_file:
    col1, col2 = st.columns([3,1])
    with col1:
        if st.button("▶️ Iniciar Transcrição", type="primary"):
            audio_path = os.path.join(os.getcwd(), f"temp_{audio_file.name}")
            try:
                with open(audio_path, "wb") as f:
                    f.write(audio_file.read())
                
                st.header("🔄 Progresso da Transcrição")
                progresso_placeholder = st.empty()
                status_placeholder = st.empty()
                progresso = progresso_placeholder.progress(0)
                status = status_placeholder
                
                atualizar_progresso(progresso, status, "🎧 Carregando modelo Whisper", 5)
                modelo = whisper.load_model(modelo_escolhido)
                
                atualizar_progresso(progresso, status, "🎧 Transcrevendo áudio", 15)
                progress_queue = queue.Queue()
                
                if chunk_processing:
                    processar_audio_chunk(modelo, audio_path, idioma_escolhido, progress_queue)
                    result_type, resultado = progress_queue.get_nowait()
                    if result_type == "erro":
                        raise Exception(resultado)
                else:
                    resultado = modelo.transcribe(audio_path, language=None if idioma_escolhido=="auto" else idioma_escolhido)
                
                atualizar_progresso(progresso, status, "✅ Transcrição concluída", 50)
                del modelo
                
                atualizar_progresso(progresso, status, "🗣️ Inicializando diarização", 55)
                processar_diarizacao(audio_path, HUGGINGFACE_TOKEN, progress_queue)
                result_type, diarization = progress_queue.get_nowait()
                if result_type == "erro":
                    raise Exception(resultado)
                
                atualizar_progresso(progresso, status, "✅ Diarização concluída", 75)
                
                # Processar segmentos
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
                    falas.append({
                        "tempo": f"{formatar_tempo(start)} - {formatar_tempo(end)}",
                        "locutor": nome_final,
                        "texto": texto
                    })
                    if i % max(1, total_segmentos // 10) == 0:
                        progress_val = 80 + int(10*(i+1)/total_segmentos)
                        atualizar_progresso(progresso, status, "🔄 Processando segmentos", progress_val)
                
                # Gerar documentos
                doc = Document()
                doc.add_heading(f'Tabela 1 - Transcrição do Áudio "{audio_file.name}".', level=1)
                tabela = doc.add_table(rows=1, cols=3)
                tabela.style = 'Table Grid'
                hdr_cells = tabela.rows[0].cells
                hdr_cells[0].text = 'Tempo'
                hdr_cells[1].text = 'Locutor'
                hdr_cells[2].text = 'Transcrição'
                for fala in falas:
                    row_cells = tabela.add_row().cells
                    row_cells[0].text = fala["tempo"]
                    row_cells[1].text = fala["locutor"]
                    row_cells[2].text = fala["texto"]
                
                doc_stream = BytesIO()
                doc.save(doc_stream)
                doc_stream.seek(0)
                st.session_state["doc_word"] = doc_stream
                st.session_state["tabela_falas"] = pd.DataFrame([{"Tempo": f["tempo"], "Locutor": f["locutor"], "Transcrição": f["texto"]} for f in falas])
                st.session_state["tabela_falas"].index += 1
                st.session_state["csv_data"] = st.session_state["tabela_falas"].to_csv(index=False)
                
                atualizar_progresso(progresso, status, "✅ Processamento concluído!", 100)
                st.success("🎉 Transcrição concluída!")
                
            except Exception as e:
                st.error(f"❌ Erro durante o processamento:")
                st.code(str(e))
                st.expander("🔍 Detalhes técnicos").code(traceback.format_exc())
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)

# Exibir resultados
if "tabela_falas" in st.session_state:
    st.header("📊 Resultados")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        st.download_button(
            label="📥 Baixar Word",
            data=st.session_state["doc_word"],
            file_name=f"{st.session_state['audio_processado'].split('.')[0]}-{modelo_escolhido}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    with col_btn2:
        st.download_button(
            label="📊 Baixar CSV",
            data=st.session_state["csv_data"],
            file_name=f"{st.session_state['audio_processado'].split('.')[0]}-{modelo_escolhido}.csv",
            mime="text/csv"
        )
    with col_btn3:
        if st.button("🧹 Limpar Resultados"):
            for key in ["tabela_falas", "doc_word", "csv_data", "audio_processado"]:
                st.session_state.pop(key, None)
            st.rerun()
    
    st.subheader("📋 Transcrição Completa")
    st.dataframe(st.session_state["tabela_falas"], use_container_width=True, height=400)
    
    with st.expander("📈 Estatísticas Detalhadas"):
        df = st.session_state["tabela_falas"]
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Total de Falas", len(df))
            st.metric("Locutores Únicos", df['Locutor'].nunique())
        with col_stat2:
            distribuicao = df['Locutor'].value_counts()
            st.write("**Distribuição de Falas por Locutor:**")
            for locutor, count in distribuicao.items():
                st.write(f"• {locutor}: {count} falas")
