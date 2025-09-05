import os
import gc
import time
import traceback
import streamlit as st
from dotenv import load_dotenv
import whisper
from pyannote.audio import Pipeline
from docx import Document
import pandas as pd
import threading
import queue
from io import BytesIO

# -------------------------------
# 0. Configuração da página
# -------------------------------
st.set_page_config(layout="wide", page_title="SPAV - Transcrição", page_icon="🎙️")
os.environ["SPEECHBRAIN_LOCAL_CACHE_STRATEGY"] = "copy"

# -------------------------------
# Configurações de otimização para máquinas com poucos recursos
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

def limpar_memoria():
    gc.collect()

def processar_audio_chunk(modelo, audio_path, language, progress_queue):
    try:
        # Ajuste: passar None para detecção automática
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
# 1. Configuração inicial
# -------------------------------
st.title("🎙️ SPAV - Transcrição de Áudio Otimizada")
st.write("Sistema otimizado para máquinas com recursos limitados")

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    st.error("❌ Token do HuggingFace não encontrado. Defina no arquivo `.env`")
    st.stop()

st.sidebar.header("⚙️ Configurações")

opcoes_modelos = {
    "tiny": {"nome": "Tiny", "descricao": "⚡ Ultra rápido, baixa precisão", "tamanho": "39MB"},
    "base": {"nome": "Base", "descricao": "⚡ Rápido, precisão moderada", "tamanho": "74MB"},
    "small": {"nome": "Small", "descricao": "⚖️ Balanceado", "tamanho": "244MB"},
    "medium": {"nome": "Medium", "descricao": "🧐 Mais lento, boa precisão", "tamanho": "769MB"},
    "large": {"nome": "Large", "descricao": "🔍 Muito lento, alta precisão", "tamanho": "1.5GB"}
}

modelo_escolhido = st.sidebar.selectbox(
    "🎯 Modelo Whisper:",
    options=list(opcoes_modelos.keys()),
    index=2,
    format_func=lambda x: opcoes_modelos[x]["nome"]
)

if modelo_escolhido == "medium":
    st.sidebar.warning(f"⚠️ Modelo '{modelo_escolhido}' será lento na máquina atual.")
elif modelo_escolhido == "large":
    st.sidebar.error(f"⚠️ Modelo '{modelo_escolhido}' será MUITO lento na máquina atual.")

st.sidebar.info(f"📊 Tamanho do modelo: {opcoes_modelos[modelo_escolhido]['tamanho']}")

opcoes_idiomas = {
    "pt": "Português (Brasil)",
    "en": "Inglês",
    "es": "Espanhol",
    "fr": "Francês",
    "de": "Alemão",
    "auto": "🌐 Detectar automaticamente"
}

idioma_escolhido_label = st.sidebar.selectbox(
    "🗣️ Idioma do áudio:",
    list(opcoes_idiomas.values()),
    index=0
)
idioma_escolhido = list(opcoes_idiomas.keys())[list(opcoes_idiomas.values()).index(idioma_escolhido_label)]

st.sidebar.header("🔧 Configurações Avançadas")
chunk_processing = st.sidebar.checkbox("📦 Processamento em chunks (recomendado)", value=True)
auto_cleanup = st.sidebar.checkbox("🧹 Limpeza automática de memória", value=True)

# -------------------------------
# 2. Upload do arquivo de áudio
# -------------------------------
st.header("📁 Upload do Arquivo")
audio_file = st.file_uploader(
    "Selecione um arquivo de áudio",
    type=["mp3", "wav", "m4a", "flac"],
    help="Formatos suportados: MP3, WAV, M4A, FLAC"
)

if audio_file is not None:
    try:
        file_size_mb = len(audio_file.read()) / (1024 * 1024)
        audio_file.seek(0)
        if file_size_mb > 100:
            st.warning("⚠️ Arquivo é grande. O processamento será lento tenha paciência.")
    except Exception:
        st.info("📏 Arquivo carregado com sucesso")
    
    if "audio_processado" not in st.session_state or st.session_state["audio_processado"] != audio_file.name:
        for key in ["tabela_falas", "nome_base", "modelo_escolhido", "doc_word", "csv_data"]:
            st.session_state.pop(key, None)
        st.session_state["audio_processado"] = audio_file.name
        if auto_cleanup:
            limpar_memoria()

# -------------------------------
# 3. Processamento do áudio
# -------------------------------
if audio_file is not None:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("▶️ Iniciar Transcrição", type="primary"):
            start_time = time.time()
            audio_path = os.path.join(os.getcwd(), f"temp_{audio_file.name}")
            
            try:
                with open(audio_path, "wb") as f:
                    f.write(audio_file.read())
                
                st.header("🔄 Progresso da Transcrição")
                col_prog1, col_prog2 = st.columns([3, 1])
                with col_prog1:
                    progresso = st.progress(0)
                    status = st.empty()
                
                atualizar_progresso(progresso, status, "🎧 Carregando modelo Whisper", 5)
                
                try:
                    modelo = whisper.load_model(modelo_escolhido)
                except Exception as e:
                    st.error(f"❌ Erro ao carregar modelo: {str(e)}")
                    st.stop()
                
                if auto_cleanup:
                    limpar_memoria()
                
                atualizar_progresso(progresso, status, "🎧 Transcrevendo áudio", 15)
                progress_queue = queue.Queue()
                
                # Passar None se detecção automática
                lang_param = None if idioma_escolhido == "auto" else idioma_escolhido
                
                if chunk_processing:
                    transcription_thread = threading.Thread(
                        target=processar_audio_chunk,
                        args=(modelo, audio_path, idioma_escolhido, progress_queue)
                    )
                    transcription_thread.start()
                    while transcription_thread.is_alive():
                        progress_val = min(15 + ((time.time() - start_time) * 2), 45)
                        atualizar_progresso(progresso, status, "🎧 Transcrevendo áudio", progress_val)
                        time.sleep(1)
                    try:
                        result_type, resultado = progress_queue.get_nowait()
                        if result_type == "erro":
                            raise Exception(resultado)
                    except queue.Empty:
                        st.error("❌ Erro na transcrição")
                        st.stop()
                else:
                    resultado = modelo.transcribe(audio_path, language=lang_param)
                
                atualizar_progresso(progresso, status, "✅ Transcrição concluída", 50)
                
                del modelo
                if auto_cleanup:
                    limpar_memoria()
                
                atualizar_progresso(progresso, status, "🗣️ Inicializando diarização", 55)
                diarization_thread = threading.Thread(
                    target=processar_diarizacao,
                    args=(audio_path, HUGGINGFACE_TOKEN, progress_queue)
                )
                diarization_thread.start()
                while diarization_thread.is_alive():
                    progress_val = min(55 + ((time.time() - start_time) * 0.5), 70)
                    atualizar_progresso(progresso, status, "🗣️ Processando locutores", progress_val)
                    time.sleep(2)
                try:
                    result_type, diarization = progress_queue.get_nowait()
                    if result_type == "erro":
                        raise Exception(resultado)
                except queue.Empty:
                    st.error("❌ Erro na diarização")
                    st.stop()
                
                atualizar_progresso(progresso, status, "✅ Diarização concluída", 75)
                
                atualizar_progresso(progresso, status, "🔄 Processando resultados", 80)
                
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
                        progress_val = 80 + int(10 * (i + 1) / total_segmentos)
                        atualizar_progresso(progresso, status, "🔄 Processando segmentos", progress_val)
                
                atualizar_progresso(progresso, status, "📄 Gerando documentos", 95)
                nome_base = os.path.splitext(audio_file.name)[0]
                
                # -------------------------------
                # Criar Word em memória
                doc = Document()
                doc.add_heading(f'Tabela 1 - Transcrição do Áudio "{audio_file.name}".', level=1)
                #info_para.add_run(f"• Arquivo: {audio_file.name}\n")
                #info_para.add_run(f"• Modelo: {modelo_escolhido}\n")
                #info_para.add_run(f"• Idioma: {idioma_escolhido_label}\n")
                #info_para.add_run(f"• Total de segmentos: {len(falas)}\n")
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
                
                info_para = doc.add_paragraph()
                info_para.add_run("\n")
                info_para.add_run("Informações da Transcrição:\n").bold = True
                info_para.add_run(f"• Locutores identificados: {len(mapa_locutores)}\n")
                info_para.add_run(f"• Processado em: {time.strftime('%d/%m/%Y %H:%M:%S')}\n")

                doc_stream = BytesIO()
                doc.save(doc_stream)
                doc_stream.seek(0)
                st.session_state["doc_word"] = doc_stream
                
                # -------------------------------
                # Guardar CSV em session_state
                st.session_state["tabela_falas"] = pd.DataFrame([
                    {"Tempo": f["tempo"], "Locutor": f["locutor"], "Transcrição": f["texto"]} 
                    for f in falas
                ])
                st.session_state["tabela_falas"].index += 1
                st.session_state["csv_data"] = st.session_state["tabela_falas"].to_csv(index=False)
                
                atualizar_progresso(progresso, status, "✅ Processamento concluído!", 100)
                
                st.success(f"🎉 **Transcrição concluída!**")
                st.warning(f"🗂️ Não esqueça de revisar a transcrição.")
                
            except Exception as e:
                st.error(f"❌ **Erro durante o processamento:**")
                st.code(str(e))
                st.expander("🔍 Detalhes técnicos").code(traceback.format_exc())
            finally:
                if os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except:
                        pass
                if auto_cleanup:
                    limpar_memoria()

# -------------------------------
# 4. Exibir resultados
# -------------------------------
if "tabela_falas" in st.session_state:
    st.header("📊 Resultados")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        st.download_button(
            label="📥 Baixar Word",
            data=st.session_state["doc_word"],
            file_name=f"{st.session_state['audio_processado'].split('.')[0]}-{modelo_escolhido}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary"
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
    st.dataframe(
        st.session_state["tabela_falas"], 
        use_container_width=True,
        height=400
    )
    
    with st.expander("📈 Estatísticas Detalhadas"):
        df = st.session_state["tabela_falas"]
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Total de Falas", len(df))
            locutores_unicos = df['Locutor'].nunique()
            st.metric("Locutores Únicos", locutores_unicos)
        with col_stat2:
            distribuicao = df['Locutor'].value_counts()
            st.write("**Distribuição de Falas por Locutor:**")
            for locutor, count in distribuicao.items():
                st.write(f"• {locutor}: {count} falas")
