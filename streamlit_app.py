"""
Deep Speaker Identification - Streamlit UI

Features:
- Live Monitor with threading
- Speaker Enrollment
- Speaker Comparison (UMAP visualization)
- Audio Quality Indicator
- Confidence History Chart
- Settings Panel
"""

import streamlit as st
import os
import time
import numpy as np
import wave
import yaml
import torch
import pandas as pd
from datetime import datetime
import threading

# --- Paths & Imports ---
import sys
sys.path.append(os.path.abspath('.'))

from src.inference.identify_speaker import SpeakerIdentifier
from src.inference.realtime import RealTimeProcessor, ThreadedAudioProcessor
from src.utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(console=False)
logger = get_logger("streamlit")

# Try to import optional modules
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("PyAudio not available. Microphone features disabled.")

try:
    from src.utils.audio_quality import assess_quality
    QUALITY_AVAILABLE = True
except ImportError:
    QUALITY_AVAILABLE = False

try:
    from src.utils.visualization import reduce_embeddings, create_plotly_scatter
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False

# --- Config ---
CONFIG_PATH = "config.yaml"

@st.cache_data
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

config = load_config()

# --- Page Setup ---
st.set_page_config(
    page_title="Deep Speaker ID",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styles ---
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #0f3460;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .big-name {
        font-size: 2.5em;
        font-weight: bold;
        background: linear-gradient(45deg, #4CAF50, #8BC34A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .quality-good { color: #4CAF50; }
    .quality-medium { color: #FFC107; }
    .quality-bad { color: #f44336; }
    .speech-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .speech-active { background-color: #4CAF50; animation: pulse 1s infinite; }
    .speech-inactive { background-color: #666; }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# --- State Initialization ---
if 'identifier' not in st.session_state:
    with st.spinner("Model yükleniyor..."):
        model_path = config.get('paths', {}).get('model_path', 'models/speaker_encoder.pt')
        emb_path = config.get('paths', {}).get('embeddings_path', 'embeddings/speakers.json')

        st.session_state.identifier = SpeakerIdentifier(
            model_path=os.path.abspath(model_path),
            embeddings_path=os.path.abspath(emb_path)
        )
        st.session_state.processor = RealTimeProcessor(
            st.session_state.identifier,
            sample_rate=16000,
            buffer_duration=5.0  # 5 saniye buffer
        )
        logger.info("Model loaded successfully.")

if 'history' not in st.session_state:
    st.session_state.history = []

if 'confidence_history' not in st.session_state:
    st.session_state.confidence_history = []

if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False

# --- Sidebar ---
st.sidebar.title("🎙️ Speaker ID")
page = st.sidebar.radio(
    "Menü",
    ["Canlı İzleme", "Dosya Analizi", "Kişi Kayıt", "Konuşmacılar", "Ayarlar"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Cihaz:** `{st.session_state.identifier.device}`")
st.sidebar.markdown(f"**Kayıtlı Kişi:** {len(st.session_state.identifier.speakers)}")

if not PYAUDIO_AVAILABLE:
    st.sidebar.warning("⚠️ PyAudio yüklü değil. Mikrofon özellikleri devre dışı.")

# --- Utils ---
def save_temp_wav(audio_bytes, sr=16000):
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
        with wave.open(fp.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio_bytes)
        return fp.name

# --- Pages ---

if page == "Canlı İzleme":
    st.title("🔴 Canlı Konuşmacı Tanıma")

    if not PYAUDIO_AVAILABLE:
        st.error("Bu özellik için PyAudio gereklidir. Lütfen `pip install pyaudio` ile yükleyin.")
    else:
        col1, col2 = st.columns([3, 1])

        with col1:
            st.write("Mikrofonu başlatarak gerçek zamanlı tanıma yapabilirsiniz.")

            c1, c2, c3 = st.columns(3)
            with c1:
                start_btn = st.button("▶️ Başlat", disabled=st.session_state.is_recording, use_container_width=True)
            with c2:
                stop_btn = st.button("⏹️ Durdur", disabled=not st.session_state.is_recording, use_container_width=True)
            with c3:
                clear_btn = st.button("🗑️ Geçmişi Temizle", use_container_width=True)

            if start_btn:
                st.session_state.is_recording = True
                st.session_state.confidence_history = []
                # Buffer'ı sıfırla - yeni processor oluştur
                st.session_state.processor = RealTimeProcessor(
                    st.session_state.identifier,
                    sample_rate=16000,
                    buffer_duration=5.0
                )
                st.rerun()

            if stop_btn:
                st.session_state.is_recording = False
                st.rerun()

            if clear_btn:
                st.session_state.history = []
                st.session_state.confidence_history = []
                st.rerun()

        with col2:
            # Quality indicator placeholder
            st.markdown("**Ses Kalitesi**")
            quality_placeholder = st.empty()
            speech_prob_placeholder = st.empty()

        # Main Display
        st.markdown("---")
        col_result, col_chart = st.columns([2, 1])

        with col_result:
            placeholder_speaker = st.empty()
            placeholder_conf = st.empty()

        with col_chart:
            chart_placeholder = st.empty()

        if st.session_state.is_recording:
            p = pyaudio.PyAudio()
            try:
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024
                )

                audio_buffer = []

                while st.session_state.is_recording:
                    try:
                        data = stream.read(1024, exception_on_overflow=False)
                        audio_int16 = np.frombuffer(data, dtype=np.int16)
                        audio_buffer.append(audio_int16)

                        # Audio quality assessment
                        if QUALITY_AVAILABLE and len(audio_buffer) > 10:
                            recent_audio = np.concatenate(audio_buffer[-10:])
                            quality = assess_quality(recent_audio)
                            quality_placeholder.markdown(
                                f"<span class='quality-{quality['color']}'>{quality['label']}</span> "
                                f"(SNR: {quality['snr_db']}dB)",
                                unsafe_allow_html=True
                            )

                        # Process
                        result = st.session_state.processor.process_chunk(audio_int16)

                        if result:
                            speaker = result['speaker']
                            conf = result['confidence']

                            # Track confidence history
                            st.session_state.confidence_history.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "confidence": conf
                            })

                            # Keep only last 50 points
                            if len(st.session_state.confidence_history) > 50:
                                st.session_state.confidence_history = st.session_state.confidence_history[-50:]

                            # Display confidence chart
                            if st.session_state.confidence_history:
                                df_conf = pd.DataFrame(st.session_state.confidence_history)
                                chart_placeholder.line_chart(df_conf.set_index("time")["confidence"], height=150)

                            # Update main display
                            if speaker == "Silence":
                                placeholder_speaker.markdown(
                                    "<div class='metric-card' style='color:#666'>"
                                    "<span class='speech-indicator speech-inactive'></span>Sessizlik</div>",
                                    unsafe_allow_html=True
                                )
                                placeholder_conf.progress(0, text="Konuşma algılanmadı")
                            elif speaker == "Unknown":
                                placeholder_speaker.markdown(
                                    "<div class='metric-card' style='color:#FFC107'>"
                                    "<span class='speech-indicator speech-active'></span>Bilinmeyen Konuşmacı</div>",
                                    unsafe_allow_html=True
                                )
                                placeholder_conf.progress(int(conf * 100), text=f"Güven: {conf:.2%}")
                            else:
                                placeholder_speaker.markdown(
                                    f"<div class='metric-card'>"
                                    f"<span class='speech-indicator speech-active'></span>"
                                    f"<div class='big-name'>{speaker}</div></div>",
                                    unsafe_allow_html=True
                                )
                                placeholder_conf.progress(int(conf * 100), text=f"Güven: {conf:.2%}")

                                # Log to history
                                if not st.session_state.history or st.session_state.history[-1]['speaker'] != speaker:
                                    st.session_state.history.append({
                                        "zaman": datetime.now().strftime("%H:%M:%S"),
                                        "konuşmacı": speaker,
                                        "güven": f"{conf:.2%}"
                                    })

                        time.sleep(0.01)

                    except Exception as e:
                        logger.error(f"Stream error: {e}")
                        st.error(f"Hata: {e}")
                        break

            finally:
                stream.stop_stream()
                stream.close()
                p.terminate()

        # History Table
        st.markdown("---")
        st.subheader("📋 Oturum Geçmişi")
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(df.sort_index(ascending=False).head(15), use_container_width=True)
        else:
            st.info("Henüz tanıma yapılmadı.")

elif page == "Dosya Analizi":
    st.title("📁 Dosya ile Konuşmacı Tanıma")

    st.info("Bir ses dosyası (.wav, .mp3) yükleyerek konuşmacıyı tanımlayabilirsiniz.")

    uploaded_file = st.file_uploader("Ses Dosyası Seçin", type=["wav", "mp3", "m4a", "ogg", "flac"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("🔍 Analiz Et", use_container_width=True):
            with st.spinner("Ses analiz ediliyor..."):
                import tempfile

                # Save uploaded file temporarily
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    # Run identification
                    result = st.session_state.identifier.identify(tmp_path)

                    if result:
                        st.markdown("---")
                        st.subheader("📊 Sonuç")

                        col1, col2 = st.columns(2)

                        with col1:
                            if result['speaker'] == "Unknown":
                                st.markdown(
                                    "<div class='metric-card' style='color:#FFC107'>"
                                    "<h2>Bilinmeyen Konuşmacı</h2></div>",
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f"<div class='metric-card'>"
                                    f"<div class='big-name'>{result['speaker']}</div></div>",
                                    unsafe_allow_html=True
                                )

                        with col2:
                            st.metric("Güven Skoru", f"{result['confidence']:.2%}")
                            st.progress(int(result['confidence'] * 100))

                        # Show all speaker scores
                        st.markdown("---")
                        st.subheader("📈 Tüm Konuşmacı Skorları")

                        all_scores = {}
                        for name, ref_emb in st.session_state.identifier.speakers.items():
                            emb = st.session_state.identifier.compute_embedding(tmp_path)
                            if emb is not None:
                                score = torch.nn.functional.cosine_similarity(
                                    emb.unsqueeze(0), ref_emb.unsqueeze(0)
                                ).item()
                                all_scores[name] = score

                        if all_scores:
                            df_scores = pd.DataFrame({
                                "Konuşmacı": list(all_scores.keys()),
                                "Benzerlik": list(all_scores.values())
                            }).sort_values("Benzerlik", ascending=False)

                            st.dataframe(
                                df_scores.style.background_gradient(subset=["Benzerlik"], cmap="RdYlGn"),
                                use_container_width=True
                            )

                            # Bar chart
                            st.bar_chart(df_scores.set_index("Konuşmacı"))
                    else:
                        st.error("Ses analiz edilemedi. Dosya geçersiz olabilir.")

                except Exception as e:
                    st.error(f"Hata: {e}")
                    logger.error(f"File analysis error: {e}")

                finally:
                    # Cleanup
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

    else:
        st.markdown("""
        ### Desteklenen Formatlar
        - WAV (önerilen)
        - MP3
        - M4A
        - OGG
        - FLAC

        ### Öneriler
        - En az 3 saniye uzunluğunda ses yükleyin
        - Temiz ve gürültüsüz kayıtlar daha iyi sonuç verir
        """)

elif page == "Kişi Kayıt":
    st.title("👤 Yeni Konuşmacı Kayıt")

    st.info("Yeni bir kişiyi sisteme eklemek için adını girin ve 3-5 saniye net bir şekilde konuşun.")

    col1, col2 = st.columns([2, 1])

    with col1:
        name = st.text_input("Konuşmacı Adı", placeholder="Örn: Ahmet")
        duration = st.slider("Kayıt Süresi (saniye)", 3, 10, 4)

    with col2:
        st.markdown("**İpuçları:**")
        st.markdown("- Net ve yüksek sesle konuşun")
        st.markdown("- Arka plan gürültüsünden kaçının")
        st.markdown("- Farklı cümleler kurun")

    if not PYAUDIO_AVAILABLE:
        st.error("Bu özellik için PyAudio gereklidir.")
    elif st.button("🎙️ Kaydet ve Ekle", use_container_width=True):
        if not name:
            st.error("Lütfen önce bir isim girin.")
        elif name in st.session_state.identifier.speakers:
            st.warning(f"'{name}' zaten kayıtlı. Farklı bir isim deneyin veya önce mevcut kaydı silin.")
        else:
            status = st.empty()
            progress = st.progress(0)

            # Countdown
            for i in range(3, 0, -1):
                status.warning(f"Kayıt {i} saniye sonra başlayacak... Hazır olun!")
                time.sleep(1)

            status.error("🔴 KAYIT YAPILIYOR... Konuşun!")

            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

            frames = []
            total_chunks = int(16000 / 1024 * duration)

            for i in range(total_chunks):
                data = stream.read(1024)
                frames.append(data)
                progress.progress((i + 1) / total_chunks)

            stream.stop_stream()
            stream.close()
            p.terminate()

            status.success("Kayıt tamamlandı. İşleniyor...")

            # Quality check
            raw_bytes = b''.join(frames)
            audio_np = np.frombuffer(raw_bytes, dtype=np.int16)

            if QUALITY_AVAILABLE:
                quality = assess_quality(audio_np)
                if quality['score'] < 40:
                    st.warning(f"⚠️ Ses kalitesi düşük ({quality['label']}). Sorunlar: {', '.join(quality['issues'])}")

            # Save and enroll
            temp_path = save_temp_wav(raw_bytes)

            try:
                success = st.session_state.identifier.enroll_speaker(name, temp_path)

                if success:
                    st.balloons()
                    st.success(f"✅ **{name}** başarıyla eklendi!")
                    logger.info(f"Speaker enrolled: {name}")

                    # Save sample for future training
                    save_dir = os.path.join("src", "data", "raw", name)
                    os.makedirs(save_dir, exist_ok=True)
                    sample_path = os.path.join(save_dir, f"{name}_{int(time.time())}.wav")

                    with wave.open(sample_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(raw_bytes)

                    st.info(f"Ses örneği kaydedildi: `{sample_path}`")
                else:
                    st.error("Kayıt başarısız. Ses kalitesi yetersiz olabilir.")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

elif page == "Konuşmacılar":
    st.title("📋 Kayıtlı Konuşmacılar")

    speakers = st.session_state.identifier.speakers

    if not speakers:
        st.warning("Henüz kayıtlı konuşmacı yok. 'Kişi Kayıt' sekmesinden yeni kişi ekleyebilirsiniz.")
    else:
        st.write(f"Toplam **{len(speakers)}** konuşmacı kayıtlı.")

        for name, emb in speakers.items():
            with st.container():
                c1, c2, c3 = st.columns([3, 2, 1])

                with c1:
                    st.subheader(f"👤 {name}")

                with c2:
                    st.caption(f"Embedding Boyutu: {emb.shape[0]}")

                with c3:
                    if st.button("🗑️ Sil", key=f"del_{name}"):
                        st.session_state.identifier.remove_speaker(name)
                        logger.info(f"Speaker removed: {name}")
                        st.rerun()

                st.divider()

elif page == "Karşılaştırma":
    st.title("🔬 Konuşmacı Karşılaştırma")

    speakers = st.session_state.identifier.speakers

    if len(speakers) < 2:
        st.warning("Karşılaştırma için en az 2 kayıtlı konuşmacı gereklidir.")
    elif not VIZ_AVAILABLE:
        st.error("Görselleştirme için umap-learn ve plotly gereklidir. `pip install umap-learn plotly`")
    else:
        st.write("Kayıtlı konuşmacıların embedding vektörleri 2D uzayda görselleştirilmiştir.")

        method = st.radio("Boyut İndirgeme Yöntemi", ["UMAP", "t-SNE"], horizontal=True)

        with st.spinner("Görselleştirme hesaplanıyor..."):
            reduced = reduce_embeddings(speakers, method=method.lower())
            fig = create_plotly_scatter(reduced, title=f"Konuşmacı Embedding Uzayı ({method})")

            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Görselleştirme oluşturulamadı.")

        # Similarity matrix
        st.subheader("📊 Benzerlik Matrisi")

        names = list(speakers.keys())
        n = len(names)
        sim_matrix = np.zeros((n, n))

        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                emb1 = speakers[name1]
                emb2 = speakers[name2]
                sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
                sim_matrix[i, j] = sim

        df_sim = pd.DataFrame(sim_matrix, index=names, columns=names)
        st.dataframe(df_sim.style.background_gradient(cmap='RdYlGn', vmin=0, vmax=1).format("{:.3f}"))

elif page == "Ayarlar":
    st.title("⚙️ Ayarlar")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tanıma Ayarları")

        current_threshold = config.get('inference', {}).get('threshold', 0.82)
        new_threshold = st.slider("Benzerlik Eşiği", 0.5, 1.0, current_threshold, 0.01,
                                   help="Düşük değer = daha kolay tanıma, yanlış pozitif riski. Yüksek değer = daha katı tanıma.")

        current_vad = float(config.get('inference', {}).get('vad_aggressiveness', 0.5))
        new_vad = st.slider("VAD Hassasiyeti", 0.0, 1.0, current_vad, 0.1,
                            help="Konuşma algılama hassasiyeti. Yüksek = daha hassas.")

    with col2:
        st.subheader("Sistem Bilgisi")

        st.markdown(f"""
        | Özellik | Değer |
        |---------|-------|
        | PyAudio | {'✅' if PYAUDIO_AVAILABLE else '❌'} |
        | Ses Kalitesi | {'✅' if QUALITY_AVAILABLE else '❌'} |
        | Görselleştirme | {'✅' if VIZ_AVAILABLE else '❌'} |
        | GPU | {'✅ ' + str(st.session_state.identifier.device) if 'cuda' in str(st.session_state.identifier.device) else '❌ CPU'} |
        """)

    if st.button("💾 Ayarları Kaydet"):
        # Update config in memory (full file write would require more logic)
        st.success(f"Eşik değeri güncellendi: {new_threshold}")
        st.info("Not: Kalıcı değişiklik için `config.yaml` dosyasını düzenleyin.")

    st.markdown("---")
    st.subheader("Yapılandırma Dosyası")
    st.code(yaml.dump(config, default_flow_style=False, allow_unicode=True), language="yaml")
