import streamlit as st
import os
import tempfile
import sys
import torch
import time

# Ensure we can import from src
sys.path.append(os.path.abspath('.'))

from src.inference.identify_speaker import SpeakerIdentifier
from src.training.training_api import (
    prepare_training_data, 
    start_training_async, 
    get_training_status,
    enroll_all_speakers
)
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title="Konuşmacı Tanıma", layout="centered")

st.title("🎙️ Konuşmacı Tanıma Sistemi")

@st.cache_resource
def get_identifier():
    """
    Load the model once and cache it.
    """
    base_dir = os.path.abspath('.')
    model_path = os.path.join(base_dir, "models", "speaker_encoder.pt")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return SpeakerIdentifier(
        model_path=model_path, 
        device=device
    )

try:
    identifier = get_identifier()
    st.success(f"Model başarıyla yüklendi: {identifier.device}")
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")
    st.stop()

st.write("---")
st.subheader("Ses Kaydı veya Yükleme")

tab1, tab2, tab3, tab4 = st.tabs(["🎤 Kayıt", "📁 Dosya Yükle", "➕ Yeni Kişi Ekle", "🎓 Model Eğit"])

temp_path = None

with tab1:
    st.write("Kaydı başlatmak için mikrofona tıklayın.")
    audio_bytes = audio_recorder(
        pause_threshold=2.0,
        sample_rate=16000
    )
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            fp.write(audio_bytes)
            temp_path = fp.name

with tab2:
    uploaded_file = st.file_uploader("Bir WAV dosyası yükleyin", type=["wav"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            fp.write(uploaded_file.getbuffer())
            temp_path = fp.name

with tab3:
    st.write("Yeni bir konuşmacıyı sisteme ekle. Birden fazla dosya seçerek modelin başarısını artırabilirsiniz.")
    new_name = st.text_input("Konuşmacı Adı")
    enroll_files = st.file_uploader("Kayıt Dosyaları (Çoklu Seçim)", type=["wav"], key="enroll_upload", accept_multiple_files=True)
    
    if st.button("Kaydet (Enroll)") and new_name and enroll_files:
         progress_bar = st.progress(0)
         success_count = 0
         
         for i, uploaded_file in enumerate(enroll_files):
             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
                fp.write(uploaded_file.getbuffer())
                enroll_path = fp.name
             
             st.text(f"İşleniyor: {uploaded_file.name}")
             
             if identifier.enroll_speaker(new_name, enroll_path):
                 success_count += 1
             
             os.remove(enroll_path)
             progress_bar.progress((i + 1) / len(enroll_files))
             
         if success_count == len(enroll_files):
             st.success(f"Tamamlandı! {success_count} adet ses dosyası kullanılarak '{new_name}' için güçlü bir profil oluşturuldu.")
             st.info("Sistem, yüklediğiniz tüm dosyaların ortalamasını (Centroid) alarak ideal vektörü hesapladı.")
         else:
             st.warning(f"İşlem bitti ancak sadece {success_count}/{len(enroll_files)} dosya başarılı oldu.")

with tab4:
    st.write("Modeli eğitmek için konuşmacıların ses dosyalarını yükleyin.")
    
    # Session state for speaker management
    if "training_speakers" not in st.session_state:
        st.session_state.training_speakers = {}
    if "speaker_count" not in st.session_state:
        st.session_state.speaker_count = 1
    
    st.markdown("### 📂 Konuşmacı Ekle")
    
    # Add new speaker input
    col1, col2 = st.columns([3, 1])
    with col1:
        new_speaker_name = st.text_input("Yeni Konuşmacı Adı", key="new_speaker_input")
    with col2:
        st.write("")  # Spacing
        st.write("")
        if st.button("➕ Ekle"):
            if new_speaker_name and new_speaker_name.strip():
                name = new_speaker_name.strip().upper()
                if name not in st.session_state.training_speakers:
                    st.session_state.training_speakers[name] = []
                    st.rerun()
    
    # File upload for each speaker
    if st.session_state.training_speakers:
        st.markdown("### 🎵 Ses Dosyaları Yükle")
        
        for speaker_name in list(st.session_state.training_speakers.keys()):
            with st.expander(f"👤 {speaker_name}", expanded=True):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    files = st.file_uploader(
                        f"{speaker_name} ses dosyaları",
                        type=["wav"],
                        accept_multiple_files=True,
                        key=f"upload_{speaker_name}"
                    )
                    if files:
                        st.session_state.training_speakers[speaker_name] = [f.getvalue() for f in files]
                        st.caption(f"{len(files)} dosya seçildi")
                
                with col2:
                    st.write("")
                    if st.button("🗑️", key=f"del_{speaker_name}"):
                        del st.session_state.training_speakers[speaker_name]
                        st.rerun()
        
        # Training parameters
        st.markdown("### ⚙️ Eğitim Ayarları")
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("Epoch Sayısı", min_value=5, max_value=50, value=10)
        with col2:
            use_pretrained = st.checkbox("Mevcut modeli kullan (Fine-tuning)", value=True)
        
        # Start training button
        st.markdown("---")
        
        # Check if we have enough data
        total_files = sum(len(files) for files in st.session_state.training_speakers.values())
        speakers_with_files = sum(1 for files in st.session_state.training_speakers.values() if files)
        
        if speakers_with_files < 2:
            st.warning("⚠️ En az 2 farklı konuşmacı için ses dosyası gereklidir.")
        elif total_files < 4:
            st.warning("⚠️ Toplam en az 4 ses dosyası gereklidir.")
        else:
            st.info(f"✅ {speakers_with_files} konuşmacı, {total_files} ses dosyası hazır.")
        
        if st.button("🚀 Eğitimi Başlat", disabled=(speakers_with_files < 2 or total_files < 4)):
            # Prepare data
            with st.spinner("Veriler hazırlanıyor..."):
                speakers_data = {
                    name: files for name, files in st.session_state.training_speakers.items() if files
                }
                data_path = prepare_training_data(speakers_data)
            
            # Start training
            model_path = os.path.join(os.path.abspath('.'), "models", "speaker_encoder.pt")
            pretrained = model_path if use_pretrained and os.path.exists(model_path) else None
            
            if start_training_async(data_path, model_path, epochs=epochs, pretrained_model=pretrained):
                st.success("Eğitim başlatıldı! İlerlemeyi aşağıda takip edin.")
            else:
                st.error("Eğitim zaten devam ediyor.")
        
        # Show training progress
        status = get_training_status()
        if status["is_running"] or status["completed"] or status["error"]:
            st.markdown("### 📊 Eğitim Durumu")
            
            if status["is_running"]:
                progress = status["current_epoch"] / status["total_epochs"] if status["total_epochs"] > 0 else 0
                st.progress(progress)
                st.write(f"**Durum:** {status['status_message']}")
                st.write(f"**Epoch:** {status['current_epoch']}/{status['total_epochs']}")
                st.write(f"**Loss:** {status['current_loss']:.4f}")
                
                # Auto-refresh
                time.sleep(1)
                st.rerun()
                
            elif status["completed"]:
                st.success(f"✅ {status['status_message']}")
                
                # Offer to enroll speakers
                if st.button("📝 Konuşmacıları Veritabanına Kaydet"):
                    with st.spinner("Konuşmacılar kaydediliyor..."):
                        base_dir = os.path.abspath('.')
                        model_path_for_enroll = os.path.join(base_dir, "models", "speaker_encoder.pt")
                        results = enroll_all_speakers("data/raw", model_path_for_enroll)
                        success = sum(1 for v in results.values() if v)
                        st.success(f"{success}/{len(results)} konuşmacı başarıyla kaydedildi.")
                        
                        # Clear cache to reload model
                        get_identifier.clear()
                        st.rerun()
                        
            elif status["error"]:
                st.error(f"❌ Hata: {status['error']}")
    else:
        st.info("👆 Yukarıdan konuşmacı ekleyerek başlayın.")

# Inference
if temp_path:
    st.write("---")
    st.subheader("Sonuçlar")
    
    with st.spinner("Ses analizi yapılıyor..."):
        try:
            result = identifier.identify(temp_path, threshold=0.75)
            
            if result:
                speaker = result.get("speaker")
                confidence = result.get("confidence")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if speaker and speaker != "Unknown":
                        st.success(f"**{speaker}**")
                    else:
                        st.warning("**Bilinmeyen Kişi**")
                
                with col2:
                    st.metric("Benzerlik Skoru", f"{confidence:.3f}")
                
            else:
                st.error("Bu ses dosyasından embedding üretilemedi (Çok kısa veya sessiz).")
                
        except Exception as e:
            st.error(f"Analiz sırasında hata: {e}")
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
