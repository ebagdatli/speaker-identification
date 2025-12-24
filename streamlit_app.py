import streamlit as st
import os
import tempfile
import sys
import torch

# Ensure we can import from src
sys.path.append(os.path.abspath('.'))

from src.inference.identify_speaker import SpeakerIdentifier
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title="Speaker ID", layout="centered")

st.title("🎙️ Speaker Identification System")

@st.cache_resource
def get_identifier():
    """
    Load the model once and cache it.
    """
    # Use absolute paths for robustness
    base_dir = os.path.abspath('.')
    model_path = os.path.join(base_dir, "models", "speaker_encoder.pt")
    emb_path = os.path.join(base_dir, "embeddings", "speakers.json")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return SpeakerIdentifier(
        model_path=model_path, 
        embeddings_path=emb_path,
        device=device
    )

try:
    identifier = get_identifier()
    st.success(f"Model loaded successfully on {identifier.device}")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.write("---")
st.subheader("Record or Upload Audio")

tab1, tab2 = st.tabs(["🎤 Record", "📁 Upload"])

temp_path = None

with tab1:
    st.write("Click the microphone to start recording.")
    audio_bytes = audio_recorder(
        pause_threshold=2.0,
        sample_rate=16000 # Trying to match training rate/librosa default mostly
    )
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            fp.write(audio_bytes)
            temp_path = fp.name

with tab2:
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            fp.write(uploaded_file.getbuffer())
            temp_path = fp.name

# Inference
if temp_path:
    st.write("---")
    st.subheader("Results")
    
    with st.spinner("Analyzing voice print..."):
        try:
            # We use a lower threshold for UI feedback, or strict?
            # Using the identify method default (0.85) or customized
            result = identifier.identify(temp_path, threshold=0.80)
            
            if result:
                speaker = result.get("speaker")
                confidence = result.get("confidence")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if speaker and speaker != "Unknown":
                        st.success(f"**{speaker}**")
                    else:
                        st.warning("**Unknown Speaker**")
                
                with col2:
                    st.metric("Similarity Score", f"{confidence:.3f}")
                
            else:
                st.error("Could not generate embedding for this audio.")
                
        except Exception as e:
            st.error(f"Error during inference: {e}")
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
