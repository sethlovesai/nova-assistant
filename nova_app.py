#!/usr/bin/env python3
import streamlit as st
import time
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
from nova import initialize_whisper, transcribe_audio, chat_with_nova, speak_text
from nova2 import NovaAssistant

from streamlit_mic_recorder import mic_recorder
import io

# Page configuration
st.set_page_config(
    page_title="Nova Assistant",
    page_icon="🌟",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    
    .nova-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .nova-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .nova-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #007bff, #6610f2);
        color: white;
        margin-left: 2rem;
    }
    
    .nova-message {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        color: #333;
        margin-right: 2rem;
    }
    
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .status-ready {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        color: #155724;
    }
    
    .status-recording {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        color: #721c24;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .voice-controls {
        text-align: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    div.stButton > button {
        background: linear-gradient(135deg, #ff6b6b, #ff8e8e);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'nova_initialized' not in st.session_state:
    st.session_state.nova_initialized = False
if 'recording' not in st.session_state:
    st.session_state.recording = False

nova = NovaAssistant()

def init_nova():
    """Initialize Nova system"""
    if not st.session_state.nova_initialized:
        with st.spinner(" Nova is initializing..."):
            if initialize_whisper():
                st.session_state.nova_initialized = True
                return True
            else:
                st.error(" Failed to initialize Nova")
                return False
    return True

def record_audio(duration=5, sample_rate=16000):
    """Record audio using sounddevice"""
    try:
        audio = sd.rec(int(duration * sample_rate), 
                      samplerate=sample_rate, 
                      channels=1, 
                      dtype=np.float32)
        sd.wait()
        return audio.flatten(), sample_rate
    except Exception as e:
        st.error(f"Recording error: {e}")
        return None, None

def record_push_to_talk():

    audio = mic_recorder(
                start_prompt="Start recording",
                stop_prompt="Stop recording",
                just_once=False,
                use_container_width=False,
                key="nova_ptt"
            )

    if audio:
        wav_bytes = audio["bytes"]         # WAV mono bytes
        sr = audio.get("sample_rate", None)       # browser-provided sample rate (often 48k)
        st.audio(wav_bytes, format="audio/wav")

        data = np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        return data, sr 

    return None, None

def add_message(role, content):
    """Add message to conversation"""
    st.session_state.conversation.append({
        'role': role,
        'content': content,
        'timestamp': time.time()
    })

def display_conversation():
    """Display conversation history"""
    if st.session_state.conversation:
        for msg in st.session_state.conversation[-8:]:  # Show last 8 messages
            if msg['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🗣️ You:</strong><br>{msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message nova-message">
                    <strong>🤖 Nova:</strong><br>{msg['content']}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="chat-message nova-message">
            <strong>🤖 Nova:</strong><br>Hello boss! I'm Nova, your personal assistant. How can I help you today?
        </div>
        """, unsafe_allow_html=True)

def handle_user_input(user_input):
    """Process user input and get Nova's response"""
    if user_input.strip():
        # Add user message
        add_message('user', user_input)
        
        # Get Nova's response
        with st.spinner("🤖 Nova is thinking..."):
            response = nova.speak_with_nova(user_input)
            add_message('nova', response)
            
            try:
                nova.speak(response)
            except Exception as e:
                st.warning(f"Could not play audio: {e}")
        
        st.rerun()

def main():
    # Header
    st.markdown("""
    <div class="nova-header">
        <div class="nova-title">Nova</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize Nova
    if not init_nova():
        st.stop()
    
    # Status
    st.markdown("""
    <div class="status-box status-ready">
        Nova is initialised.
    </div>
    """, unsafe_allow_html=True)
    
    # Voice Controls
    st.markdown("## Talk to Nova")


    if st.button(" Speak", use_container_width=True):
        # Show recording status
        status_placeholder = st.empty()
        status_placeholder.markdown(f"""
        <div class="status-box status-recording">
                Speaking with Nova
        </div>
        """, unsafe_allow_html=True)
        nova.run_nova()
        # Record audio
        # audio, sr = record_audio()            
        # if audio is not None:
        #     status_placeholder.markdown("""
        #     <div class="status-box">
        #          Processing your voice...
        #     </div>
        #     """, unsafe_allow_html=True)
            
        #     # Transcribe
        #     user_input = transcribe_audio(audio, sr)
        #     # user_input = transcribe_audio(data, sr_loaded)  # use your current function
            
        #     if user_input:
        #         status_placeholder.markdown(f"""
        #         <div class="status-box status-ready">
        #              Heard: "{user_input}"
        #         </div>
        #         """, unsafe_allow_html=True)
                
        #         # Process the input
        #         handle_user_input(user_input)
        #     else:
        #         status_placeholder.markdown("""
        #         <div class="status-box">
        #              Could not understand the audio. Please try again.
        #         </div>
        #         """, unsafe_allow_html=True)
        # else:
        #     status_placeholder.empty()

    # Text Input
    st.markdown("## Chat with Nova")
    
    with st.form("text_form", clear_on_submit=True):
        text_input = st.text_area(
            "Type your message:",
            placeholder="Ask Nova anything...",
            height=100
        )
        
        submitted = st.form_submit_button("Enter", use_container_width=True)
        
        if submitted and text_input:
            handle_user_input(text_input)

    st.markdown("## Conversation History")
    display_conversation()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; opacity: 0.6; padding: 1rem;">
        <small>Nova Assistant | Built with Streamlit & OpenAI</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
