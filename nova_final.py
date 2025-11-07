#!/usr/bin/env python3
"""
Nova Assistant - Final Beautiful UI
- Toggle recording: Press to start, press again to stop (no time limit)
- Exact design from the image
- Simple and reliable
"""

import streamlit as st
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
from nova import initialize_whisper, transcribe_audio, chat_with_nova, speak_text

# Page configuration
st.set_page_config(
    page_title="Nova AI",
    page_icon="🌟",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS matching the exact design from the image
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        height: 100%;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0f1c 0%, #1a1f35 50%, #0a0f1c 100%);
        background-attachment: fixed;
        min-height: 100vh;
    }
    
    .main {
        padding: 0;
        height: 100vh;
        display: flex;
        flex-direction: column;
    }
    
    /* Animated stars */
    .stars {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 1;
    }
    
    .star {
        position: absolute;
        width: 2px;
        height: 2px;
        background: white;
        border-radius: 50%;
        opacity: 0.8;
        animation: twinkle 4s infinite linear;
    }
    
    @keyframes twinkle {
        0%, 100% { opacity: 0.3; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.2); }
    }
    
    /* Top bar */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem 2rem;
        position: relative;
        z-index: 10;
    }
    
    .logo {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        color: white;
        font-size: 1.25rem;
        font-weight: 600;
    }
    
    .logo-icon {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #22d3ee, #06b6d4);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
    }
    
    .status {
        color: #10b981;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    /* Main content */
    .content {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 2rem;
        position: relative;
        z-index: 10;
    }
    
    .title {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #22d3ee, #a855f7);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.125rem;
        font-weight: 400;
        max-width: 500px;
        margin-bottom: 3rem;
        line-height: 1.6;
    }
    
    /* Voice orb */
    .voice-orb-container {
        margin: 2rem 0 4rem 0;
    }
    
    .voice-orb {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        background: linear-gradient(135deg, #22d3ee, #06b6d4);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        color: white;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 20px 60px rgba(34, 211, 238, 0.3);
        position: relative;
        animation: gentle-float 6s ease-in-out infinite;
    }
    
    .voice-orb:hover {
        transform: scale(1.05);
        box-shadow: 0 25px 80px rgba(34, 211, 238, 0.4);
    }
    
    .voice-orb.recording {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        animation: pulse-recording 2s infinite, gentle-float 6s ease-in-out infinite;
        box-shadow: 0 20px 60px rgba(239, 68, 68, 0.4);
    }
    
    .voice-orb.recording::before {
        content: '';
        position: absolute;
        width: 220px;
        height: 220px;
        border: 2px solid rgba(239, 68, 68, 0.3);
        border-radius: 50%;
        animation: ripple 2s infinite;
    }
    
    @keyframes gentle-float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }
    
    @keyframes pulse-recording {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }
    
    @keyframes ripple {
        0% { transform: scale(1); opacity: 0.3; }
        100% { transform: scale(1.15); opacity: 0; }
    }
    
    /* Bottom input section */
    .bottom-section {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(10, 15, 28, 0.95);
        backdrop-filter: blur(20px);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        z-index: 100;
    }
    
    .input-container {
        max-width: 600px;
        margin: 0 auto;
    }
    
    .input-row {
        display: flex;
        gap: 0.75rem;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .text-input {
        flex: 1;
        padding: 1rem 1.25rem;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 25px;
        color: white;
        font-size: 1rem;
        outline: none;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }
    
    .text-input:focus {
        border-color: rgba(34, 211, 238, 0.5);
        box-shadow: 0 0 20px rgba(34, 211, 238, 0.2);
    }
    
    .text-input::placeholder {
        color: #64748b;
    }
    
    .send-btn {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: linear-gradient(135deg, #22d3ee, #06b6d4);
        border: none;
        color: white;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        transition: all 0.3s ease;
    }
    
    .send-btn:hover {
        transform: scale(1.1);
        box-shadow: 0 8px 25px rgba(34, 211, 238, 0.3);
    }
    
    .mic-btn {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #22d3ee;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        transition: all 0.3s ease;
    }
    
    .mic-btn:hover {
        background: rgba(34, 211, 238, 0.1);
        border-color: #22d3ee;
    }
    
    .mic-btn.recording {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        border-color: #ef4444;
        color: white;
        animation: pulse-mic 2s infinite;
    }
    
    @keyframes pulse-mic {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .input-hint {
        text-align: center;
        color: #64748b;
        font-size: 0.75rem;
        margin-bottom: 0.5rem;
    }
    
    .credit {
        text-align: center;
        color: #475569;
        font-size: 0.75rem;
    }
    
    /* Conversation */
    .conversation {
        position: absolute;
        top: 120px;
        left: 50%;
        transform: translateX(-50%);
        width: 90%;
        max-width: 800px;
        max-height: calc(100vh - 300px);
        overflow-y: auto;
        z-index: 10;
        padding-bottom: 2rem;
    }
    
    .message {
        margin: 1rem 0;
        max-width: 75%;
        animation: fadeInUp 0.3s ease;
    }
    
    .user-msg {
        margin-left: auto;
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        padding: 0.875rem 1.25rem;
        border-radius: 1.25rem 1.25rem 0.25rem 1.25rem;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .nova-msg {
        margin-right: auto;
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.875rem 1.25rem;
        border-radius: 1.25rem 1.25rem 1.25rem 0.25rem;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Status messages */
    .status-msg {
        text-align: center;
        padding: 1rem;
        margin: 1rem auto;
        max-width: 400px;
        border-radius: 0.75rem;
        font-weight: 500;
        font-size: 0.875rem;
    }
    
    .status-recording {
        background: rgba(239, 68, 68, 0.1);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.3);
        animation: pulse-status 2s infinite;
    }
    
    .status-processing {
        background: rgba(251, 191, 36, 0.1);
        color: #fcd34d;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.1);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    @keyframes pulse-status {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Hide Streamlit elements */
    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    .stDeployButton,
    footer {
        display: none !important;
    }
    
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Custom Streamlit button styling */
    div.stButton > button {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        color: inherit !important;
        width: auto !important;
        height: auto !important;
    }
    
    .stTextInput > div > div > input {
        display: none !important;
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
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None

def init_nova():
    """Initialize Nova"""
    if not st.session_state.nova_initialized:
        with st.spinner("🌟 Nova is initializing..."):
            if initialize_whisper():
                st.session_state.nova_initialized = True
                return True
            else:
                st.error("Failed to initialize Nova")
                return False
    return True

def toggle_recording():
    """Toggle recording on/off"""
    if not st.session_state.recording:
        # Start recording
        st.session_state.recording = True
        st.session_state.recorded_audio = None
        st.rerun()
    else:
        # Stop recording and process
        st.session_state.recording = False
        
        # Record audio now
        try:
            sample_rate = 16000
            st.info("🔄 Processing your voice...")
            
            # Record for a reasonable duration (max 30 seconds as safety)
            duration = 10  # Allow longer recordings
            audio = sd.rec(int(duration * sample_rate), 
                          samplerate=sample_rate, 
                          channels=1, 
                          dtype=np.float32)
            sd.wait()
            
            st.session_state.recorded_audio = audio.flatten()
            
            # Transcribe
            user_input = transcribe_audio(st.session_state.recorded_audio, sample_rate)
            
            if user_input.strip():
                st.success(f"✅ Heard: '{user_input}'")
                handle_user_input(user_input)
            else:
                st.warning("⚠️ Could not understand the audio. Please try again.")
                
        except Exception as e:
            st.error(f"Recording error: {e}")
        
        st.rerun()

def add_message(role, content):
    """Add message to conversation"""
    st.session_state.conversation.append({
        'role': role,
        'content': content,
        'timestamp': time.time()
    })

def handle_user_input(user_input):
    """Process user input"""
    if user_input.strip():
        add_message('user', user_input)
        
        with st.spinner("🤖 Nova is thinking..."):
            response = chat_with_nova(user_input)
            add_message('nova', response)
            
            try:
                speak_text(response)
            except Exception as e:
                st.warning(f"Could not play audio: {e}")

def main():
    # Add animated stars

    # Initialize Nova
    if not init_nova():
        st.stop()
    
    # Top bar
    st.markdown("""
    <div class="top-bar">
        <div class="logo">
            <div class="logo-icon">🌟</div>
            Nova AI
        </div>
        <div class="status">Ready</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content area
    st.markdown("""
    <div class="content">
        <h1 class="title">✨ Hey, I'm Nova ✨</h1>
        <p class="subtitle">
            Your personal AI voice assistant. Tap the microphone or type below to start.
        </p>
        
        <div class="voice-orb-container">
    """, unsafe_allow_html=True)
    
    # Voice button with toggle functionality
    orb_class = "voice-orb recording" if st.session_state.recording else "voice-orb"
    orb_icon = "⏹️" if st.session_state.recording else "🎤"
    
    # Create a hidden button that we'll style with CSS
    if st.button(orb_icon, key="voice_orb", help="Click to start/stop recording"):
        toggle_recording()
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Recording status
    if st.session_state.recording:
        st.markdown("""
        <div class="status-msg status-recording">
            🔴 Recording... Click the button again to stop
        </div>
        """, unsafe_allow_html=True)
    
    # Conversation display
    if st.session_state.conversation:
        st.markdown('<div class="conversation">', unsafe_allow_html=True)
        
        for msg in st.session_state.conversation[-8:]:  # Show last 8 messages
            if msg['role'] == 'user':
                st.markdown(f"""
                <div class="message">
                    <div class="user-msg">🗣️ {msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message">
                    <div class="nova-msg">🤖 {msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Bottom input section
    st.markdown("""
    <div class="bottom-section">
        <div class="input-container">
            <div class="input-row">
                <!-- Custom input will go here -->
            </div>
            <p class="input-hint">Type or speak to Nova</p>
            <p class="credit">Voice-powered by Nova AI • Always learning, always improving</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Text input (positioned to work with the bottom section)
    with st.form("text_form", clear_on_submit=True):
        text_input = st.text_input("", placeholder="Type your message...", label_visibility="collapsed")
        submitted = st.form_submit_button("Send")
        
        if submitted and text_input:
            handle_user_input(text_input)
            st.rerun()

if __name__ == "__main__":
    main()
