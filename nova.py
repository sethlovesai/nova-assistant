import os
import ssl
import urllib.request
import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from openai import OpenAI
import subprocess
import tempfile
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

# Fix SSL certificate verification for Whisper model download
ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVEN_LABS_KEY"))

# Global variables
whisper_model = None
conversation_history = []



def initialize_whisper():
    """Initialize Whisper model once"""
    global whisper_model
    if whisper_model is None:
        print("Nova is booting up...")
        try:
            whisper_model = whisper.load_model("base")
            print("Nova ready")
        except Exception as e:
            print(f"Error: {e}")
            return False
    return True

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone"""
    try:
        print(f"Recording for {duration} seconds... Speak now!")
        
        audio = sd.rec(int(duration * sample_rate), 
                        samplerate=sample_rate, 
                        channels=1, 
                        dtype=np.float32)
        sd.wait()
        print("⏹️  Recording finished")
        return audio.flatten(), sample_rate
    except Exception as e:
        print(f"Recording error: {e}")
        return None, None

def transcribe_audio(audio, sample_rate):
    """Convert speech to text using OpenAI Whisper"""

    if audio is None:
        return ""
    
    try:
        temp_filename = "temp_audio.wav"
        write(temp_filename, sample_rate, audio)
        
        result = whisper_model.transcribe(temp_filename)
        
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        return result["text"].strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")
        return ""

def speak_text(text):
    """Convert text to speech"""
    if not text.strip():
        return False
        
    try:
        # Generate speech
        audio_stream = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id="HE9Vblt34asUwmFv9IWS",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        
        audio = b"".join(audio_stream)

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f: 
            f.write(audio)
            temp_filename = f.name

        # Use safe macOS audio player
        subprocess.run(['afplay', temp_filename], check=True)
        os.unlink(temp_filename)

        return True
    except Exception as e:
        print(f"Speech error: {e}")
        return False

def chat_with_nova(user_input):
    """Get response from Nova using OpenAI"""
    global conversation_history
    
    try:
        # Add user message to conversation
        conversation_history.append({"role": "user", "content": user_input})
        
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
        
        # System prompt for Nova
        messages = [
            {
                "role": "system", 
                "content": """You are Nova, a friendly 28-year-old personal assistant working in a tech company. 
                You are bubbly yet professional, keep responses concise and conversational. 
                Address the user as 'boss' or 'Seth'. Be helpful and engaging."""
            }
        ] + conversation_history
        
        # Get response from OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        
        nova_response = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": nova_response})
        
        return nova_response
    except Exception as e:
        print(f"Chat error: {e}")
        return "I'm sorry, I'm having trouble processing that right now."

def run_nova():
    """Main voice assistant loop"""
    print("Nova Voice Assistant Starting...")
    
    # Initialize Whisper
    if not initialize_whisper():
        print("❌ Failed to initialize. Exiting.")
        return
    
    print("✅ Nova is ready! Press Ctrl+C to exit.")
    
    # Welcome message
    speak_text("Hello boss! How can I help you today?")
    
    try:
        while True:
            print("\n" + "="*50)
            
            # Record and transcribe
            audio, sr = record_audio(duration=5)
            if audio is None:
                continue
                
            user_input = transcribe_audio(audio, sr)
            print(f"🗣️  You: '{user_input}'")
            
            if not user_input.strip():
                speak_text("I didn't catch that. Could you repeat?")
                continue
                
            # Check for exit commands
            if any(word in user_input.lower() for word in ['goodbye', 'bye', 'exit', 'quit', 'stop']):
                speak_text("Goodbye boss! Have a great day!")
                break
            
            # Get Nova's response
            print("🤖 Nova is thinking...")
            response = chat_with_nova(user_input)
            print(f"🤖 Nova: {response}")
            
            # Speak the response
            speak_text(response)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        speak_text("Goodbye boss!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_system():
    """Test all components"""
    print("🧪 Testing Nova Voice Assistant...")
    
    # Test Whisper
    if not initialize_whisper():
        return False
    
    # Test text-to-speech
    print("🔊 Testing voice output...")
    if speak_text("Nova voice system test successful!"):
        print("✅ All systems working!")
        return True
    else:
        print("❌ Voice system failed")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_system()
    else:
        run_nova()
