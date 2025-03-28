import os
import re
import time
import queue
import numpy as np
import pyaudio
import riva.client
from riva.client.proto.riva_audio_pb2 import AudioEncoding
import requests
from dotenv import load_dotenv

# Load environment variables from .env file to securely manage API keys
load_dotenv()

class AI_Receptionist:
    def __init__(self):
        # Audio stream configuration for voice input
        self.rate = 16000  # Sample rate (16kHz)
        self.chunk = 1600  # Audio buffer size
        self.format = pyaudio.paInt16  # 16-bit audio format
        self.channels = 1  # Mono audio
        
        # NVIDIA Riva API configuration
        self.api_key = os.getenv("NVDIA_API_KEY")
        self.server = "grpc.nvcf.nvidia.com:443"
        self.use_ssl = True

        # Groq API configuration for AI response generation
        self.groq_api_key = os.getenv("groq_api_key")
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama3-8b-8192"
        
        # Initial system prompt to define AI receptionist's persona
        self.chat_history = [
            {"role": "system", "content": "Alice, you are the best receptionist at ABC Labs—efficient, friendly, and always ready to assist! Provide direct and concise responses to queries while maintaining a professional and welcoming tone. Keep answers within 20 words. Your exceptional communication skills and warm personality make you a valuable asset to the team!"}
        ]
        
        # Authentication for NVIDIA Riva ASR (Speech-to-Text) service
        self.auth_asr = riva.client.Auth(ssl_cert=None, use_ssl=self.use_ssl, uri=self.server, metadata_args=[
            ("function-id", "1598d209-5e27-4d3c-8079-4751568b1081"),
            ("authorization", f"Bearer {self.api_key}")
        ])
        self.asr_service = riva.client.ASRService(self.auth_asr)
        
        # Authentication for NVIDIA Riva TTS (Text-to-Speech) service
        self.auth_tts = riva.client.Auth(ssl_cert=None, use_ssl=self.use_ssl, uri=self.server, metadata_args=[
            ("function-id", "877104f7-e885-42b9-8de8-f6e4c6303969"),
            ("authorization", f"Bearer {self.api_key}")
        ])
        self.tts_service = riva.client.SpeechSynthesisService(self.auth_tts)
        
        # ASR configuration for speech recognition
        self.asr_config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding="LINEAR_PCM",
                sample_rate_hertz=self.rate,
                language_code="en-US",
                max_alternatives=1,  # Return top transcription result
                enable_automatic_punctuation=True,
                verbatim_transcripts=True
            ),
            interim_results=False,  # Only return final transcription
        )
        
        # Initialize audio queue and PyAudio stream
        self.audio_queue = queue.Queue()
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self.callback
        )
        
    def callback(self, in_data, frame_count, time_info, status):
        # Audio input callback with noise filtering
        audio_array = np.frombuffer(in_data, dtype=np.int16)
        
        # Only add audio to queue if it exceeds noise threshold
        if np.max(np.abs(audio_array)) > 500:
            self.audio_queue.put(in_data)

        return (in_data, pyaudio.paContinue)

    def start_transcription(self):
        print("🎤 Listening... Speak now.")
        self.stream.start_stream()
        
        # Silence and pause detection parameters
        silence_threshold = 800
        pause_duration = 1
        
        try:
            while True:
                audio_buffer = []
                last_speech_time = time.time()
                
                # Continuous audio capture with pause detection
                while True:
                    if not self.audio_queue.empty():
                        data = self.audio_queue.get()
                        audio_array = np.frombuffer(data, dtype=np.int16)
                        
                        # Update last speech time when audio exceeds threshold
                        if np.max(np.abs(audio_array)) > silence_threshold:
                            last_speech_time = time.time()
                        
                        audio_buffer.append(data)
                    
                    # Break if silence persists beyond pause duration
                    if time.time() - last_speech_time > pause_duration:
                        break
                
                # Process captured audio
                if audio_buffer:
                    raw_audio = b''.join(audio_buffer)
                    
                    # Measure ASR processing time
                    start_time = time.time()
                    responses = self.asr_service.streaming_response_generator(
                        audio_chunks=[raw_audio], streaming_config=self.asr_config
                    )
                    asr_time = time.time() - start_time
                    
                    # Extract final transcription
                    full_text = ""
                    for response in responses:
                        for result in response.results:
                            if result.is_final:
                                full_text = result.alternatives[0].transcript
                    
                    print(f"📝 Recognized Text: {full_text} (ASR Time: {asr_time:.2f}s)")
                    
                    # Generate AI response if text is detected
                    if full_text:
                        ai_response = self.generate_ai_response(full_text)
                        self.generate_audio(ai_response)
                    
                    print("🎤 Listening... Speak now.")
                    
                    # Clear remaining audio queue
                    while not self.audio_queue.empty():
                        self.audio_queue.get()
        
        except KeyboardInterrupt:
            print("🛑 Transcription stopped.")
            self.stream.stop_stream()
    
    def clean_text(self, text):
        """
        Sanitizes text by:
        - Removing markdown-style symbols
        - Replacing multiple whitespaces with single space
        - Trimming leading/trailing whitespace
        """
        text = re.sub(r"[*_#`]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def generate_ai_response(self, text):
        # Add user input to chat history
        self.chat_history.append({"role": "user", "content": text})
        
        # Prepare Groq API request headers and payload
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": self.chat_history,
            "temperature": 0.7  # Balanced creativity
        }
        
        # Measure AI response generation time
        start_time = time.time()
        response = requests.post(self.groq_url, json=payload, headers=headers)
        ai_time = time.time() - start_time
        
        # Process AI response
        if response.status_code == 200:
            ai_response = response.json()["choices"][0]["message"]["content"]
            cleaned_response = self.clean_text(ai_response)
            self.chat_history.append({"role": "assistant", "content": cleaned_response})
            print(f"🤖 AI Response Time: {ai_time:.2f}s")
            return cleaned_response
        else:
            print(f"Error: {response.text}")
            return "I'm sorry, I couldn't process that."
    
    def generate_audio(self, text):
        print(f"🎙 AI Receptionist: {text}")
        
        # Measure TTS synthesis time
        start_time = time.time()
        responses = self.tts_service.synthesize_online(
            text=self.clean_text(text),
            voice_name="Magpie-Multilingual.EN-US.Female.Female-1",
            language_code="en-US",
            sample_rate_hz=44100,
            encoding=AudioEncoding.LINEAR_PCM
        )
        tts_time = time.time() - start_time
        
        print(f"🔊 TTS Synthesis Time: {tts_time:.2f}s")
        
        # Audio playback using PyAudio
        p_out = pyaudio.PyAudio()
        stream_out = p_out.open(format=pyaudio.paInt16, channels=1, rate=44100, output=True)
        for response in responses:
            stream_out.write(response.audio)
        stream_out.stop_stream()
        stream_out.close()
        p_out.terminate()

if __name__ == "__main__":
    # Initialize and start the AI receptionist
    receptionist = AI_Receptionist()
    greeting = "Thank you for calling. My name is Alice, how may I assist you?"
    receptionist.generate_audio(greeting)
    time.sleep(1)  # Brief pause before starting transcription
    receptionist.start_transcription()