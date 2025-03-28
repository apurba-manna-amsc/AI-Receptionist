# 🤖 AI Receptionist: Voice-Assisted Conversational Assistant

## Overview
An advanced voice-interactive application leveraging cutting-edge technologies:

- 🎙️ Speech-to-Text: NVIDIA Riva ASR
- 💬 Large Language Model: Llama 3 (8B) via Groq API
- 🔊 Text-to-Speech: NVIDIA Riva TTS

## Workflow
1. **Voice Input**: Captures user speech via microphone.
2. **Speech-to-Text (STT)**: Converts speech into text using NVIDIA Riva ASR.
3. **AI Response Generation**: Processes text input using Llama 3 (Groq API) to generate an intelligent response.
4. **Text-to-Speech (TTS)**: Converts AI-generated response back to speech using NVIDIA Riva TTS.
5. **Audio Output**: Plays the synthesized speech to the user.

## Technology Stack
- **STT**: NVIDIA Riva ASR
- **LLM**: Llama 3 (8B) via Groq API
- **TTS**: NVIDIA Riva TTS
- **Framework**: Python

## Prerequisites

### Hardware Requirements
- High-performance CPU/GPU
- Microphone for voice input
- Speakers or other audio output device

### Software Dependencies
- Python 3.8+
- PyAudio
- NVIDIA Riva SDK
- Groq API Client

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/ai-receptionist.git
cd ai-receptionist
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Unix/macOS
# venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install nvidia-riva-client
```

### 4. Download NVIDIA Riva Python Client
```bash
git clone https://github.com/nvidia-riva/python-clients.git
```

### 5. Configure Environment Variables
Create a `.env` file in the project directory and add the following:
```plaintext
NVDIA_API_KEY=your_nvidia_riva_api_key
GROQ_API_KEY=your_groq_api_key
```

## Running the Application
```bash
python ai_receptionist.py
```

## Configuration Parameters
- **Sample Rate**: 16kHz
- **Audio Chunk**: 1600 samples
- **Language**: English (US)
- **Voice**: Magpie Multilingual Female

## Performance Characteristics
- **Speech Recognition (ASR) Latency**: ~0.5-1.5 seconds
- **Response Generation (LLM)**: <2 seconds
- **Voice Synthesis (TTS)**: ~1 second

## References

### 1. NVIDIA Riva ASR (Speech-to-Text)
- Advanced neural speech recognition
- High accuracy in noisy environments
- Supports multiple languages
- **Documentation**: [NVIDIA Riva ASR](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-overview.html)

### 2. Llama 3 (Large Language Model) via Groq API
- Developed by Meta AI
- Optimized for conversational AI
- **Key Features**:
  - Contextual understanding
  - Concise response generation
  - Multi-turn conversation support
- **References**:
  - [Meta AI Research](https://ai.meta.com/research/publications/)
  - [Llama 3 Overview](https://ai.meta.com/blog/meta-llama-3/)

### 3. NVIDIA Riva TTS (Text-to-Speech)
- Multilingual TTS engine
- Natural voice synthesis
- Low-latency audio generation
- **Documentation**: [NVIDIA Riva TTS](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tts/tts-overview.html)

### 4. Groq API
- High-performance inference platform
- Low-latency model serving
- **Resources**:
  - [Groq Developer Portal](https://wow.groq.com/)
  - [API Documentation](https://console.groq.com/docs)

### 5. Additional Resources
- **NVIDIA AI Models**: [NVIDIA Build](https://build.nvidia.com/models)
- **Groq API Overview**: [Groq API Docs](https://console.groq.com/docs/overview)
- **AssemblyAI YouTube Channel**: [AssemblyAI](https://www.youtube.com/@AssemblyAI)

## Troubleshooting
- Verify API connectivity and ensure keys are correct
- Check microphone permissions and system settings
- Ensure a stable internet connection
- Validate environment variables in the `.env` file

## Future Enhancements
- **Multi-language support**: Expand ASR and TTS to additional languages.
- **Emotion Detection**: Adjust TTS tone based on sentiment analysis.
- **Cloud Integration**: Deploy as a cloud-based voice agent.
- **GUI Interface**: Develop a user-friendly dashboard for monitoring interactions.


## Acknowledgments
- NVIDIA AI Labs
- Meta AI Research
- Groq Infrastructure Team

---

### Recommended `requirements.txt`
```plaintext
pyaudio
numpy
python-dotenv
requests
riva-client
groq
```

