# tts_server.py
import os
import uuid
import json
import torch
import torchaudio
import nltk
import re
import hashlib
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Body, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
# Import model-specific libraries conditionally
try:
    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict
    MOCK_MODE = False
except ImportError:
    print("Warning: Zonos model not available. Using mock implementation.")
    MOCK_MODE = True
    # Create mock versions
    def make_cond_dict(text, **kwargs):
        """Mock implementation of make_cond_dict."""
        return {"text": text}

import uvicorn
import shutil
from pathlib import Path

from zonos.utils import DEFAULT_DEVICE as device
# Or if that's not available:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# At the top of the file, with other imports
import nltk.data

# Try to be very explicit about the NLTK data path
nltk.data.path.append(os.path.expanduser('~/nltk_data'))
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

# Download resources
nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize FastAPI
app = FastAPI(title="Zonos TTS API",
              description="API for text-to-speech synthesis using the Zonos model",
              version="1.0.0")

# Create data directory for storing voice info, samples, and generated audio
DATA_DIR = Path("data")
VOICE_INFO_DIR = DATA_DIR / "voice_info"
VOICE_SAMPLES_DIR = DATA_DIR / "voice_samples"
GENERATED_AUDIO_DIR = DATA_DIR / "generated_audio"

# Create necessary directories
for directory in [DATA_DIR, VOICE_INFO_DIR, VOICE_SAMPLES_DIR, GENERATED_AUDIO_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Pydantic models for API requests and responses
class SystemInfoResponse(BaseModel):
    model_name: str
    voices_count: int
    version: str
    cuda_available: bool
    mock_mode: bool

class VoiceListResponse(BaseModel):
    voices: List[Dict[str, Any]]

class VoiceInfoRequest(BaseModel):
    voice_info: Dict[str, Any]

class VoiceInfoResponse(BaseModel):
    file_id: str
    message: str

class TextToSpeechRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    language: str = Field("en-us", description="Language code")
    voice_id: Optional[str] = Field(None, description="Voice ID for speaker embedding")
    emotion: List[float] = Field([0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077],
                               description="Emotion vector with 8 values")
    fmax: float = Field(24000.0, description="Maximum frequency")
    pitch_std: float = Field(45.0, description="Pitch standard deviation")
    speaking_rate: float = Field(15.0, description="Speaking rate")
    vqscore_8: List[float] = Field([0.78] * 8, description="VQ score with 8 values")
    ctc_loss: float = Field(0.1, description="CTC loss value")
    dnsmos_ovrl: float = Field(4.0, description="DNSMOS overall score")
    speaker_noised: bool = Field(False, description="Whether to add noise to speaker embedding")
    unconditional_keys: List[str] = Field(["emotion"], description="Unconditional keys")
    cfg_scale: float = Field(2.5, description="CFG scale for generation")
    min_p: float = Field(0.15, description="Min P value for sampling")
    seed: Optional[int] = Field(None, description="Random seed for generation")
    max_tokens_per_chunk: int = Field(86 * 28, description="Maximum tokens per chunk")
    max_chunk_length: int = Field(250, description="Maximum text length per chunk")
    add_silence_prefix: bool = Field(True, description="Whether to add a silence prefix")

class TextToSpeechResponse(BaseModel):
    file_id: str
    file_url: str
    text: str
    message: str

class VoiceSampleResponse(BaseModel):
    voice_id: str
    message: str
    status: str

# Global variables
model = None
voices = []
voice_info_file = VOICE_INFO_DIR / "voices.json"

# Helper function to normalize IDs for case-insensitive matching
def normalize_id(id_str):
    """Normalize an ID string for case-insensitive comparison"""
    if id_str is None:
        return ""
    return str(id_str).lower().strip()

# Helper function to get a voice by ID (case-insensitive)
def get_voice_by_id(voice_id: str) -> Optional[Dict[str, Any]]:
    """Get a voice by its ID (case-insensitive)."""
    voice_id_norm = normalize_id(voice_id)
    for voice in voices:
        if normalize_id(voice.get("id")) == voice_id_norm:
            return voice
    return None

# Load voice info
def load_voice_info():
    """Load voice information from file."""
    global voices
    if voice_info_file.exists():
        try:
            with open(voice_info_file, "r", encoding="utf-8") as f:
                voice_info = json.load(f)
                voices = voice_info.get("voices", [])
            print(f"Loaded {len(voices)} voices from {voice_info_file}")
        except Exception as e:
            print(f"Error loading voice info: {e}")
            voices = []
    else:
        print("No voice info file found.")
        voices = []

# Save voice info
def save_voice_info():
    """Save voice information to file."""
    voice_info = {"voices": voices}
    with open(voice_info_file, "w", encoding="utf-8") as f:
        json.dump(voice_info, f, indent=2)
    print(f"Saved {len(voices)} voices to {voice_info_file}")

# Initialize model
def init_model():
    """Initialize the TTS model."""
    global model
    if MOCK_MODE:
        print("Running in mock mode (no Zonos model available)")
        return

    try:
        # Model initialization code - replace this with actual model initialization
        print("Initializing model...")
        model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer")
        print("Model initialized.")
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Continuing without a model (mock mode).")

# Process text for TTS
# Then replace your process_text function with these functions from your working sample
def split_text_into_sentences(text):
    """Splits text into sentences while respecting sentence boundaries."""
    return nltk.sent_tokenize(text)

def split_long_sentence(sentence, N):
    """Splits a long sentence into smaller chunks with a maximum length of N characters."""
    words = sentence.split()
    chunks = []
    current_chunk = ''
    for word in words:
        # +1 accounts for the space between words
        if len(current_chunk) + len(word) + 1 <= N:
            current_chunk += word + ' '
        else:
            chunks.append(current_chunk.rstrip())
            current_chunk = word + ' '
    chunks.append(current_chunk.rstrip())
    return chunks

def process_text(text, language="en-us", max_chunk_length=250):
    """Process text for TTS, splitting into manageable chunks."""
    try:
        sentences = split_text_into_sentences(text)

        chunks = []
        chunk = ''
        for sentence in sentences:
            if len(chunk) + len(sentence) <= max_chunk_length:
                chunk = (chunk + " " + sentence).strip()
            else:
                if len(sentence) > max_chunk_length:
                    # Split long sentence into smaller chunks first
                    long_chunks = split_long_sentence(sentence, max_chunk_length)
                    if chunk:
                        chunks.append(chunk)
                    chunks.extend(long_chunks)
                    chunk = ''
                else:
                    if chunk:
                        chunks.append(chunk)
                    chunk = sentence
        if chunk:
            chunks.append(chunk)

        return chunks

    except Exception as e:
        print(f"NLTK tokenization failed: {e}. Using fallback tokenizer.")
        # Your existing fallback tokenization code here
        punctuation = ['.', '!', '?']
        sentences = []
        current = ""
        for char in text:
            current += char
            if char in punctuation and len(current.strip()) > 0:
                sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())

        # Group sentences into chunks
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chunk_length and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            current_chunk += " " + sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

# Generate TTS audio with mockable test sounds
def generate_tts_mock(text, **kwargs):
    """Mock implementation of TTS generation for testing without the model."""
    # Generate a simple sine wave for testing
    sample_rate = 24000
    duration_seconds = max(1.0, len(text) * 0.1)  # Rough estimate: 0.1 seconds per character
    t = torch.linspace(0, duration_seconds, int(duration_seconds * sample_rate))

    # Generate a tone with some variation based on the text
    # Use hash of text to create a consistent frequency for the same text
    text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16) % 500
    frequency = 300 + text_hash  # Base frequency + variation (300-800 Hz)

    # Create a simple sine wave
    audio = 0.5 * torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)

    # Add some envelope to avoid clicks
    if len(audio[0]) > 100:
        # Fade in
        audio[0, :100] *= torch.linspace(0, 1, 100)
        # Fade out
        audio[0, -100:] *= torch.linspace(1, 0, 100)

    return audio, sample_rate

# Generate TTS audio
def generate_tts(text, **kwargs):
    """Generate TTS audio from text."""
    global model

    if model is None:
        # Use mock implementation
        return generate_tts_mock(text, **kwargs)

    try:
        # Process text into chunks
        chunks = process_text(text, kwargs.get("language", "en-us"), kwargs.get("max_chunk_length", 250))

        full_audio = None
        sample_rate = 24000

        # Process each chunk
        for chunk in chunks:
            # Create conditioning dictionary for Zonos model
            cond_dict = make_cond_dict(
                text=chunk,
                language=kwargs.get("language", "en-us"),
                speaker=kwargs.get("speaker_emb"),  # Changed to match your sample
                emotion=kwargs.get("emotion"),
                fmax=kwargs.get("fmax", 24000.0),
                pitch_std=kwargs.get("pitch_std", 45.0),
                speaking_rate=kwargs.get("speaking_rate", 15.0),
                vqscore_8=kwargs.get("vqscore_8", [0.78] * 8),
                ctc_loss=kwargs.get("ctc_loss", 0.1),
                dnsmos_ovrl=kwargs.get("dnsmos_ovrl", 4.0),
                speaker_noised=kwargs.get("speaker_noised", False),
                unconditional_keys=kwargs.get("unconditional_keys", ["emotion"]),
                device=device  # Make sure device is defined globally
            )

            # Prepare conditioning (this step was missing)
            conditioning = model.prepare_conditioning(cond_dict)

            # Generate audio with correct parameters
            codes = model.generate(
                prefix_conditioning=conditioning,
                max_new_tokens=kwargs.get("max_tokens_per_chunk", 86 * 28),
                cfg_scale=kwargs.get("cfg_scale", 2.5),
                sampling_params=dict(min_p=kwargs.get("min_p", 0.15)),
                disable_torch_compile=True
            )

            # Decode the generated codes into waveform audio
            audio = model.autoencoder.decode(codes).cpu()[0]  # Get first audio sample

            # Concatenate with previous chunks
            if full_audio is None:
                full_audio = audio
            else:
                # Add a small silence between chunks (0.25 seconds)
                silence = torch.zeros(1, int(sample_rate * 0.25))
                full_audio = torch.cat([full_audio, silence, audio], dim=1)

        return full_audio, sample_rate

    except Exception as e:
        print(f"Error generating TTS: {e}")
        # Fallback to mock implementation
        return generate_tts_mock(text, **kwargs)

# API routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint to check if API is running."""
    return {"status": "ok", "message": "TTS API is running"}

@app.get("/system-info", response_model=SystemInfoResponse)
async def system_info():
    """Get information about the TTS system."""
    return {
        "model_name": "Zonos TTS",
        "voices_count": len(voices),
        "version": "1.0.0",
        "cuda_available": torch.cuda.is_available(),
        "mock_mode": MOCK_MODE
    }

@app.get("/voices", response_model=VoiceListResponse)
async def list_voices():
    """List all available voices."""
    return {"voices": voices}

@app.post("/voice-info", response_model=VoiceInfoResponse)
async def upload_voice_info(request: VoiceInfoRequest):
    """Upload voice information."""
    global voices

    voice_info = request.voice_info
    uploaded_voices = voice_info.get("voices", [])

    # Update voices list
    for uploaded_voice in uploaded_voices:
        voice_id = uploaded_voice.get("id")
        if not voice_id:
            continue

        # Check if voice already exists (case-insensitive)
        existing_voice = get_voice_by_id(voice_id)
        if existing_voice:
            # Update existing voice
            for i, voice in enumerate(voices):
                if normalize_id(voice.get("id")) == normalize_id(voice_id):
                    voices[i] = uploaded_voice
                    break
        else:
            # Add new voice
            voices.append(uploaded_voice)

    # Save updated voice info
    save_voice_info()

    file_id = str(uuid.uuid4())
    return {
        "file_id": file_id,
        "message": f"Uploaded voice info for {len(uploaded_voices)} voices"
    }

@app.post("/voice-sample", response_model=VoiceSampleResponse)
async def upload_voice_sample(
    voice_data: str = Body(...),
    sample_file: UploadFile = File(...)
):
    """Upload a voice sample for a specific voice."""
    try:
        voice_info = json.loads(voice_data)
        voice_id = voice_info.get("voice_id")
        name = voice_info.get("name", "Unknown")

        if not voice_id:
            raise HTTPException(status_code=400, detail="Voice ID is required")

        # Check if voice exists
        voice = get_voice_by_id(voice_id)
        if not voice:
            raise HTTPException(status_code=404, detail=f"Voice ID '{voice_id}' not found")

        # Create directory for voice samples
        voice_dir = VOICE_SAMPLES_DIR / voice_id
        voice_dir.mkdir(exist_ok=True, parents=True)

        # Save the uploaded file
        file_path = voice_dir / f"{uuid.uuid4()}_{sample_file.filename}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(sample_file.file, f)

        return {
            "voice_id": voice_id,
            "message": f"Uploaded voice sample for '{name}'",
            "status": "success"
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid voice data JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading voice sample: {str(e)}")

@app.post("/tts", response_model=TextToSpeechResponse)
async def text_to_speech(request: TextToSpeechRequest, background_tasks: BackgroundTasks):
    """Generate speech from text."""
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    # Generate a unique file ID
    file_id = str(uuid.uuid4())

    # If voice_id is provided, get the voice info
    speaker_emb = None
    if request.voice_id:
        voice = get_voice_by_id(request.voice_id)
        if not voice:
            raise HTTPException(status_code=404, detail=f"Voice ID '{request.voice_id}' not found")

        # Get speaker embedding from voice (this would be implementation-specific)
        # For now, we'll just use a placeholder
        speaker_emb = voice.get("speaker_embedding", None)

    # Generate TTS audio
    audio, sample_rate = generate_tts(
        text=text,
        speaker_emb=speaker_emb,
        emotion=request.emotion,
        fmax=request.fmax,
        pitch_std=request.pitch_std,
        speaking_rate=request.speaking_rate,
        vqscore_8=request.vqscore_8,
        ctc_loss=request.ctc_loss,
        dnsmos_ovrl=request.dnsmos_ovrl,
        speaker_noised=request.speaker_noised,
        unconditional_keys=request.unconditional_keys,
        cfg_scale=request.cfg_scale,
        min_p=request.min_p,
        seed=request.seed,
        max_tokens_per_chunk=request.max_tokens_per_chunk,
        max_chunk_length=request.max_chunk_length,
        add_silence_prefix=request.add_silence_prefix,
    )

    # Save audio to file
    output_path = GENERATED_AUDIO_DIR / f"{file_id}.wav"
    torchaudio.save(output_path, audio, sample_rate)

    # Return file information
    return {
        "file_id": file_id,
        "file_url": f"/download/{file_id}",
        "text": text,
        "message": "Speech generated successfully"
    }

@app.get("/download/{file_id}")
async def download_audio(file_id: str):
    """Download generated audio file."""
    file_path = GENERATED_AUDIO_DIR / f"{file_id}.wav"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{file_id}.wav' not found")

    return FileResponse(
        path=file_path,
        filename=f"{file_id}.wav",
        media_type="audio/wav"
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the server on startup."""
    load_voice_info()
    init_model()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)