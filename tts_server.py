import os
import uuid
import json
import torch
import torchaudio
import nltk
import re
import hashlib
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Body, Query, Form
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

try:
    from zonos.utils import DEFAULT_DEVICE as device
except ImportError:
    # If that's not available:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# At the top of the file, with other imports
import nltk.data

# Try to be very explicit about the NLTK data path
nltk.data.path.append(os.path.expanduser('~/nltk_data'))
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

# Download resources
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
except Exception as e:
    print(f"Warning: Failed to download NLTK data: {e}")

# Initialize FastAPI
app = FastAPI(title="Zonos TTS API",
              description="API for text-to-speech synthesis using the Zonos model",
              version="1.0.0")

# Create data directory for storing voice info, samples, and generated audio
DATA_DIR = Path("data")
VOICE_INFO_DIR = DATA_DIR / "voice_info"
VOICE_SAMPLES_DIR = DATA_DIR / "voice_samples"
GENERATED_AUDIO_DIR = DATA_DIR / "generated_audio"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
ASSETS_DIR = Path("assets")

# Create necessary directories
for directory in [DATA_DIR, VOICE_INFO_DIR, VOICE_SAMPLES_DIR, GENERATED_AUDIO_DIR, EMBEDDINGS_DIR, ASSETS_DIR]:
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
    emotion_vector: Optional[List[float]] = Field(None, description="Emotion vector with 8 values")
    speaking_rate: Optional[float] = Field(None, description="Speaking rate")
    pitch: Optional[float] = Field(None, description="Pitch adjustment")
    pitch_std: Optional[float] = Field(None, description="Pitch standard deviation")
    fmax: Optional[float] = Field(None, description="Maximum frequency")
    vqscore_8: Optional[List[float]] = Field(None, description="VQ score with 8 values")
    ctc_loss: Optional[float] = Field(None, description="CTC loss value")
    dnsmos_ovrl: Optional[float] = Field(None, description="DNSMOS overall score")
    speaker_noised: Optional[bool] = Field(None, description="Whether to add noise to speaker embedding")
    cfg_scale: Optional[float] = Field(None, description="CFG scale for generation")
    min_p: Optional[float] = Field(None, description="Min P value for sampling")
    seed: Optional[int] = Field(None, description="Random seed for generation")
    add_silence_prefix: Optional[bool] = Field(None, description="Whether to add a silence prefix")

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
system_settings = {}
audio_prefix_codes = None

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
    global voices, system_settings
    if voice_info_file.exists():
        try:
            with open(voice_info_file, "r", encoding="utf-8") as f:
                voice_info = json.load(f)
                voices = voice_info.get("voices", [])
                system_settings = voice_info.get("tts_system_settings", {})
            print(f"Loaded {len(voices)} voices from {voice_info_file}")
        except Exception as e:
            print(f"Error loading voice info: {e}")
            voices = []
            system_settings = {}
    else:
        print("No voice info file found.")
        voices = []
        system_settings = {}

# Save voice info
def save_voice_info():
    """Save voice information to file."""
    voice_info = {
        "version": "1.0",
        "voices": voices,
        "tts_system_settings": system_settings
    }
    with open(voice_info_file, "w", encoding="utf-8") as f:
        json.dump(voice_info, f, indent=2)
    print(f"Saved {len(voices)} voices to {voice_info_file}")

# Initialize model
def init_model():
    """Initialize the TTS model."""
    global model, audio_prefix_codes
    if MOCK_MODE:
        print("Running in mock mode (no Zonos model available)")
        return

    try:
        # Model initialization code
        print("Initializing model...")
        zonos_defaults = system_settings.get("zonos_defaults", {})
        model_path = zonos_defaults.get("model_path", "Zyphra/Zonos-v0.1-hybrid")
        model = Zonos.from_pretrained(model_path)
        print(f"Model initialized from {model_path}.")

        # Load audio prefix if enabled
        audio_prefix = system_settings.get("audio_prefix", {})
        if audio_prefix.get("enabled", False):  # Changed to false by default
            prefix_file = audio_prefix.get("default_file", "assets/silence_100ms.wav")
            if os.path.exists(prefix_file):
                try:
                    wav_prefix, sr_prefix = torchaudio.load(prefix_file)
                    wav_prefix = wav_prefix.mean(0, keepdim=True)
                    wav_prefix = model.autoencoder.preprocess(wav_prefix, sr_prefix)
                    wav_prefix = wav_prefix.to(device, dtype=torch.float32)
                    audio_prefix_codes = model.autoencoder.encode(wav_prefix.unsqueeze(0))
                    print(f"Loaded audio prefix from {prefix_file}")
                except Exception as e:
                    print(f"Error loading audio prefix: {e}")
                    audio_prefix_codes = None
            else:
                print(f"Audio prefix file not found: {prefix_file}")
                audio_prefix_codes = None
        else:
            audio_prefix_codes = None
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Continuing without a model (mock mode).")

# Process text for TTS
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
        # Fallback tokenization code
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
def generate_tts(text, voice_id=None, **kwargs):
    """Generate TTS audio from text."""
    global model, audio_prefix_codes, system_settings

    # Initialize parameters with defaults from system settings
    zonos_defaults = system_settings.get("zonos_defaults", {})
    params = {
        "language": zonos_defaults.get("language", "en-us"),
        "fmax": zonos_defaults.get("fmax", 24000.0),
        "pitch_std": zonos_defaults.get("pitch_std", 45.0),
        "speaking_rate": zonos_defaults.get("speaking_rate", 15.0),
        "vqscore_8": zonos_defaults.get("vqscore_8", [0.78] * 8),
        "ctc_loss": zonos_defaults.get("ctc_loss", 0.1),
        "dnsmos_ovrl": zonos_defaults.get("dnsmos_ovrl", 4.0),
        "speaker_noised": zonos_defaults.get("speaker_noised", False),
        "unconditional_keys": zonos_defaults.get("unconditional_keys", ["emotion"]),
        "cfg_scale": zonos_defaults.get("cfg_scale", 2.5),
        "min_p": zonos_defaults.get("min_p", 0.15),
        "max_tokens_per_chunk": zonos_defaults.get("max_tokens_per_chunk", 2408),
        "add_silence_prefix": zonos_defaults.get("add_silence_prefix", True),
        "emotion_vector": [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077]
    }

    # Override with voice-specific parameters if voice_id is provided
    if voice_id:
        voice = get_voice_by_id(voice_id)
        if voice:
            zonos_params = voice.get("zonos_parameters", {})
            for key, value in zonos_params.items():
                if key == "emotion_vector":
                    params["emotion"] = value
                else:
                    params[key] = value

    # Override with request parameters
    for key, value in kwargs.items():
        if value is not None:
            if key == "emotion_vector":
                params["emotion"] = value
            else:
                params[key] = value

    # Get speaker embedding for voice cloning
    speaker_emb = None
    if voice_id:
        voice = get_voice_by_id(voice_id)
        if voice:
            cloning = voice.get("cloning", {})
            if cloning.get("enabled", False):
                embedding_path = cloning.get("embedding_path")
                if embedding_path and os.path.exists(embedding_path):
                    try:
                        speaker_emb = torch.load(embedding_path, map_location=device)
                        print(f"Loaded speaker embedding from {embedding_path}")
                    except Exception as e:
                        print(f"Error loading speaker embedding: {e}")

    if model is None:
        # Use mock implementation
        return generate_tts_mock(text, **params)

    try:
        # Process text into chunks
        chunk_settings = system_settings.get("chunk_processing", {})
        max_chunk_length = chunk_settings.get("max_chunk_length", 250)
        chunks = process_text(text, params.get("language", "en-us"), max_chunk_length)

        full_audio = None
        # Use the model's native sampling rate instead of hardcoding
        sample_rate = model.autoencoder.sampling_rate

        # Process each chunk
        for chunk in chunks:
            # Create conditioning dictionary for Zonos model
            cond_dict = make_cond_dict(
                text=chunk,
                language=params.get("language", "en-us"),
                speaker=speaker_emb,
                emotion=params.get("emotion", params.get("emotion_vector")),
                fmax=params.get("fmax", 24000.0),
                pitch_std=params.get("pitch_std", 45.0),
                speaking_rate=params.get("speaking_rate", 15.0),
                vqscore_8=params.get("vqscore_8", [0.78] * 8),
                ctc_loss=params.get("ctc_loss", 0.1),
                dnsmos_ovrl=params.get("dnsmos_ovrl", 4.0),
                speaker_noised=params.get("speaker_noised", False),
                unconditional_keys=params.get("unconditional_keys", ["emotion"]),
                device=device
            )

            # Prepare conditioning
            conditioning = model.prepare_conditioning(cond_dict)

            # Generate audio with correct parameters
            codes = model.generate(
                prefix_conditioning=conditioning,
                audio_prefix_codes=audio_prefix_codes if params.get("add_silence_prefix", True) else None,
                max_new_tokens=params.get("max_tokens_per_chunk", 2408),
                cfg_scale=params.get("cfg_scale", 2.5),
                sampling_params=dict(min_p=params.get("min_p", 0.15)),
                disable_torch_compile=True
            )

            # Decode the generated codes into waveform audio
            audio = model.autoencoder.decode(codes).cpu()[0]  # Get first audio sample

            # Concatenate with previous chunks
            if full_audio is None:
                full_audio = audio
            else:
                # Add a small silence between chunks
                silence_ms = chunk_settings.get("silence_between_chunks_ms", 100)  # Reduced from 250
                silence = torch.zeros(1, int(sample_rate * silence_ms / 1000))
                full_audio = torch.cat([full_audio, silence, audio], dim=1)

        return full_audio, sample_rate

    except Exception as e:
        print(f"Error generating TTS: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to mock implementation
        return generate_tts_mock(text, **params)

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
    global voices, system_settings

    voice_info = request.voice_info
    uploaded_voices = voice_info.get("voices", [])

    # Update system settings if provided
    if "tts_system_settings" in voice_info:
        system_settings = voice_info.get("tts_system_settings", {})

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
    voice_data: str = Form(...),
    sample_file: UploadFile = File(...)
):
    """Upload a voice sample for a specific voice."""
    global model
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

        # Generate a unique filename
        sample_filename = f"{uuid.uuid4()}_{sample_file.filename}"
        file_path = voice_dir / sample_filename

        # Save the uploaded file
        with open(file_path, "wb") as f:
            shutil.copyfileobj(sample_file.file, f)

        # Update voice cloning info
        cloning = voice.get("cloning", {})
        if not cloning:
            cloning = {"enabled": True}
            voice["cloning"] = cloning

        cloning["enabled"] = True
        cloning["audio_file"] = str(file_path)

        # Generate speaker embedding if model is available
        if model and not MOCK_MODE:
            try:
                # Create embeddings directory if needed
                os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

                # Generate embedding filename
                embedding_filename = f"{voice_id}_embedding.pt"
                embedding_path = EMBEDDINGS_DIR / embedding_filename

                # Load audio and generate embedding
                wav, sample_rate = torchaudio.load(file_path)
                speaker_emb = model.make_speaker_embedding(wav, sample_rate)

                # Save embedding
                torch.save(speaker_emb, embedding_path)

                # Update voice with embedding path
                cloning["embedding_path"] = str(embedding_path)

                # Save voice info
                save_voice_info()

                return {
                    "voice_id": voice_id,
                    "message": f"Uploaded voice sample for '{name}' and generated speaker embedding",
                    "status": "success"
                }

            except Exception as e:
                print(f"Error generating speaker embedding: {e}")
                import traceback
                traceback.print_exc()

        # If we get here, either model is unavailable or embedding generation failed
        cloning["audio_file"] = str(file_path)
        save_voice_info()

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

    # Extract parameters
    params = {
        "language": request.language,
        "emotion_vector": request.emotion_vector,
        "speaking_rate": request.speaking_rate,
        "pitch": request.pitch,
        "pitch_std": request.pitch_std,
        "fmax": request.fmax,
        "vqscore_8": request.vqscore_8,
        "ctc_loss": request.ctc_loss,
        "dnsmos_ovrl": request.dnsmos_ovrl,
        "speaker_noised": request.speaker_noised,
        "cfg_scale": request.cfg_scale,
        "min_p": request.min_p,
        "seed": request.seed,
        "add_silence_prefix": request.add_silence_prefix
    }

    # Filter out None values
    params = {k: v for k, v in params.items() if v is not None}

    # Generate TTS audio
    audio, sample_rate = generate_tts(
        text=text,
        voice_id=request.voice_id,
        **params
    )

    # Save audio to file - use the correct sample rate
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