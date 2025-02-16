import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

import random

#model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

# wav, sampling_rate = torchaudio.load("assets/exampleaudio.mp3")
wav, sampling_rate = torchaudio.load("assets/michael-01.wav")
speaker = model.make_speaker_embedding(wav, sampling_rate)

# torch.manual_seed(421)

# Define the conditioning parameters
text = "Tropical Storm Gabrielle was a short-lived tropical cyclone that passed over North Carolina before tracking out to sea. The seventh named storm of the 2007 Atlantic hurricane season, Gabrielle developed as a subtropical cyclone on September 8 about 385 miles (620 km) southeast of Cape Lookout, North Carolina."
language = "en-us"
emotion = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077]  # Example emotion vector
fmax = 24000.0
pitch_std = 45.0
speaking_rate = 15.0
vqscore_8 = [0.78] * 8
ctc_loss = 0.0
dnsmos_ovrl = 4.0
speaker_noised = False
# unconditional_keys = {"vqscore_8", "dnsmos_ovrl"}  # Example unconditional keys
unconditional_keys = {"emotion"}
# unconditional_keys = {}

# Create the conditioning dictionary
cond_dict = make_cond_dict(
    text=text,
    language=language,
    speaker=speaker,
    emotion=emotion,
    fmax=fmax,
    pitch_std=pitch_std,
    speaking_rate=speaking_rate,
    vqscore_8=vqscore_8,
    ctc_loss=ctc_loss,
    dnsmos_ovrl=dnsmos_ovrl,
    speaker_noised=speaker_noised,
    unconditional_keys=unconditional_keys,
    device=device
)

conditioning = model.prepare_conditioning(cond_dict)

# Generation parameters
cfg_scale = 2.0
min_p = 0.15
seed = 420
randomize_seed = True
progress = None

if randomize_seed:
    seed = random.randint(0, 2**32 - 1)
    print(f"Randomized seed: {seed}")
torch.manual_seed(seed)

# Load optional prefix audio
prefix_audio_path = "assets/silence_100ms.wav"
audio_prefix_codes = None
if prefix_audio_path:
    wav_prefix, sr_prefix = torchaudio.load(prefix_audio_path)
    wav_prefix = wav_prefix.mean(0, keepdim=True)
    wav_prefix = model.autoencoder.preprocess(wav_prefix, sr_prefix)
    wav_prefix = wav_prefix.to(device, dtype=torch.float32)
    audio_prefix_codes = model.autoencoder.encode(wav_prefix.unsqueeze(0))

codes = model.generate(
    prefix_conditioning=conditioning,
    # audio_prefix_codes=audio_prefix_codes,
    max_new_tokens=86 * 30,
    cfg_scale=cfg_scale,
    sampling_params=dict(min_p=min_p),
    #progress=progress,
     disable_torch_compile=True
)

wavs = model.autoencoder.decode(codes).cpu()
torchaudio.save("sample.wav", wavs[0], model.autoencoder.sampling_rate)