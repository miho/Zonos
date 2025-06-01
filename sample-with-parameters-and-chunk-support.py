# sample-with-parameters-and-chunk-support.py
import re
import nltk
import torch
import torchaudio
import random
import gc
import os
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

# check if using GPU
if str(device).startswith("cuda"):
    print(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    print("Using CPU")

# Download necessary NLTK tokenizers and resources
nltk.download('punkt')
nltk.download('punkt_tab')

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

# Input text to be processed
text = (
    "Tropical Storm Gabrielle was a short-lived tropical cyclone that passed over North Carolina "
    "before tracking out to sea. The seventh named storm of the 2007 Atlantic hurricane season, Gabrielle "
    "developed as a subtropical cyclone on September 8 about 385 miles (620 km) southeast of Cape Lookout, North Carolina. "
    "Unfavorable wind shear affected the storm for much of its duration, although a temporary decrease in the shear allowed "
    "the cyclone to become a tropical storm. On September 9, Gabrielle made landfall at Cape Lookout National Seashore in the "
    "Outer Banks of North Carolina with winds of 60 mph (97 km/h). Turning to the northeast, the storm quickly weakened "
    "and dissipated on September 11. In advance of the storm, tropical cyclone watches and warnings were issued for coastal areas, "
    "while rescue teams and the U.S. Coast Guard were put on standby. The storm dropped heavy rainfall near its immediate landfall location "
    "but little precipitation elsewhere. Along the coast of North Carolina, high waves, rip currents, and storm surge were reported. "
    "Slight localized flooding was reported. Gusty winds also occurred, though no wind damage was reported. One person drowned in rough "
    "surf caused by the storm in Florida. Overall damage was minor. A cold front moved off the southeast coast of the United States on September 1."
    "[1] Gradually decaying, the front degenerated into an area of cloudiness and showers just east of the Georgia coast on September 2.[2] "
    "Tracking eastward, a weak low-pressure area developed the next day.[1] It slowly became better organized as its motion became erratic,[3] "
    "and by late on September 4 the convection had become concentrated to the east of the center.[4] On September 5, a Hurricane Hunters flight "
    "indicated the system had not acquired the characteristics of a tropical or subtropical cyclone. Interaction with an upper-level trough resulted "
    "in moderate wind shear which suppressed further development,[5] and by September 6 the thunderstorm activity lost much organization.[6] "
    "Then, upper-level winds became increasingly favorable, allowing the convection to concentrate about halfway between North Carolina and Bermuda.[7] "
    "With a deep-layer ridge to its north, the system turned to a steady west-northwest track. A reconnaissance aircraft flight late on September 7 reported"
    " a very elongated center, with peak flight winds of 55 mph (89 km/h) about 100 miles (160 km) northeast of the center. Subsequent to the flight, the center"
    " became slightly better organized, and based on the large wind field and the presence of an upper-level low to its west-southwest, "
    "the National Hurricane Center classified the system as Subtropical Storm Gabrielle early on September 8 while located about 385 miles (620 km) southeast of Cape Lookout, North Carolina.[8] "
    "Upon becoming a subtropical cyclone, Gabrielle was located in an area of cooler air to its north, dry air to its south and west, southerly wind shear, and cooler water temperatures along its path.[8] "
    "Despite these unfavorable conditions, a curved convective band developed in its northern and western quadrants,[9] and the circulation became better defined.[10] Subsequently, the rainbands in its "
    "northeastern quadrant dissipated, leaving the well-defined center far removed from the convection. By later that day, the circulation began to become more involved with the remaining convection. "
    "Based on evidence of a weak warm-core, the system was re-designated as Tropical Storm Gabrielle late on September 8 about 185 miles (298 km) southeast of Cape Lookout, North Carolina.[11] Vertical "
    "wind shear decreased as the storm passed over the Gulf Stream, allowing a strong convective burst to develop near the center.[12] As it approached the coast of North Carolina, the center re-developed "
    "within the deep convection underneath the mid-level circulation,[13] although increased northerly wind shear displaced the center of Gabrielle to the north of the thunderstorm activity.[14] Based on "
    "reports from Hurricane Hunters, it is estimated Gabrielle moved ashore at Cape Lookout National Seashore at 1530 UTC on September 9 with winds of 60 mph (97 km/h), though due to the shear the strongest "
    "winds remained offshore.[1] Tracking around the ridge over the western Atlantic, the storm turned to the north and north-northeast,[15] emerging into the ocean near Kill Devil Hills, North Carolina early "
    "on September 10 as a poorly organized system with convection far to the south of the center.[16] Gabrielle weakened to a tropical depression shortly thereafter,[17] and maintained scattered convection despite "
    "unfavorable wind shear as it tracked along the northern portion of the Gulf Stream.[18]"
     " By midday on September 11, the circulation had become ill-defined and elongated; failing to meet the criteria of a tropical cyclone, the National Hurricane Center declared Gabrielle dissipating well to the south of Nova Scotia.[19]"
    "By early the next day, the remnants of Gabrielle were absorbed by an approaching cold front.[20]"
)

# replace [1], [2], etc. with empty string
text = re.sub(r'\[\d+\]', '', text)

# Create sentence chunks ensuring they don't exceed the maximum character length (N)
sentences = split_text_into_sentences(text)
N = 250  # Maximum character length per chunk
chunks = []
chunk = ''
for sentence in sentences:
    if len(chunk) + len(sentence) <= N:
        chunk = (chunk + " " + sentence).strip()
    else:
        if len(sentence) > N:
            # Split long sentence into smaller chunks first
            long_chunks = split_long_sentence(sentence, N)
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

print("Generated text chunks:")
for i, c in enumerate(chunks, 1):
    print(f"Chunk {i}: {c}")

# --------------------------------------------------------------------
# TTS Setup
# --------------------------------------------------------------------
# Load the TTS model
# model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)


# Load a reference audio to obtain the speaker embedding
# wav, sampling_rate = torchaudio.load("assets/michael-01.wav")
# wav, sampling_rate = torchaudio.load("assets/hello-and-welcome-to-zonos-t-american-female-2025-02-16.wav")
wav, sampling_rate = torchaudio.load("assets/female-professional-sample.wav")
speaker = model.make_speaker_embedding(wav, sampling_rate)

# Set the TTS parameters
language = "en-us"
emotion = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077]  # Example emotion vector
fmax = 24000.0
pitch_std = 45.0
speaking_rate = 15.0
vqscore_8 = [0.78] * 8
ctc_loss = 0.1
dnsmos_ovrl = 4.0
speaker_noised = False
unconditional_keys = {"emotion"}
# unconditional_keys = {}
cfg_scale = 2.5
min_p = 0.15

# Load optional prefix audio (if required) to prepend a brief silence or cue
prefix_audio_path = "assets/silence_100ms.wav"
audio_prefix_codes = None
if prefix_audio_path:
    wav_prefix, sr_prefix = torchaudio.load(prefix_audio_path)
    wav_prefix = wav_prefix.mean(0, keepdim=True)
    wav_prefix = model.autoencoder.preprocess(wav_prefix, sr_prefix)
    wav_prefix = wav_prefix.to(device, dtype=torch.float32)
    audio_prefix_codes = model.autoencoder.encode(wav_prefix.unsqueeze(0))

# Set a randomized seed for varied generation per chunk
seed = random.randint(0, 2**32 - 1)
# 3782800710 works well for the sample text
print(f"Randomized seed for chunks {seed}")
torch.manual_seed(seed)    


# ensure that the directory exists    
if not os.path.exists("chunk_out"):
        os.makedirs("chunk_out")

# clear content of the directory
for filename in os.listdir("chunk_out"):
    file_path = os.path.join("chunk_out", filename)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)


# Process each text chunk individually with TTS and save the resulting audio
for i, chunk_text in enumerate(chunks, 1):
    print(f"\nProcessing chunk {i} of {len(chunks)}:")

    with torch.no_grad():
        cond_dict = make_cond_dict(
            text=chunk_text,
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

        codes = model.generate(
            prefix_conditioning=conditioning,
            # Uncomment the following line if you wish to include the prefix audio codes:
            audio_prefix_codes=audio_prefix_codes,
            max_new_tokens=86 * 28,
            # max_new_tokens=86 * 30,
            cfg_scale=cfg_scale,
            sampling_params=dict(min_p=min_p),
            disable_torch_compile=True
        )

        # Decode the generated codes into waveform audio and extract the first batch element
        wavs = model.autoencoder.decode(codes).cpu()
        chunk_wav = wavs[0]

    # Save the individual chunk audio file to chunk_out/ directory
    chunk_filename = f"chunk_out/chunk_{i}.wav"
    torchaudio.save(chunk_filename, chunk_wav, model.autoencoder.sampling_rate)
    print(f"Saved chunk audio as {chunk_filename}")

    wav = None
    chunk_wav = None

    # all_chunk_wavs.append(chunk_wav)

    # Free up memory used during TTS generation for this chunk
    del cond_dict, conditioning, codes, wavs, chunk_wav
    gc.collect()

    # If using a GPU, clear the unused cached memory
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

# Merge all individual chunk audio files together into a single output file

# load all the chunk wavs
all_chunk_wavs = []
for i in range(1, len(chunks) + 1):
    chunk_filename = f"chunk_out/chunk_{i}.wav"
    wav, sr = torchaudio.load(chunk_filename)
    all_chunk_wavs.append(wav)

final_audio = torch.cat(all_chunk_wavs, dim=1)
output_filename = "merged_sample.wav"
torchaudio.save(output_filename, final_audio, model.autoencoder.sampling_rate)
print(f"\nFinal merged audio saved as {output_filename}")

gc.collect()

# If using a GPU, clear the unused cached memory
if str(device).startswith("cuda"):
    torch.cuda.empty_cache()