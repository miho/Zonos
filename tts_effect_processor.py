import argparse
import json
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize
from pedalboard import (
    Pedalboard, Reverb, Compressor, Delay, PitchShift, Limiter,
    HighpassFilter, LowpassFilter, PeakFilter, HighShelfFilter, LowShelfFilter
)
import noisereduce as nr
import warnings

# --- Graceful import for newer pedalboard effects ---
try:
    from pedalboard import Flanger
    FLANGER_AVAILABLE = True
except ImportError:
    FLANGER_AVAILABLE = False

try:
    from pedalboard import Deesser
    DEESSER_AVAILABLE = True
except ImportError:
    DEESSER_AVAILABLE = False

# Suppress warnings from pydub about FFmpeg/avconv
warnings.filterwarnings("ignore", category=UserWarning, module='pydub.utils')

def apply_pedalboard_effect(audio, board):
    """Helper to apply a pedalboard effect."""
    # Convert pydub segment to numpy array for pedalboard
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    
    # Pedalboard expects mono or stereo, handle different channel counts
    if audio.channels > 2:
        samples = samples.reshape((-1, audio.channels))[:, :2]
    elif audio.channels == 1:
        samples = samples.reshape((-1, 1))

    effected_samples = board(samples, audio.frame_rate)
    
    # Convert back to pydub AudioSegment
    return AudioSegment(
        effected_samples.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )

# --- Effect Implementations ---

def effect_fade_in(audio, params):
    """Applies a fade-in effect."""
    duration_ms = params.get('duration_ms', 1000)
    print(f"  Applying: Fade-In ({duration_ms} ms)")
    return audio.fade_in(duration_ms)

def effect_fade_out(audio, params):
    """Applies a fade-out effect."""
    duration_ms = params.get('duration_ms', 1000)
    print(f"  Applying: Fade-Out ({duration_ms} ms)")
    return audio.fade_out(duration_ms)

def effect_silence(audio, params):
    """Adds silence to the beginning or end of the audio."""
    duration_ms = params.get('duration_ms', 1000)
    position = params.get('position', 'end').lower()
    print(f"  Applying: Adding {duration_ms}ms of silence to the {position}")
    
    pause = AudioSegment.silent(duration=duration_ms)
    
    if position == 'start':
        return pause + audio
    else: # Default to end
        return audio + pause

def effect_telephone(audio, params):
    """Simulates a telephone effect using band-pass filtering."""
    print("  Applying: Telephone Effect")
    low_passed = audio.low_pass_filter(params.get('high_cutoff_hz', 3400))
    high_passed = low_passed.high_pass_filter(params.get('low_cutoff_hz', 300))
    return high_passed.apply_gain(params.get('gain_db', 3))

def effect_announcement(audio, params):
    """Simulates a loudspeaker announcement."""
    print("  Applying: Announcement Effect")
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=params.get('low_cutoff_hz', 400)),
        LowpassFilter(cutoff_frequency_hz=params.get('high_cutoff_hz', 3000)),
        Reverb(
            room_size=params.get('room_size', 0.6),
            damping=params.get('damping', 0.7),
            wet_level=params.get('wet_level', 0.33),
            dry_level=params.get('dry_level', 0.4)
        ),
        Compressor(threshold_db=-10, ratio=4)
    ])
    return apply_pedalboard_effect(audio, board)

def effect_reverb(audio, params):
    """Applies a simple reverb effect."""
    print("  Applying: Reverb")
    board = Pedalboard([Reverb(**params)])
    return apply_pedalboard_effect(audio, board)

def effect_noise_reduction(audio, params):
    """Applies noise reduction."""
    print("  Applying: Noise Reduction")
    samples = np.array(audio.get_array_of_samples())
    reduced_noise_samples = nr.reduce_noise(
        y=samples,
        sr=audio.frame_rate,
        prop_decrease=params.get('strength', 0.95)
    )
    return AudioSegment(
        reduced_noise_samples.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    
def effect_deesser(audio, params):
    """Applies a de-esser to reduce sibilance."""
    print("  Applying: De-Esser")
    if not DEESSER_AVAILABLE:
        print("  Warning: Deesser effect not available. Please upgrade pedalboard: pip install --upgrade pedalboard. Skipping effect.")
        return audio
    board = Pedalboard([Deesser(**params)])
    return apply_pedalboard_effect(audio, board)

def effect_high_pass_filter(audio, params):
    """Applies a high-pass filter."""
    print(f"  Applying: High-Pass Filter at {params.get('cutoff_hz')} Hz")
    return audio.high_pass_filter(params.get('cutoff_hz', 80))

def effect_equalizer(audio, params):
    """Applies multi-band equalization."""
    print("  Applying: Equalizer")
    bands = []
    for band in params.get('bands', []):
        band_type = band.get('type', 'peak').lower()
        freq = band.get('frequency')
        gain = band.get('gain_db')
        q = band.get('q', 1.0)
        
        if not all([freq, gain is not None]):
            continue
            
        if band_type == 'peak':
            bands.append(PeakFilter(cutoff_frequency_hz=freq, gain_db=gain, q=q))
        elif band_type == 'low_shelf':
            bands.append(LowShelfFilter(cutoff_frequency_hz=freq, gain_db=gain, q=q))
        elif band_type == 'high_shelf':
            bands.append(HighShelfFilter(cutoff_frequency_hz=freq, gain_db=gain, q=q))
            
    if not bands:
        return audio
        
    board = Pedalboard(bands)
    return apply_pedalboard_effect(audio, board)

def effect_compressor(audio, params):
    """Applies a compressor."""
    print("  Applying: Compressor")
    board = Pedalboard([Compressor(**params)])
    return apply_pedalboard_effect(audio, board)

def effect_pitch_shift(audio, params):
    """Applies pitch shifting."""
    print(f"  Applying: Pitch Shift ({params.get('semitones')} semitones)")
    board = Pedalboard([PitchShift(**params)])
    return apply_pedalboard_effect(audio, board)

def effect_delay(audio, params):
    """Applies a delay/echo effect."""
    print("  Applying: Delay")
    board = Pedalboard([Delay(**params)])
    return apply_pedalboard_effect(audio, board)

def effect_panning(audio, params):
    """Applies stereo panning."""
    print(f"  Applying: Panning ({params.get('pan')})")
    return audio.pan(params.get('pan', 0.0))

def effect_loudness_normalization(audio, params):
    """Applies RMS-based loudness normalization."""
    print(f"  Applying: Loudness Normalization to {params.get('headroom_db')} dBFS")
    return normalize(audio, headroom=params.get('headroom_db', 0.1))

def effect_limiter(audio, params):
    """Applies a limiter."""
    print(f"  Applying: Limiter (Ceiling: {params.get('threshold_db')} dB)")
    board = Pedalboard([Limiter(**params)])
    return apply_pedalboard_effect(audio, board)

def effect_walkie_talkie(audio, params):
    """Simulates a walkie-talkie effect."""
    print("  Applying: Walkie-Talkie Effect")
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=params.get('high_pass_hz', 400)),
        LowpassFilter(cutoff_frequency_hz=params.get('low_pass_hz', 2800)),
        Compressor(threshold_db=-25, ratio=8),
    ])
    distorted = apply_pedalboard_effect(audio, board)
    return distorted.apply_gain(params.get('gain_db', 0.1))

def effect_robot(audio, params):
    """Simulates a robot voice."""
    print("  Applying: Robot Effect")
    if not FLANGER_AVAILABLE:
        print("  Warning: Flanger effect not available. Please upgrade pedalboard: pip install --upgrade pedalboard. Skipping effect.")
        return audio
        
    board = Pedalboard([
        Flanger(rate_hz=params.get('flanger_rate_hz', 0.8), depth=params.get('flanger_depth', 0.9)),
        PitchShift(semitones=params.get('pitch_semitones', -2))
    ])
    return apply_pedalboard_effect(audio, board)

def effect_ghost(audio, params):
    """Simulates a ghostly, ethereal voice."""
    print("  Applying: Ghost Effect")
    board = Pedalboard([
        Reverb(
            room_size=params.get('reverb_room_size', 0.9),
            wet_level=params.get('reverb_wet_level', 0.7),
            dry_level=params.get('reverb_dry_level', 0.1),
            damping=params.get('reverb_damping', 0.8)
        ),
        Delay(
            delay_seconds=params.get('delay_ms', 500) / 1000.0,
            feedback=params.get('delay_feedback', 0.35),
            mix=params.get('delay_mix', 0.25)
        )
    ])
    return apply_pedalboard_effect(audio, board)

def effect_muffled(audio, params):
    """Simulates a voice heard through a wall."""
    print("  Applying: Muffled Effect")
    filtered = audio.low_pass_filter(params.get('low_pass_hz', 700))
    return filtered.apply_gain(params.get('volume_reduction_db', -8))


# Effect dispatcher dictionary
EFFECT_DISPATCHER = {
    'fade_in': effect_fade_in,
    'fade_out': effect_fade_out,
    'silence': effect_silence,
    'telephone': effect_telephone,
    'announcement': effect_announcement,
    'reverb': effect_reverb,
    'noise_reduction': effect_noise_reduction,
    'de-esser': effect_deesser,
    'high_pass_filter': effect_high_pass_filter,
    'equalizer': effect_equalizer,
    'compressor': effect_compressor,
    'pitch_shift': effect_pitch_shift,
    'delay': effect_delay,
    'panning': effect_panning,
    'loudness_normalization': effect_loudness_normalization,
    'limiter': effect_limiter,
    'walkie_talkie': effect_walkie_talkie,
    'robot': effect_robot,
    'ghost': effect_ghost,
    'muffled': effect_muffled,
}


def process_jobs(jobs, presets):
    """
    Processes a list of audio jobs using a dictionary of presets.
    """
    for i, item in enumerate(jobs):
        source_path = item.get('source')
        target_path = item.get('target')
        
        if not source_path or not target_path:
            print(f"Skipping job {i+1} due to missing 'source' or 'target' path.")
            continue

        print(f"\n--- Starting Job {i+1}: {source_path} -> {target_path} ---")

        try:
            audio = AudioSegment.from_file(source_path)
            print(f"Successfully loaded '{source_path}'")
        except FileNotFoundError:
            print(f"Error: Source file not found at {source_path}. Skipping job.")
            continue
        except Exception as e:
            print(f"Error loading {source_path}: {e}. Skipping job.")
            continue

        effects_to_apply = []
        preset_name = item.get('apply_preset')

        if preset_name:
            if preset_name in presets:
                print(f"Applying preset: '{preset_name}'")
                effects_to_apply = presets[preset_name].get('effects', [])
            else:
                print(f"Warning: Preset '{preset_name}' not found. Skipping job.")
                continue
        else:
            # Fallback to inline effects if no preset is specified
            effects_to_apply = item.get('effects', [])
            if effects_to_apply:
                 print("Applying inline effects.")

        if not effects_to_apply:
            print("Warning: No effects or valid preset found for this job. Skipping.")
            continue
            
        # Apply the chain of effects
        processed_audio = audio
        for effect in effects_to_apply:
            effect_name = effect.get('name')
            parameters = effect.get('parameters', {})
            
            effect_function = EFFECT_DISPATCHER.get(effect_name)
            
            if effect_function:
                try:
                    processed_audio = effect_function(processed_audio, parameters)
                except Exception as e:
                    print(f"  Error applying effect '{effect_name}': {e}")
            else:
                print(f"  Warning: Effect '{effect_name}' not recognized. Skipping.")

        # Export the final processed audio
        try:
            file_format = target_path.split('.')[-1]
            processed_audio.export(target_path, format=file_format)
            print(f"Successfully processed and saved to '{target_path}'")
        except Exception as e:
            print(f"Error exporting file to {target_path}: {e}")
            
    print("\n--- All jobs completed. ---")


def main():
    """
    Main function to parse arguments and run the audio processing.
    """
    parser = argparse.ArgumentParser(
        description="A command-line tool to apply a chain of audio effects to files based on a JSON configuration.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "jobs_file",
        help="Path to the JSON file containing the audio processing jobs."
    )
    parser.add_argument(
        "--presets",
        dest="presets_file",
        help="Optional path to a separate JSON file containing effect presets."
    )
    
    args = parser.parse_args()
    
    # Load jobs file
    try:
        with open(args.jobs_file, 'r') as f:
            jobs_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Jobs file not found at {args.jobs_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.jobs_file}. Please check its format.")
        return

    jobs = jobs_data.get('audio_processing_jobs', [])
    presets = {}

    # Load presets
    if args.presets_file:
        print(f"Loading presets from separate file: {args.presets_file}")
        try:
            with open(args.presets_file, 'r') as f:
                presets_data = json.load(f)
                presets = presets_data.get('effect_presets', {})
        except FileNotFoundError:
            print(f"Warning: Presets file not found at {args.presets_file}. Continuing without presets.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {args.presets_file}. Continuing without presets.")
    else:
        # Look for presets embedded in the main jobs file
        if 'effect_presets' in jobs_data:
            print("Loading presets embedded in jobs file.")
            presets = jobs_data.get('effect_presets', {})

    if not presets:
        print("No presets loaded. Only inline effects will be processed.")

    process_jobs(jobs, presets)


if __name__ == '__main__':
    main()
