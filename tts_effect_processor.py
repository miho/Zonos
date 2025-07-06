#!/usr/bin/env python3
"""
audio_processor_v2.py
A JSON-driven batch audio FX renderer with preset support.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize
import noisereduce as nr

# -- Pedalboard imports -------------------------------------------------------
from pedalboard import (
    Pedalboard, Reverb, Compressor, Delay, PitchShift, Limiter, Chorus,
    HighpassFilter, LowpassFilter, PeakFilter, HighShelfFilter, LowShelfFilter,
)

# Optional PB FX
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

# Silence the ffmpeg warning Pydub likes to emit
warnings.filterwarnings("ignore", category=UserWarning, module="pydub.utils")

###############################################################################
# Utility – AudioSegment ↔ NumPy                                              #
###############################################################################

INT_RANGES = {1: 128, 2: 32768, 4: 2147483648}  # absolute max for int{8,16,32}

def seg_to_float32(seg: AudioSegment) -> tuple[np.ndarray, int]:
    """
    AudioSegment ➜ float32 np.ndarray in range −1 … 1
    Returns (samples, sample_rate)
    """
    raw = np.frombuffer(seg._data, f"<i{seg.sample_width}")
    floats = raw.astype(np.float32) / INT_RANGES[seg.sample_width]
    if seg.channels > 1:
        floats = floats.reshape(-1, seg.channels)
    return floats, seg.frame_rate

def float32_to_seg(samples: np.ndarray, sr: int, template: AudioSegment) -> AudioSegment:
    """
    float32 array (−1 … 1) ➜ AudioSegment, copying meta from *template*.
    """
    # De-NaN & clip
    samples = np.nan_to_num(samples)
    samples = np.clip(samples, -1.0, 1.0)
    ints = (samples * INT_RANGES[template.sample_width]).astype(f"<i{template.sample_width}")
    if ints.ndim == 2:
        ints = ints.flatten()
    return AudioSegment(
        data=ints.tobytes(),
        frame_rate=sr,
        sample_width=template.sample_width,
        channels=samples.shape[1] if samples.ndim == 2 else 1,
    )

def run_board(seg: AudioSegment, board: Pedalboard) -> AudioSegment:
    x, sr = seg_to_float32(seg)
    y = board(x, sr)
    return float32_to_seg(y, sr, seg)

###############################################################################
# Individual Effects                                                          #
###############################################################################

def fx_fade_in(seg, p):           return seg.fade_in(p.get("duration_ms", 1000))
def fx_fade_out(seg, p):          return seg.fade_out(p.get("duration_ms", 1000))

def fx_silence(seg, p):
    dur = p.get("duration_ms", 1000)
    pos = p.get("position", "end").lower()
    silence = AudioSegment.silent(dur)
    return silence + seg if pos == "start" else seg + silence

def fx_telephone(seg: AudioSegment, p: dict) -> AudioSegment:
    """
    Realistic PSTN / cheap-speaker sound.
    JSON parameters (any optional):
        hp           : 300      # high-pass edge
        lp           : 3400     # low-pass edge
        slope_stages : 4        # number of HP & LP stages (each ≈12 dB/oct)
        comp_thresh  : -20      # compressor threshold dB
        comp_ratio   : 6
        crush_bits   : 8        # 4-16, lower = noisier
        resonance_hz : 1100
        resonance_db : 5
        gain_db      : 0
    """
    cfg = {
        "hp": 300, "lp": 3400,
        "slope_stages": 4,
        "comp_thresh": -20, "comp_ratio": 6,
        "crush_bits": 8,
        "resonance_hz": 1100, "resonance_db": 5,
        "gain_db": 0,
    }
    cfg.update(p or {})

    # ------------------------------------------------------------------ #
    # 1) build a very steep band-pass by cascading HP & LP filters       #
    # ------------------------------------------------------------------ #
    filters = []
    for _ in range(cfg["slope_stages"]):
        filters.append(HighpassFilter(cfg["hp"]))
    for _ in range(cfg["slope_stages"]):
        filters.append(LowpassFilter(cfg["lp"]))

    # add mid-band resonance + compression
    filters += [
        PeakFilter(cutoff_frequency_hz=cfg["resonance_hz"],
                   gain_db=cfg["resonance_db"], q=2.0),
        Compressor(threshold_db=cfg["comp_thresh"],
                   ratio=cfg["comp_ratio"]),
    ]

    seg = run_board(seg, Pedalboard(filters))

    # ------------------------------------------------------------------ #
    # 2) force telephone sample-rate (8 kHz) then resample back          #
    # ------------------------------------------------------------------ #
    orig_sr = seg.frame_rate
    seg = seg.set_frame_rate(8000).set_frame_rate(orig_sr)

    # ------------------------------------------------------------------ #
    # 3) μ-law-style bit-crush                                           #
    # ------------------------------------------------------------------ #
    bits = max(4, min(16, int(cfg["crush_bits"])))
    x, sr = seg_to_float32(seg)
    μ = 255.0
    x_mu = np.sign(x) * (np.log1p(μ * np.abs(x)) / np.log1p(μ))
    step = 2 ** bits
    x_mu = np.round(x_mu * step) / step
    x_out = np.sign(x_mu) * (1/μ) * ((1 + μ) ** np.abs(x_mu) - 1)
    seg = float32_to_seg(x_out, sr, seg)

    # ------------------------------------------------------------------ #
    # 4) optional make-up gain                                           #
    # ------------------------------------------------------------------ #
    if cfg["gain_db"]:
        seg = seg.apply_gain(cfg["gain_db"])

    return seg

###############################################################################
# Helper – correct-format white noise                                         #
###############################################################################
def _make_static(seg: AudioSegment, level_db: float) -> AudioSegment:
    """
    Generate white noise the same length/format as *seg* and trim it to *level_db*.
    Works for any sample-width / channel count.
    """
    n_samples = len(seg.get_array_of_samples())
    dtype = f"<i{seg.sample_width}"
    rng_max = INT_RANGES[seg.sample_width]

    noise_int = np.random.randint(-rng_max, rng_max, n_samples, dtype=dtype)
    if seg.channels > 1:
        noise_int = noise_int.reshape((-1, seg.channels)).flatten()

    noise_seg = AudioSegment(
        noise_int.tobytes(),
        frame_rate=seg.frame_rate,
        sample_width=seg.sample_width,
        channels=seg.channels,
    ).apply_gain(level_db)

    return noise_seg


###############################################################################
# WALKIE-TALKIE v2                                                            #
###############################################################################
def fx_walkie_talkie(seg: AudioSegment, p: dict) -> AudioSegment:
    """
    Much harsher, unmistakable radio sound.
    Parameters you can pass in JSON (all optional, defaults shown):
      hp, lp            : 400 / 2800  – band-pass edges (Hz)
      thresh            : -22         – compressor threshold dB
      ratio             : 10          – compressor ratio
      gain_db           : 2           – make-up gain after compression
      crush_bits        : 8           – bit depth after crushing (4-16)
      am_depth          : 0.25        – 0-1, squared 30 Hz AM for squelch
      static_level_db   : -35         – noise loudness
    """

    cfg = {             # defaults
        "hp": 400, "lp": 2800,
        "thresh": -22, "ratio": 10, "gain_db": 2,
        "crush_bits": 8,
        "am_depth": .25,
        "static_level_db": -35,
    }
    cfg.update(p or {})

    # 1) Band-pass + heavy comp
    board = Pedalboard([
        HighpassFilter(cfg["hp"]),
        LowpassFilter(cfg["lp"]),
        Compressor(threshold_db=cfg["thresh"], ratio=cfg["ratio"]),
    ])
    seg = run_board(seg, board).apply_gain(cfg["gain_db"])

    # 2) Down-sample to 8 kHz and back (telephone bandwidth)
    orig_sr = seg.frame_rate
    seg = seg.set_frame_rate(8000).set_frame_rate(orig_sr)

    # 3) Bit-crush to N bits
    crush_bits = int(cfg["crush_bits"])
    if 4 <= crush_bits <= 16:
        x, sr = seg_to_float32(seg)
        step = 2 ** crush_bits
        x = np.round(x * step) / step              # quantise
        seg = float32_to_seg(x, sr, seg)

    # 4) Optional AM “squelch” modulation
    depth = float(cfg["am_depth"])
    if depth > 0:
        x, sr = seg_to_float32(seg)
        t = np.arange(x.shape[0]) / sr
        am = 1 + depth * np.sign(np.sin(2 * np.pi * 30 * t))   # 30 Hz square
        x *= am[:, None] if x.ndim == 2 else am
        seg = float32_to_seg(x, sr, seg)

    # 5) Add proper static
    if cfg["static_level_db"] < 0:
        static_seg = _make_static(seg, cfg["static_level_db"])
        seg = seg.overlay(static_seg)

    return seg

def fx_reverb(seg, p):            return run_board(seg, Pedalboard([Reverb(**p)]))
def fx_noise_reduction(seg, p):
    x, sr = seg_to_float32(seg)
    y = nr.reduce_noise(y=x, sr=sr, prop_decrease=p.get("strength", .95),
                        n_channels=seg.channels)
    return float32_to_seg(y, sr, seg)

def fx_deesser(seg, p):
    if not DEESSER_AVAILABLE:
        print("Warning: De-esser unavailable – skipping.")
        return seg
    return run_board(seg, Pedalboard([Deesser(**p)]))

def fx_high_pass(seg, p):         return seg.high_pass_filter(p.get("cutoff_hz", 80))
def fx_low_pass(seg, p):          return seg.low_pass_filter(p.get("cutoff_hz", 5000))

def fx_equalizer(seg, p):
    bands = []
    for b in p.get("bands", []):
        typ = b.get("type", "peak")
        b_args = {k: v for k, v in b.items() if k != "type"}
        if typ == "peak":
            bands.append(PeakFilter(**b_args))
        elif typ == "low_shelf":
            bands.append(LowShelfFilter(**b_args))
        elif typ == "high_shelf":
            bands.append(HighShelfFilter(**b_args))
    return seg if not bands else run_board(seg, Pedalboard(bands))

def fx_compressor(seg, p):        return run_board(seg, Pedalboard([Compressor(**p)]))
def fx_pitch_shift(seg, p):       return run_board(seg, Pedalboard([PitchShift(**p)]))
def fx_delay(seg, p):             return run_board(seg, Pedalboard([Delay(**p)]))
def fx_panning(seg, p):           return seg.pan(p.get("pan", 0.0))
def fx_loudness(seg, p):          return normalize(seg, p.get("headroom_db", 0.1))
def fx_limiter(seg, p):           return run_board(seg, Pedalboard([Limiter(**p)]))

def fx_static(seg: AudioSegment, p: dict) -> AudioSegment:
    """
    Generate static noise and overlay it on the input segment.
    JSON parameters (all optional, defaults shown):
        level_db : -35
        

    """
    level_db = p.get("level_db", -35)
    if level_db >= 0:
        print("Warning: Static noise level must be negative – skipping.")
        return seg
    static = _make_static(seg, level_db)
    return seg.overlay(static)

def fx_robot(seg: AudioSegment, p: dict) -> AudioSegment:
    """
    Robot voice:
      • Flanger (if available) *or* a chorus that mimics a flanger
      • Optional pitch-shift to drop the formants
    JSON parameters (all optional, defaults shown):
        rate_hz          : 0.8
        depth            : 0.9
        feedback         : 0.0        (ignored by Chorus)
        centre_delay_ms  : 6          (only for Chorus)
        mix              : 1.0        (wet/dry, Chorus only)
        pitch_semitones  : -2
    """
    rate   = p.get("rate_hz",        0.8)
    depth  = p.get("depth",          0.9)
    fb     = p.get("feedback",       0.0)
    cdel   = p.get("centre_delay_ms", 6)
    mix    = p.get("mix",            1.0)
    pitch  = p.get("pitch_semitones", -2)

    if FLANGER_AVAILABLE:
        print("  Using real Flanger")
        fx = Flanger(rate_hz=rate, depth=depth, feedback=fb)
    else:
        print("  Flanger not found → using Chorus emulation")
        #  A very short delay and 100 % wet mix approximates a flanger sweep
        fx = Chorus(rate_hz=rate,
                    depth=depth,
                    centre_delay_ms=cdel,
                    mix=mix)

    board = Pedalboard([
        fx,
        PitchShift(semitones=pitch)
    ])
    return run_board(seg, board)

def fx_ghost(seg, p):
    board = Pedalboard([
        Reverb(room_size=p.get("room_size", .9),
               wet_level=p.get("wet_level", .7),
               dry_level=p.get("dry_level", .1),
               damping=p.get("damping", .8)),
        Delay(delay_seconds=p.get("delay_ms", 500)/1000,
              feedback=p.get("feedback", .35),
              mix=p.get("mix", .25)),
    ])
    return run_board(seg, board)

def fx_muffled(seg, p):
    return seg.low_pass_filter(p.get("low_pass_hz", 700)).apply_gain(
        p.get("gain_db", -8)
    )

###############################################################################
# Dispatcher dictionary                                                       #
###############################################################################

FX = {
    "fade_in": fx_fade_in,
    "fade_out": fx_fade_out,
    "silence": fx_silence,
    "telephone": fx_telephone,
    "walkie_talkie": fx_walkie_talkie,
    "reverb": fx_reverb,
    "noise_reduction": fx_noise_reduction,
    "deesser": fx_deesser,
    "high_pass_filter": fx_high_pass,
    "low_pass_filter": fx_low_pass,
    "equalizer": fx_equalizer,
    "compressor": fx_compressor,
    "pitch_shift": fx_pitch_shift,
    "delay": fx_delay,
    "panning": fx_panning,
    "loudness_normalization": fx_loudness,
    "limiter": fx_limiter,
    "robot": fx_robot,
    "ghost": fx_ghost,
    "muffled": fx_muffled,
    "static": fx_static,
}

###############################################################################
# Core engine                                                                 #
###############################################################################

def apply_chain(seg: AudioSegment, chain: list[dict]) -> AudioSegment:
    out = seg
    for step in chain:
        name = step.get("name")
        params = step.get("parameters", {})
        fx = FX.get(name)
        if not fx:
            print(f"  !! Unknown effect '{name}' – skipped.")
            continue
        print(f"  → {name}")
        try:
            out = fx(out, params)
        except Exception as e:
            print(f"    Error: {e} (effect skipped)")
    return out

def pad_audio(seg: AudioSegment, pre_ms: int = 0, post_ms: int = 0) -> AudioSegment:
    """
    Return *seg* with <pre_ms> ms of silence at the start and
    <post_ms> ms of silence at the end.
    """
    if pre_ms:
        seg = AudioSegment.silent(pre_ms, frame_rate=seg.frame_rate) + seg
    if post_ms:
        seg = seg + AudioSegment.silent(post_ms, frame_rate=seg.frame_rate)
    return seg


def process_jobs(jobs: list[dict], presets: dict):
    for idx, job in enumerate(jobs, 1):
        src, dst = Path(job.get("source")), Path(job.get("target"))
        if not src or not dst:
            print(f"Job {idx}: missing source/target – skipped.")
            continue

        print(f"\nJob {idx}: {src} → {dst}")

        # ---------------- Get preset (if any) --------------------
        preset_name  = job.get("apply_preset")
        preset_data  = presets.get(preset_name, {}) if preset_name else {}
        if preset_name and not preset_data:
            print(f"  !! Preset '{preset_name}' not found - skipped.")
            continue

        try:
            audio = AudioSegment.from_file(src)
        except Exception as e:
            print(f"  Cannot open source: {e} - skipped.")
            continue
        

# --------------- Resolve padding ------------------------
        pre  = job.get("pad_start_ms",
                       preset_data.get("pad_start_ms", 0))
        post = job.get("pad_end_ms",
                       preset_data.get("pad_end_ms",   0))

        pre  = int(pre)  if pre  else 0
        post = int(post) if post else 0

        if pre or post:
            print(f"  Padding: {pre} ms head  /  {post} ms tail")
            audio = pad_audio(audio, pre, post)
        # --------------------------------------------------------

        # Resolve effect list
        chain = job.get("effects")
        if not chain:
            preset = job.get("apply_preset")
            chain = presets.get(preset, {}).get("effects") if preset else None
            if chain:
                print(f"  Using preset '{preset}'")
        if not chain:
            print("  No effects specified – skipped.")
            continue

        processed = apply_chain(audio, chain)

        try:
            processed.export(dst, format=dst.suffix.lstrip("."))
            print(f"  ✓ Saved to {dst}")
        except Exception as e:
            print(f"  Export error: {e}")


###############################################################################
# CLI                                                                         #
###############################################################################

import regex as re


###############################################################################
# CLI helpers                                                                 #
###############################################################################
def _strip_json_comments(src: str) -> str:
    """
    Remove // and /* */ comments that are *outside* quoted strings.
    """
    out, i, ln = [], 0, len(src)
    in_str, esc = False, False

    while i < ln:
        c = src[i]

        # toggle string context
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":                # escape next char
                esc = True
            elif c == '"':                 # closing quote
                in_str = False
            i += 1
            continue

        if c == '"':                       # opening quote
            in_str = True
            out.append(c)
            i += 1
            continue

        # we are *outside* a string ------------------------------------------
        if c == "/" and i + 1 < ln:
            nxt = src[i + 1]
            if nxt == "/":                 # single-line comment
                i += 2
                while i < ln and src[i] not in ("\n", "\r"):
                    i += 1
                continue
            if nxt == "*":                 # block comment
                i += 2
                while i + 1 < ln and not (src[i] == "*" and src[i + 1] == "/"):
                    i += 1
                i += 2                     # skip closing */
                continue

        out.append(c)
        i += 1

    return "".join(out)


def _strip_trailing_commas(src: str) -> str:
    """
    Remove trailing commas before } or ] (again, outside quoted strings).
    """
    pattern = []
    in_str, esc = False, False
    for c in src:
        if in_str:
            pattern.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            pattern.append(c)
            continue
        # outside string
        if c == ",":
            pattern.append("\u0000")       # sentinel so we can lookahead
        else:
            pattern.append(c)

    cleaned = []
    it = iter(range(len(pattern)))
    for idx in it:
        ch = pattern[idx]
        if ch == "\u0000":                 # our sentinel
            # look ahead for next non-space
            j = idx + 1
            while j < len(pattern) and pattern[j].isspace():
                j += 1
            if j < len(pattern) and pattern[j] in ("]", "}"):
                # skip comma
                continue
            cleaned.append(",")
        else:
            cleaned.append(ch)
    return "".join(cleaned)


def load_json(path: str | Path) -> dict:
    """
    Load a JSON / JSONC file:
      • supports // and /* */ comments
      • supports trailing commas
      • leaves anything inside quotes untouched
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        txt = _strip_json_comments(txt)
        txt = _strip_trailing_commas(txt)
        return json.loads(txt)
    except Exception as e:
        print(f"Could not load {path}: {e}")
        return {}


def main():
    ap = argparse.ArgumentParser(
        description="Batch audio processor with preset support")
    ap.add_argument("jobs_file", help="JSON describing jobs (and optionally presets)")
    ap.add_argument("--presets", help="Separate JSON file holding presets")
    args = ap.parse_args()

    jobs_json = load_json(args.jobs_file)
    presets_json = load_json(args.presets) if args.presets else jobs_json

    jobs = jobs_json.get("audio_processing_jobs", [])
    presets = presets_json.get("effect_presets", {})

    if not jobs:
        print("No jobs found – nothing to do.")
        return

    process_jobs(jobs, presets)


if __name__ == "__main__":
    main()