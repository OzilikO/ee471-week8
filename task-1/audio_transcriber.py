import os
import sys
import wave
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import audio

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm")
SAMPLE_AUDIO = os.path.join(os.path.dirname(__file__), "sample_data", "sample_speech.wav")


def load_audio_samples(audio_path: str) -> np.ndarray:
    with wave.open(audio_path, "rb") as wf:
        raw_frames = wf.readframes(wf.getnframes())
        audio_data = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_data


def run(audio_path: str = SAMPLE_AUDIO):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{os.path.basename(MODEL_PATH)}' not found in models/.")
        print("Please ensure the model is downloaded.")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = audio.AudioTranscriberOptions(base_options=base_options)
    
    with audio.AudioTranscriber.create_from_options(options) as transcriber:
        audio_samples = load_audio_samples(audio_path)
        result = transcriber.transcribe(audio_samples)
        
        print(f"--- Audio Transcriber Results ({os.path.basename(audio_path)}) ---")
        if result and result.segments:
            for idx, seg in enumerate(result.segments):
                print(f"  Segment #{idx+1} [{seg.from_timestamp_ms}ms - {seg.to_timestamp_ms}ms]: \"{seg.transcript}\"")
        else:
            print("  Transcription finished (no speech detected).")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else SAMPLE_AUDIO
    run(p)
