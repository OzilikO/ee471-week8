import os
import sys
import wave
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import audio

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "kokoro_int8_fp16.litertlm")
DEFAULT_TEXT = "Hello! Welcome to MediaPipe on-device text to speech synthesis."
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "sample_data", "tts_output.wav")


def run(input_text: str = DEFAULT_TEXT, out_file: str = OUTPUT_PATH):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{os.path.basename(MODEL_PATH)}' not found in models/.")
        print("Please ensure the model has finished downloading.")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = audio.TextToSpeechOptions(base_options=base_options)
    
    with audio.TextToSpeech.create_from_options(options) as tts:
        result = tts.synthesize(input_text)
        
        print(f"--- Text-to-Speech Results ---")
        print(f"Input Text: \"{input_text}\"")
        if result and result.samples is not None:
            # Convert float32 [-1.0, 1.0] to int16 PCM and save as WAV
            int16_samples = (result.samples * 32767.0).clip(-32768, 32767).astype(np.int16)
            sample_rate = result.sample_rate_hz if result.sample_rate_hz else 24000
            
            with wave.open(out_file, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(int16_samples.tobytes())
            
            print(f"Synthesized {len(result.samples)} audio samples ({sample_rate}Hz).")
            print(f"Audio saved to: {out_file}")
        else:
            print("Speech synthesis finished.")


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT
    run(t)
