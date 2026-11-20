#takes raw audio clips or streams and categorizes 
#them into over 500 predefined sound events—such as speech, music, applause, or animal sounds—using a pre-trained model like YAMNet
import os
import sys
import wave
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import audio
from mediapipe.tasks.python.components import containers

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "yamnet.tflite")
SAMPLE_AUDIO = os.path.join(os.path.dirname(__file__), "sample_data", "sample_speech.wav")


def load_audio_clip(audio_path: str) -> containers.AudioData:
    with wave.open(audio_path, "rb") as wf:
        rate = float(wf.getframerate())
        raw_frames = wf.readframes(wf.getnframes())
        audio_data = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
    return containers.AudioData.create_from_array(audio_data, sample_rate=rate)


def run(audio_path: str = SAMPLE_AUDIO):
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = audio.AudioClassifierOptions(base_options=base_options, max_results=3)
    
    with audio.AudioClassifier.create_from_options(options) as classifier:
        audio_clip = load_audio_clip(audio_path)
        result = classifier.classify(audio_clip)
        
        print(f"--- Audio Classifier Results ({os.path.basename(audio_path)}) ---")
        if result and result[0].classifications:
            for cat in result[0].classifications[0].categories:
                print(f"  Category: '{cat.category_name}' (Score: {cat.score:.3f})")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else SAMPLE_AUDIO
    run(p)
