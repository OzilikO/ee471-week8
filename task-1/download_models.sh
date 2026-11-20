#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
mkdir -p "$MODELS_DIR"

echo "=========================================================="
echo " Checking & Downloading All 22 MediaPipe Models to $MODELS_DIR"
echo "=========================================================="

download() {
    local url="$1"
    local filename="$2"
    local dest="$MODELS_DIR/$filename"

    if [ -f "$dest" ]; then
        echo "[✓] Already exists: $filename"
    else
        echo "[↓] Downloading $filename..."
        curl -f -L -C - --retry 3 "$url" -o "$dest" || echo "[!] Warning: failed to download $filename from $url"
    fi
}

# --- Vision Models (11 tasks) ---
download "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite" "blaze_face_short_range.tflite"
download "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" "pose_landmarker.task"
download "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" "face_landmarker.task"
download "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task" "gesture_recognizer.task"
download "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" "hand_landmarker.task"
download "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task" "holistic_landmarker.task"
download "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/latest/efficientnet_lite0.tflite" "efficientnet_lite0.tflite"
download "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/latest/mobilenet_v3_small.tflite" "mobilenet_v3_small.tflite"
download "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite" "deeplab_v3.tflite"
download "https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite" "magic_touch.tflite"
download "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite" "efficientdet_lite0.tflite"

# --- Audio Models (3 tasks) ---
download "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite" "yamnet.tflite"
download "https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3/resolve/main/parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm" "parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm"
download "https://huggingface.co/litert-community/Kokoro-82M/resolve/main/kokoro_int8_fp16.litertlm" "kokoro_int8_fp16.litertlm"

# --- Text Models (5 tasks) ---
download "https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/latest/language_detector.tflite" "language_detector.tflite"
download "https://storage.googleapis.com/mediapipe-models/text_classifier/bert_classifier/float32/latest/bert_classifier.tflite" "bert_classifier.tflite"
download "https://storage.googleapis.com/mediapipe-models/text_embedder/universal_sentence_encoder/float32/latest/universal_sentence_encoder.tflite" "universal_sentence_encoder.tflite"
download "https://storage.googleapis.com/mediapipe-models/text_summarizer/gemma_200m/latest/summarization_quant_200m_2modes.litertlm" "summarization_quant_200m_2modes.litertlm"
download "https://huggingface.co/litert-community/MediaPipe-TextProofreader-200M/resolve/main/proofread_quant_200m.litertlm" "proofread_quant_200m.litertlm"

# --- Retrieval & Decision Models (3 tasks) ---
download "https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm/resolve/main/embeddinggemma-2-text-270m.litertlm" "embeddinggemma-2-text-270m.litertlm"
download "https://huggingface.co/litert-community/embeddinggemma-2-text-vision-440m-litert-lm/resolve/main/embeddinggemma-2-text-vision-440m.litertlm" "embeddinggemma-2-text-vision-440m.litertlm"

echo "=========================================================="
echo " All MediaPipe models verified!"
echo "=========================================================="
