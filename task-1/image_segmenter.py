import os
import sys
import ctypes
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_gles = os.path.expanduser("~/.local/lib/libGLESv2.so.2")
if os.path.exists(_gles):
    ctypes.CDLL(_gles, mode=ctypes.RTLD_GLOBAL)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "deeplab_v3.tflite")
DEFAULT_IMG = os.path.join(os.path.dirname(__file__), "sample_data", "sample_face.png")
OUTPUT_IMG = os.path.join(os.path.dirname(__file__), "sample_data", "annotated_image_segmenter.png")


def run(image_path: str = DEFAULT_IMG, output_path: str = OUTPUT_IMG):
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        output_category_mask=True
    )
    
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        image = mp.Image.create_from_file(image_path)
        result = segmenter.segment(image)
        
        print(f"--- Image Segmenter Results ({os.path.basename(image_path)}) ---")
        if result.category_mask:
            mask = result.category_mask.numpy_view()
            print(f"  Segmented mask shape: {mask.shape}")
            print(f"  Detected category IDs: {list(set(mask.flatten()))[:5]}")
            
            # Create color overlay on image
            orig = cv2.imread(image_path)
            h, w = orig.shape[:2]
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            color_mask = np.zeros_like(orig)
            color_mask[mask_resized > 0] = [0, 255, 128] # greenish overlay
            blended = cv2.addWeighted(orig, 0.6, color_mask, 0.4, 0)
            cv2.imwrite(output_path, blended)
            print(f"Annotated image saved to: {output_path}")
        else:
            print("No segmentation mask produced.")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    run(img)
