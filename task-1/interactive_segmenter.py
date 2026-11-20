import os
import sys
import ctypes
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.keypoint import NormalizedKeypoint

_gles = os.path.expanduser("~/.local/lib/libGLESv2.so.2")
if os.path.exists(_gles):
    ctypes.CDLL(_gles, mode=ctypes.RTLD_GLOBAL)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "magic_touch.tflite")
DEFAULT_IMG = os.path.join(os.path.dirname(__file__), "sample_data", "sample_face.png")
OUTPUT_IMG = os.path.join(os.path.dirname(__file__), "sample_data", "annotated_interactive_segmenter.png")


def run(image_path: str = DEFAULT_IMG, output_path: str = OUTPUT_IMG):
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.InteractiveSegmenterLegacyOptions(
        base_options=base_options,
        output_category_mask=True
    )

    with vision.InteractiveSegmenterLegacy.create_from_options(options) as segmenter:
        image = mp.Image.create_from_file(image_path)
        ROI = vision.InteractiveSegmenterLegacyRegionOfInterest
        roi = ROI(
            format=ROI.Format.KEYPOINT,
            keypoint=NormalizedKeypoint(x=0.5, y=0.5)
        )
        result = segmenter.segment(image, roi)
        
        print(f"--- Interactive Segmenter Results ({os.path.basename(image_path)}) ---")
        if result.category_mask:
            mask = result.category_mask.numpy_view()
            print(f"  Interactive segment mask shape: {mask.shape}")
            print(f"  Segmented object pixels: {(mask == 0).sum()} pixels")
            
            orig = cv2.imread(image_path)
            h, w = orig.shape[:2]
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            color_mask = np.zeros_like(orig)
            color_mask[mask_resized == 0] = [255, 128, 0] # blue-ish overlay
            # Mark the interactive click point
            cv2.circle(orig, (w // 2, h // 2), 5, (0, 0, 255), -1)
            blended = cv2.addWeighted(orig, 0.6, color_mask, 0.4, 0)
            cv2.imwrite(output_path, blended)
            print(f"Annotated image saved to: {output_path}")
        else:
            print("No interactive segmentation mask produced.")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    run(img)
