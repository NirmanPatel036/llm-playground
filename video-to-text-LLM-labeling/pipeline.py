import cv2
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

def extract_keyframes(video_path, num_frames=5):
    """
    Extracts evenly spaced frames from the video.
    """
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frames = []
    for idx in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, image = vidcap.read()
        if success:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img))
    vidcap.release()
    return frames

def caption_image(image, processor, model, device):
    """
    Generates a caption for a single image using BLIP.
    """
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def describe_video(video_path, num_frames=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    frames = extract_keyframes(video_path, num_frames)
    captions = [caption_image(frame, processor, model, device) for frame in frames]

    # Aggregate captions into a summary
    summary = " ".join(captions)
    return summary

# Usage example
print(describe_video("video_path.avi", num_frames=5))
