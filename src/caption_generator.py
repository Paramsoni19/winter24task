from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def generate_image_caption(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    caption = model.generate(**inputs)
    return processor.decode(caption[0], skip_special_tokens=True)