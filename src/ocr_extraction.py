import numpy as np
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def extract_text_from_masked_region(image, mask):
    mask_3d = np.expand_dims(mask, axis=-1) if mask.ndim == 2 else mask
    masked_image = image * mask_3d

    cropped_image = Image.fromarray(masked_image.astype(np.uint8))
    return pytesseract.image_to_string(cropped_image)