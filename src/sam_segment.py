import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

def load_sam_model(checkpoint_path, device="cuda"):
    sam = sam_model_registry["vit_l"](checkpoint=checkpoint_path).to(device)
    predictor = SamPredictor(sam)
    return predictor

def segment_image(predictor, image_path):
    image = np.array(Image.open(image_path))
    predictor.set_image(image)

    masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=None, multimask_output=False)
    return image, masks[0]