import os
from src.sam_segment import load_sam_model, segment_image
from src.ocr_extraction import extract_text_from_masked_region
from src.caption_generator import generate_image_caption
from src.feature_encoder import get_feature_encoding

def main():
    checkpoint_path = "./checkpoints/sam_vit_l_0b3195.pth"
    image_path = "./images/example.jpg"
    output_path = "./outputs/extracted_data.txt"

    # Load SAM model
    predictor = load_sam_model(checkpoint_path)

    # Segment image and extract text
    image, mask = segment_image(predictor, image_path)
    ocr_text = extract_text_from_masked_region(image, mask)

    # Generate caption
    caption = generate_image_caption(image_path)

    # Combine OCR text and caption
    combined_text = f"Caption: {caption}\nOCR Text: {ocr_text}"
    print("Combined Text:\n", combined_text)

    # Save combined text
    os.makedirs("./outputs", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(combined_text)

    # Generate feature encoding
    feature_encoding = get_feature_encoding(combined_text)
    print("Feature Encoding Shape:", feature_encoding.shape)

if __name__ == "__main__":
    main()