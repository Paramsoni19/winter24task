from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def get_feature_encoding(input_text, model_name="bigscience/bloom"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    return hidden_states[-1]  # Return the last hidden state