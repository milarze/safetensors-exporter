import os
import shutil
import json
import torch
from safetensors.torch import save_model
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast, T5Tokenizer
from huggingface_hub import snapshot_download
from sys import argv

def convert_and_save_model(model_id: str, output_dir: str):
    """
    Convert a PyTorch model to Safetensors format and save config and tokenizer files.
    
    Args:
        model_id (str): The Hugging Face model ID.
        output_dir (str): Directory to save the exported model.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model {model_id}...")
    
    # Download the complete model directory
    model_path = snapshot_download(repo_id=model_id)
    print(f"Downloaded model files to {model_path}")
    
    # Load the model using transformers
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    print("Model loaded successfully!")
    
    # Save the model in safetensors format
    model_file = os.path.join(output_dir, "model.safetensors")
    save_model(model, model_file)
    print(f"Model weights saved to {model_file}")
    
    # Check if tokenizer.json already exists
    tokenizer_json_path = os.path.join(model_path, "tokenizer.json")
    if os.path.exists(tokenizer_json_path):
        # Copy the existing tokenizer.json
        dst_tokenizer = os.path.join(output_dir, "tokenizer.json")
        shutil.copyfile(tokenizer_json_path, dst_tokenizer)
        print(f"Copied existing tokenizer.json to {dst_tokenizer}")
    else:
        # No tokenizer.json, we need to create it from the SentencePiece model
        print("No tokenizer.json found. Attempting to create one from SentencePiece model...")
        
        # First try to load with the slow tokenizer and convert to fast tokenizer
        try:
            # Load the slow tokenizer
            slow_tokenizer = T5Tokenizer.from_pretrained(model_path)
            print("Loaded T5 tokenizer (slow version)")
            
            # Convert to fast tokenizer
            fast_tokenizer = T5TokenizerFast.from_pretrained(model_path, from_slow=True)
            print("Successfully converted to fast tokenizer")
            
            # Save the tokenizer.json
            fast_tokenizer.save_pretrained(output_dir)
            print(f"Created and saved tokenizer.json to {output_dir}")
            
        except Exception as e:
            print(f"Could not convert to fast tokenizer: {e}")
            print("Falling back to copying SentencePiece model files...")
            
            # Copy necessary tokenizer files
            essential_files = [
                "spiece.model", 
                "tokenizer_config.json",
                "special_tokens_map.json"
            ]
            
            for file in essential_files:
                src_path = os.path.join(model_path, file)
                if os.path.exists(src_path):
                    dst_path = os.path.join(output_dir, file)
                    shutil.copyfile(src_path, dst_path)
                    print(f"Copied {file} to {dst_path}")
    
    # Copy config.json
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        dst_config = os.path.join(output_dir, "config.json")
        shutil.copyfile(config_path, dst_config)
        print(f"Copied config.json to {dst_config}")
    else:
        # Save the config if it doesn't exist
        model.config.save_pretrained(output_dir)
        print(f"Saved config to {os.path.join(output_dir, 'config.json')}")
    
    # Try one more approach to generate tokenizer.json if it still doesn't exist
    if not os.path.exists(os.path.join(output_dir, "tokenizer.json")):
        try:
            # Try using the sentencepiece_model_to_tokenizer_json script
            print("Attempting to convert SentencePiece model to tokenizer.json...")
            
            # Import the conversion function directly to avoid subprocess
            from transformers.tools.sentencepiece_model_to_tokenizer_json import convert_sentencepiece_to_tokenizer_json
            
            spiece_path = os.path.join(output_dir, "spiece.model")
            if os.path.exists(spiece_path):
                # Read tokenizer_config.json for extra tokens if available
                extra_tokens = {}
                config_path = os.path.join(output_dir, "tokenizer_config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        if 'additional_special_tokens' in config:
                            extra_tokens = {'additional_special_tokens': config['additional_special_tokens']}
                
                output_json = os.path.join(output_dir, "tokenizer.json")
                convert_sentencepiece_to_tokenizer_json(
                    spiece_path, 
                    output_json,
                    extra_tokens=extra_tokens
                )
                print(f"Successfully created tokenizer.json at {output_json}")
            else:
                print("No spiece.model found, cannot create tokenizer.json")
                
        except Exception as e:
            print(f"Failed to convert SentencePiece to tokenizer.json: {e}")
    
    print(f"Model conversion complete. All files saved to {output_dir}")
    return output_dir

if __name__ == "__main__":
    # Example usage
    print(argv)
    if len(argv) < 2:
        print("Usage: python script.py <model_id> [output_directory]")
        exit(1)
    
    model_id = argv[1]
    output_dir = argv[2] if len(argv) > 2 else "/exported_model"
    
    print(f"Converting model {model_id} to Safetensors format...")
    convert_and_save_model(model_id, output_dir)

