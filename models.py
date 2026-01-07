import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_embedding_model(model_name):
    """Loads the sentence transformer model."""
    print(f"Loading embedding model from: {model_name}...")
    print(f"HF_HOME set to: {os.environ.get('HF_HOME', 'Not set')}")
    try:
        # Load model (supports local path or Hugging Face model name)
        model = SentenceTransformer(model_name)
        print("✅ Embedding model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        print(f"❌ Full error details: {type(e).__name__}: {str(e)}")
        print("⚠️ Ensure the model path is correct and contains all necessary files.")
        return None


def load_generation_model(model_name):
    """Loads the Hugging Face generative model and tokenizer."""
    print(f"Loading generation model from: {model_name}...")
    print(f"HF_HOME set to: {os.environ.get('HF_HOME', 'Not set')}")
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✅ Tokenizer loaded successfully.")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="cpu",  # Force CPU to ensure compatibility
            dtype=torch.float32,  # Use float32 for CPU compatibility
        )
        print("✅ Model loaded successfully.")
        
        if tokenizer.pad_token is None:
             tokenizer.pad_token = tokenizer.eos_token
        print("✅ Generation model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Failed to load generation model: {e}")
        print(f"❌ Full error details: {type(e).__name__}: {str(e)}")
        print("⚠️ Ensure the model path is correct and contains all necessary files.")
        return None, None 