import torch
from transformers import AutoModelForSeq2SeqLM, T5EncoderModel, AutoTokenizer

class ModelHandler:
    def __init__(self, model_name='google/flan-t5-base', device=None):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.original_model = None
        self.encoder_model = None
        self.tokenizer = None

    def load_seq2seq_model(self):
        """Load the original sequence-to-sequence model."""
        self.original_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        print(f"Original model '{self.model_name}' loaded.")
        return self.original_model

    def load_encoder_model(self):
        """Load the encoder-only version of the model."""
        self.encoder_model = T5EncoderModel.from_pretrained(self.model_name)
        self.encoder_model.to(self.device)
        self.encoder_model.eval()
        print(f"Encoder model '{self.model_name}' loaded and set to evaluation mode on {self.device}.")
        return self.encoder_model

    def load_tokenizer(self):
        """Load the tokenizer for the model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print(f"Tokenizer for '{self.model_name}' loaded.")
        return self.tokenizer
