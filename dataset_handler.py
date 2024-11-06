from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

class CustomConversationalDataset(Dataset):
    def __init__(self, embeddings, targets, tokenizer, max_length=256):
        self.embeddings = embeddings
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        input_embedding = self.embeddings[idx]
        target_text = self.targets[idx]
        
        labels = self.tokenizer(
            target_text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )['input_ids'].squeeze(0)
        
        return {
            'input_embedding': input_embedding,
            'labels': labels
        }

class CustomDataCollatorWithPadding:
    def __init__(self):
        pass
    
    def __call__(self, features):
        if 'input_embedding' not in features[0]:
            raise KeyError("Expected 'input_embedding' in features but not found.")
        
        input_embeddings = torch.stack([f['input_embedding'] for f in features])
        labels = pad_sequence([f['labels'] for f in features], batch_first=True, padding_value=-100)
        
        batch = {
            'inputs_embeds': input_embeddings,
            'labels': labels,
            'attention_mask': torch.ones(input_embeddings.size()[:-1], dtype=torch.long)
        }
        return batch
