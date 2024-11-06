import torch

class EmbeddingCreator:
    def __init__(self, encoder_model, tokenizer, device):
        self.encoder_model = encoder_model
        self.tokenizer = tokenizer
        self.device = device

    def create_incremental_embeddings(self, conversations, max_sequence_length=256):
        """Create incremental conversation embeddings with emotional context."""
        incremental_embeddings = []
        targets = []

        for conv in conversations:
            current_context = "Analyze emotional change and respond:\n"
            for turn in conv['conversation']:
                if turn['speaker'] == 'User':
                    current_context += f"{turn['speaker']}: {turn['text']} (Emotion: {', '.join(turn['emotion_label'])})\n"
                    embedding_input_text = current_context + "Chatbot:"

                    # Tokenize input
                    inputs = self.tokenizer(
                        embedding_input_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.device)

                    # Get embeddings
                    with torch.no_grad():
                        outputs = self.encoder_model(**inputs)
                        last_hidden_state = outputs.last_hidden_state.to(self.device)

                        # Adjust shape to be uniform
                        if last_hidden_state.shape[1] > max_sequence_length:
                            last_hidden_state = last_hidden_state[:, :max_sequence_length, :]
                        elif last_hidden_state.shape[1] < max_sequence_length:
                            padding = torch.zeros(
                                (1, max_sequence_length - last_hidden_state.shape[1], last_hidden_state.shape[2]),
                                device=last_hidden_state.device
                            )
                            last_hidden_state = torch.cat((last_hidden_state, padding), dim=1)

                        incremental_embeddings.append(last_hidden_state.detach().cpu())

                if turn['speaker'] == 'Chatbot':
                    current_context += f"{turn['speaker']}: {turn['text']} (Emotion: {', '.join(turn['emotion_label'])})\n"
                    target_with_emotion = f"{turn['text']} (Emotion: {', '.join(turn['emotion_label'])})"
                    targets.append(target_with_emotion)

        return torch.stack(incremental_embeddings), targets
