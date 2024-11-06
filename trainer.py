from transformers import Trainer
import torch

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs_embeds = inputs.pop('inputs_embeds').to(model.device)
        labels = inputs['labels'].to(model.device)
        outputs = model(inputs_embeds=inputs_embeds, labels=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def _remove_unused_columns(self, dataset, description):
        return dataset
