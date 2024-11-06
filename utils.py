from transformers import TrainerCallback
import torch

class ClearCUDACacheCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        print("Cleared CUDA cache after evaluation.")
