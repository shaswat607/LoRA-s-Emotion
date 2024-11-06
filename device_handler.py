import torch

class DeviceHandler:
    @staticmethod
    def get_device():
        # Check if GPU is available and set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        return device