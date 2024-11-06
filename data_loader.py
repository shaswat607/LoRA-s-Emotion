import json

class DataLoader:
    @staticmethod
    def load_data(file_path):
        """Load data from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['conversations']
