import os
import json
from torch.utils.data import Dataset

class CrossVLAD_Dataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.label_files = sorted([file for file in os.listdir(labels_dir) if file.endswith('.json')])
        
    def __len__(self):

        return len(self.label_files)
    
    def __getitem__(self, idx):

        label_file_path = os.path.join(self.labels_dir, self.label_files[idx])
        
        with open(label_file_path, 'r') as f:
            label_data = json.load(f)
        
        image_id = label_data["image_id"]
        image_file_path = os.path.join(self.images_dir, f"{str(image_id).zfill(12)}.jpg")
        
        sample = {
            "image_path": image_file_path,
            "label": label_data
        }
        return sample
    
def custom_collate_fn(batch):
    image_paths = [item['image_path'] for item in batch]
    labels = [item['label'] for item in batch]
    return {"image_path": image_paths, "labels": labels}

