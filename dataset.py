import os
import json
from torch.utils.data import Dataset

class CrossVLAD_Dataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        """
        初始化数据集，加载图像路径和对应的标注文件路径。
        
        参数:
            images_dir (str): 存放图像的目录。
            labels_dir (str): 存放标注文件的目录。
            transform (callable, optional): 应用于图像的转换操作（如数据增强等）。
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        
        # 获取所有标注文件列表
        self.label_files = sorted([file for file in os.listdir(labels_dir) if file.endswith('.json')])
        
    def __len__(self):
        """
        返回数据集的大小（即样本数量）。
        
        返回:
            int: 数据集的大小。
        """
        return len(self.label_files)
    
    def __getitem__(self, idx):
        """
        根据索引获取图像及其标注数据。
        
        参数:
            idx (int): 索引。
        
        返回:
            dict: 包含图像张量和标注信息的字典。
        """
        # 获取标注文件路径
        label_file_path = os.path.join(self.labels_dir, self.label_files[idx])
        
        # 读取标注文件
        with open(label_file_path, 'r') as f:
            label_data = json.load(f)
        
        # 获取图像文件路径
        image_id = label_data["image_id"]
        image_file_path = os.path.join(self.images_dir, f"{str(image_id).zfill(12)}.jpg")
        
        # # 加载图像
        # image = Image.open(image_file_path).convert("RGB")
        
        # # 应用图像转换（如果有）
        # if self.transform:
        #     image = self.transform(image)
        
        # 返回图像和标注数据
        sample = {
            "image_path": image_file_path,
            "label": label_data
        }
        return sample
    
# 自定义 collate_fn
def custom_collate_fn(batch):
    """
    自定义批量处理函数，处理不规则的数据。
    """
    image_paths = [item['image_path'] for item in batch]
    labels = [item['label'] for item in batch]
    return {"image_path": image_paths, "labels": labels}

