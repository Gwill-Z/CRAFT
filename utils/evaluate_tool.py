from transformers import CLIPProcessor, CLIPModel
import torch
from typing import List, Dict, Union
from tqdm import tqdm
import numpy as np

class ClipAttackEvaluator:
    def __init__(self):
        # 加载CLIP模型和处理器
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def calculate_similarity(self, generated_text: str, target_text: str) -> float:
        """使用CLIP计算生成文本和目标文本的余弦相似度"""
        # 使用CLIP处理器处理文本
        inputs = self.processor(
            text=[generated_text, target_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77  # CLIP的默认最大长度
        )
        
        # 将输入移到正确的设备上
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 获取文本特征
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        # 计算余弦相似度
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = text_features[0] @ text_features[1].T
        
        return similarity.item()
    

class SBERTAttackEvaluator:
    def __init__(self):
        # 加载 Sentence-BERT 模型
        # from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def calculate_similarity(self, generated_text: str, target_text: str) -> float:
        """使用 Sentence-BERT 计算生成文本和目标文本的余弦相似度"""
        # 直接使用模型编码文本
        with torch.no_grad():
            # 获取文本嵌入
            embeddings = self.model.encode(
                [generated_text, target_text],
                convert_to_tensor=True,
                device=self.device
            )
        
        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0),
            embeddings[1].unsqueeze(0)
        )
        
        return similarity.item()
    
    
    
# 创建评估器实例
evaluator = ClipAttackEvaluator()
s_evaluator = SBERTAttackEvaluator()

# # 评估单个样本
# gen_text = "A hippo standing in a field next to a bus"
# # target_text = "A person is playing the guitar in the park."
# target_text = "A bus grazing on a fenced field with grass"
# target_object = "park"

cat1 = "cat"
cat2 = "cow"

similarity = evaluator.calculate_similarity(cat1, cat2)
s_similarity = s_evaluator.calculate_similarity(cat1, cat2)
print(f"CLIP Similarity: {similarity}")
print(f"SBERT Similarity: {s_similarity}")

# print(result)

