from typing import List

from PIL import Image
import torch

from transformers import AutoProcessor, AutoModelForCausalLM 
from utils.eval_model import BaseEvalModel
from torchvision import transforms
from transformers.image_utils  import OPENAI_CLIP_MEAN,OPENAI_CLIP_STD



class EvalModel(BaseEvalModel):
    """Florence model evaluation.
    
    Attributes:
        model (nn.Module): Underlying Torch model.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for model.
        device: Index of GPU to use, or the string "cpu"
        """
        
    def __init__(self, model_args):
        assert (
            "model_path" in model_args
            and "processor_path" in model_args
            and "device" in model_args
        ), "Florence requires model_path, processor_path, and device arguments to be specified"
        
        self.device = model_args["device"]
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_args["processor_path"], local_files_only=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_args["model_path"], local_files_only=True, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.model.to(self.device, self.torch_dtype)
        self.processor.tokenizer.padding_side = "left"
        
    def get_pixel_values(self, image: Image.Image, bbox=None) -> torch.Tensor:
        """Get pixel values for an image.
        
        Args:
            image: Image to process.
            
        Returns:
            Tensor of pixel values.
            """
        pixel_values = self.processor.image_processor(image, return_tensors="pt", do_normalize=True)["pixel_values"].to(self.device, self.torch_dtype)
        
        if bbox is not None:
            orig_size = image.size
            target_size = (768, 768)
            bbox = self.convert_bbox(bbox, orig_size, target_size)
        
        return pixel_values, bbox
        
    def get_patch_indices(self, box, image_size=768, grid=24):
        patch_size = image_size // grid
        x1, y1, x2, y2 = box
        
        # 向上取整得到第一个完全在框内的patch
        patch_x1 = (x1 + patch_size - 1) // patch_size
        patch_y1 = (y1 + patch_size - 1) // patch_size
        
        # 向下取整得到最后一个完全在框内的patch
        patch_x2 = x2 // patch_size
        patch_y2 = y2 // patch_size
        
        # 检查是否有完全在框内的patch
        if patch_x1 > patch_x2 or patch_y1 > patch_y2:
            return []
            
        patch_indices = []
        for i in range(patch_y1, patch_y2 + 1):
            for j in range(patch_x1, patch_x2 + 1):
                if 0 <= i < grid and 0 <= j < grid:
                    token_idx = i * grid + j
                    patch_indices.append(token_idx)
        token_indices = [idx + 1 for idx in patch_indices]
        return token_indices
    
    def convert_bbox(self, bbox, orig_size, target_size):
        orig_w, orig_h = orig_size
        target_w, target_h = target_size
        x, y, w, h = bbox
        
        w_scale, h_scale = target_w / orig_w, target_h / orig_h
        x1 = int(x * w_scale)
        y1 = int(y * h_scale)
        x2 = int((x + w) * w_scale)
        y2 = int((y + h) * h_scale)
        
        return (max(0, min(x1, target_w)), 
                max(0, min(y1, target_h)), 
                max(0, min(x2, target_w)), 
                max(0, min(y2, target_h)))
        
    def get_mask(self, pixel_box, pixel_values):
        mask = torch.zeros_like(pixel_values)
        x1, y1, x2, y2 = pixel_box
        mask[:, :, y1:y2, x1:x2] = 1
        return mask
            
   
    def generate_caption(self, pixel_values, img):
        caption_inputs = self.processor(text=["<CAPTION>"], images=img, return_tensors="pt", do_normalize=True).to(self.device, self.torch_dtype)
        generate_ids = self.model.generate(
                    input_ids=caption_inputs["input_ids"],
                    pixel_values=pixel_values,
                    max_length=108,
                    num_beams=3
                )
        caption = self.processor.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        return caption
    
    def get_image_features(self, pixel_values):
        outputs = self.model._encode_image(pixel_values)
        return outputs
    
    # def get_text_embeds(self, text):
    #     texts = [template.format(text) for template in imagenet_templates]
    #     # text = self.processor._construct_prompts(text)
    #     inputs_ids = self.tokenizer(texts, return_tensors="pt", padding="longest", truncation=True, max_length=128)["input_ids"].to(self.device)
    #     inputs_ids.requires_grad = False
    #     outputs = self.model.get_input_embeddings()(inputs_ids)
    #     pooled_output = outputs
    #     pooled_output /= pooled_output.norm(dim=-1, keepdim=True)
    #     pooled_output = pooled_output.mean(dim=0, keepdim=True)

    #     return pooled_output
    
    def get_text_embeds(self, text):
        if text is list:
            text = self.processor._construct_prompts(text)
        elif text is str:
            text = self.processor._construct_prompts([text])
        inputs_ids = self.tokenizer(text, return_tensors="pt", padding="longest", truncation=True, max_length=128)["input_ids"].to(self.device)
        outputs = self.model.get_input_embeddings()(inputs_ids)
        return outputs[:, 1:-1, :].to(self.device, self.torch_dtype)
    
    def get_output_loss(self, pixel_values, label_text, img):
        inputs_ids = self.processor(text="<CAPTION>", images=img, return_tensors="pt", do_normalize=True).to(self.device, self.torch_dtype)
        label_ids = self.processor.tokenizer.encode(label_text)
        labels = torch.tensor(label_ids).unsqueeze(0).to(self.device)
        outputs = self.model( 
            input_ids=inputs_ids["input_ids"],
            pixel_values=pixel_values,
            labels=labels
        )
        return outputs.loss
        
        
        