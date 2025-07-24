import argparse
import importlib
import json
import os
from pathlib import Path
from datetime import datetime
import traceback
import numpy as np
from dataset import CrossVLAD_Dataset
from tqdm import tqdm
from utils.attack_tool import load_model
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp
import random
import sys
# 当前目录为根目录/train, 获取项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 添加所有可能的路径
sys.path.extend([
    str(root_dir),
    str(root_dir + '/unifiedio2'),

])

def tensor_to_image(tensor):
    # 确保张量在CPU上
    tensor = tensor.detach().cpu()
    if tensor.shape[0] == 3:  # 如果第一维是通道数
        tensor = tensor.permute(1, 2, 0)
    
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    tensor = (tensor * std + mean)
    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
    array = tensor.numpy()
    return Image.fromarray(array)

class Trainer:
    def __init__(self, model, dataset, gpu_id, alpha=0.01, epsilon=0.1, sample_nums=1000, iters=10, do_normalize=True, mask=True, num_negative=1, model_name="florence2"):
        """
        Initialize Trainer with specific GPU ID.
        """
        self.model = model
        self.dataset = dataset
        self.sample_nums = sample_nums
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.alpha = alpha
        self.epsilon = epsilon
        self.iters = iters
        self.do_normalize = do_normalize
        self.random_start = False
        self.mask = mask
        self.num_negative = num_negative
        self.model_name = model_name

    def setup_data(self, all_indices, start_idx, end_idx):
        """
        Setup data for this GPU's portion of work
        """
        self.indices = all_indices[start_idx:end_idx]
        self.dataset = Subset(self.dataset, self.indices)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

    def attack(self, img, img_id, label, alpha, epsilon, iters, do_normalize, save_path):
        try:
            bbox = label["source_object_boxes"][0]
            pixel_values, pixel_box = self.model.get_pixel_values(img, bbox)
            pixel_values = pixel_values.to(self.device)
            
            token_indices = self.model.get_patch_indices(pixel_box)
            if self.mask:
                if self.model_name == "florence2":
                    mask = self.model.get_mask(pixel_box, pixel_values)
                else:
                    raise ValueError("Invalid model name")
            else:
                mask = 1
                
            delta = torch.empty_like(pixel_values).uniform_(-epsilon, epsilon).to(self.device)
            delta = delta * mask
            delta.requires_grad = True
            
            change_category_name = label["target_object_category"]
            original_category_name = label["source_object_category"]

            positive_name = change_category_name
            if self.num_negative == 1:
                negative_name = original_category_name
                negative_embeds = self.model.get_text_embeds(negative_name)
                negative_embeds = torch.mean(negative_embeds, dim=1)
            elif self.num_negative == 2:
                # v2, use all other categories as negative
                mscoco_categories_path = "datasets/mscoco_categories.json"
                with open(mscoco_categories_path, "r") as f:
                    categories = json.load(f)["categories"]
                    negative_names = [cat["name"] for cat in categories if cat["name"] != positive_name]
                negative_embeds = torch.empty(0, 1024).to(self.device)
                for negative_name in negative_names:
                    neg_embed = self.model.get_text_embeds(negative_name)
                    neg_embed = torch.mean(neg_embed, dim=1)
                    negative_embeds = torch.cat((negative_embeds, neg_embed), dim=0)
                
            elif self.num_negative == 0:
                negative_embeds = None
            else:
                raise ValueError("Invalid number of negative samples")
            
            positive_embeds = self.model.get_text_embeds(positive_name)
            positive_embeds = torch.mean(positive_embeds, dim=1)

            best_delta = None
            best_loss = float("inf")
            margin = 0.9
            for _ in tqdm(range(iters), desc="Attacking"):
                # get CLIP loss 
                image_features = self.model.get_image_features(pixel_values + delta).to(self.device)
                patch_features = image_features[:, token_indices, :]
                patch_features = patch_features.mean(dim=1)

                positive_similarity = F.cosine_similarity(patch_features, positive_embeds, dim=-1)
                if self.num_negative == 1:
                    negative_similarity = F.cosine_similarity(patch_features, negative_embeds, dim=-1)
                    loss = F.relu(negative_similarity - positive_similarity + margin).mean()
                elif self.num_negative == 2:
                    patch_features_expanded = patch_features.unsqueeze(1)
                    negative_embeds_expanded = negative_embeds.unsqueeze(0)
                    negative_similarities = F.cosine_similarity(patch_features_expanded, negative_embeds_expanded, dim=2)
                    loss = F.relu(negative_similarities - positive_similarity.unsqueeze(1) + margin).mean()
                elif self.num_negative == 0:
                    loss = F.relu(margin - positive_similarity).mean()

                # get the gradients
                delta.grad = torch.autograd.grad(loss, delta)[0]
                with torch.no_grad():
                    delta.grad.sign_()
                    delta.grad = delta.grad * mask
                    delta.data = delta.data - alpha * delta.grad
                    delta.data.clamp_(-epsilon, epsilon)
                    delta.grad.zero_()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_delta = delta.detach().clone()

            adv_pixel_values = pixel_values + best_delta
            adv_image = tensor_to_image(adv_pixel_values[0])

            image_save_dir = Path(save_path) / "adv_images"
            os.makedirs(image_save_dir, exist_ok=True)
            image_save_path = image_save_dir / f"{str(img_id.item()).zfill(12)}.png"
            adv_image.save(image_save_path)
            print(f"Saved adversarial image to {image_save_path}")
            return True, None
        except Exception as e:
            error_msg = f"Error processing image {img_id}: {str(e)}\n{traceback.format_exc()}"
            return False, error_msg

    def train(self, output_dir):
        """
        Execute training with error handling and progress tracking
        """
        failed_images = []
        processed_images = []

        # set random seed
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        for i, sample in enumerate(self.dataloader):
            try:
                img_path = sample["image_path"][0]
                img = Image.open(img_path)
                label = sample["label"]
                img_id = label["image_id"]

                success, error_msg = self.attack(
                    img=img,
                    img_id=img_id,
                    label=label,
                    alpha=self.alpha,
                    epsilon=self.epsilon,
                    iters=self.iters,
                    do_normalize=self.do_normalize,
                    save_path=output_dir
                )

                if success:
                    processed_images.append(img_id.item())
                    print(f"GPU {self.gpu_id}: Successfully processed image {i + 1}/{len(self.dataset)} (Image ID: {img_id})")
                else:
                    failed_images.append({"image_id": img_id.item(), "error": error_msg})
                    print(f"GPU {self.gpu_id}: Failed to process image {img_id}: {error_msg}")

            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
                failed_images.append({"image_id": img_id.item(), "error": error_msg})
                print(f"GPU {self.gpu_id}: {error_msg}")

            # Save progress after each image
            self._save_progress(output_dir, processed_images, failed_images)

        return processed_images, failed_images

    def _save_progress(self, output_dir, processed_images, failed_images):
        """
        Save progress information to files
        """
        progress_dir = output_dir / f"gpu_{self.gpu_id}_progress"
        progress_dir.mkdir(exist_ok=True)

        # Save processed images
        with open(progress_dir / "processed_images.json", 'w') as f:
            json.dump(processed_images, f)

        # Save failed images
        with open(progress_dir / "failed_images.json", 'w') as f:
            json.dump(failed_images, f)

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for attack model')
    
    # Dataset arguments
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument('--images_dir', type=str, default='datasets/CrossVLAD/transforming/images',
                           help='Path to images directory')
    data_group.add_argument('--labels_dir', type=str, default='datasets/CrossVLAD/transforming/labels',
                           help='Path to labels directory')
    data_group.add_argument('--sample_nums', type=int, default=1000,
                           help='Number of samples to process')
    
    # Model arguments
    model_group = parser.add_argument_group('Model Configuration') 
    model_group.add_argument('--model_name', type=str, default='florence2',
                            help='Name of model')
    model_group.add_argument('--num_gpus', type=int, default=2,
                            help='Number of GPUs to use')
    
    # Attack arguments
    attack_group = parser.add_argument_group('Attack Configuration')
    attack_group.add_argument('--alpha', type=float, default=0.00392156862745098,
                             help='Attack step size')
    attack_group.add_argument('--epsilon', type=float, default=0.06274509803921569,
                             help='Maximum perturbation')
    attack_group.add_argument('--iters', type=int, default=300,
                             help='Number of attack iterations')
    attack_group.add_argument('--mask', type=lambda x: x.lower() == 'true', default=True,
                             help='Whether to use mask')
    attack_group.add_argument('--num_negative', type=int, default=1,
                             help='Number of negative samples')
    
    # Processing arguments
    proc_group = parser.add_argument_group('Processing Configuration')
    proc_group.add_argument('--do_normalize', type=lambda x: x.lower() == 'true', default=True,
                           help='Whether to normalize images')
    
    # Output arguments
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--output_base_dir', type=str, default='output',
                             help='Base output directory')

    return parser.parse_args()

def setup_gpu_process(rank, world_size, dataset, args, output_dir, sample_indices):
    try:
        torch.cuda.set_device(rank)

        model_name = args.model_name
        module = importlib.import_module(f"models.{model_name}")
        model = load_model(f"cuda:{rank}", module, model_name=model_name)

        samples_per_gpu = len(sample_indices) // world_size
        start_idx = rank * samples_per_gpu
        end_idx = start_idx + samples_per_gpu if rank != world_size - 1 else len(sample_indices)

        trainer = Trainer(
            model, dataset, rank,
            alpha=args.alpha,
            epsilon=args.epsilon,
            sample_nums=args.sample_nums,
            iters=args.iters,
            do_normalize=args.do_normalize,
            mask=args.mask,
            num_negative=args.num_negative,
            model_name=model_name
        )

        trainer.setup_data(sample_indices, start_idx, end_idx)
        return trainer.train(output_dir=output_dir)

    except Exception as e:
        print(f"Error in GPU {rank}: {str(e)}\n{traceback.format_exc()}")
        return [], [{"error": str(e)}]

def main():
    args = parse_args()
    
    # Set up dataset
    dataset = CrossVLAD_Dataset(images_dir=args.images_dir, labels_dir=args.labels_dir)

    # Generate and save random indices
    all_indices = np.random.choice(len(dataset), args.sample_nums, replace=False)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    output_dir = Path(args.output_base_dir) / f"craft_{args.model_name}_mask_{args.mask}_neg_{args.num_negative}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save selected indices
    np.save(output_dir / "selected_indices.npy", all_indices)

    # Initialize multiprocessing
    mp.spawn(
        setup_gpu_process,
        args=(args.num_gpus, dataset, args, output_dir, all_indices),
        nprocs=args.num_gpus,
        join=True
    )

    # Save script
    os.system(f"cp {__file__} {output_dir}/")
    # Save arguments
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f)

if __name__ == "__main__":
    main()