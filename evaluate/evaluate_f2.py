import os
import json
import sys
import numpy as np
import torch
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.attack_tool import load_model
from dataset import CrossVLAD_Dataset
from tqdm import tqdm
import logging
from pathlib import Path
import importlib
import argparse
from datetime import datetime

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def setup_logging(output_dir):
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    log_save_path = output_dir / "evaluate.log"
    file_handler = logging.FileHandler(log_save_path)
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    tqdm_handler = TqdmLogger()
    tqdm_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.addHandler(tqdm_handler)
    
    return logger

class TqdmLogger(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)

def preprocess_image_with_noise(model, image_path):
    img = Image.open(image_path)
    inputs_x = model.processor.image_processor(images=img, do_normalize=True, return_tensors="pt")["pixel_values"].to(device, torch_dtype)

    adversarial_image = inputs_x
    return inputs_x, adversarial_image

def get_input_ids(model, image_path, task_prompt, text=None):
    img = Image.open(image_path)
    processor = model.processor
    if text is not None:
        input_ids = processor(text=task_prompt + text, images=img, return_tensors="pt", do_normalize=True).to(device, torch_dtype)["input_ids"]
    else:
        input_ids = processor(text=task_prompt, images=img, return_tensors="pt", do_normalize=True).to(device, torch_dtype)["input_ids"]
    return input_ids

def calculate_iou(box1, box2):
            x1_min, y1_min, x1_max, y1_max = box1
            x2, y2, w2, h2 = box2
            
            # 计算box2的右下角坐标
            x2_max = x2 + w2
            y2_max = y2 + h2
            
            # 计算交集的坐标
            xA = max(x1_min, x2)
            yA = max(y1_min, y2)
            xB = min(x1_max, x2_max)
            yB = min(y1_max, y2_max)
            
            # 计算交集的面积
            interArea = max(0, xB - xA) * max(0, yB - yA)
            
            # 计算box1的面积
            box1Area = (x1_max - x1_min) * (y1_max - y1_min)
            
            # 计算box2的面积
            box2Area = w2 * h2
            
            # 计算IoU
            iou = interArea / float(box1Area + box2Area - interArea)
            return iou

class Evaluator:
    def __init__(self, model):
        self.model = model
        self.processor = model.processor

    def is_successful_on_caption(self, label, input_ids, pixel_values):
        raise NotImplementedError

    def is_successful_on_od(self, label, input_ids, pixel_values, inputs_x, img_size):
        raise NotImplementedError

    def is_successful_on_region_to_category(self, label, input_ids, pixel_values, image_path):
        raise NotImplementedError

    def is_successful_on_open_vocabulary_detection(self, label, input_ids, pixel_values, img_size):
        raise NotImplementedError

    def get_open_vocabulary_text(self, label):
        raise NotImplementedError

class TargetChangeEvaluator(Evaluator):
    def is_successful_on_caption(self, label, input_ids, pixel_values, inputs_x):
        caption_original_result = self.model.model.generate(
            input_ids=input_ids,
            pixel_values=inputs_x,
            max_length=128,
            num_beams=3,
            do_sample=False,
        )
        caption_result = self.model.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_length=128,
            num_beams=3,
            do_sample=False,
        )
        caption_result = self.processor.batch_decode(caption_result, skip_special_tokens=True)[0].lower()
        caption_original_result = self.processor.batch_decode(caption_original_result, skip_special_tokens=True)[0].lower()
        original_category = label["source_object_category"].lower()
        change_category = label["target_object_category"].lower()

        attack_success = change_category in caption_result and original_category not in caption_result
        no_attack_success = change_category in caption_original_result and original_category not in caption_original_result
        return attack_success, no_attack_success
    
    def is_successful_on_od(self, label, input_ids, pixel_values, inputs_x, img_size):
        od_original_result = self.model.model.generate(
            input_ids=input_ids,
            pixel_values=inputs_x,
            max_length=128,
            num_beams=3,
            do_sample=False,
        )
        
        od_result = self.model.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_length=128,
            num_beams=3,
            do_sample=False,
        )

        od_original_result = self.processor.batch_decode(od_original_result, skip_special_tokens=False)[0]
        od_result = self.processor.batch_decode(od_result, skip_special_tokens=False)[0]
        
        post_od_original_result = self.processor.post_process_generation(od_original_result, task="<OD>", image_size=img_size)
        post_od_result = self.processor.post_process_generation(od_result, task="<OD>", image_size=img_size)
        
        original_detected_labels = post_od_original_result["<OD>"]['labels']
        original_detected_boxes = post_od_original_result["<OD>"]['bboxes']
        detected_labels = post_od_result["<OD>"]['labels']
        detected_boxes = post_od_result["<OD>"]['bboxes']
        
        original_category = label["source_object_category"]
        change_category = label["target_object_category"]
        source_object_boxes = label["source_object_boxes"][0]

        
        # evaluate no attack
        if original_category not in original_detected_labels and change_category in original_detected_labels:
            change_category_index = original_detected_labels.index(change_category)
            change_box = original_detected_boxes[change_category_index]
            iou = calculate_iou(change_box, source_object_boxes)
            if iou >= 0.6:
                no_attack_success = True
            else:
                no_attack_success = False
        else:
            no_attack_success = False

        # evaluate attack
        if original_category not in detected_labels and change_category in detected_labels:
            change_category_index = detected_labels.index(change_category)
            change_box = detected_boxes[change_category_index]
            iou = calculate_iou(change_box, source_object_boxes)
            if iou >= 0.6:
                attack_success = True
            else:
                attack_success = False
        else:
            attack_success = False
        
        return attack_success, no_attack_success

    def is_successful_on_region_to_category(self, label, input_ids, pixel_values, image_path, inputs_x):
        region_to_category_original_result = self.model.model.generate(
            input_ids=input_ids,
            pixel_values=inputs_x,
            max_length=128,
            num_beams=3,
            do_sample=False,
        )
        region_to_category_result = self.model.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_length=128,
            num_beams=3,
            do_sample=False,
        )
        region_to_category_result = self.processor.batch_decode(region_to_category_result, skip_special_tokens=True)[0]
        region_to_category_original_result = self.processor.batch_decode(region_to_category_original_result, skip_special_tokens=True)[0]
        change_category = label["target_object_category"]
    
        attack_success = change_category in region_to_category_result
        no_attack_success = change_category in region_to_category_original_result
        return attack_success, no_attack_success

    def is_successful_on_open_vocabulary_detection(self, label, input_ids, pixel_values, img_size, inputs_x, is_target=True):
        open_vocabulary_detection_original_result = self.model.model.generate(
            input_ids=input_ids,
            pixel_values=inputs_x,
            max_length=128,
            num_beams=3,
            do_sample=False,
        )
        open_vocabulary_detection_result = self.model.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_length=128,
            num_beams=3,
            do_sample=False,
        )
        open_vocabulary_detection_result = self.processor.batch_decode(open_vocabulary_detection_result, skip_special_tokens=False)[0]
        open_vocabulary_detection_result = self.processor.post_process_generation(
            open_vocabulary_detection_result, task="<OPEN_VOCABULARY_DETECTION>", image_size=img_size)
        
        open_vocabulary_detection_original_result = self.processor.batch_decode(open_vocabulary_detection_original_result, skip_special_tokens=False)[0]
        open_vocabulary_detection_original_result = self.processor.post_process_generation(
            open_vocabulary_detection_original_result, task="<OPEN_VOCABULARY_DETECTION>", image_size=img_size)
        
        attack_success = False
        no_attack_success = False
        true_box = label["source_object_boxes"][0]

        try:
            predict_box = open_vocabulary_detection_result["<OPEN_VOCABULARY_DETECTION>"]["bboxes"][0]
            iou = calculate_iou(predict_box, true_box)
            if is_target:
                if iou >= 0.6:
                    attack_success = True
            else:
                if iou < 0.6:
                    attack_success = True
        except Exception as e:
            print(open_vocabulary_detection_result)
            print(e)
            print(label["image_id"])
            attack_success = False

        try:
            predict_box_original = open_vocabulary_detection_original_result["<OPEN_VOCABULARY_DETECTION>"]["bboxes"][0]
            iou_original = calculate_iou(predict_box_original, true_box)
            if is_target:
                if iou_original >= 0.6:
                    no_attack_success = True
            else:
                if iou_original < 0.6:
                    no_attack_success = True
            
        except Exception as e:
            print(open_vocabulary_detection_original_result)
            print(e)
            print(label["image_id"])
            no_attack_success = False

        return attack_success, no_attack_success

    def get_open_vocabulary_text(self, label):
        return label["target_object_category"]

def evaluate_attack(model, output_dir, IMAGE_DIR, LABELS_DIR):
    logger = setup_logging(output_dir)
    logger.info(f"Starting evaluation in directory: {output_dir}")
    month_and_day = datetime.now().strftime("%m%d")
    ADV_IMAGE = output_dir / "adv_images"
    OUTPUT_JSON = output_dir / f"evaluate_result_{month_and_day}.json"

    evaluator = TargetChangeEvaluator(model)
   
    results = {
        "caption_success": 0,
        "od_success": 0,
        "region_to_category_success": 0,
        "open_vocabulary_detection_success": 0,
        "caption_no_attack_success": 0,
        "od_no_attack_success": 0,
        "region_to_category_no_attack_success": 0,
        "open_vocabulary_detection_no_attack_success": 0
    }
    total_samples = 0

    # Load dataset
    dataset = CrossVLAD_Dataset(images_dir=IMAGE_DIR, labels_dir=LABELS_DIR)

    cnt_4 = 0
    cnt_3 = 0
    no_attack_cnt4 = 0
    no_attack_cnt3 = 0
    for sample in tqdm(dataset, desc=f"Evaluating"):
        img_id = sample["label"]["image_id"]
        label = sample["label"]

        success_nums = 0
        no_attack_success_nums = 0

        # Load original image and adv image
        img_id_str = str(int(img_id)).zfill(12)
        image_path = Path(IMAGE_DIR) / f"{img_id_str}.jpg"
        adv_image_path = ADV_IMAGE / f"{img_id_str}.png"
    
        if not image_path.exists() or not adv_image_path.exists():
            continue
        
        img = Image.open(image_path)
        adv_image = Image.open(adv_image_path)
        inputs_x = model.processor.image_processor(images=img, do_normalize=True, return_tensors="pt")["pixel_values"].to(device, torch_dtype)
        adversarial_image = model.processor.image_processor(images=adv_image, do_normalize=True, return_tensors="pt")["pixel_values"].to(device, torch_dtype)

        # Caption task
        task_prompt = "<CAPTION>"
        input_ids = get_input_ids(model, str(image_path), task_prompt)
        attack_success, no_attack_success = evaluator.is_successful_on_caption(label, input_ids, adversarial_image, inputs_x)
        if attack_success:
            results["caption_success"] += 1
            success_nums += 1
        if no_attack_success:
            results["caption_no_attack_success"] += 1
            no_attack_success_nums += 1

        # OD task
        task_prompt = "<OD>"
        img = Image.open(str(image_path))
        img_size = (img.width, img.height)
        input_ids = get_input_ids(model, str(image_path), task_prompt)

        attack_success, no_attack_success = evaluator.is_successful_on_od(label, input_ids, adversarial_image, inputs_x, img_size)
        if attack_success:
            results["od_success"] += 1
            success_nums += 1
        if no_attack_success:
            results["od_no_attack_success"] += 1
            no_attack_success_nums += 1

        # Region_to_Category task
        img_size = img.size
        img_size_tensor = torch.tensor(img_size).to(device, torch_dtype)
        bbox = label["source_object_boxes"][0] # x, y, w, h
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        box_tensor = torch.tensor(bbox).to(device, torch_dtype)
        quad_box = model.processor.post_processor.box_quantizer.quantize(box_tensor, img_size_tensor)
        text = "".join([f"<loc_{coord}>" for coord in quad_box.tolist()])
        task_prompt = "<REGION_TO_CATEGORY>"
        input_ids = get_input_ids(model, str(image_path), task_prompt, text)

        attack_success, no_attack_success = evaluator.is_successful_on_region_to_category(label, input_ids, adversarial_image, image_path, inputs_x)
        if attack_success:
            results["region_to_category_success"] += 1
            success_nums += 1
        if no_attack_success:
            results["region_to_category_no_attack_success"] += 1
            no_attack_success_nums += 1

        # Open Vocabulary Detection task
        task_prompt = "<OPEN_VOCABULARY_DETECTION>"

        target_text = evaluator.get_open_vocabulary_text(label)
        target_input_ids = get_input_ids(model, str(image_path), task_prompt, target_text)
        origin_text = label["source_object_category"]
        origin_input_ids = get_input_ids(model, str(image_path), task_prompt, origin_text)

        target_attack_success, target_no_attack_success = evaluator.is_successful_on_open_vocabulary_detection(label, target_input_ids, adversarial_image, img_size, inputs_x, is_target=True)
        origin_attack_success, origin_no_attack_success = evaluator.is_successful_on_open_vocabulary_detection(label, origin_input_ids, inputs_x, img_size, inputs_x, is_target=False)
        if target_attack_success :
            results["open_vocabulary_detection_success"] += 1
            success_nums += 1
        if target_no_attack_success and origin_no_attack_success:
            results["open_vocabulary_detection_no_attack_success"] += 1
            no_attack_success_nums += 1


        total_samples += 1
        
        if success_nums == 4:
            cnt_4 += 1

        if success_nums >= 3:
            cnt_3 += 1

        if no_attack_success_nums == 4:
            no_attack_cnt4 += 1
        
        if no_attack_success_nums >= 3:
            no_attack_cnt3 += 1
      

    # Compute attack success rate
    no_attack_sum = 0
    attack_sum = 0
    if total_samples > 0:
        for key in results.keys():
            results[key] = results[key] / total_samples
            if "no_attack" in key:
                no_attack_sum += results[key]
            else:
                attack_sum += results[key]

    # avg results of no attack
    results["no_attack_avg"] = no_attack_sum / 4
    # avg results of attack
    results["attack_avg"] = attack_sum / 4
        
    results["CTSR_4"] = cnt_4 / total_samples
    results["CTSR_3"] = cnt_3 / total_samples
    results["no_attack_CTSR_4"] = no_attack_cnt4 / total_samples
    results["no_attack_CTSR_3"] = no_attack_cnt3 / total_samples

    # Save results to JSON file
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)

    print("Evaluation complete. Results saved to", OUTPUT_JSON)

def main():
    parser = argparse.ArgumentParser(description='Evaluate attack results')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory path')
    parser.add_argument('--IMAGE_DIR', type=str, required=True,
                        help='Path to the directory containing images')
    parser.add_argument('--LABELS_DIR', type=str, required=True,
                        help='Path to the directory containing labels')
    
    args = parser.parse_args()
    
    # Convert output_dir to Path
    output_dir = Path(args.output_dir)
    
    # Load the model
    model_name = "florence2"
    module = importlib.import_module("models.florence2")
    model = load_model(device, module, model_name)
    
    # Run evaluation
    evaluate_attack(model, output_dir, args.IMAGE_DIR, args.LABELS_DIR)

if __name__ == "__main__":
    main()
