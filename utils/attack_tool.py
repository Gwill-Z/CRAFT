import os
import json
import random
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from PIL import Image
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def load_florence2_model(device,module):
    model_args = {
        "model_path": "Florence-2-large",
        "processor_path": "Florence-2-large",
        "device": device
    }
    eval_model = module.EvalModel(model_args)
    return eval_model

def load_uio2_model(device, module):
    model_args = {
        "model_path": "allenai/uio2-large",
        "processor_path": "unifiedio2/uio2-preprocessor",
        "tokenizer": "unifiedio2/tokenizer.model",
        "device": device
    }
    eval_model = module.EvalModel(model_args)
    return eval_model

def load_gpt_4v_model(module):
    eval_model = module.EvalModel()
    return eval_model

def load_model(device, module,model_name):
    if model_name == "florence2":
        return load_florence2_model(device,module)
    elif model_name == "uio2":
        return load_uio2_model(device, module)
    elif model_name == "gpt4v":
        return load_gpt_4v_model(module)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
