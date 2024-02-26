__author__      = "Adnan Karol"
__version__ = "1.0.1"
__maintainer__ = "Adnan Karol"
__email__ = "adnanmushtaq5@gmail.com"
__status__ = "Dev"


# Import Dependencies
import numpy as np
import torch
import cv2
import os
import json
from ultralytics import YOLO


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "yolov8l.pt")
    model = YOLO(model_path)
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == "image/jpeg":
        jpg_as_np = np.frombuffer(request_body, dtype=np.uint8)
        img = cv2.imdecode(jpg_as_np, flags=-1)
        return img
    else:
        raise Exception("Unsupported content type: " + request_content_type)
    

def predict_fn(input_data, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        result = model(input_data)
    return result


def output_fn(prediction_output, accept):
    infer = {}
    for result in prediction_output:
        if result.boxes:
            infer['boxes'] = result.boxes.cpu().numpy().tolist()
        if result.masks:
            infer['masks'] = result.masks.cpu().numpy().tolist()
        if result.probs:
            infer['probs'] = result.probs.cpu().numpy().tolist()
    return json.dumps(infer)

