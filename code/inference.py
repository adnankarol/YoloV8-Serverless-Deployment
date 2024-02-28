__author__      = "Adnan Karol"
__version__ = "1.0.1"
__maintainer__ = "Adnan Karol"
__email__ = "adnanmushtaq5@gmail.com"
__status__ = "Dev"


# Import Dependencies
import numpy as np
import torch, os, json, io, cv2, time
from ultralytics import YOLO


def model_fn(model_dir):
    model = YOLO('yolov8l.pt')
    print("Loaded the YoloV8 Model !!!")
    return model


def input_fn(request_body, request_content_type):
    if request_content_type:
        jpg_original = np.load(io.BytesIO(request_body),
                               allow_pickle=True)
        jpg_as_np = np.frombuffer(jpg_original, 
		                          dtype=np.uint8)
        img = cv2.imdecode(jpg_as_np, flags=-1)
    else:
        raise Exception("Unsupported content type: " + request_content_type)
    return img


def predict_fn(input_data, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        result = model(input_data)
    return result


def output_fn(prediction_output, content_type):
    infer = {}
    for result in prediction_output:
        if 'boxes' in result._keys and result.boxes is not None:
            infer['boxes'] = result.boxes.numpy().data.tolist()
        if 'masks' in result._keys and result.masks is not None:
            infer['masks'] = result.masks.numpy().data.tolist()
        if 'keypoints' in result._keys and result.keypoints is not None:
            infer['keypoints'] = result.keypoints.numpy().data.tolist()
        if 'probs' in result._keys and result.probs is not None:
            infer['probs'] = result.probs.numpy().data.tolist()
    
    return json.dumps(infer)