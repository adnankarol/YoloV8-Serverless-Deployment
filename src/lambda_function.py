__author__      = "Adnan Karol"
__version__ = "1.0.1"
__maintainer__ = "Adnan Karol"
__email__ = "adnanmushtaq5@gmail.com"
__status__ = "Dev"


# Import Dependencies
import json
import boto3
import base64
import numpy as np
import cv2
from sagemaker.pytorch import PyTorchPredictor
from sagemaker.deserializers import JSONDeserializer

# Initialize the SageMaker client
sagemaker_client = boto3.client('sagemaker-runtime')
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    # Read the image data from the specified S3 object
    response = s3_client.get_object(Bucket='playgrounddatascience', Key='cat.jpg')
    image_data = response['Body'].read()
    
    # Decode the image data using OpenCV
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Resize image
    model_height, model_width = 300, 300
    resized_image = cv2.resize(img, (model_height, model_width))
    
    # Prepare payload for SageMaker endpoint
    payload = cv2.imencode('.jpg', resized_image)[1].tobytes()
    
    # Specify the endpoint name
    endpoint_name = 'yolo-endpoint'
    
    # Initialize PyTorchPredictor
    predictor = PyTorchPredictor(endpoint_name=endpoint_name, deserializer=JSONDeserializer())
    
    # Predict using SageMaker endpoint
    result = predictor.predict(payload)
    
    # Print the result
    print(result)
    
    # Return the result as the Lambda response
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
