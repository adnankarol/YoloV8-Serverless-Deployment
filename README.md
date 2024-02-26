# Deploying YOLOv8 Model on Amazon SageMaker Endpoints and Integrating with AWS Lambda and API Gateway

This repository contains instructions and code for hosting YOLOv8 PyTorch models on Amazon SageMaker Endpoints, integrated with AWS Lambda and API Gateway. The YOLOv8 model, renowned for its efficiency and accuracy in object detection, is deployed using SageMaker endpoints, complemented by AWS Lambda for serverless processing and API Gateway for creating a scalable and accessible RESTful API. This setup provides a comprehensive solution for deploying machine learning models in production environments.

# Solution Overview

The solution leverages AWS services and follows the steps outlined below:

1. **GitHub Repository**: Two notebooks (`Notebooks/Model_Deploy.ipynb` and `Notebooks/Test_Endpoint.ipynb`) are provided in the `Notebooks/` directory of the GitHub repository.
2. **SageMaker Notebook**: Downloads the YOLOv8 PyTorch model and stores it along with custom inference code in an Amazon Simple Storage Service (Amazon S3) bucket.
3. **SageMaker Endpoint**: Hosts the YOLOv8 PyTorch model and the custom inference code.
4. **Testing Endpoint**: Utilizes `Notebooks/Test_Endpoint.ipynb` to test the endpoint and gather results.

# Prerequisites

Ensure you have the following:

1. **Permissions**: 
   - AWS Account with IAM roles providing access to Amazon SageMaker, Amazon Lambda, Amazon API Gateway, and Amazon S3.

2. **Create an S3 Bucket**: 
   - Create an Amazon S3 bucket where you will upload the model artifacts (`model.tar.gz`) and any other necessary files.
   - Ensure that the bucket has proper permissions configured to allow access from the SageMaker service.

3. **Create a SageMaker Notebook Instance**:
   - Set up a SageMaker notebook instance where you will run the provided notebooks.
   - Assign an IAM role to the notebook instance with the necessary permissions to access SageMaker, S3, and any other AWS services required for your workflow.
   - Ensure the notebook instance has internet access to download dependencies and access AWS services.

Ensure these prerequisites are fulfilled before proceeding with the deployment process. Detailed instructions for fulfilling these prerequisites can be found in the AWS documentation for Amazon S3 and Amazon SageMaker.

# Running the Setup Bash Script

To set up the environment for deploying YOLOv8 models, follow these steps to run the provided bash script:

1. **Open Terminal**: Open a terminal window on your local machine.

2. **Navigate to Repository**: Use the `cd` command to navigate to the directory where the repository containing the setup bash script is located.

3. **Make Bash Script Executable (if needed)**: If the bash script does not have execute permissions, you can make it executable using the following command:

    ```bash
    chmod +x setup_environment.sh
    ```

4. **Run the Bash Script**: Execute the bash script by running the following command:

    ```
    bash ./set_environment.sh
    ```

5. **Installation of Dependencies**: The script will then install the Python dependencies specified in the `requirements.txt` file using pip.

6. **Deactivation**: Once the setup is complete, the virtual environment will be deactivated automatically.

7. **Completion Message**: You will receive a message indicating that the setup process has finished successfully.

8. **Start your Environment**: Use the command below to activate your virtual environment:
    ```
    conda activate yolo
    ```


By running the bash script, you will create the necessary environment for deploying and testing your YOLOv8 PyTorch models on Amazon SageMaker Endpoints. Ensure you have fulfilled the prerequisites and follow any additional instructions provided during the script execution.

# Steps to Host YOLOv8 on a SageMaker Endpoint

1. **Custom Inference Code**: The `model/code/inference.py` file contains functions for model loading, input parsing, inference, and output processing. Modify these functions according to your pipeline and workflow.

2. **Create `model.tar.gz`**: Create a tarball containing the model and inference code:

    ```
    $ tar -czvf model.tar.gz code/ yolov8l.pt
    ```

3. **Host `model.tar.gz` to SageMaker Endpoint**: Upload the `model.tar.gz` to an S3 bucket, then create a SageMaker PyTorchModel and deploy it to an endpoint.

## Testing the SageMaker Endpoint

Once the endpoint is hosted, follow the steps in `Notebooks/Test_Endpoint.ipynb` to run inference and gather results. The output can be plotted accordingly.


# Creating Lambda Function

To integrate your YOLOv8 PyTorch model with API Gateway, follow these steps to create a Lambda function:

- **Navigate to Lambda Console**: Open the AWS Lambda console in your browser.

- **Click "Create function"**: Start by clicking on the "Create function" button.

- **Choose Author from Scratch**: Select "Author from scratch" as the function blueprint.

- **Configure Basic Settings**: Enter a name for your Lambda function and select the runtime (e.g., Python 3.8).

- **Choose or Create an Execution Role**: Either choose an existing role with appropriate permissions or create a new role. This role will allow the Lambda function to interact with other AWS services like SageMaker.

- **Write or Upload Code**: Write your Lambda function code or upload a ZIP file containing your code. A sample code is stored at `src/lambda.py`.

- **Configure Triggers (Optional)**: If you haven't configured API Gateway as a trigger during function creation, you can add it later in the function configuration.

- **Save and Deploy**: Once your Lambda function is configured, click on "Save" and then "Deploy" to deploy the function.

# Create an API Gateway

1. **Create an API Gateway**: Navigate to the API Gateway console and follow the steps to create a new REST API.
2. **Integrate with Lambda**: Add a resource and method (e.g., POST) to your API, and integrate it with the Lambda function that invokes the SageMaker endpoint.
3. **Deploy API**: Deploy your API to a stage to make it accessible via an endpoint URL.

## Test your API

1. Use the provided endpoint URL to make HTTP requests to your API, passing input data as needed.
2. Monitor the execution and performance of your API using AWS CloudWatch logs and metrics.

# Clean Up

Remember to delete the endpoint and its configuration and other created resources to save costs.

# Next Steps

Consider automating the deployment process using AWS CloudFormation templates. This can streamline resource provisioning and ensure consistency across deployments.

# Conclusion

This repository provides a comprehensive guide on hosting pre-trained YOLOv8 PyTorch models on Amazon SageMaker endpoints, integrating them with AWS Lambda and API Gateway for seamless deployment and inference. By following the instructions and code provided, you can deploy your YOLOv8 model in a scalable and cost-effective manner, leveraging the power of serverless computing and RESTful APIs.

For further enhancements, explore features such as authentication, caching, and custom domain names in API Gateway, and optimize Lambda function performance using CloudWatch monitoring.

For additional reference, you can also refer to the article [Hosting YOLOv8 PyTorch Model on Amazon SageMaker Endpoints](https://aws.amazon.com/blogs/machine-learning/hosting-yolov8-pytorch-model-on-amazon-sagemaker-endpoints/). 

Refer to the GitHub repository for detailed code examples and instructions. Delve deeper into SageMaker endpoints, Lambda functions, and API Gateway by consulting the respective AWS documentation for insights and best practices. Additionally, explore CloudFormation support for SageMaker to automate the deployment process and streamline your machine learning workflow.

# Contact

In case of any questions, feel free to contact me over Linkedin at [Adnan](https://www.linkedin.com/in/adnan-karol-aa1666179/).