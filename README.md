# Malware Detection using Deep Learning

This project implements a deep learning model to classify benign and malware files using TensorFlow and Scikit-learn. The project includes a Dockerfile to containerize the environment, making it easier to deploy and run the model on cloud platforms such as Azure.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Dockerization](#dockerization)
- [Azure Deployment](#azure-deployment)
- [Prerequisites](#prerequisites)
- [Results](#results)
- [Contributing](#contributing)

## Overview
The project involves building a deep learning model to detect malware based on system call metrics. The model is trained using a dataset of system metrics and leverages TensorFlow to implement a neural network. 

The model's workflow includes:
1. Data loading and preprocessing.
2. Model creation and training.
3. Evaluation and results.
4. Containerization using Docker for easy deployment.

## Dataset
The dataset used in this project contains system metrics labeled as benign or malware and was fetched from Kaggle. The key columns used for classification include various performance metrics collected from processes. The dataset is loaded in the script and preprocessed using pandas and scikit-learn.

## Model Architecture
- Input size: 27 features
- Hidden layers: 6 layers with 50 neurons each, using the ReLU activation function.
- Output layer: 2 neurons with softmax activation for binary classification.

The model is trained with sparse categorical crossentropy loss and evaluated based on accuracy.

## Project Structure
|-- Dockerfile |-- requirements.txt |-- mal_det_dl.py |-- Malware dataset.csv
- **Dockerfile**: Contains the configuration to set up the Python environment and dependencies for running the project inside a container.
- **mal_det_dl.py**: Python script containing the deep learning model for malware detection.
- **requirements.txt**: List of required Python packages.

## Dockerization
The Docker image for this project contains a Python 3.9 environment and the necessary dependencies (numpy, pandas, scikit-learn, and tensorflow). The Dockerfile is provided to build and run the project inside a containerized environment.

### Build the Docker Image
```bash
docker build -t malware-detector .
```

## Azure Deployment
This project was temporarily deployed on Azure using a Docker container. The Docker image was built and pushed to the Azure Container Registry, and the container was hosted on an Azure Container Instance.

**Steps to Deploy on Azure:**
**Create Azure Container Registry (ACR):**

1.Navigate to Azure portal and create a new Container Registry.

2.Push your local Docker image to ACR using:
```bash
docker tag malware-detector <acr-name>.azurecr.io/malware-detector:v1
docker push <acr-name>.azurecr.io/malware-detector:v1
```

**Deploy to Azure Container Instance (ACI):**

1.Create a new Azure Container Instance and pull the image from ACR.

2.Use the following Azure CLI command to deploy:
```bash
az container create --resource-group <resource-group> --name malware-detector --image <acr-name>.azurecr.io/malware-detector:v1 --registry-login-server <acr-name>.azurecr.io --registry-username <acr-username> --registry-password <acr-password> --dns-name-label malware-detector --ports 80
```
The container instance will be deployed and can be accessed via the DNS name provided.

## Prerequisites
- Docker
- Python 3.9
- TensorFlow
- Azure CLI (for deployment)

## Installation
Clone the repository:
```bash
git clone https://github.com/harshjoshi-2003/Docker-Image-Of-DL-Model.git
```
Build the Docker image:
```bash
docker build -t malware-detector .
```
Install the required Python packages (if running locally):
```bash
pip install -r requirements.txt
```

## Usage
To run the model inside Docker:
```bash
docker run -it malware-detector
```
Alternatively, if running locally, execute the Python script:
```bash
python mal_det_dl.py
```
## Results
The model achieved satisfactory accuracy in distinguishing between benign and malware files. Performance metrics can be printed after evaluation as shown in the script.

## Contributing
Feel free to fork this repository, open issues, or submit pull requests with any enhancements or bug fixes.
