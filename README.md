# Customer Churn Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![Jenkins](https://img.shields.io/badge/Jenkins-CI%2FCD-red)
![GitHub](https://img.shields.io/badge/GitHub-Version%20Control-black)

---

## Project Overview

Customer churn prediction is a Machine Learning project that predicts whether a customer is likely to leave a telecom service based on customer demographics, account information, and service usage.

The project demonstrates an end-to-end ML workflow, including data preprocessing, feature engineering, model training, model deployment using Streamlit, Docker containerization, and CI/CD automation using Jenkins.

---

## Features

- Customer churn prediction using Machine Learning
- Data preprocessing and feature engineering
- Interactive Streamlit web application
- Model serialization using Pickle
- Docker containerization
- Jenkins CI/CD pipeline
- GitHub version control
- Responsive prediction interface

---

## Tech Stack

### Programming Language
- Python

### Machine Learning
- Scikit-learn
- Pandas
- NumPy

### Data Visualization
- Matplotlib
- Seaborn

### Web Application
- Streamlit

### DevOps
- Docker
- Jenkins
- Git
- GitHub

---

## Project Structure

```
customer-churn-prediction/
│
├── app/
│   ├── app.py
│   ├── predict.py
│   └── preprocessing.py
│
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── encoder.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_training.ipynb
│
├── src/
│
├── Dockerfile
├── Jenkinsfile
├── requirements.txt
├── README.md
└── .streamlit/
```

---

## Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Data Preprocessing
6. Model Training
7. Model Evaluation
8. Model Serialization
9. Streamlit Deployment
10. Docker Containerization
11. Jenkins CI/CD Pipeline

---

## Installation

Clone the repository

```bash
git clone https://github.com/sanjanahp16/Customer-Churn-Prediction.git
```

Navigate to the project

```bash
cd Customer-Churn-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app/app.py
```

---

## Docker

Build Docker Image

```bash
docker build -t customer-churn-app .
```

Run Docker Container

```bash
docker run -d -p 8501:8501 --name customer-churn customer-churn-app
```

Open the application

```
http://localhost:8501
```

---

## Jenkins CI/CD Pipeline

The Jenkins pipeline automates the deployment process.

Pipeline Stages

- Checkout Source Code
- Install Dependencies
- Build Docker Image
- Stop Existing Docker Container
- Run New Docker Container
- Deploy Updated Streamlit Application

CI/CD Flow

```
Developer
      │
      ▼
Git Push
      │
      ▼
GitHub Repository
      │
      ▼
Jenkins Pipeline
      │
      ▼
Docker Build
      │
      ▼
Docker Container
      │
      ▼
Streamlit Application
```

---

## Model Inputs

The application predicts customer churn using features such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract Type
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

---

## Future Enhancements

- Cloud Deployment using AWS EC2
- Docker Hub Integration
- GitHub Actions CI/CD
- Kubernetes Deployment
- Model Monitoring
- Automated Model Retraining
- User Authentication
- Database Integration

---

## Author

**Sanjana H P**

- GitHub: https://github.com/sanjanahp16
- LinkedIn: *(Add your LinkedIn profile URL here)*

---

## License

This project is developed for educational and portfolio purposes.