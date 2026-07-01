pipeline {
    agent any

    environment {
        IMAGE_NAME = "customer-churn-app"
        CONTAINER_NAME = "customer-churn-container"
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                bat '''
                python -m pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t %IMAGE_NAME% .'
            }
        }

        stage('Stop Existing Container') {
            steps {
                bat '''
                docker stop %CONTAINER_NAME% || exit /b 0
                docker rm %CONTAINER_NAME% || exit /b 0
                '''
            }
        }

        stage('Install Dependencies') {
    steps {
        bat '''
        "C:\\Users\\SINCHANA\\AppData\\Local\\Programs\\Python\\Python311\\python.exe" -m pip install --upgrade pip
        "C:\\Users\\SINCHANA\\AppData\\Local\\Programs\\Python\\Python311\\python.exe" -m pip install -r requirements.txt
        '''
    }
}
    }

    post {
        success {
            echo 'Deployment Successful!'
        }
        failure {
            echo 'Deployment Failed!'
        }
    }
}