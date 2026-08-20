pipeline {
    agent any

    environment {
        OPENAI_API_KEY = credentials('OPENAI_API_KEY')
        CONFIDENT_API_KEY = credentials('CONFIDENT_API_KEY')
    }

    stages {
        stage('Setup') {
            steps {
                sh 'python --version'
                sh 'python -m pip install --upgrade pip'
                sh 'python -m pip install -r requirements.txt'
            }
        }

        stage('Unit test') {
            steps {
                dir('Session1_Intro') {
                    sh 'python -m pytest test_run_deepeval_eval.py -v'
                }
            }
        }

        stage('Run evaluation') {
            steps {
                dir('Session1_Intro') {
                    sh 'python run_deepeval_eval.py'
                }
            }
        }
    }
}
