pipeline {
    agent any 
    stages {
        stage('Data acquisition') { 
            steps {
                
                sh ('chmod +x ./src/real_time_twitter_collection.py')
            }
        }
        stage('nlp_pipeline_model') { 
            steps {
                // 
                sh ('chmod +x ./src/nlp_pipeline.py')
            }
        }
       
    } 
}
