# Project Overview
- This project consists of codes chiefly of four components: The PySpark part, Airflow, MLFlow and FastAPI(including Prometheus, Grafana). 
- The part consisting of Apache Spark has the function data preprocessing functions such as min max scaling with the help of pyspark. This requires java to be installed on the system. This is in the modular_LSTM.py code. 
- The airflow part has the DAG consisting of the pipeline having three components: getting input which consists of pre-processing data, predict_temp and printing out the output. This is present in project_airflow.py
- MLFlow is implemented in a Jupyter notebook. We have kept six experiments varying the model structure, learning rate and number of epochs for training. We first establish the server using   
mlflow server --host localhost --port 5000
- The below 2 codes are used for running in the .ipynb cells:
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment('name')
- We create autologging as well as track the various parameters and metrics. 
- The Bangalore_LSTM.ipnyb file is the code without any of the above features which was initially developed. This has a description of the dataset used here, and LSTM applied for certain data. 
- The FASTAPI-pipeline implements grafana and prometheus together. Collectively, this consists of prometheus_data, docker-compose.yml and src. src consists of the app folder which has the code for api. It also has the dockerfile and requirements.txt which state the modules to be imported in the docker image.
- This is executed by pulling prometheus into docker with the following command:
docker pull prom/prometheus
- Prometheus can be accessed at http://localhost:9090/
- Set the prometheus.yml file in prometheus_data to the correct configuration. Additionally specify multiple port requirements(avoiding conflicts) in docker-compose.yml. 
Now run the app using docker-compose up -d --build command. 
- Verify that Prometheus is scraping metrics from FastAPI app by visiting the following url: http://localhost:9090/targets
- Grafana will be present in the port specified in docker-compose.yml(In this code set at 4000). Hence go to localhost:4000 .
- You can choose the dash board after sign in and get the appropriate metrics to be plotted based on that have been kept in the fast api app(LSTM_api.py)
