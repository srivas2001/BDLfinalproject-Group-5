from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

import torch
import torch.nn as nn
import numpy as np

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': datetime(2024, 3, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# # Disable oneDNN optimization
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Function to load the model
def load_model(model_path: str) -> nn.Module:
     # Ensure LSTM is available globally
    model=LSTM(1, 128, 2, 1)
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to inverse transform the data
def inverse_transform(y, min, max):
    return y * (max - min) + min

# Function to forward transform the data
def forward_transform(data, min, max):
    data = np.array(data)
    data = (data - min) / (max - min)
    return torch.tensor(data, dtype=torch.float32)

# Function to predict temperature
def predict_temperature(data_point, min, max):
    data = forward_transform(data_point, min, max)
    with torch.no_grad():
        input_tensor = torch.FloatTensor(data).unsqueeze(0).unsqueeze(-1).to(next(model.parameters()).device)
        prediction = model(input_tensor).item()
        prediction = inverse_transform(prediction, min, max)
    return str(prediction)

def predict_temp(text_input, model_path_str):
    input_size, hidden_size, num_layers, output_size = 1, 128, 2, 1
    
    global model
    # Load the model if not already loaded
    # model = LSTM(input_size, hidden_size, num_layers, output_size)

    #model=nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
    model_path = model_path_str
    
    model = load_model(model_path)

    # Split the text input by comma and convert to floats
    float_list = [float(content) for content in text_input.split(',')]

    # Predict temperature
    temp = predict_temperature(float_list, 18.4, 32.2)

    return_stat = f'Predicted Temperature is {temp} Degree Celcius'
    
    return return_stat

def get_input(file_path):
    with open(file_path, 'r') as file:
        inputs_7 = file.readline().strip()
    return inputs_7

def write_output(prediction, output_file_path):
    with open(output_file_path, 'w') as file:
        file.write(prediction)


model_path = "/home/arun/airflow/dags/model/model2_new.pth"
file_path = '/home/arun/airflow/dags/fetched_data/weather_7.txt'
output_file_path = '/home/arun/airflow/dags/prediction.txt'


with DAG('project_DAG', 
         default_args=default_args,
         start_date=datetime(2024, 5, 17, 9),
         schedule='*/2 * * * *',
         catchup=False) as dag:

    get_input_obj = PythonOperator(
        task_id='get_input',
        python_callable=get_input,
        op_kwargs={'file_path':file_path}
    )

    predict_temp_obj = PythonOperator(
        task_id='predict_temp',
        python_callable=predict_temp,
        op_kwargs={'text_input':"{{ ti.xcom_pull(task_ids='get_input') }}",'model_path_str':model_path}
    )

    write_output_obj = PythonOperator(
        task_id='write_output',
        python_callable=write_output,
        op_kwargs={'prediction':"{{ ti.xcom_pull(task_ids='predict_temp') }}",'output_file_path':output_file_path}
    )

    get_input_obj >> predict_temp_obj >> write_output_obj















