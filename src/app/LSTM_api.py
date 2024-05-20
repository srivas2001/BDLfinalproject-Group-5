from fastapi import FastAPI, Request, Query
import uvicorn
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge
import time
import psutil
from LSTM1 import LSTM

# Disable oneDNN optimization
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define Prometheus metrics
API_USAGE_COUNTER = Counter("api_usage_counter", "API usage counter", ["client_ip"])
PROCESSING_TIME_GAUGE = Gauge("processing_time_gauge", "Processing time of the API", ["client_ip"])
CPU_UTIL_TIME = Gauge("cpu_utilization_gauge", "CPU utilization during processing", ["client_ip"])
MEMORY_UTILIZATION_GAUGE = Gauge("MEMORY_UTILIZATION_GAUGE", "Memory utilization during processing", ["client_ip"])
NETWORK_IO_BYTES_GAUGE = Gauge("NETWORK_IO_BYTES_GAUGE", "Network I/O bytes during processing", ["client_ip"])
NETWORK_IO_BYTES_RATE_GAUGE = Gauge("NETWORK_IO_BYTES_RATE_GAUGE", "Network I/O bytes rate during processing", ["client_ip"])
API_RUNTIME_GAUGE = Gauge("API_RUNTIME_GAUGE", "API runtime", ["client_ip"])
API_TL_TIME_GAUGE = Gauge("API_TL_TIME_GAUGE", "API T/L time", ["client_ip"])

# Set device for torch
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize FastAPI app
app = FastAPI()
Instrumentator().instrument(app).expose(app)

# Define LSTM model class

# Function to load the model
def load_model(model_path: str) -> nn.Module:
     # Ensure LSTM is available globally
    model=LSTM(1, 128, 2, 1)
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to inverse transform the data
def inverse_transform(y: torch.Tensor, min: float, max: float) -> float:
    return y * (max - min) + min

# Function to forward transform the data
def forward_transform(data: list, min: float, max: float) -> torch.Tensor:
    data = np.array(data)
    data = (data - min) / (max - min)
    return torch.tensor(data, dtype=torch.float32)

# Function to predict temperature
def predict_temperature(data_point: list, min: float, max: float) -> str:
    data = forward_transform(data_point, min, max)
    with torch.no_grad():
        input_tensor = torch.FloatTensor(data).unsqueeze(0).unsqueeze(-1).to(next(model.parameters()).device)
        prediction = model(input_tensor).item()
        prediction = inverse_transform(prediction, min, max)
    return str(prediction)

# Function to calculate processing time
def calculate_processing_time(start_time: float, length: int) -> float:
    end_time = time.time()
    total_time = end_time - start_time
    return total_time / length * 1e6

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get('/predict', response_model=None)
async def predict_temp(request: Request, text_input: str = Query(...)):
    start_time = time.time()
    input_size, hidden_size, num_layers, output_size = 1, 128, 2, 1
    global model
    # Load the model if not already loaded
    #model = LSTM(input_size, hidden_size, num_layers, output_size)
    #model=nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
    model_path = os.getenv("MODEL_PATH", "/app/model6ml.pth")
    model = load_model(model_path)

    # Split the text input by comma and convert to floats
    float_list = [float(content) for content in text_input.split(',')]
    client_ip = request.client.host

    # Increment API usage counter
    API_USAGE_COUNTER.labels(client_ip=client_ip).inc()

    # Predict temperature
    temp = predict_temperature(float_list, 18.4, 30.6)

    # Get CPU and memory usage
    cpu_percent = psutil.cpu_percent(interval=1)
    CPU_UTIL_TIME.labels(client_ip=client_ip).set(cpu_percent)
    memory_info = psutil.virtual_memory()
    MEMORY_UTILIZATION_GAUGE.labels(client_ip=client_ip).set(memory_info.percent)

    # Get network I/O
    net_io = psutil.net_io_counters()
    NETWORK_IO_BYTES_GAUGE.labels(client_ip=client_ip).set(net_io.bytes_sent + net_io.bytes_recv)
    NETWORK_IO_BYTES_RATE_GAUGE.labels(client_ip=client_ip).set((net_io.bytes_sent + net_io.bytes_recv) / (time.time() - start_time))

    # Calculate processing time
    processing_time = calculate_processing_time(start_time, len(float_list))
    PROCESSING_TIME_GAUGE.labels(client_ip=client_ip).set(processing_time)

    # Calculate API runtime
    api_runtime = time.time() - start_time
    API_RUNTIME_GAUGE.labels(client_ip=client_ip).set(api_runtime)

    # Calculate API T/L time
    api_tltime = api_runtime / len(float_list)
    API_TL_TIME_GAUGE.labels(client_ip=client_ip).set(api_tltime)

    return {"predicted_temperature": temp}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
