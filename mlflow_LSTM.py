import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.init as init
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset,DataLoader
from copy import deepcopy as dc
def read_data(path):
    data = pd.read_csv(path)
    return data
def extract_T(data):
    data_T = data[['datetime', 'temp']]
    data_T['datetime'] = pd.to_datetime(data_T['datetime'], format='%Y-%m-%d')
    return data_T
#Now to put in format easily readable by pytorch for lstm
def prep_df_lstm(df, n_steps):
    df = dc(df)
    df.set_index('datetime', inplace=True)
    for i in range(1, n_steps + 1):
        df['t-' + str(i)] = df['temp'].shift(i)
    df.dropna(inplace=True)
    return df
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Steps to be followed in the pipeline
#1. Read data -Done
#2. Extract the temperature and date part-Done
#3. do shifting of data(from prep_df_lstm)-Done
#4. Then convert to numpy array
#5. Then do min max scaling with sklearn 
#6. Take the X,y part separately, split into train, test, validation as shown in the .ipynb file
#7. Reshape the train, test and val
#8. Now convert to torch tensor
#9. Use weather dataset, dataloader
#10. Define the LSTM model
#11. Define training epochs
#12. Train
#13. Plot the functions
#14. Predict temperature given date
#15. Plot the predictions
def convertscale(data):
    data_np=data.to_numpy()
    scaler=MinMaxScaler()
    data_scaled=scaler.fit_transform(data_np)
    return data_scaled,scaler
def prep_data(data,n_steps):
    #Split into X, y here
    X=data[:,1:]
    y=data[:,0]
    X=dc(np.flip(X,axis=1))
    return X,y
def split_data(X,y,split_index_1,split_index_2):
    X_train=X[:split_index_1]
    y_train=y[:split_index_1]
    X_test=X[split_index_1:split_index_2]
    y_test=y[split_index_1:split_index_2]
    X_val=X[split_index_2:]
    y_val=y[split_index_2:]
    return X_train,y_train,X_test,y_test,X_val,y_val
def reshape_data_tensor(X_train,y_train,X_test,y_test,X_val,y_val,time_period,device): #Here time period is number of days used for predictions in the shifted data
    X_train=X_train.reshape(-1,time_period,1) #Requirement for LSTM in pytorch
    X_test=X_test.reshape(-1,time_period,1)
    y_train=y_train.reshape(-1,1)
    y_test=y_test.reshape(-1,1)
    X_val=X_val.reshape(-1,time_period,1)
    y_val=y_val.reshape(-1,1)
    X_train=torch.tensor(X_train).float().to(device)
    y_train=torch.tensor(y_train).float().to(device)
    X_test=torch.tensor(X_test).float().to(device)
    y_test=torch.tensor(y_test).float().to(device)
    X_val=torch.tensor(X_val).float().to(device)
    y_val=torch.tensor(y_val).float().to(device)
    return X_train,y_train,X_test,y_test,X_val,y_val
class WeatherDataset(Dataset):
    def __init__(self,X,y):
        self.X=X
        self.y=y
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]
def data_load(X_train,y_train,X_test,y_test,X_val,y_val,batch_size):
    train_dataset=WeatherDataset(X_train,y_train)
    test_dataset=WeatherDataset(X_test,y_test)
    #val_dataset=WeatherDataset(X_val,y_val)
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
    #val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=True)
    return train_loader,test_loader
class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size):
        super(LSTM,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,output_size)
    def forward(self,x):
        h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        c0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        out,_=self.lstm(x,(h0,c0))
        out=self.fc(out[:,-1,:])
        return out
def train_epoch(epoch):
    model.train(True)
    print(f'Epoch {epoch+1}')
    running_loss=0
    for batch_index,batch in enumerate(train_loader):
        X_batch,y_batch=batch
        output=model(X_batch)
        loss=criterion(output,y_batch)
        running_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_index%100==99:
            avg_loss=running_loss/100
            print(f'Batch : {batch_index+1} Loss : {avg_loss}')
            loss_train.append(avg_loss)
            plt.plot(loss_train)
            plt.show()
            running_loss=0
def validate_epoch():
    model.train(False)
    running_loss=0
    for batch_index,batch in enumerate(test_loader):
        X_batch,y_batch=batch
        with torch.no_grad():
            output=model(X_batch)
            loss=criterion(output,y_batch)
            running_loss+=loss.item()
    avg_loss=running_loss/len(test_loader)
    print(f'Validation Loss : {avg_loss}')
def train_model(lr,num_epochs,criterion,optimizer,scheduler,model,train_loader,test_loader):
    loss_train=[]
    for epoch in range(num_epochs):
        train_epoch(epoch)
        validate_epoch()
        scheduler.step()
def prediction(model,X_input,y_input,time_period,scaler):
    with torch.no_grad():
        predicted=model(X_input).to('cpu').numpy()
    train_predictions=predicted.flatten()
    dummies=np.zeros((X_input.to('cpu').shape[0],time_period+1))
    dummies[:,0]=train_predictions
    dummies=scaler.inverse_transform(dummies)
    train_predictions=dc(dummies[:,0])
    dummies=np.zeros((X_input.to('cpu').shape[0],time_period+1))
    dummies[:,0]=y_input.to('cpu').numpy().flatten()
    dummies=scaler.inverse_transform(dummies)
    new_y_train=dc(dummies[:,0]) 
    return train_predictions,new_y_train
def plot_predictions(train_predictions,new_y_train):
    plt.plot(new_y_train,label='Actual')
    plt.plot(train_predictions,label='Predicted')
    plt.legend()
    plt.show()
def mean_error(train_predictions,new_y_train):
    error=np.mean(np.abs(train_predictions-new_y_train))
    print(f'Mean Error : {error}')
    return error
if __name__ == "__main__":
    data=read_data(r"C:\Users\sriva\OneDrive\Documents\GitHub\FinalProject_BDL\Bangalore,India 2021-10-30 to 2024-04-13.csv")
    data_T=extract_T(data)
    data_T=prep_df_lstm(data_T,7)
    data_scaled,scaler=convertscale(data_T)
    X,y=prep_data(data_scaled,7)
    split_index_1=int(0.7*len(X))
    split_index_2=int(0.85*len(X))
    X_train,y_train,X_test,y_test,X_val,y_val=split_data(X,y,split_index_1,split_index_2)
    X_train,y_train,X_test,y_test,X_val,y_val=reshape_data_tensor(X_train,y_train,X_test,y_test,X_val,y_val,7,device)
    train_loader,test_loader=data_load(X_train,y_train,X_test,y_test,X_val,y_val,64)
    model=LSTM(1,64,2,1).to(device)
    criterion=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    scheduler=lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)
    train_model(0.001,10,criterion,optimizer,scheduler,model,train_loader,test_loader)
    train_predictions,new_y_train=prediction(model,X_train,y_train,7,scaler)
    plot_predictions(train_predictions,new_y_train)
    mean_error(train_predictions,new_y_train)
    test_predictions,new_y_test=prediction(model,X_test,y_test,7,scaler)
    plot_predictions(test_predictions,new_y_test)
    mean_error(test_predictions,new_y_test)
    val_predictions,new_y_val=prediction(model,X_val,y_val,7,scaler)
    plot_predictions(val_predictions,new_y_val)
    mean_error(val_predictions,new_y_val)


