import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load Stock Data from CSV File
file_path = 'IBM.csv'
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data[['Close']]  # Using only the 'Close' column for simplicity
data = data.dropna()

#plt.plot(data, label='Actual Prices')
#plt.title('Historical Stock Prices')
#plt.xlabel('Time')
#plt.ylabel('Stock Price [USD]')
#plt.legend()
#plt.show()

# Identify the split point (one year before the last date in the dataset)
split_date = data.index.max() - pd.DateOffset(years=1)

# Split the data
train_data = data[data.index <= split_date]
test_data = data[data.index > split_date]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# Prepare Data for LSTM
def create_dataset(dataset, time_step):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 5
X_train, y_train = create_dataset(train_scaled, time_step)
X_test, y_test = create_dataset(test_scaled, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# Define LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the Model
epochs = 200
for i in range(epochs):
    for seq, labels in zip(X_train, y_train):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(torch.tensor(seq, dtype=torch.float32))
        single_loss = loss_function(y_pred, torch.tensor(labels, dtype=torch.float32))
        single_loss.backward()
        optimizer.step()

    if i%25 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

# Making Predictions
model.eval()
test_predictions = []
for seq in X_test:
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        test_predictions.append(model(torch.tensor(seq, dtype=torch.float32)).item())


# Inverse transform to get actual prices
test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))

# Calculate Mean Squared Error
test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
mse = mean_squared_error(test_actual, test_predictions)
print(f'Test Mean Squared Error: {mse}')

# Plot the results
plt.plot(test_actual, label='Actual Prices')
plt.plot(test_predictions, label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time [days]')
plt.ylabel('Stock Price [USD]')
plt.legend()
plt.show()