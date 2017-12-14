
# coding: utf-8

# In[1]:

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import optimizers
import time

plt.style.use('ggplot')


# In[2]:

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX), numpy.array(dataY)


# In[3]:

# fix random seed for reproducibility
numpy.random.seed(7)


# In[5]:

# load the dataset
dataframe = read_csv('clean_train_data.csv',header=0,squeeze=True)


# In[6]:

import pandas as pd
# 日期格式转换函数
def convert_to_date(s):
    d = pd.to_datetime(s, format='%Y-%m-%d')
    return d

dataframe['Date'] = dataframe['Date'].apply(convert_to_date)
dataframe.index = dataframe['Date']
raw_dataframe = dataframe.copy()

# 2005-01 2016-12 2017-09
start_date = pd.to_datetime('2005-01')
end_date = pd.to_datetime('2017-09')
mask = (dataframe['Date'] > start_date) & (dataframe['Date'] <= end_date)
dataframe = dataframe.loc[mask]
dataframe = dataframe['Num']


# In[7]:

plt.figure(figsize=(12,5))
dataframe.plot()
plt.ylabel('price')
plt.show()


# In[8]:

dataset = dataframe.values
#dataset


# In[9]:

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# In[39]:

# split into train and test sets
train_size = int(len(dataset) * 0.965)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]


# In[40]:

#train.shape


# In[41]:

#test.shape


# In[42]:

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# In[43]:

#trainX


# In[44]:

#trainY


# In[45]:

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# In[46]:

trainX


# In[47]:

# create and fit the LSTM network
model = Sequential()

model.add(LSTM(input_dim=1,output_dim=100,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(100,return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(output_dim=1))
model.add(Activation('linear'))

start=time.time()
model.compile(loss='mean_squared_error',optimizer='Adam')
print ('compilation time:',time.time()-start)

history=model.fit(trainX,trainY,batch_size=1,nb_epoch=50,validation_split=0.2,verbose=2)


# In[48]:

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[49]:

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[50]:

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# In[51]:

# shift train predictions for plotting
trainPredictPlot=numpy.empty_like(dataset)
trainPredictPlot[:]=numpy.nan
trainPredictPlot=numpy.reshape(trainPredictPlot,(dataset.shape[0],1))
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict


# In[52]:

# shift test predictions for plotting
testPredictPlot=numpy.empty_like(dataset)
testPredictPlot[:]=numpy.nan
testPredictPlot=numpy.reshape(testPredictPlot,(dataset.shape[0],1))
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# In[53]:

#summarize history for loss
fig=plt.figure(figsize=(12,8))
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# In[54]:

#plot predictions
fig=plt.figure(figsize=(12,8))
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.ylabel('price')
plt.xlabel('date')
plt.show()


# In[55]:

# 单独观察预测部分
Pred = testPredict[:,0]
Real = testY[0]


# In[56]:

plt.plot(Pred, 'b')
plt.plot(Real, 'g')
plt.show()

