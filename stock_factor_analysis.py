# -*- coding: utf-8 -*-
import numpy as np
import tushare as ts
import pandas as pd
import datetime
import math
import matplotlib.pyplot as plt
import tensorflow as tf

#Get the data

#test stock
stock_id = '600016'
#trainning data time period
start_time = '2016-01-01'
end_time = '2018-02-01'

#test data time period
t_start_time = '2018-01-02'
t_end_time = '2018-02-01'


###获取数据

"""
#上证指数
df_000001 = ts.get_k_data(code='000001',  index=True, start=start_time, end=end_time)[['date','close','volume']]
df_000001.columns = ['date','000001_close','000001_volume']
#深证指数
df_399001 = ts.get_k_data(code='399001',  index=True, start=start_time, end=end_time)[['date','close','volume']]
df_399001.columns = ['date','399001_close','399001_volume']
#创业板指
df_399006 = ts.get_k_data(code='399006',  index=True, start=start_time, end=end_time)[['date','close','volume']]
df_399006.columns = ['date','399006_close','399006_volume']
#沪深300
# df_1B0300= ts.get_k_data(code='1B0300',  index=True, start=start_time, end=end_time)[['date','close','volume']]
# df_399006.columns = ['date','1B0300_close','1B0300_volume']
#上证50
# df_1B0016 = ts.get_k_data(code='sz50',  index=True, start=start_time, end=end_time)[['date','close','volume']]
# df_1B0016.columns = ['date','1B0016_close','1B0016_volume']
#中证500
# df_1B0905 = ts.get_k_data(code='1B0905',  index=True, start=start_time, end=end_time)[['date','close','volume']]
# df_1B0905.columns = ['date','1B0905_close','1B0905_volume']

# print(df_000001)
# print(df_399001)
# print(df_399006)
# print(df_1B0300)
# print(df_1B0016)
# print(df_1B0905)


#合并数据注意时间段的长度要一致,证券市场是同一个，以免产生时区错位。
#另外有些个股会停牌，最好采用上证指数的时间作为基准交易时间。
df_index = df_000001.merge(df_399001, on='date', how='left')
df_index = df_index.merge(df_399006, on='date', how='left')

#获取要分析的个股数据
df_stock = ts.get_k_data(code=stock_id, start=start_time, end=end_time)[['date','code','close','volume']]

#将个股数据与影响因素数据合并,这里需要把个股放在前面，进行左连接，去掉个股停牌的日期
df1 = df_stock.merge(df_index, on='date', how='left')
print(df1)
#数据保存
df1.to_csv(r'c:\smartkline\output\dataset.csv')
"""

#从磁盘读取数据
df1 = pd.read_csv(r'c:\smartkline\output\dataset.csv', index_col=0)
###数据规范化
#去除日期、代码列，求变化百分比
df1 = df1.drop(['date','code'], axis=1)
df1 = df1.pct_change(periods=1, fill_method='pad')
#删除第一行空值,并重置索引
df1 = df1.drop([0])
df1 = df1.reset_index(drop=True)
# print(df1)

# np_data = df1.values
# print(np_data)



#设置模型超参数
num_steps = 7
timeperiod_window_size = 1
num_features = 8
input_shape = (None, num_steps, timeperiod_window_size, num_features)

#把数据分割为训练集、验证集、测试集
split_prop =(0.7, 0.9, 1.0)

#计算一个sequence的元素个数
one_sequence_elements = num_steps*timeperiod_window_size*num_features
num_sequences = len(df1)//one_sequence_elements

features_data = df1.values.ravel()
features_data = features_data[0:num_sequences*one_sequence_elements]
features_data = features_data.reshape([num_sequences,1,num_steps*timeperiod_window_size,num_features])

trainning_features = features_data[0:math.floor(num_sequences*split_prop[0])]
validation_features = features_data[math.floor(num_sequences*split_prop[0]):math.floor(num_sequences*split_prop[1])]
test_features = features_data[math.floor(num_sequences*split_prop[1]):math.floor(num_sequences*split_prop[2])]

target_data =df1['close'].values.ravel()
target_data = [target_data[i] for i in range(timeperiod_window_size,num_sequences*num_steps*timeperiod_window_size,timeperiod_window_size)]
#如果最后一个target值越界，那么赋值为0%
if len(target_data)< num_sequences*num_steps:
    target_data = np.concatenate((target_data,np.array([0.])))

trainning_targets = target_data[0:len(trainning_features)*num_steps]
trainning_targets = trainning_targets.reshape(len(trainning_features),-1)
validation_targets = target_data[len(trainning_features)*num_steps:(len(trainning_features)+len(validation_features))*num_steps]
validation_targets = validation_targets.reshape(len(validation_features),-1)
test_targets = target_data[(len(trainning_features)+len(validation_features))*num_steps:]
test_targets = test_targets.reshape(len(test_features),-1)
# print(features_data)
print(np.shape(features_data))
print(np.shape(trainning_features))
print(np.shape(test_features))
print(np.shape(target_data))
print(trainning_targets.shape)
print(test_targets.shape)



lstm_units = 256
learning_rate = 0.0001
#Building the computational graph

from keras.layers import Conv2D,Dense,LSTM,ConvLSTM2D,Reshape,LSTMCell,RNN,Flatten
from keras.models import Sequential
from keras import losses,optimizers,metrics
from keras.callbacks import ModelCheckpoint,EarlyStopping

model = Sequential()
model.add(Conv2D(filters=1, kernel_size=(timeperiod_window_size,num_features), strides=(timeperiod_window_size,num_features),
                 padding='valid', data_format='channels_first', activation='tanh', use_bias=False,
                 input_shape=(1,num_steps*timeperiod_window_size,num_features)))
model.add(Reshape((num_steps,1)))
cells = [LSTMCell(lstm_units),LSTMCell(lstm_units)]
model.add(RNN(cells,return_sequences=True))
model.add(Dense(1))
model.add(Flatten())
rmsprop = optimizers.RMSprop(lr=learning_rate)
model.summary()
model.compile(optimizer=rmsprop,
              loss='mean_squared_error')

#Running the computational graph
checkpointer = ModelCheckpoint(filepath=r'.\factor_analysis_weights.hdf5', verbose=1, save_best_only=True)
earlystopping = EarlyStopping(monitor='val_loss', patience=100)
model.fit(trainning_features, trainning_targets,batch_size=5,
          epochs=6000,validation_data=(validation_features, validation_targets), callbacks=[checkpointer,earlystopping])
score = model.evaluate(test_features, test_targets, batch_size=1)
print('test loss percent: {:.4%}'.format(math.sqrt(score)))

#显示卷积层的各个影响因素的权重
print(model.get_weights()[0])
