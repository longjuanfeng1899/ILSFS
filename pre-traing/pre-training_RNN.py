import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU,SimpleRNN
from tensorflow.keras.layers import Dense, Dropout,Reshape,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from shap_ILSFS.utils import _shap_importances
import scipy as sp
from scipy.signal import savgol_filter
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# image_dir="./limo_consumption/"

all_df = pd.read_csv('../dataset/VMD/limo_relation_data_6.14_2.csv', index_col='TIMESTAMP')
all_df.index = pd.to_datetime(all_df.index, unit='ms')



df=all_df.iloc[:,5:]
df=np.abs(df)
np.random.seed(42)

#
# for c in range(1,4):
#     df[f"RANDOM_{c}"] = np.random.normal(0,5, (df.shape[0],1))

df_copy=df

n_real=len(df.columns)
n_features=df.shape[1]
alpha=0.05

df_shadow=df.apply(np.random.permutation)
df_shadow.columns=['shadow_' + feat for feat in df.columns]
df=pd.concat([df,df_shadow],axis=1)

df=df[:2000]
df_for_training=df[:1800]
df_for_testing=df[1800:]

print(df.columns[:50])
# df_for_training["ZD_P"]=savgol_filter(df_for_training["ZD_P"],5,3,mode='nearest')
scaler = MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled=scaler.transform(df_for_testing)

def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)

time_step=60
trainX,trainY=createXY(df_for_training_scaled,time_step)
testX,testY=createXY(df_for_testing_scaled,time_step)

# trainX=np.array(df_for_training_scaled)
# testX=np.array(df_for_testing_scaled)
# trainY=np.array(df_for_training_scaled[""])

# trainX=np.array(df_for_training_scaled)


print("trainX Shape-- ",trainX.shape)
print("trainY Shape-- ",trainY.shape)


n_inputs=df_for_training_scaled.shape[1]
input=Input(shape=(time_step,n_inputs,))
# l1=LSTM(units=64,return_sequences=True)(input)
l1=SimpleRNN(units=64,return_sequences=True)(input)
l1=Dropout(0.1)(l1)
l2=SimpleRNN(units=64)(l1)
l3=Dropout(0.1)(l2)
l4=BatchNormalization()(l3)
output=Dense(1)(l3)
model=Model(input,output)
model.compile(optimizer="adam",loss="mse")

start_time=time.time()

history = model.fit(trainX, trainY, epochs=30, batch_size=20, verbose=2,
                        validation_split=0.2)
fig,ax1 = plt.subplots()
# ax2 = ax1.twinx()

ax1.plot(history.history['loss'],
         'b',
         label='Training loss')
ax1.plot(history.history['val_loss'],
         'r',
         label='Validation loss')
# ax2.plot(history.history['acc'],
#          'g',
#          label='Training acc')
# ax2.plot(history.history['val_acc'],
#          'y',
#          label='Validation acc')

fig.legend(loc='upper right')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss, [mse]')
# ax2.set_ylabel('Accuracy,[acc]')
ax1.set_ylim([0,0.1])

prediction=model.predict(trainX)
prediction_copy=np.repeat(prediction,n_inputs,axis=-1)
pred=scaler.inverse_transform(prediction_copy)[:,0]

original_copy = np.repeat(trainY,n_inputs, axis=-1)
original=scaler.inverse_transform(np.reshape(original_copy,(len(trainY),n_inputs)))[:,0]
fn=pd.DataFrame()
fn["real"]=original
fn["fitted"]=pred
fn.to_csv("result_fitted.csv")

# plt.plot(original, color = 'red', label = 'Real  P')
# plt.plot(pred, color = 'blue', label = 'Fitted  P')
# plt.title(' limo Power Prediction')
# plt.xlabel('Time')
# plt.ylabel(' Power')
# plt.legend()
# plt.show()
#
# print(shap_values)
import shap
shap.initjs()
explainer=shap.GradientExplainer(model,trainX)
coefs = explainer.shap_values(trainX)
shap_values=coefs[0]
end_time=time.time()
print("程序运行时间：%.2f秒" % (end_time - start_time))
if isinstance(coefs, list):
    coefs = list(map(lambda x: np.abs(x).mean(0), coefs))
    coefs = np.sum(coefs, axis=0)
else:
    coefs = np.abs(coefs).mean(0)
shap.summary_plot(shap_values=shap_values[0],
                  features=trainX[0],
                  feature_names=df.columns)
