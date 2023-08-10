import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
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

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
image_dir="./limo_consumption/"

all_df = pd.read_csv('dataset/VMD/limo_relation_data_6.14_2.csv', index_col='TIMESTAMP')
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

# h=pd.DataFrame()
#
# h["real"]=df_for_training["ZD_P"]
df_for_training["ZD_P"]=savgol_filter(df_for_training["ZD_P"],5,3,mode='nearest')
# h["denoising"]=df_for_training["ZD_P"]
# h.to_csv("data_processing.csv")
# df_for_training=savgol_filter(df_for_training,5,3,mode='nearest')
scaler = MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled=scaler.transform(df_for_testing)
def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            # dataY.append(dataset[i,0])
            dataY.append(dataset[i, [0]])
    return np.array(dataX),np.array(dataY)

def _do_tests( dec_reg, hit_reg, iter_id):
        """Private method to operate Bonferroni corrections on the feature
        selections."""

        active_features = np.where(dec_reg >= 0)[0]
        hits = hit_reg[active_features]
        # get uncorrected p values based on hit_reg
        to_accept_ps = sp.stats.binom.sf(hits - 1, iter_id, .5).flatten()
        to_reject_ps = sp.stats.binom.cdf(hits, iter_id, .5).flatten()

        # Bonferroni correction with the total n_features in each iteration
        to_accept = to_accept_ps <= alpha / float(len(dec_reg))
        to_reject = to_reject_ps <= alpha / float(len(dec_reg))

        # find features which are 0 and have been rejected or accepted
        to_accept = np.where((dec_reg[active_features] == 0) * to_accept)[0]
        to_reject = np.where((dec_reg[active_features] == 0) * to_reject)[0]

        # updating dec_reg
        dec_reg[active_features[to_accept]] = 1
        dec_reg[active_features[to_reject]] = -1

        return dec_reg
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
l1=LSTM(units=64,return_sequences=True)(input)
l2=Dropout(0.1)(l1)
l2=LSTM(units=64)(l2)
l3=Dropout(0.1)(l2)
l4=BatchNormalization()(l3)
output=Dense(1)(l3)
model=Model(input,output)
model.compile(optimizer="adam",loss="mse")
#
# history = model.fit(trainX, trainY, epochs=100, batch_size=20, verbose=2,
#                         validation_split=0.2)
# prediction=model.predict(trainX)
# prediction_copy=np.repeat(prediction,n_inputs,axis=-1)
# pred=scaler.inverse_transform(prediction_copy)[:,0]
#
# original_copy = np.repeat(trainY,n_inputs, axis=-1)
# original=scaler.inverse_transform(np.reshape(original_copy,(len(trainY),n_inputs)))[:,0]
#
# plt.plot(original, color = 'red', label = 'Real  P')
# plt.plot(pred, color = 'blue', label = 'Predicted  P')
# plt.title(' limo Power Prediction')
# plt.xlabel('Time')
# plt.ylabel(' Power')
# plt.legend()
# plt.show()

# print(shap_values)
supports = np.zeros((df_copy.shape[1],))
n_trials=10
for i in range(n_trials):
    print("----- TRIAL {} -----".format(i + 1))
    max_iter = 20
    hit_reg = np.zeros(n_features, dtype=np.int)
    imp_history = np.zeros(n_features, dtype=np.float)
    # holds the decision about each feature:
    # default (0); accepted (1); rejected (-1)
    dec_reg = np.zeros(n_features, dtype=np.int)
    dec_history = np.zeros((max_iter, n_features), dtype=np.int)
    sha_max_history = []
    for i in range(max_iter):
        if (dec_reg != 0).all():
            print("All Features analyzed. stop!")
            break
        history = model.fit(trainX, trainY, epochs=50, batch_size=20, verbose=0,
                            validation_split=0.2)

        coefs = _shap_importances(model, trainX)
        print(coefs)

        imp_sha = coefs[n_real:]
        imp_real = coefs[:n_real]

        # get the threshold of shadow importances used for rejection
        imp_sha_max = np.percentile(imp_sha, 100)
        # record importance history
        sha_max_history.append(imp_sha_max)
        imp_history = np.vstack((imp_history, imp_real))
        # register which feature is more imp than the max of shadows
        hit_reg[np.where(imp_real[~np.isnan(imp_real)] > imp_sha_max)[0]] += 1

        dec_reg = _do_tests(dec_reg, hit_reg, i + 1)
        dec_history[i] = dec_reg

    confirmed = np.where(dec_reg == 1)[0]
    tentative = np.where(dec_reg == 0)[0]

    support_ = np.zeros(n_features, dtype=np.bool)
    ranking_ = np.ones(n_features, dtype=np.int) * 4
    n_features_ = confirmed.shape[0]
    importance_history_ = imp_history[1:]

    if tentative.shape[0] > 0:
        tentative_median = np.nanmedian(imp_history[1:, tentative], axis=0)
        tentative_low = tentative[
            np.where(tentative_median <= np.median(sha_max_history))[0]]
        tentative_up = np.setdiff1d(tentative, tentative_low)

        ranking_[tentative_low] = 3
        if tentative_up.shape[0] > 0:
            ranking_[tentative_up] = 2

    if confirmed.shape[0] > 0:
        support_[confirmed] = True
        ranking_[confirmed] = 1

    if (~support_).all():
        raise RuntimeError(
            "Boruta didn't select any feature. Try to increase max_iter or "
            "increase (if not None) early_stopping_boruta_rounds or "
            "decrese perc.")
    print(support_)
    print(support_.astype(int))
    supports+=support_.astype(int)

print(supports)
plt.figure(figsize=(8,5))
plt.bar(range(len(supports)), supports,
         color=['red' if 'RANDOM' in c else 'blue' for c in df_copy])
plt.xticks(range(len(supports)), df_copy.columns,rotation=315); plt.ylabel('trials')
plt.xlabel("features")
plt.title('how many times a feature is selected')
plt.show()

plt.savefig('features_selection.png')
