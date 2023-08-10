from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from datetime import datetime

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import seaborn as sns

row_mark=1800
time_step=1
opt = K.optimizers.Adam(learning_rate=0.00001, epsilon=1e-6, amsgrad=True)
id = datetime.now().strftime("%d%m%Y_%H%M%S")
def vae_loss(original, out, z_log_sigma, z_mean,sequence_length):
    reconstruction = K.backend.mean(K.backend.square(original - out)) * sequence_length
    kl = -0.5 * K.backend.mean(1 + z_log_sigma - K.backend.square(z_mean) - K.backend.exp(z_log_sigma))

    return reconstruction + kl
def reshape(da):
    return da.reshape(-1, time_step, da.shape[1]).astype("float32")
    # return da.reshape(da.shape[0],  da.shape[1]).astype("float32")
def sample_z(args):
    mu, log_sigma = args
    # eps = tf.random.normal(shape=(1,1), mean=0., stddev=1.)
    # return mu + tf.exp(log_sigma / 2) * eps
    sigma = tf.exp(log_sigma * 0.5)
    epsilon = tf.random.normal(shape=(10, 36), mean=0.0, stddev=1.0)
    return mu + epsilon * sigma
def plot_log_likelihood(df_log_px):
    plt.figure(figsize=(14, 6), dpi=80)
    plt.title("Log likelihood")
    sns.set_color_codes()
    sns.distplot(df_log_px, hist=True, kde=True, rug=True, color='blue')
    # plt.savefig('output/VMD/log_likelihood_VMD_'+id+'.png')
def main():
    all_df = pd.read_csv('dataset/VMD/limo_relation_data_6.14_2.csv', index_col='TIMESTAMP', )
    all_df.index = pd.to_datetime(all_df.index, unit='ms')
    all_df = np.abs(all_df)
    all_df=all_df.iloc[:,5:]
    all_df=all_df[:4950]
    # all_df = all_df[:6300]
    train_df = all_df[:row_mark]
    test_df = all_df[row_mark:]

    scaler = MinMaxScaler()
    # scaler.fit(np.array(all_df)[:, 1:])
    scaler.fit(np.array(all_df))
    train_scaled = scaler.transform(np.array(train_df))
    test_scaled = scaler.transform(np.array(test_df))
    # train_scaled, test_scaled = split_normalize_data(all_df)
    n_inputs = train_scaled.shape[1]
    print("train and test data shape after scaling: ", train_scaled.shape, test_scaled.shape)

    train_X = reshape(train_scaled)
    # train_X1=train_X.reshape(10,-1,4)
    test_X = reshape(test_scaled)


    inputs = Input(n_inputs,)
    r =Reshape((time_step,n_inputs,))(inputs)
    e = LSTM(64,activation='softplus',return_sequences=False)(r)
    mu = Dense(n_inputs)(e)
    log_sigma = Dense(n_inputs)(e)
    # z = Lambda(sample_z)([mu, log_sigma])
    z = Lambda(sample_z,output_shape=(n_inputs,))([mu, log_sigma])

    d = RepeatVector(time_step)(z)
    d = LSTM(64,activation='softplus', return_sequences=True)(d)

    d=Dense(n_inputs,activation='linear')(d)
    output=Reshape((n_inputs,))(d)

    # define autoencoder model
    model = Model(inputs=inputs, outputs=output)
    model.add_loss(vae_loss(inputs,output,log_sigma,mu,n_inputs))
    # compile autoencoder model
    model.compile(optimizer=opt, loss=None)
    # plot the autoencoder
    # plot_model(model, './shap_file/'+'autoencoder_compress_consumption1.png', show_shapes=True)
    # fit the autoencoder model to reconstruct input
    history = model.fit(train_scaled, train_scaled, epochs=100,  batch_size=10, verbose=2, validation_data=(test_scaled[:500], test_scaled[:500]))
    # model.save(model_dir+'111.h5')
    train_predict_x=model.predict(train_X, batch_size=10)
    train_predict_x_a=scaler.inverse_transform(np.array(train_predict_x))
    # train_log_px = train_log_px.reshape(train_log_px.shape[0], train_log_px.shape[2])
    df_train_rct_px = pd.DataFrame()
    df_train_rct_px['rct_px'] = np.mean(np.abs(train_predict_x-train_scaled), axis=1)

    plot_log_likelihood(df_train_rct_px)

    test_rct_px = model.predict(test_X, batch_size=10)
    # test_log_px = test_log_px.reshape(test_log_px.shape[0], test_log_px.shape[2])
    df_rct_px = pd.DataFrame()
    df_rct_px['rct_px'] = np.mean(np.abs(test_rct_px-test_scaled), axis=1)
    df_log_px = pd.concat([df_train_rct_px, df_rct_px])
    # df_log_px['threshold_max'] = df_train_log_px['log_px'].max()
    # df_log_px['threshold_min'] =df_train_log_px['log_px'].min()
    df_log_px['threshold'] =df_train_rct_px['rct_px'].max()
    rule = df_log_px['rct_px'] > df_log_px['threshold']
    df_log_px=pd.concat([df_log_px,rule],axis=1)
    df_log_px=df_log_px.rename(columns={0:'anaomaly'})
    df_log_px.index = all_df.index

    df_log_px.plot(logy=True, figsize=(24, 9), color=['blue', 'red','red'])
    # plt.savefig('output/VMD/anomaly_lstm_vae_VMD_'+id+'.png')
    df_log_px.to_csv("output/VMD/Anomaly_lstm_vae_VMD_0_"+id+".csv")





    # import shap
    # shap.initjs()
    # feature_names = ['ZD_P', 'GF_P', 'XF_P', 'ZP_P', 'TD_P', 'GD_P']
    # explainer = shap.GradientExplainer(model, train_scaled[:300])
    # # print(explainer.expected_value)
    # # explainer = shap.KernelExplainer(model, train_scaled[:800])
    # #
    # # # shap_values = explainer.shap_values(train_scaled[:300])
    # shap_values=explainer.shap_values(train_scaled[:800])
    # print("shap_values",shap_values)
    #
    # shap.summary_plot(shap_values=shap_values[0],
    #                   features=train_scaled[:800],
    #                   feature_names=feature_names)
    # # shap.waterfall_plot(shap_values=shap_values)
    # # shap.force_plot(base_value=explainer.expected_value[0], shap_values=shap_values[0][0], features=train_scaled[:300][10],
    # #                 feature_names=feature_names, matplotlib=True)


if __name__ == "__main__":
    main()
