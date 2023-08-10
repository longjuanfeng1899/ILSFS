import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from shap_ILSFS.utils import _shap_importances
import scipy as sp
import os

# 计算混淆矩阵
def compute_confusion_matrix(precited,expected):
    part = precited ^ expected             # 对结果进行分类，亦或使得判断正确的为0,判断错误的为1
    pcount = np.bincount(part)             # 分类结果统计，pcount[0]为0的个数，pcount[1]为1的个数
    tp_list = list(precited & expected)    # 将TP的计算结果转换为list
    fp_list = list(precited & ~expected)   # 将FP的计算结果转换为list
    tp = tp_list.count(1)                  # 统计TP的个数
    fp = fp_list.count(1)                  # 统计FP的个数
    tn = pcount[0] - tp                    # 统计TN的个数
    fn = pcount[1] - fp                    # 统计FN的个数
    return tp, fp, tn, fn

# 计算常用指标
def compute_indexes(tp, fp, tn, fn):
    accuracy = (tp+tn) / (tp+tn+fp+fn)     # 准确率
    precision = tp / (tp+fp)               # 精确率
    recall = tp / (tp+fn)                  # 召回率
    F1 = (2*precision*recall) / (precision+recall)    # F1
    return accuracy, precision, recall, F1
def evaluation_curves(tp,tn,fp,fn,y_true,y_pred,mode):

    #  -- ROC curve with annotated decision point
    fp_rates, tp_rates, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fp_rates, tp_rates)
    plt.figure(figsize=[5, 4])
    plt.plot(fp_rates, tp_rates, color='orange',
             lw=1, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='grey')

    plt.plot(fp / (fp + tn), tp / (tp + fn), 'bo', markersize=8, label='Decision Threshold Point')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', size=13)
    plt.ylabel('True Positive Rate', size=13)
    plt.title('ROC Curve '+mode, size=15)
    plt.legend(loc="lower right")
    plt.subplots_adjust(wspace=.3)
    # plt.show()
    plt.savefig("ROC_Curve"+mode+".png")
    accuracy, precision, recall, F1=compute_indexes(tp, fp, tn, fn)
    results = {
        "Precision": precision, "Recall": recall,
        "F1 Score": F1, "Accuracy": accuracy, "AUC": roc_auc
    }
    return results

def multi_curves(list,y_true,mode):
    plt.figure(figsize=[10, 8])
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='grey')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', size=13)
    plt.ylabel('True Positive Rate', size=13)
    plt.title('ROC Curve - '+mode, size=15)
    plt.subplots_adjust(wspace=.3)
    for ln in list:
        print(ln)
        tp, tn, fp, fn =ln[0]
        y_pred=ln[1]
        color =ln[2]
        label=ln[3]
        type=ln[4]
        fp_rates, tp_rates, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fp_rates, tp_rates)
        plt.plot(fp_rates, tp_rates,type, color=color,
                 lw=1, label=label,markersize=5)
        # plt.plot(fp / (fp + tn), tp / (tp + fn), 'bo', markersize=5, color=color,label='Decision Threshold Point of '+label)
        plt.legend(loc="lower right")

    plt.savefig("output/VMD/ROC_Curve_pro"+mode+".png")
    return mode


def split_list_by_n(list_collection, n):
    """
    将集合均分，每份n个元素
    :param list_collection:
    :param n:
    :return:返回的结果为评分后的每份可迭代对象
    """
    for i in range(0, len(list_collection), n):
        yield list_collection[i: i + n]


def adjust(list_temp,n,k):
    # time.sleep(1)
    temp = split_list_by_n(list_temp, n)
    a=[]
    for i in temp:
        i=np.array(i)
        b=np.where(i==1)
        if np.array(b).size>0:
            if b[0][0]<k:
                i[i != 1] = 1
            else:
                i[i != 0] = 0
        a.extend(i)
    return a

label_list=pd.read_csv("dataset/VMD/label.csv", index_col="TIMESTAMP")
label_act=np.array(label_list["anomaly_act"])

detection_list_0=pd.read_csv("output/VMD/Anomaly_lstm_vae_VMD_0_09082023_173102.csv", index_col="TIMESTAMP")
result_0=np.array(detection_list_0["anaomaly"])
result_0 =result_0+0


detection_list_1=pd.read_csv("output/VMD/Anomaly_lstm_vae_VMD_IFS_3.csv", index_col="TIMESTAMP")
result_1=np.array(detection_list_1["anaomaly"])
result_1=result_1+0

detection_list_2=pd.read_csv("output/VMD/BRU/Anomaly_lstm_vae_VMD_BRU_30.csv",index_col="TIMESTAMP")
result_2=np.array(detection_list_2["anaomaly"])
result_2=result_2+0
#
detection_list_3=pd.read_csv("output/VMD/RFE/Anomaly_lstm_vae_VMD_RFE_30_09082023_210757.csv", index_col="TIMESTAMP")
result_3=np.array(detection_list_3["anaomaly"])
result_3=result_3+0
#
detection_list_4=pd.read_csv("output/VMD/GA/Anomaly_lstm_vae_VMD_GA_50_26072023_135054.csv",index_col="TIMESTAMP")
result_4=np.array(detection_list_4["anaomaly"])
result_4=result_4+0

detection_list_5=pd.read_csv("output/VMD/XGBSVFIR/Anomaly_lstm_vae_VMD_XGBSVFIR_50_04082023_202929.csv",index_col="TIMESTAMP")
result_5=np.array(detection_list_5["anaomaly"])
result_5=result_5+0

detection_list_6=pd.read_csv("output/VMD/ShapHT+/Anomaly_lstm_vae_VMD_ShapHT_70_04082023_200410.csv",index_col="TIMESTAMP")
result_6=np.array(detection_list_6["anaomaly"])
result_6=result_6+0
# result=[0,0,0,0,0,1,1,0,1,1,0,0,0,0,0,1,0,0,0,0]
result_0=np.array(adjust(result_0,10,1))
result_1=np.array(adjust(result_1,10,1))
result_2=np.array(adjust(result_2,10,1))
result_3=np.array(adjust(result_3,10,5))
result_4=np.array(adjust(result_4,10,5))
result_5=np.array(adjust(result_5,10,1))
result_6=np.array(adjust(result_6,10,3))


tp0, fp0, tn0, fn0=compute_confusion_matrix(label_act,result_0)

tp1, fp1, tn1, fn1=compute_confusion_matrix(label_act,result_1)

tp2, fp2, tn2, fn2=compute_confusion_matrix(label_act,result_2)
#
tp3, fp3, tn3, fn3=compute_confusion_matrix(label_act,result_3)
#
tp4, fp4, tn4, fn4=compute_confusion_matrix(label_act,result_4)

tp5, fp5, tn5, fn5=compute_confusion_matrix(label_act,result_5)
#
tp6, fp6, tn6, fn6=compute_confusion_matrix(label_act,result_6)

list=[
    [[tp0,tn0,fp0,fn0],result_0,"green","None",'bo-'],
    [[tp1,tn1,fp1,fn1],result_1,"orange","ILSFS",'gv-'],
    [[tp2,tn2,fp2,fn2],result_2,"red","LGBM-Boruta",'ys-'],
    [[tp3,tn3,fp3,fn3],result_3,"blue","SVM-RFE",'ch-'],
    [[tp4,tn4,fp4,fn4],result_4,"purple","GA",'mD-'],
    [[tp5,tn5,fp5,fn5],result_5,"black","XGBSVFIR",'-d'],
    [[tp6,tn6,fp6,fn6],result_6,"brown","ShapHT+",'-p']
]

roc=multi_curves(list,label_act,"VMD")

print(roc)


results0=evaluation_curves(tp0,tn0,fp0,fn0,label_act,result_0,"(a)")
#
results1=evaluation_curves(tp1,tn1,fp1,fn1,label_act,result_1,"(b)")
#
results2=evaluation_curves(tp2,tn2,fp2,fn2,label_act,result_2,"(c)")
#
results3=evaluation_curves(tp3,tn3,fp3,fn3,label_act,result_3,"(d)")
#
results4=evaluation_curves(tp4,tn4,fp4,fn4,label_act,result_4,"(e)")

results5=evaluation_curves(tp5,tn5,fp5,fn5,label_act,result_5,"(f)")
#
results6=evaluation_curves(tp6,tn6,fp6,fn6,label_act,result_6,"(g)")
#
print(results0)
print(results1)
print(results2)
print(results3)
print(results4)
print(results5)
print(results6)




