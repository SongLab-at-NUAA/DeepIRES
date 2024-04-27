from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,accuracy_score,recall_score,matthews_corrcoef,confusion_matrix,roc_curve, precision_recall_curve
import numpy as np
import pandas as pd
import os
from sklearn.metrics import auc
import keras.backend as k
from scipy import stats
from model import deepires_model
#忽略
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import tensorflow as tf
def plotROC(test,score):
    fpr,tpr,threshold = roc_curve(test, score)
    auc_roc = roc_auc_score(test, score)
    plt.figure()
    font = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 22,
         }
    lw = 3
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='auROC = %f' %auc_roc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.tick_params(labelsize=20)
    plt.xlabel('False Positive Rate',font)
    plt.ylabel('True Positive Rate',font)
    plt.title('Receiver operating characteristic curve',font)
    plt.legend(loc="lower right")
    plt.savefig('x.jpg',dpi=350)
    plt.show()
test = np.load('../dataset/test/balanced_test_set.npz')
#test=np.load('C:/Users/czw20/Desktop/python/linear/iresite.npz')
X_tes,y_tes = test['X_tes'], test['y_tes']
model=deepires_model()
weight_path='../weights/new/'
model.load_weights(weight_path).expect_partial()
y_pred_1 = model.predict(X_tes)
y_pred=np.where(y_pred_1>0.5,1,0)
acc = accuracy_score(y_tes, y_pred)
sn = recall_score(y_tes, y_pred)
mcc = matthews_corrcoef(y_tes, y_pred)
tn, fp, fn, tp = confusion_matrix(y_tes, y_pred).ravel()
sp = tn / (tn + fp)
auroc = roc_auc_score(y_tes, y_pred_1)
f1 = f1_score(y_tes, y_pred.reshape(-1))
lr_precision, lr_recall, _ = precision_recall_curve(y_tes, y_pred_1)
aupr=auc(lr_recall,lr_precision)

print("ACC : ", acc)
print("SN : ", sn)
print("SP : ", sp)
print("MCC : ", mcc)
print("AUC : ", auroc)
print("F1-sorce : ", f1)
print("AUPR : ", aupr)

'''''
ires=pd.read_csv('C:/Users/czw20/Desktop/python/linear/5utr.csv')
pred=ires.pred
label=ires.label
pred1=ires.pred3
fpr1,tpr1,threshold1 = roc_curve(label, pred1)
fpr2,tpr2,threshold2 = roc_curve(label, pred)
auc_roc1 = roc_auc_score(label, pred1)
auc_roc2= roc_auc_score(label, pred)
plt.figure()
font = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 22,
         }
lw = 3
plt.figure(figsize=(8,8))
plt.plot(fpr1, tpr1, color='darkorange',lw=lw, label='ourmodel auROC = %f' %auc_roc1)
plt.plot(fpr2, tpr2, color='green',lw=lw, label='irespy auROC = %f' %auc_roc2)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.tick_params(labelsize=20)
plt.xlabel('False Positive Rate',font)
plt.ylabel('True Positive Rate',font)
plt.title('Receiver operating characteristic curve',font)
plt.legend(loc="lower right")
plt.show()
'''''