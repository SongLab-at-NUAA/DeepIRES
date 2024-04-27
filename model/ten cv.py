from numpy import interp
from scipy.stats import stats
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.regularizers import l1, l2
from model import ResBlock1
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,accuracy_score,recall_score,matthews_corrcoef,confusion_matrix,roc_curve, precision_recall_curve,auc
import numpy as np
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from datetime import datetime
from sklearn.metrics import roc_auc_score,average_precision_score, f1_score,recall_score,matthews_corrcoef,confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
from matplotlib import pyplot
from sklearn.utils import class_weight
from model import deepires_model
batch_size=64
t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
names = ['first']
name=names[0]
train=np.load('../dataset/train/train_set.npz')
X_tra,y_tra = train['X_tra'], train['y_tra']
class_weight = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(y_tra),y=y_tra.reshape(-1))
cw = dict(enumerate(class_weight))
#print(cw)
acc=np.zeros(10)
sn=np.zeros(10)
sp=np.zeros(10)
mcc=np.zeros(10)
f1=np.zeros(10)
auroc=np.zeros(10)
aupr=np.zeros(10)
mean_fpr = np.linspace(0, 1, 100)
tprs=[]
aucs=[]
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
model_prediction_result_cv=[]
for i,(tra, val) in enumerate(kfold.split(X_tra, y_tra)):
    print('\n\n第%d折' % i)
    model=deepires_model()
    X_train=X_tra[tra]
    y_train=y_tra[tra]
    X_test=X_tra[val]
    y_test=y_tra[val]
    Reduce = ReduceLROnPlateau(monitor='val_auc',
                               factor=0.1,
                               patience=10,
                               verbose=1,
                               mode='max',
                               min_delta=0.0001,
                               cooldown=0,
                               min_lr=0)
    filepath = 'cv/%d/'%i
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False, save_weights_only=True)#, mode='max',monitor='val_auc')
    callbacks_list = [checkpoint]
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3), metrics=['accuracy','AUC'])
    #back = EarlyStopping(monitor='val_auc', patience=30, verbose=1, mode='max')
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=60, batch_size=64, class_weight=cw,
                        callbacks=[callbacks_list, Reduce])
    t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    print("开始时间:"+t1+"结束时间："+t2)
    model=deepires_model()
    model.load_weights("cv/%d/"%i).expect_partial()
    y_pred1 = model.predict(X_test)
    y_pred = np.where(y_pred1 > 0.5, 1, 0)
    y_pred = y_pred.reshape(-1,)
    y_pred1 = y_pred1.reshape(-1,)
    y_test = y_test.reshape(-1,)
    tmp_result = np.zeros((len(y_test), 3))
    tmp_result[:, 0], tmp_result[:, 1], tmp_result[:, 2] = y_test, y_pred, y_pred1
    model_prediction_result_cv.append(tmp_result)
    acc[i] = accuracy_score(y_test, y_pred)
    sn[i] = recall_score(y_test, y_pred)
    mcc[i] = matthews_corrcoef(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sp[i] = tn / (tn + fp)
    auroc[i] = roc_auc_score(y_test, y_pred1)
    aupr[i] = average_precision_score(y_test, y_pred1)
    f1[i] = f1_score(y_test, np.round(y_pred1.reshape(-1)))
    fpr, tpr, threshold = roc_curve(y_test, y_pred1)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
            # 计算auc
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.2f)' % (i, roc_auc))

plt.plot([0,1],[0,1],linestyle='--',lw=2,color='r',label='Luck',alpha=.8)
mean_tpr=np.mean(tprs,axis=0)
mean_tpr[-1]=1.0
mean_auc=auc(mean_fpr,mean_tpr)#计算平均AUC值
std_auc=np.std(tprs,axis=0)
plt.plot(mean_fpr,mean_tpr,color='b',label=r'Mean ROC (area=%0.2f)'%mean_auc,lw=2,alpha=.8)
std_tpr=np.std(tprs,axis=0)
tprs_upper=np.minimum(mean_tpr+std_tpr,1)
tprs_lower=np.maximum(mean_tpr-std_tpr,0)
plt.fill_between(mean_tpr,tprs_lower,tprs_upper,color='gray',alpha=.2)
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc='lower right')
plt.show()
print("ACC : ", np.mean(acc))
print("SN : ", np.mean(sn))
print("SP : ", np.mean(sp))
print("MCC : ", np.mean(mcc))
print("AUC : ", np.mean(auroc))
print("AUPR : ", np.mean(aupr))
print("f1_score : ", np.mean(f1))