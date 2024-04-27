
# In[ ]:
import os
from numpy import interp
from scipy.stats import stats
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.regularizers import l1, l2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,accuracy_score,recall_score,matthews_corrcoef,confusion_matrix,roc_curve, precision_recall_curve,auc
import numpy as np
from model import CNN_model,GRU_model,CNN_GRU_model,CNN_GRU_ATT_model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from datetime import datetime
from sklearn.metrics import roc_auc_score,average_precision_score, f1_score,recall_score,matthews_corrcoef,confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
from model import binary_focal_loss,deepires_model
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
X_train,X_test,y_train, y_test =train_test_split(X_tra,y_tra ,test_size=0.1, random_state=42)
model=deepires_model()
model.summary()
filepath = '../weights/new/'
checkpoint = ModelCheckpoint(filepath, monitor='val_auc',verbose=1, save_best_only=True, save_weights_only=True,mode='max')
callbacks_list = checkpoint
back = EarlyStopping(monitor='val_auc', patience=20, verbose=1, mode='max')
Reduce=ReduceLROnPlateau(monitor='val_auc',
                         factor=0.1,
                         patience=10,
                         verbose=1,
                         mode='max',
                         min_delta=0.0001,
                         cooldown=0,
                         min_lr=0)
model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),metrics=['accuracy','AUC'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10,batch_size=64,class_weight=cw,
                    callbacks=[callbacks_list,back,Reduce])
acc = history.history['val_accuracy']
loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
pyplot.title('Model%d.tf' )
pyplot.plot(epochs, acc, 'red', label='Validation acc')
pyplot.plot(epochs, loss, 'blue', label='Validation loss')
pyplot.legend()
pyplot.show()
y_pred1 = model.predict(X_test)
y_pred = np.where(y_pred1>0.5,1,0)
acc = accuracy_score(y_test, y_pred)
sn = recall_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
pre=tp/(tp+fp)
sp = tn / (tn + fp)
auroc = roc_auc_score(y_test, y_pred1)
aupr = average_precision_score(y_test, y_pred)
#f1 = f1_score(y_test, np.round(y_pred1.reshape(-1)))
fpr, tpr, threshold = roc_curve(y_test, y_pred)
tprs=[]
aucs=[]
tprs.append(interp(mean_fpr, fpr, tpr))
tprs[-1][0] = 0.0
            # 计算auc
roc_auc = auc(fpr, tpr)
aucs.append(roc_auc)

print("ACC : ", acc)
print("SN : ", sn)
print("SP : ", sp)
print("PRE : ", pre)
print("MCC : ", mcc)
print("AUC : ", auroc)
print("AUPR : ", aupr)
#print("f1_score : ", f1)
