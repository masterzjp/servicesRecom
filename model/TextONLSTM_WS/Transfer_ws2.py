# coding=utf-8
import keras
from keras import Model, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

import pickle as pl
import numpy as np

from model.TextONLSTM_WS.textONlstm2 import TextONLSTM2

maxlen = 102
max_features = 89098
# max_features = 52695
batch_size = 64
embedding_dims = 300
epochs = 100
#######################调用GPU#################
"""GPU设置为按需增长"""
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
# config.gpu_options.per_process_gpu_memory_fraction = 0.6 # 每个GPU现存上届控制在60%以内
sess = tf.Session(config=config)
KTF.set_session(sess)
##############################查看正在使用的GPU##########################
print (tf.__version__)
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

##################################################################################################################师兄训练的embedding
pretrained_w2v, _, _ = pl.load(open(r'D:\workplaces\pycharmworkDir\ServicesRecommend\data\emb_matrix_glove_300', 'rb'))
################################################################
print('Loading data...')

from sklearn.model_selection import train_test_split


x,y1,y2,y1_pad,y2_pad = pl.load(open(r'D:\workplaces\pycharmworkDir\ServicesRecommend\data\ws_txt_vector300dim_y1y2_2len_100len_zjp0145','rb'))
pre_y1_pad = pl.load(open(r'D:\workplaces\pycharmworkDir\ServicesRecommend\data\predict_labels\py1_id_pad_2','rb'))
# ##################################################################################
# ########################################################################################10240000000000000
x_train, x_test, y2_train, y2_test = train_test_split( x, y2, test_size=0.2, random_state=42)
x_train,x_test,pre_y1_train_pad,pre_y1_test_pad=train_test_split( x, pre_y1_pad, test_size=0.2, random_state=42)
# #嵌入真实父标签########################################################################################################################
# x_train, x_test, y1_train_pad, y1_test_pad = train_test_split( x, y1_pad, test_size=0.2, random_state=42)
#
# emb_label_train = list(np.column_stack((y1_train_pad,x_train)))
# emb_label_test = list(np.column_stack((y1_test_pad,x_test)))

############################
# 20191119-------注释掉 嵌入预测父标签
emb_label_train = list(np.column_stack((pre_y1_train_pad,x_train)))
emb_label_test = list(np.column_stack((pre_y1_test_pad,x_test)))

###############################################################################################1024

#######################################################
# emb_label_train = list(np.column_stack((y1_train_pad,x_train)))
# emb_label_test = list(np.column_stack((y1_test_pad,x_test)))20191022
#############################################################################
model = TextONLSTM2(maxlen, max_features, embedding_dims, pretrained_w2v).get_model()
###################  20191022  ##############################

model.load_weights(r"D:\workplaces\pycharmworkDir\ServicesRecommend\data\weights\Ay1pad_y2_best_weights.h5",by_name=True)

###############################################################################
# adam = optimizers.Adam(lr=0.001)#adam默认学习率0.001 Fine tune 的话学习率比预训练模型小10倍
# model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy',f1])
def acc_top5(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy',acc_top5])
################################################################################


model.summary()
print('Train...')
fileweights = r"D:\workplaces\pycharmworkDir\ServicesRecommend\data\weights\Acontent_best_weights.h5"
checkpoint = ModelCheckpoint(fileweights, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
# checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max')

# 当评价指标不在提升时，减少学习率
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=5, mode='auto')
####################y1标签嵌入到x_train文本中#################################################
# model.fit(x_train, y2_train,
#           # validation_split=0.1,
#           batch_size=batch_size,
#           epochs=epochs,
#           callbacks=[early_stopping, checkpoint,reduce_lr],
#           validation_data=(x_test, y2_test),
#           shuffle= True)
############################################0926########################
model.fit([emb_label_train], y2_train,
          # validation_split=0.1,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[early_stopping, checkpoint,reduce_lr],
          validation_data=([emb_label_test], y2_test),
          shuffle= True)
###################################################################
print('Test...')
# result = model.predict(x_test)
# print(result)
