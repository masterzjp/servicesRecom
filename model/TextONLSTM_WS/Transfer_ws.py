# coding=utf-8
import keras
from keras import Model, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import pickle as pl
import numpy as np
# from metrics import f1
from model.TextONLSTM_WS.textONlstm import TextONLSTM

maxlen = 100

max_features = 89098
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
############################################################################################################################
pretrained_w2v, _, _ = pl.load(open(r'D:\workplaces\pycharmworkDir\ServicesRecommend\data\emb_matrix_glove_300', 'rb'))
#############################################################################################################################
print('Loading data...')
x,y1,y2,y1_pad,y2_pad =pl.load(open(r'D:\workplaces\pycharmworkDir\ServicesRecommend\data\ws_txt_vector300dim_y1y2_2len_100len_zjp0145','rb'))
x_train,x_test,y2_train,y2_test = train_test_split( x, y2, test_size=0.2, random_state=42)
x_train,x_test,y1_train,y1_test = train_test_split( x, y1, test_size=0.2, random_state=42)
##########################################################################################################################

print('Build model...')

model = TextONLSTM(maxlen, max_features, embedding_dims, pretrained_w2v).get_model()
######################设置top5精度######################################################################################
def acc_top5(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)
##########################################################################################################################
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy',acc_top5])

model.summary()
print('Train...')
fileweights = r"D:\workplaces\pycharmworkDir\ServicesRecommend\data\weights\Ay1pad_y2_best_weights.h5"
checkpoint = ModelCheckpoint(fileweights, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max')
# 当评价指标不在提升时，减少学习率
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=3, mode='auto')
###########训练层次标签之间权重##########################################################################################
model.fit(x_train, y1_train,
          # validation_split=0.1,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[early_stopping, checkpoint,reduce_lr],
          validation_data=(x_test, y1_test),
          shuffle= True)
####################################
# model.fit([x_train], y1,
#           # validation_split=0.1,
#           batch_size=batch_size,
#           epochs=epochs,
#           callbacks=[early_stopping, checkpoint],
#           validation_data=([x_test], ty1),
#           shuffle= True)
###################################################################20191206注释下面全部
print('Test...')
predict = model.predict([x])
predict = np.argmax(predict, axis=1)
print(predict)
print(np.shape(predict))
pl.dump(predict, open('D:\workplaces\pycharmworkDir\ServicesRecommend\data\predict_labels\ws_layer1_predict1_2', 'wb'))

# result = model.predict(x_test)
# print(result)
######################################
import pickle as pl
import tensorflow.contrib.keras as kr
pretrained_w2v, word_to_id, _ = pl.load(
    open(r'D:\workplaces\pycharmworkDir\ServicesRecommend\data\emb_matrix_glove_300', 'rb'))

y1 = ['business', 'communications', 'computer', 'data management', 'digital media', 'other services', 'recreational activities', 'social undertakings', 'traffic']

y1_id_pad = []
label1_id =pl.load(open(r'D:\workplaces\pycharmworkDir\ServicesRecommend\data\predict_labels\ws_layer1_predict1_2','rb'))

for i in label1_id:
    y1_id_pad.append([word_to_id[x] for x in y1[i].split(' ') if x in word_to_id])
    # print(y1[i])
print(len(y1_id_pad))
print(y1_id_pad[:10])
y1_length = 2
y1_pad = kr.preprocessing.sequence.pad_sequences(y1_id_pad, y1_length, padding='post', truncating='post')
with open(r'D:\workplaces\pycharmworkDir\ServicesRecommend\data\predict_labels\py1_id_pad_2', 'wb') as f:
    pl.dump(y1_pad, f)
#######################################