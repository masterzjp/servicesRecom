
import keras
import pickle as pl
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from model.TextONLSTM_WS.textONlstm import TextONLSTM

maxlen = 100
max_features = 89098
# max_features = 52695
batch_size = 64
embedding_dims = 300
epochs = 100
def get_y1(DATA_text,pretrained_w2v):



    model = TextONLSTM(maxlen, max_features, embedding_dims, pretrained_w2v).get_model()

    model.load_weights(r"D:\workplaces\pycharmworkDir\ServicesRecommend\data\weights\Ay1pad_y2_best_weights.h5",by_name=True)

    # 单独评估一个本来分类
    # DATA_text = []
    # DATA_text.append((x_test[0]))
    # DATA_text = np.array(DATA_text)

    predict = model.predict(DATA_text,batch_size = 1,verbose = 1)

    predict = np.argmax(predict, axis=1)
    import pickle as pl
    pl.dump(predict, open('D:\workplaces\pycharmworkDir\ServicesRecommend\data\inputPredict\ws_layer1_predict_y1', 'wb'))
    ######################index2word####################################

    pretrained_w2v, word_to_id, _ = pl.load(
        open(r'D:\workplaces\pycharmworkDir\ServicesRecommend\data\emb_matrix_glove_300', 'rb'))
    # y1 = ['biochemistry', 'civil', 'computer science', 'electrical', 'mechanical', 'medical', 'psychology']
    y1 = ['business', 'communications', 'computer', 'data management', 'digital media', 'other services', 'recreational activities', 'social undertakings', 'traffic']
    y1_id_pad = []
    label1_id =pl.load(open(r'D:\workplaces\pycharmworkDir\ServicesRecommend\data\inputPredict\ws_layer1_predict_y1','rb'))

    for i in label1_id:
        y1_id_pad.append([x for x in y1[i].split(' ') if x in word_to_id])
        # print(y1[i])
    # print(y1_id_pad)
    return y1_id_pad
