
import pickle as pl
from sklearn.model_selection import train_test_split
cont_pad,y1_index,y2_index,y1_pad,y2_pad = pl.load(open(r'D:\赵鲸朋\pycharmModel0905\pycharmModel0905\mymodel - Englishtext\data\wos\WOSDATA_txt_vector500dimsy1y2_10dim_zjp','rb'))
x_train, x_test, y2_train, y2_test = train_test_split( cont_pad, y2_index, test_size=0.2, random_state=42)
print(x_train[:3])
print(x_test[:3])
x_train, x_test, y1_train, y1_test = train_test_split( cont_pad, y1_index, test_size=0.2, random_state=42)
print(x_train[:3])
print(x_test[:3])
x_train, x_test, y1_train_pad, y1_test_pad = train_test_split( cont_pad, y1_pad, test_size=0.2, random_state=42)
x_train, x_test, y2_train_pad, y2_test_pad = train_test_split( cont_pad, y2_pad, test_size=0.2, random_state=42)
with open(r'D:\赵鲸朋\pycharmModel0905\pycharmModel0905\mymodel - Englishtext\model_others\TextONLSTM_wos\wos_train_test_split\train_txt_len_500_ypad_10_2', 'wb') as f:
    pl.dump((x_train,y1_train,y2_train,y1_train_pad,y2_train_pad), f)
with open(r'D:\赵鲸朋\pycharmModel0905\pycharmModel0905\mymodel - Englishtext\model_others\TextONLSTM_wos\wos_train_test_split\test_txt_len_500_ypad_10_2', 'wb') as f:
    pl.dump((x_test,y1_test,y2_test,y1_test_pad,y2_test_pad), f)