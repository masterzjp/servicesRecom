import xlrd
from sklearn.model_selection import train_test_split

import pickle as pl
import numpy as np
# from model_others.webServiceRecommend.get_label import get_labelset
from recommend.predict_y1 import get_y1
from recommend.predict_y2 import get_y2
from data_process.dataloader import process_input
pretrained_w2v, word_to_id, _ = pl.load(open(r'D:\workplaces\pycharmworkDir\ServicesRecommend\data\emb_matrix_glove_300', 'rb'))

cont_file = r"D:\workplaces\pycharmworkDir\ServicesRecommend\recommend\description.txt"

seq_length = 100
x_test= process_input(cont_file,word_to_id,seq_length)
######################################输入服务描述x_test[0]##############################################
serviceDescription = x_test[0]
########################################################################################################
DATA_text = []
DATA_text.append((serviceDescription))
DATA_text = np.array(DATA_text)
def get_labelset():
    label_y1 = get_y1(DATA_text,pretrained_w2v)
    label_y2 = get_y2(DATA_text,pretrained_w2v)
    # print(label_y1)
    # print(label_y2)
    label_set = set()
    y1 = label_y1[0][0]
    y2 = label_y2[0][0]
    label_set.add(y1)
    label_set.add(y2)
    print('该服务所属类别为：')
    print(label_set)
    return label_set

def get_RecommendService(excelpath,txtpath,count):
    excel = xlrd.open_workbook(excelpath, encoding_override='utf-8')
    sheet_excel = excel.sheets()[0]  # 选定表
    nrows = sheet_excel.nrows  # 获取行号
    ncols = sheet_excel.ncols  # 获取列号
    # print(nrows)
    dicts = dict()
    for i in range(1,nrows):
        cache_row = set()
        cache_row.add(sheet_excel.cell_value(i,1).lower())
        cache_row.add(sheet_excel.cell_value(i,2).lower())
        # print(cache_row)
        dicts[i] = (len(input&cache_row)/len(input|cache_row))
        # print(dicts[i])
    L = sorted(dicts.items(), key=lambda item: item[1], reverse=True)
    L = L[:count]
    print('推荐的服务为：（服务id,相似度）')
    print(L)
    # # print(cache_alldata)
    with open(txtpath,'w') as f:
        f.write(str(L))
input = get_labelset()
#服务仓库路径
excelpath = r"D:\workplaces\pycharmworkDir\ServicesRecommend\data\services_Repository\Service0415.xlsx"
#推荐服务ID存储路径
txtpath = r"D:\workplaces\pycharmworkDir\ServicesRecommend\recommend\output_Reco_Services\recommendServices.txt"
count = 5
#输出格式为（服务id,该服务与需求服务描述的相似度）
get_RecommendService(excelpath,txtpath,count)

