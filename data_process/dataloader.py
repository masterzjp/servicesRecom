#encoding:utf-8
from collections import  Counter
import tensorflow.contrib.keras as kr
# import keras as kr
import numpy as np
import codecs
import re

from nltk import *
from nltk.corpus import stopwords

from sklearn import preprocessing

def isSymbol(inputString):
    return bool(re.match(r'[^\w]', inputString))
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'rb').readlines()]
    return stopwords
def read_file(filename):
    """
    Args:
        filename:trian_filename,test_filename,val_filename 
    Returns:
        two list where the first is lables and the second is contents cut by jieba
        
    """
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z]+)")  # the method of cutting text by punctuation(匹配中文 大小写字母)
    contents,labels=[],[]
    with codecs.open(filename,'r',encoding='gbk') as f:
        for line in f:
            try:
                line=line.rstrip()
                ###############多标签############
                label = []
                labels_content = line.split('\t')
                #工单数据格式：label+"\t"+keywords+"\t"+abstract
                for i in range(len(labels_content[:-1])):
                    if labels_content[i] != '':
                        label.append(labels_content[i])
                labels.append(label)
                #labels_content[-2:]=keywords Abstract
                content = labels_content[-1:]
                # print(content)
                ###########################
                # stopWords = set(stopwords.words('english'))
                stopWords = stopwordslist(r'D:\赵鲸朋\pycharmModel0905\pycharmModel0905\PycharmProjects\Wos-Metadata2txt\output\stopwordds')
                # print(stopWords)
                wordsFiltered = []
                for items in content:
                    # print('items:'+items)
                    word_tok_list = items.split(' ')
                    # word_tok = word_tokenize(items)
                    # print(word_tok_list)
                    for w in word_tok_list:
                        if w not in stopWords and not isSymbol(w) and not hasNumbers(w) and len(w)>=2:
                            wordsFiltered.append(w.rstrip(';').rstrip('.').rstrip(',').rstrip('."').lstrip('"'))
                #####################################################
                # print(word)
                # print(wordsFiltered)
                #####################################################
                # blocks = re_han.split(content)
                # word = []
                # for blk in blocks:
                #     if re_han.match(blk):
                #         for w in jieba.cut(blk):
                #             if len(w)>=2:
                #                 word.append(w)
                contents.append(wordsFiltered)
            except:
                pass
    return labels,contents
# filename = r"D:\赵鲸朋\pycharmModel0905\pycharmModel0905\PycharmProjects\Wos-Metadata2txt\output\test.txt"
# labels,contents = read_file(filename)
# print(labels)
# print(contents)

# config = TextConfig()
# file = r"E:\Workspaces\PycharmProjects\DataProcess\data\cnews.mullabel.txt"
# labels,contents = read_file(file)
# print(labels)
# print(contents)
#labels = ['教育','体育'，'社会'，……]
#contents = [['我们'，'北京','上大学','研究生'],['今天','巴萨','战胜','火箭'],['住房','公积金','上涨']……]

def build_vocab(filenames,vocab_dir,vocab_size):
    """
    Args:
        filename:trian_filename,test_filename,val_filename
        vocab_dir:path of vocab_filename
        vocab_size:number of vocabulary
    Returns:
        writting vocab to vocab_filename

    """
    all_data = []
    for filename in filenames:
        _,data_train=read_file(filename)
        for content in data_train:
            all_data.extend(content)
    counter=Counter(all_data)
    count_pairs=counter.most_common(vocab_size-1)
    words, _ =list(zip(*count_pairs))
    words=['<PAD>']+list(words)

    with codecs.open(vocab_dir,'w',encoding='utf-8') as f:
        f.write('\n'.join(words)+'\n')
# filenames = [filename]
# vocab_dir = r"./output/testvocab2.txt"
# build_vocab(filenames,vocab_dir,8000)
def read_vocab(vocab_dir):
    """
    Args:
        filename:path of vocab_filename
    Returns:
        words: a list of vocab
        word_to_id: a dict of word to id
        
    """
    words=codecs.open(vocab_dir,'r',encoding='utf-8').read().strip().split('\n')
    word_to_id=dict(zip(words,range(len(words))))
    return words,word_to_id

def read_category():
    """
    Args:
        None
    Returns:
        categories: a list of label
        cat_to_id: a dict of label to id

    """
    ##DBPedia
    # y1 = ['agent', 'device', 'event', 'place', 'species', 'sportsseason', 'topicalconcept', 'unitofwork', 'work']
    # y2 = ['actor', 'amusementparkattraction', 'animal', 'artist', 'athlete', 'bodyofwater', 'boxer', 'britishroyalty', 'broadcaster', 'building', 'cartoon', 'celestialbody', 'cleric', 'clericaladministrativeregion', 'coach', 'comic', 'comicscharacter', 'company', 'database', 'educationalinstitution', 'engine', 'eukaryote', 'fictionalcharacter', 'floweringplant', 'footballleagueseason', 'genre', 'gridironfootballplayer', 'group', 'horse', 'infrastructure', 'legalcase', 'motorcyclerider', 'musicalartist', 'musicalwork', 'naturalevent', 'naturalplace', 'olympics', 'organisation', 'organisationmember', 'periodicalliterature', 'person', 'plant', 'politician', 'presenter', 'race', 'racetrack', 'racingdriver', 'routeoftransportation', 'satellite', 'scientist', 'settlement', 'societalevent', 'software', 'song', 'sportfacility', 'sportsevent', 'sportsleague', 'sportsmanager', 'sportsteam', 'sportsteamseason', 'station', 'stream', 'tournament', 'tower', 'venue', 'volleyballplayer', 'wintersportplayer', 'wrestler', 'writer', 'writtenwork']
    # y3 = ['academicjournal', 'adultactor', 'airline', 'airport', 'album', 'amateurboxer', 'ambassador', 'americanfootballplayer', 'amphibian', 'animangacharacter', 'anime', 'arachnid', 'architect', 'artificialsatellite', 'artistdiscography', 'astronaut', 'australianfootballteam', 'australianrulesfootballplayer', 'automobileengine', 'badmintonplayer', 'band', 'bank', 'baronet', 'baseballleague', 'baseballplayer', 'baseballseason', 'basketballleague', 'basketballplayer', 'basketballteam', 'beachvolleyballplayer', 'beautyqueen', 'biologicaldatabase', 'bird', 'bodybuilder', 'brewery', 'bridge', 'broadcastnetwork', 'buscompany', 'businessperson', 'canadianfootballteam', 'canal', 'canoeist', 'cardinal', 'castle', 'cave', 'chef', 'chessplayer', 'christianbishop', 'classicalmusicartist', 'classicalmusiccomposition', 'collegecoach', 'comedian', 'comicscreator', 'comicstrip', 'congressman', 'conifer', 'convention', 'cricketer', 'cricketground', 'cricketteam', 'crustacean', 'cultivatedvariety', 'curler', 'cycad', 'cyclingrace', 'cyclingteam', 'cyclist', 'dam', 'dartsplayer', 'diocese', 'earthquake', 'economist', 'election', 'engineer', 'entomologist', 'eurovisionsongcontestentry', 'fashiondesigner', 'fern', 'figureskater', 'filmfestival', 'fish', 'footballmatch', 'formulaoneracer', 'fungus', 'gaelicgamesplayer', 'galaxy', 'glacier', 'golfcourse', 'golfplayer', 'golftournament', 'governor', 'grandprix', 'grape', 'greenalga', 'gymnast', 'handballplayer', 'handballteam', 'historian', 'historicbuilding', 'hockeyteam', 'hollywoodcartoon', 'horserace', 'horserider', 'horsetrainer', 'hospital', 'hotel', 'icehockeyleague', 'icehockeyplayer', 'insect', 'jockey', 'journalist', 'judge', 'lacrosseplayer', 'lake', 'lawfirm', 'legislature', 'library', 'lighthouse', 'magazine', 'manga', 'martialartist', 'mayor', 'medician', 'memberofparliament', 'militaryconflict', 'militaryperson', 'militaryunit', 'mixedmartialartsevent', 'model', 'mollusca', 'monarch', 'moss', 'mountain', 'mountainpass', 'mountainrange', 'museum', 'musical', 'musicfestival', 'musicgenre', 'mythologicalfigure', 'nascardriver', 'nationalfootballleagueseason', 'ncaateamseason', 'netballplayer', 'newspaper', 'noble', 'officeholder', 'olympicevent', 'painter', 'philosopher', 'photographer', 'planet', 'play', 'playboyplaymate', 'poem', 'poet', 'pokerplayer', 'politicalparty', 'pope', 'president', 'primeminister', 'prison', 'publictransitsystem', 'publisher', 'racecourse', 'racehorse', 'radiohost', 'radiostation', 'railwayline', 'railwaystation', 'recordlabel', 'religious', 'reptile', 'restaurant', 'river', 'road', 'roadtunnel', 'rollercoaster', 'rower', 'rugbyclub', 'rugbyleague', 'rugbyplayer', 'saint', 'school', 'screenwriter', 'senator', 'shoppingmall', 'single', 'skater', 'skier', 'soapcharacter', 'soccerclubseason', 'soccerleague', 'soccermanager', 'soccerplayer', 'soccertournament', 'solareclipse', 'speedwayrider', 'sportsteammember', 'squashplayer', 'stadium', 'sumowrestler', 'supremecourtoftheunitedstatescase', 'swimmer', 'tabletennisplayer', 'televisionstation', 'tennisplayer', 'tennistournament', 'theatre', 'town', 'tradeunion', 'university', 'videogame', 'village', 'voiceactor', 'volcano', 'winery', 'womenstennisassociationtournament', 'wrestlingevent']
    # y1_to_id=dict(zip(y1,range(len(y1))))
    # y2_to_id = dict(zip(y2, range(len(y2))))
    # y3_to_id = dict(zip(y3, range(len(y3))))
    ##ws
    y1 = ['business', 'communications', 'computer', 'data management', 'digital media', 'other services', 'recreational activities', 'social undertakings', 'traffic']
    # y1 = ['communications', 'data processing', 'digital media', 'economic', 'information technology', 'logistics', 'office', 'organization', 'other services', 'recreational activities', 'social undertakings']
    y2 = ['advertising', 'analytics', 'application development', 'backend', 'banking', 'bitcoin', 'chat', 'cloud', 'data', 'database', 'domains', 'ecommerce', 'education', 'email', 'enterprise', 'entertainment', 'events', 'file sharing', 'financial', 'games', 'government', 'images', 'internet of things', 'mapping', 'marketing', 'media', 'medical', 'messaging', 'music', 'news services', 'other', 'payments', 'photos', 'project management', 'real estate', 'reference', 'science', 'search', 'security', 'shipping', 'social', 'sports', 'stocks', 'storage', 'telephony', 'tools', 'transportation', 'travel', 'video', 'weather']
    y1_to_id=dict(zip(y1,range(len(y1))))
    y2_to_id = dict(zip(y2, range(len(y2))))
    return y1,y1_to_id,y2,y2_to_id

def read_files(filename):
    contents, labels1, labels2 = [], [], []
    i = 0
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                content = line.split(' ')
                stopWords = stopwordslist(
                    r'D:\赵鲸朋\pycharmModel0905\pycharmModel0905\PycharmProjects\Wos-Metadata2txt\output\stopwordds')
                # print(stopWords)
                ###################contents############################
                # wordsFiltered = []
                # for w in content:
                #     if w not in stopWords and not isSymbol(w) and not hasNumbers(w) and len(w) >= 2:
                #         wordsFiltered.append(w.rstrip('\n').rstrip('\r'))
                # contents.append(wordsFiltered)
                #####################label_y1 y2###################################
                wordsFiltered = []
                for w in content:
                    if len(w)>=2:
                        wordsFiltered.append(w.rstrip('\n').rstrip('\r'))
                contents.append(wordsFiltered)
                #######################################################
                i=i+1
            except:
                pass
    print(len(contents))
    return contents
# filename = r"D:\赵鲸朋\pycharmModel0905\pycharmModel0905\PycharmProjects\Wos-Metadata2txt\data\wos\wos_clear_content.txt"
# cont = read_files(filename)
# # print(y1[:5])
# # # print(y2[:5])
# print(cont[:5])
##20200415注释
# y1_file = r"D:\赵鲸朋\pycharmModel0905\pycharmModel0905\PycharmProjects\Wos-Metadata2txt\data\wos\wos_clear_y1.txt"
# y1 = read_files(y1_file)
# print(y1)
def process_input(cont_file,word_to_id,max_length=100):
    contents = read_files(cont_file)
    data_id = []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
    cont_pad=kr.preprocessing.sequence.pad_sequences(data_id,max_length,padding='post', truncating='post')
    return cont_pad


def process_file(cont_file,y1_file,y2_file, word_to_id,y1_to_id,y2_to_id, max_length=300,y1_length = 2,y2_length = 2):
    """
    Args:
        filename:train_filename or test_filename or val_filename
        word_to_id:get from def read_vocab()
        cat_to_id:get from def read_category()
        max_length:allow max length of sentence 
    Returns:
        x_pad: sequence data from  preprocessing sentence 
        y_pad: sequence data from preprocessing label

    """
    contents=read_files(cont_file)
    y1 = read_files(y1_file)
    y2 = read_files(y2_file)
    # y3 = read_files(y3_file)
    data_id,y1_id,y2_id,y3_id=[],[],[],[]
    y1_id_pad,y2_id_pad,y3_id_pad = [],[],[]
    label_y1 = []
    label_y2 = []
    label_y3 = []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        y1_id_pad.append([word_to_id[x] for x in y1[i] if x in word_to_id])
        y2_id_pad.append([word_to_id[x] for x in y2[i] if x in word_to_id])
        # y3_id_pad.append([word_to_id[x] for x in y3[i] if x in word_to_id])
        ##############y[i]=['computer','science']转化为y[i]=['computer science']#################################

        str = ""
        for label in y1[i]:
            str = str+ label + " "
        label_y1.append(str.rstrip(' '))
        # label_id.append(label_idd)

        str2 = ""
        for label in y2[i]:
            str2 = str2 + label + " "
        label_y2.append(str2.rstrip(' '))

        # str3 = ""
        # for label in y3[i]:
        #     str3 = str3 + label + " "
        # label_y3.append(str3.rstrip(' '))

        y1_id.append(y1_to_id[label_y1[i]])
        y2_id.append(y2_to_id[label_y2[i]])
        # y3_id.append(y3_to_id[label_y3[i]])
        ###############################################
        # y1_id.append(y1_to_id[y1[i]])
        # y2_id.append(y2_to_id[y2[i]])
    cont_pad=kr.preprocessing.sequence.pad_sequences(data_id,max_length,padding='post', truncating='post')
    y1_pad = kr.preprocessing.sequence.pad_sequences(y1_id_pad, y1_length, padding='post', truncating='post')
    y2_pad = kr.preprocessing.sequence.pad_sequences(y2_id_pad, y2_length, padding='post', truncating='post')
    # y3_pad = kr.preprocessing.sequence.pad_sequences(y3_id_pad, y3_length, padding='post', truncating='post')
    ##################################
    y1_index = kr.utils.to_categorical(y1_id)
    y2_index = kr.utils.to_categorical(y2_id)
    # y3_index = kr.utils.to_categorical(y3_id)
    #####################################

    return cont_pad,y1_index,y2_index,y1_pad,y2_pad

def batch_iter(x,y,batch_size=64):
    """
    Args:
        x: x_pad get from def process_file()
        y:y_pad get from def process_file()
    Yield:
        input_x,input_y by batch size

    """

    data_len=len(x)
    num_batch=int((data_len-1)/batch_size)+1

    indices=np.random.permutation(np.arange(data_len))
    x_shuffle=x[indices]
    y_shuffle=y[indices]

    for i in range(num_batch):
        start_id=i*batch_size
        end_id=min((i+1)*batch_size,data_len)
        yield x_shuffle[start_id:end_id],y_shuffle[start_id:end_id]

def export_word2vec_vectors(vocab, word2vec_dir,trimmed_filename):
    """
    Args:
        vocab: word_to_id 
        word2vec_dir:file path of have trained word vector by word2vec
        trimmed_filename:file path of changing word_vector to numpy file
    Returns:
        save vocab_vector to numpy file
        
    """
    file_r = codecs.open(word2vec_dir, 'r', encoding='utf-8')
    line = file_r.readline()
    voc_size, vec_dim = map(int, line.split(' '))
    embeddings = np.zeros([len(vocab), vec_dim])
    line = file_r.readline()
    while line:
        try:
            items = line.split(' ')
            word = items[0]
            vec = np.asarray(items[1:], dtype='float32')
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(vec)
        except:
            pass
        line = file_r.readline()
    np.savez_compressed(trimmed_filename, embeddings=embeddings)

def get_training_word2vec_vectors(filename):
    """
    Args:
        filename:numpy file
    Returns:
        data["embeddings"]: a matrix of vocab vector
    """
    with np.load(filename) as data:
        return data["embeddings"]
