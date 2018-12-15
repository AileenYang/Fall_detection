# order = 5
# fs = 200.0      # sample rate, Hz
# cutoff = 5.0    # desired cutoff frequency of the filter, Hz
# IIR implemention
from scipy import integrate
from scipy.signal import butter, lfilter, freqz
from sklearn.decomposition import PCA
from scipy.signal import butter, lfilter
import os
import sklearn
import math
import numpy as np
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff=5, fs=200, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
# to get physical characteristics
def get_user_propties(source_path='SisFall_dataset/physical_ch.txt'):
    f = open(source_path, mode='r', encoding='utf-8')
    datas = {}
    for line in f:
        line = line.split('|')
        line = [x.strip() for x in line]
        key = line[0]
        value = line[1:-1]
        for i in range(len(value) - 1):
            value[i] = float(value[i])
        if value[-1] == 'M':
            value[-1] = 0
        else:
            value[-1] = 1
        datas[key] = value
    f.close()
    return datas

# min-max normalization
def min_max(a):
    amin, amax = a.min(), a.max()
    a = (a - amin) / (amax - amin)
    return a

# return file names
def get_all_data(path):
    if not os.path.isdir(path):
        return []
    results = os.listdir(path)
    files = []
    for file in results:
        if file[0] in ['D', 'F']:
            files.append(file)
    files = [path + "/" + file for file in files]
    return files

# get data
def get_all_datas(path, datas):
    files = os.listdir(path)
    files = [path + "/" + file for file in files]
    for file in files:
        data = get_all_data(file)
        datas.extend(data)
    print(len(datas))

# process each line of file
def process_line(line):
    line = line.strip()
    line = line[:-1]
    nums = line.split(',')
    try:
        nums = [int(num) for num in nums]
    except:
        print('--',line)
    return nums

# feature extraction
def GetC1(datas):
    datan = np.array(datas)
    datan = datan ** 2
    datan = np.sum(datan, axis=1)
    datan = datan ** 0.5
    datan = np.mean(datan)
    return datan

def Get_std(datas):
    datan = np.array(datas)
    datan = np.std(datan, axis=0)
    horiz_std_mag9 = np.sqrt(datan[0]**2 + datan[2]**2)
    std_mag9 = np.sqrt(datan[0]**2 + datan[1]**2 + datan[2]**2)
    diff_std_mag9 = np.sqrt(datan[3]**2 + datan[4]**2 + datan[5]**2)
    horiz_std_mag2 = np.sqrt(datan[3]**2 + datan[5]**2)
    gyro_horiz_std_mag = np.sqrt(datan[6]**2 + datan[8]**2)
    gyro_std_mag = np.sqrt(datan[6]**2 + datan[7]**2 + datan[8]**2)
    datan = np.append([datan], [gyro_std_mag])
    datan = np.append([datan], [gyro_horiz_std_mag])
    datan = np.append([datan], [horiz_std_mag2])
    datan = np.append([datan], [diff_std_mag9])
    datan = np.append([datan], [std_mag9])
    datan = np.append([datan],[horiz_std_mag9])
    return datan

def Get_SigMagArea(datas):
    ntime = len(datas)
    sum = 0
    for i in range(3):
        sum += np.sum(datas[:,i])
    return sum/ntime

def Get_HorizSigMagArea(datas):
    ntime = len(datas)
    sum = 0
    for i in [0, 2]:
        sum += np.sum(datas[:, i])
    return sum / ntime

def Get_Amax(datas):
    datan = np.array(datas)
    datan = np.max(datan, axis=0)
    return datan

def Get_Amin(datas):
    datan = np.array(datas)
    datan = np.min(datan, axis=0)
    return datan

def Get_peak_diff(datas):
    datan = np.array(datas)
    return np.max(datan, axis=0) - np.min(datan, axis=0)

def Get_angle(datas):
    datan = np.array(datas)
    angle_from_horiz = np.arctan2(np.sqrt(datan[0] ** 2 + datan[2] ** 2), -datan[1]) * 180 / np.pi
    angle = np.append([np.max(angle_from_horiz)],[np.min(angle_from_horiz)])
    angle = np.append(angle, [np.mean(angle_from_horiz)])
    return angle

def GetC10(datas):
    f = lambda x: x
    N = len(datas)
    datan = np.array(datas)
    for i in range(9):
        if 1 % 3 == 2:
            continue
        datan[:, i] = datan[:, i] ** 2
    dataf = []
    for i in range(3):
        k = i * 3
        x = []
        for j in range(N):
            x.append(math.atan2((datan[j, k + 1] + datan[j, k]) ** 0.5, datan[j, k + 2]))
        x = np.mean(x)
        dataf.append(x)
    return np.array(dataf)

'''
file-->train data
a) file name-->train data: D means ADL, F means fall
b) cal avg
c) add extracted feature
d) add user info
'''
def get_one_data(path, pros):
    relaPath = path.split('/')[-1]
    proKey = path.split('/')[-2].strip()
    label = -1
    datas = []
    if relaPath[0] == 'D':
        label = 0
    elif relaPath[0] == 'F':
        label = 1
    else:
        return None
    f = open(path, 'r', encoding='utf-8')
    for line in f:
        key = process_line(line)
        if len(key) != 9:
            continue
        datas.append(key)
    f.close()
    tupleDatas = []

    # for i in range(0,len(datas)-256, 128):
    #     datan = datas[i:i+256]
    #     datan = np.array(datan)
    #     datatmp = np.array(datan)
    #     c1 = GetC1(datan)
    #     c10 = GetC10(datan)
    #     # for i in range(len(datas)):
    #         # datas[i,:] = butter_lowpass_filter(datas[i,:])
    #     datan = np.mean(datan, axis=0)
    #     datan = np.append([datan], [Get_SigMagArea(datatmp)])
    #     datan = np.append([datan], [Get_std(datatmp)])
    #     datan = np.append([datan], [Get_Amax(datatmp)])
    #     datan = np.append([datan], [Get_Amin(datatmp)])
    #
    #     datan = np.std(datan,axis = 0)
    #     datan = np.append([datan],[c1])
    #     datas = np.append([datas],[c10])
    #     tupleDatas.append((datan,label))

    datan = datas[:]
    datan = np.array(datan)
    datatmp = np.array(datan)
    c1 = GetC1(datan)
    c10 = GetC10(datan)
    datan = np.mean(datan, axis=0)
    datan = np.append([datan], [Get_SigMagArea(datatmp)])
    datan = np.append([datan], [Get_std(datatmp)])
    datan = np.append([datan], [Get_Amax(datatmp)])
    datan = np.append([datan], [Get_Amin(datatmp)])
    datan = np.append([datan], [Get_peak_diff(datatmp)])
    datan = np.append([datan], [Get_angle(datatmp)])
    datan = np.append([datan],[c1])
    # user = pros.get(proKey)
    # datan = np.append([datan], [user])
    tupleDatas.append((datan,label))

    return tupleDatas


def get_data_and_labels(trainDatas, testDatas, files):
    datas = []
    index = 0
    pros = get_user_propties()
    for file in files:
        data = get_one_data(file, pros)
        if data:
            datas.extend(data)
    length = len(datas)
    trainDatas.extend(datas[:int(length * 0.7)])
    testDatas.extend(datas[int(length * 0.7):])
    print('trainDatas', len(trainDatas))
    print('testDatas', len(testDatas))

def getXandY(datas):
    X = []
    Y = []
    for data in datas:
        x = data[0]
        Y.append(data[1])
        where_are_nan = np.isnan(x)
        where_are_inf = np.isinf(x)
        x[where_are_nan] = 0
        x[where_are_inf] = 0
        X.append(x)
    return X, Y

def norm_pro(trainX,testX):
    len1 = len(trainX)
    len2 = len(testX)
    X = []
    X.extend(trainX)
    X.extend(testX)
    X = np.array(X)
    print(X.shape)
    pca = PCA(n_components=20)
    print('begin pca', X.shape)
    X = pca.fit_transform(X)
    print('after pca',X.shape)
    return X[:len1],X[len1:]

def getProbability(trainY,testY):
    sumP1 = 0.0
    sumP2 = 0.0
    for y in trainY:
        if y == 1:
            sumP1 += 1
        else:
            sumP2 += 1
    for y  in testY:
        if y == 1:
            sumP1 += 1
        else:
            sumP2 += 1
    Pro1 = sumP1/(sumP1 + sumP2)
    Pro2 = sumP2/(sumP1 + sumP2)
    print('pro of two catagories', Pro1, Pro2)
