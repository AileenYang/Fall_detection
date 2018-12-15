from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# from preprocess import butter_lowpass_filter
from preprocess import get_all_datas,get_data_and_labels,getXandY,norm_pro,getProbability

def scoreother_measured(pro, Y):
    y_ = np.argmax(pro, axis=1)
    sum = len(Y)
    fenzi = 0.0
    Tp = 0
    Tn = 0
    Fp = 0
    Fn = 0
    for i in range(len(Y)):
        if Y[i] == 1 and y_[i] == 1:
            Tp += 1
        elif Y[i] == 1 and y_[i] == 0:
            Fn += 1
        elif Y[i] == 0 and y_[i] == 1:
            Fp += 1
        elif Y[i] == 0 and y_[i] == 0:
            Tn += 1
    precision   = Tp*100/(Tp+Fp) if Tp+Fp!=0 else 'Tp+Fp = 0'
    sensitivity = Tp*100/(Tp+Fn) if Tp+Fn!=0 else 'Tp+Fn = 0'
    specificity = Tn*100/(Fp+Tn) if Fp+Tn!=0 else 'Fp+Tn = 0'
    if isinstance(precision,str) or isinstance(sensitivity,str):
        f1 = 'could not calculate it correctly'
    else:
        f1 = 2.0*sensitivity*precision*100 /(sensitivity + precision)
    return {'sensitivity(recall)': sensitivity, 'precision': precision, 'F1': f1, 'specificity': specificity}

if __name__ == '__main__':
    trainDatas = []
    testDatas = []
    datas = []
    path = "./SisFall_dataset"
    get_all_datas(path, datas)
    get_data_and_labels(trainDatas, testDatas, datas)
    trainX, trainY = getXandY(trainDatas)
    testX, testY = getXandY(testDatas)
    trainX,testX = norm_pro(trainX, testX)
    getProbability(trainY, testY)
    print(np.array(testX).shape)
    print(np.array(trainX).shape)


    clf = GaussianNB()
    clf.fit(trainX, trainY)
    proY = clf.predict_proba(testX)
    print("\n====== NB ======")
    print(">>> Gaussian NB")
    print('other measure: ',scoreother_measured(proY,testY,))
    print('acc: ', clf.score(np.array(testX), np.array(testY)))

    print(">>> Bernoulli NB")
    clf = BernoulliNB()
    clf.fit(trainX, trainY)
    proY = clf.predict_proba(testX)
    print('other measure: ', scoreother_measured(proY, testY))
    print('acc: ', clf.score(np.array(testX), np.array(testY)))

    print("\n====== MLP ======")
    print("------ #layer  + #size------")
    for c in [(50,), (100,), (150,), (100, 100,), (100, 100, 100,)]:
        clf = MLPClassifier(hidden_layer_sizes=c)  #
        clf.fit(trainX, trainY)
        proY = clf.predict_proba(testX)
        print(">>> layer, size: ", c)
        print('other measure: ', scoreother_measured(proY, testY))
        print('acc: ', clf.score(np.array(testX), np.array(testY)))

    print("------ activation function ------")
    for c in ['identity', 'logistic', 'tanh', 'relu']:
        clf = MLPClassifier(activation=c)  #
        clf.fit(trainX, trainY)
        proY = clf.predict_proba(testX)
        print(">>> activation function: ", c)
        print('other measure: ', scoreother_measured(proY, testY))
        print('acc: ', clf.score(np.array(testX), np.array(testY)))

    print("\n====== Decision Tree ======")
    print("criterion")
    for c in ['gini', 'entropy']:
        clf = DecisionTreeClassifier(criterion=c)
        clf.fit(trainX, trainY)
        proY = clf.predict_proba(testX)
        print(">>> criterion: ", c)
        print('other measure: ', scoreother_measured(proY, testY))
        print('acc: ',clf.score(np.array(testX), np.array(testY)))
    for c in ['best', 'random']:
        clf = DecisionTreeClassifier(splitter=c)
        clf.fit(trainX, trainY)
        proY = clf.predict_proba(testX)
        print('add other_measured  acc', scoreother_measured(proY, testY))
        print('DecisionTreeClassifier acc when splitter = ', c, clf.score(np.array(testX), np.array(testY)))
    for c in [10, 20, 30, None]:
        clf = DecisionTreeClassifier(max_depth=c)
        clf.fit(trainX, trainY)
        proY = clf.predict_proba(testX)
        print(">>> depth of tree: ", c)
        print('other measure: ', scoreother_measured(proY, testY))
        print('acc: ',clf.score(np.array(testX), np.array(testY)))
    for c in [2,5,7,9,10,12]:
        clf = DecisionTreeClassifier(min_samples_split=c)
        clf.fit(trainX, trainY)
        proY = clf.predict_proba(testX)
        print('add other_measured  acc', scoreother_measured(proY, testY))
        print('DecisionTreeClassifier acc when min_samples_split = ', c, clf.score(np.array(testX), np.array(testY)))
    for c in [1,2,3,4,5]:
        clf = DecisionTreeClassifier(min_samples_leaf=c)
        clf.fit(trainX, trainY)
        proY = clf.predict_proba(testX)
        print('add other_measured  acc', scoreother_measured(proY, testY))
        print('DecisionTreeClassifier acc when min_samples_leaf = ', c, clf.score(np.array(testX), np.array(testY)))

    print("\n====== k-NN ======")
    for c in [3,5,7,9]:
        clf = KNeighborsClassifier(n_neighbors=c)
        clf.fit(trainX, trainY)
        proY = clf.predict_proba(testX)
        print(">>> k-value: ", c)
        print('other measure: ', scoreother_measured(proY, testY))
        print('acc:', clf.score(np.array(testX), np.array(testY)))

    print("\n====== Random Forest ======")
    for c in [10,30,50,70,90,100,120]:
        clf = RandomForestClassifier(n_estimators=c)
        clf.fit(trainX, trainY)
        proY = clf.predict_proba(testX)
        print(">>> # tree: ", c)
        print('other measure: ', scoreother_measured(proY, testY))
        print("acc: ", clf.score(np.array(testX), np.array(testY)))


    print("\n====== SVM ======")
    # for c in [0.8, 1.0]:
    #     clf = SVC(C =c,probability = True)
    #     clf.fit(trainX, trainY)
    #     clf.probability = True
    #     proY = clf.predict_proba(testX)
    #     print('add other_measured  acc', scoreother_measured(proY, testY))
    #     print('SVM  acc when penalty C = ', c, clf.score(np.array(testX), np.array(testY)))
    print("----- kernel -----")
    print("----this will take more time, please be patient....")
    for c in ['linear', 'poly', 'rbf']:
        clf = SVC(kernel=c,probability=True)#, probability = True)
        clf.fit(trainX, trainY)
        clf.probability = True
        # proY = clf.predict_proba(testX)
        proY = clf.predict_proba(testX)
        print(">>> kernel:", c)
        print('other measure: ', scoreother_measured(proY, testY))
        print('acc:', clf.score(np.array(testX), np.array(testY)))

