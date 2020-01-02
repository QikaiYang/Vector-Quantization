import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import seaborn as sns

def change_size(input_array,length_vec):
    temp = input_array.copy()
    temp = np.reshape(temp, int(len(temp)*len(temp[0])))
    new_width = length_vec*3
    out = int(len(temp)%new_width)
    for i in range(out):
        temp = np.delete(temp, len(temp)-1)
    temp = np.reshape(temp,(int(len(temp)/new_width),int(new_width)))
    return temp

def dist(p1,p2):
    d = 0.0
    for i in range(len(p1)):
        d += (p1[i]-p2[i])**2
    d = float(math.sqrt(d))
    return d

def center(iinput):
    cent = np.zeros(len(iinput[0]))
    for i in range(len(cent)):
        for j in range(len(iinput)):
            cent[i] += iinput[j][i]
    for i in range(len(cent)):
        cent[i] /= len(iinput)
    return cent

def locatebiggest(iinput):
    result = 0.0
    spot = 0
    for i in range(len(iinput)):
        if(iinput[i]>result):
            spot = i
    return spot

def get_error(array,k_model):
    result = 0.0
    for i in range(len(array)):
        result += dist(array[i], k_model[getwhichpart(array[i], k_model)]) 
    return result
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#__________________________________________________________________________________________________load data____________________________________________________________________________________________________

#please read me!!!!!!!!!!!!
#
#if you want to change the way to read in the data, you should modify the absolute address!!!!  
#
#the first part of the address is in the variable strr, which is the one should be modified!!!
#

NAMES = ['Use_telephone', 'Standup_chair', 'Walk', 'Climb_stairs', 'Sitdown_chair', 'Brush_teeth', 'Comb_hair', 'Eat_soup', 'Pour_water', 'Descend_stairs', 'Eat_meat', 'Drink_glass', 'Getup_bed', 'Liedown_bed']
strr = 'C:/Users/16502/Desktop/cs361/project/'      #absolute address
length_NAMES = np.zeros(len(NAMES))
for i in range(len(NAMES)):
    real_address = strr+NAMES[i]+'/'
    g = os.walk(real_address)
    m = 0
    for path,dir_list,file_list in g:  
        for file_name in file_list:
            name = NAMES[i] + str(m)
            locals()[NAMES[i] + str(m)] = np.loadtxt(real_address+file_name)
            m += 1
    length_NAMES[i] = len(file_list)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#__________________________________________________________________________________________________split_data___________________________________________________________________________________________________
for i in range(len(length_NAMES)):
    all_data = list(range(0,int(length_NAMES[i])))
    name = 'train'+str(i)
    locals()['train' + str(i)] = random.sample(all_data, int(0.65*length_NAMES[i]))
    locals()['test+valid' + str(i)] = list(set(all_data).difference(set(locals()['train' + str(i)])))
    locals()['test'+str(i)] = random.sample(locals()['test+valid' + str(i)], int(0.5*(length_NAMES[i]-int(0.65*length_NAMES[i]))))
    locals()['valid'+str(i)] = list(set(locals()['test+valid' + str(i)]).difference(set(locals()['test'+str(i)])))
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#_________________________________________________________________________________________________VQbianma______________________________________________________________________________________________________

def getwhichpart(point, centerlist):
    record = 100000000.0
    spot = 0
    for i in range(len(centerlist)):
        if(float(dist(point,centerlist[i]))<=float(record)):
            record = float(dist(point,centerlist[i]))
            spot = i
    return spot

def VQ(input, k_model):
    result = np.zeros(len(k_model))
    for i in range(len(input)):
        result[int(getwhichpart(input[i], k_model))] += 1
    total = 0.0
    return result

def get_accuracy(y_predict, y_real):
    right = 0.0
    wrong = 0.0
    for i in range(len(y_predict)):
        if(y_predict[i]==y_real[i]):
            right += 1.0
        else:
            wrong += 1.0
    return(right/(right+wrong))
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#___________________________________________________________________________________________________usekmean___________________________________________________________________________________________________
compare = list(range(25,36))
VQ_length = [10,20,30,40,50,60]
accuracy_rate1 = np.zeros((5,len(VQ_length), len(compare)))
accuracy_rate2 = np.zeros((5,len(VQ_length), len(compare)))
accuracy_rate3 = np.zeros((5,len(VQ_length), len(compare)))
under_count = 0
ancount = 0
for times in range(5):
    ancount = 0
    for vec_length in VQ_length:
        under_count = 0
        for count in compare:
            m = 0
            for i in range(len(length_NAMES)):
                for j in (locals()['train' + str(i)]):
                    if(m==0):
                        all_train = change_size(locals()[NAMES[i]+str(j)], vec_length)
                        #all_train = locals()[NAMES[i]+str(j)]
                        m += 1
                    else:
                        all_train = np.concatenate((all_train, change_size(locals()[NAMES[i]+str(j)],vec_length)),axis=0)
                        #all_train = np.concatenate((all_train, locals()[NAMES[i]+str(j)]),axis=0)
                        m += 1
            print(len(all_train))
            #all_train = change_size(all_train,vec_length)
            estimator = KMeans(n_clusters=count)
            estimator.fit(all_train)
            centroids = estimator.cluster_centers_
            #print(centroids)
            print(len(centroids))
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#___________________________________________________________________________________________________use___VQbianma_____________________________________________________________________________________________
            m = 0
            for i in range(len(length_NAMES)):
                for j in (locals()['train' + str(i)]):
                    if(m==0):
                        x_train = VQ(change_size(locals()[NAMES[i]+str(j)],vec_length), centroids)
                        x_train_label = np.array([i])
                        m += 1
                    else:
                        x_train = np.append(x_train, VQ(change_size(locals()[NAMES[i]+str(j)], vec_length), centroids))
                        x_train_label = np.append(x_train_label, i)
                        m += 1
            l = len(x_train)
            x_train = x_train.reshape(int(l/len(centroids)),len(centroids))
            #print(len(x_train))
            #print(len(x_train_label))

            m = 0
            for i in range(len(length_NAMES)):
                for j in (locals()['test' + str(i)]):
                    if(m==0):
                        x_test = VQ(change_size(locals()[NAMES[i]+str(j)],vec_length), centroids)
                        x_test_label = np.array([i])
                        m += 1
                    else:
                        x_test = np.append(x_test, VQ(change_size(locals()[NAMES[i]+str(j)], vec_length), centroids))
                        x_test_label = np.append(x_test_label, i)
                        m += 1
            l = len(x_test)
            x_test = x_test.reshape(int(l/len(centroids)),len(centroids))
            #print(len(x_test))
            #print(len(x_test_label))
   
            m = 0
            for i in range(len(length_NAMES)):
                for j in (locals()['valid' + str(i)]):
                    if(m==0):
                        x_val = VQ(change_size(locals()[NAMES[i]+str(j)],vec_length), centroids)
                        x_val_label = np.array([i])
                        m += 1
                    else:
                        x_val = np.append(x_val, VQ(change_size(locals()[NAMES[i]+str(j)],vec_length), centroids))
                        x_val_label = np.append(x_val_label, i)
                        m += 1
            l = len(x_val)
            x_val = x_val.reshape(int(l/len(centroids)),len(centroids))
            #print(len(x_val))
            #print(len(x_val_label))
            print("-----------------times=",times,"---------------k=",count,"--------------------d=",vec_length,"----------------------")
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#_____________________________________________________________________________________________________train1____________________________________________________________________________________________________
            rf=RandomForestClassifier()
            rf.fit(x_train, x_train_label)
            y_predict1 = rf.predict(x_val)
            CM1 = confusion_matrix(x_val_label, y_predict1)
            accuracy_rate1[times][ancount][under_count] = get_accuracy(y_predict1, x_val_label)
            print(accuracy_rate1[times][ancount][under_count])
            #print(CM1)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#_____________________________________________________________________________________________________train2____________________________________________________________________________________________________
            clf = svm.SVC(gamma=0.001, C=100.0)
            clf.fit(x_train, x_train_label)
            y_predict2 = clf.predict(x_val)
            CM2 = confusion_matrix(x_val_label, y_predict2)
            accuracy_rate2[times][ancount][under_count] = get_accuracy(y_predict2, x_val_label)
            print(accuracy_rate2[times][ancount][under_count])
            #print(CM2)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#_____________________________________________________________________________________________________train3___________________________________________________________________________________________________
            bay = GaussianNB().fit(x_train, x_train_label)
            y_predict3 = bay.predict(x_val)
            CM3 = confusion_matrix(x_val_label, y_predict3)
            accuracy_rate3[times][ancount][under_count] = get_accuracy(y_predict3, x_val_label)
            print(accuracy_rate3[times][ancount][under_count])
            #print(CM3)
            under_count+=1
        ancount += 1
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#_____________________________________________________________________________________________choose_the_best_amounts__________________________________________________________________________________________
df = pd.DataFrame(columns=['k', 'd', 'accuracy'])
for i in range(len(accuracy_rate2)):
    for j in range(len(accuracy_rate2[0])):
        for q in range(len(accuracy_rate2[0][0])):
            df.loc[i*len(accuracy_rate2[0])*len(accuracy_rate2[0][0])+j*len(accuracy_rate2[0][0])+q] = [compare[q], VQ_length[j],accuracy_rate2[i][j][q]]
sns.set(style="darkgrid")
sns.relplot(x="k", y = "accuracy", kind="line", data=df[df["d"]==10])
plt.title("d = 10")

sns.set(style="darkgrid")
sns.relplot(x="k", y = "accuracy", kind="line", data=df[df["d"]==20])
plt.title("d = 20")

sns.set(style="darkgrid")
sns.relplot(x="k", y = "accuracy", kind="line", data=df[df["d"]==30])
plt.title("d = 30")

sns.set(style="darkgrid")
sns.relplot(x="k", y = "accuracy", kind="line", data=df[df["d"]==40])
plt.title("d = 40")

sns.set(style="darkgrid")
sns.relplot(x="k", y = "accuracy", kind="line", data=df[df["d"]==50])
plt.title("d = 50")

sns.set(style="darkgrid")
sns.relplot(x="k", y = "accuracy", kind="line", data=df[df["d"]==60])
plt.title("d = 60")

plt.show()
