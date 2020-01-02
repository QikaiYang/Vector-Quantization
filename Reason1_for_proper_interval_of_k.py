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
compare = list(range(2,61))
VQ_length = [10,20,30,40,50,60]
error_array = np.zeros((len(VQ_length), len(compare)))
accuracy_rate = np.zeros((3,len(compare),len(VQ_length))) #method, number of k-means, length of VQ
under_count = 0
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
        error_array[ancount][under_count] = get_error(all_train, centroids)#####################
        print("the cost error is", error_array[ancount][under_count])
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#___________________________________________________________________________________________________use___VQbianma_____________________________________________________________________________________________
        print("---------------d=",vec_length,"-------------------k=",count,"------------------------------------------")
        under_count+=1
    ancount += 1
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#_____________________________________________________________________________________________choose_the_best_amounts__________________________________________________________________________________________

#-------------------------------------------------------------
plt.plot(compare,error_array[0])
plt.plot(compare,error_array[1])
plt.plot(compare,error_array[2])
plt.plot(compare,error_array[3])
plt.plot(compare,error_array[4])
plt.plot(compare,error_array[5])
plt.xlabel("amount of clusters' centers")
plt.ylabel("value of cost function")
plt.title("The value of K-means for deiiferent D's value and k's value")
plt.legend(labels = ['d = 10', 'd = 20', 'd = 30','d = 40','d = 50','d = 60'], loc = 0)
plt.show()
#-----------------------------------------------------finally_we_test_the_model_on_the_test_set-------------------------------
