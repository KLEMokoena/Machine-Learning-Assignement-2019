import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
import logisticRegression_functions as lg
import itertools
import sklearn.metrics as sk

def creatematrix(counters,vocab): #creates matrix for training
    xMatrix = np.ones((len(vocab)+1,len(counters)))
    iteration = 0
    for x in counters:
        temp_counter = x[1]
        for r in range(1, len(vocab)+1):
            xMatrix[r][iteration] = temp_counter.get(vocab[r-1], 0)
        iteration = iteration + 1
    return xMatrix

def confusion_matrix(y_test,y_pred,arr_label):
    confusion = sk.confusion_matrix(y_test,y_pred,labels=[k for k in arr_label])
    print(confusion)
    plt.imshow(confusion, interpolation='nearest',cmap=plt.cm.Blues)
    tick_marks = np.arange(confusion.shape[1])
    plt.xticks(tick_marks)
    ax = plt.gca()
    ax.set_xticklabels([k for k in arr_label])
    plt.yticks(tick_marks)
    ax.set_yticklabels([k for k in arr_label])
    thresh = confusion.max() / 2.
    for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
        plt.text(j, i, format(confusion[i, j], '.1f'),
                 horizontalalignment="center",
                 color="white" if confusion[i, j] > thresh else "black")
    plt.tight_layout()
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")

    diag = [confusion[i][i] for i in range(len(confusion))]
    accuracy=(sum(list(diag))/len(y_test))*100
    print("Accuracy:"+str(accuracy)+" %") # this outputs the accuracy of the model
    plt.show()
    return

def test_logisticAlg(X_test,vocab1,theta_list,arr_label):
    y_predict = [] # stores the predictions
    for i in range(0,len(X_test)):
        currX = X_test[i]
        currXencode = np.zeros((len(vocab1)+1,1))
        currXencode[0] = 1
        for k in range(0,len(vocab1)):
            if vocab1[k] in currX:
                currXencode[k+1] = 1
        arr_prob = [] #stores the probabilities outputed by model
        for k in theta_list:
            temp_val = lg.hypothesisfunction(k,currXencode)
            arr_prob.append(temp_val)
        index = arr_prob.index(max(arr_prob)) #this finds the index of the largest probability in the arr_prob
        y_predict.append(arr_label[index]) #this stores the predicted output of the logistic model.
    return y_predict


def main():
    with open("latest.csv",'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader) #skips first line in csv file
        typee = []
        med = []
        for row in csv_reader:
            typee.append(row[1]) #all the types in a list
            med.append(row[5]) #all the medical benefits in a list
        
        X_train, X_test, y_train, y_test = train_test_split(med, typee, test_size = 0.2) #splitting data into trainig and testing
    
        t = pd.DataFrame(y_train,columns={"Label"}) #all the types as Label from 0 to 1881 because trainging data
        m = pd.Series(X_train) #all the med benefits from 0 to 1881
        t['Features'] = m.values #all med benefits as a feature
        categories = np.unique(y_train) #list of different types, hybrid, sativa and indica
    
        vocab = []
        words_list = []
        for cat in categories: #for h,s,i do...
            temp = t.loc[t["Label"] == cat] #split dataframe by types of h,i,s including corresponding medical benefits
            size = len(temp.index) #returns the size of each spilt of h,i,s
            words = []
            for x in temp.Features:
                vocab.append(x)
                words.append(x)
                words_list.append([cat,vocab,size]) # returs only sativa as a list with all med benefits and size of list *****this is a problem****
    
        vocab1=np.unique(vocab) #list of unique med benefits
    
        counters = []
        arr_label = []
        i = 0
    
        for cat in words_list:
            temp_counter = Counter(cat[1]) #number of each med benefit in each category (only sativa for now)
            temp_total = cat[2] #size of each cat
            i = i + 1
            for key in temp_counter:
                temp_counter[key] /=temp_total #prob of each med bene
                counters.append([cat[0],temp_counter]) #only for sativa for now, but prob of each bed ben in sativa
                arr_label.append(cat[0]) #returns sativa
        
        yvector = np.identity(3) #creating output category of one vs all logistic regression algorithm
    
        xMatrix = creatematrix(counters,vocab1) #calling the createXmatrix function
        print("Data split.")
        print("Training Started.")
        theta_list = lg.trainlogisticR(xMatrix,yvector)
        print("Logistic Model Trained.")
        print("Testing Logistic Model")
        y_predict = test_logisticAlg(X_test,vocab1,theta_list,arr_label)
        print("Printing Confusing Matrix")
        confusion_matrix(y_test,y_predict,arr_label)
        print (confusion_matrix)

        return 0
    
if __name__ == "__main__":
    main()