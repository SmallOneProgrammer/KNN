# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
#import libraries 
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()

X = iris['data']
y = iris['target']

#split data 
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, random_state=42)
mu = np.mean(Xtrain, 0)
sigma = np.std(Xtrain, 0)

#standardize
Xtrain = (Xtrain - mu) / sigma

#We use the same mean and SD as the one of X_train as we dont know the mean of X_test

Xtest = (Xtest - mu) / sigma
muy = np.mean(ytrain, 0)
sigmay = np.std(ytrain, 0, ddof = 0)
muytest = np.mean(ytest, 0)
sigmaytest = np.std(ytest, 0, ddof = 0)
ytest = (ytest-muytest)/ sigmaytest
ytrain = (ytrain - muy) / sigmay


#create class KNN transformer 
class KNNtransform:
    def __init__(self, Xtrain, ytrain, Xtest, ytest):
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xtest = Xtest
        self.ytest = ytest

    def euclideanDistance(self):
        diff = np.sqrt(((self.Xtrain[:, :, None]-self.Xtest[:, :, None].T)**2).sum(1))
        print(self.Xtrain[:, : None].shape)
        print(self.Xtest[:, :, None].T.shape)
        return diff

    def transform(self, k):
        distance = self.euclideanDistance()
        sortdistance = np.argsort(distance, axis = 0)
        print(sortdistance.shape)
        y_pred = np.zeros(self.ytest.shape)
        for row in range(len(self.Xtest)):
            y_pred[row] = self.ytrain[sortdistance[:, row][:k]].mean()*np.std(ytrain, 0, ddof = 0) + np.mean(self.ytrain, 0)
        RMSE = np.sqrt(np.mean((self.ytest-y_pred)**2))
        return RMSE, y_pred

class KNNfit(KNNtransform):
    def __init__(self, Xtrain, ytrain, Xtest, ytest):
        super().__init__(Xtrain, ytrain, Xtest, ytest)

    def fit(self):
        return KNNtransform(self.Xtrain, self.ytrain, self.Xtest, self.ytest)


class Pipeline:
    def __init__(self, *args):
        self.lst = args

    def fit(self, Xtrain, ytrain, Xtest, ytest):
        alllst = []
        for clas in self.lst:
            KN = clas(Xtrain, ytrain, Xtest, ytest).fit()
            for i in range(15):
                RMSE, y_pred = KN.transform(i)
                alllst.append(RMSE)
        return alllst, y_pred



def fitModel():
    pipeline = Pipeline(KNNfit)
    lst, y_pred = pipeline.fit(Xtrain, ytrain, Xtest, ytest)
    print(y_pred)
    count = 0
    for num in range(len(y_pred)):
        if round(y_pred[num]) == round(ytest[num]):
            count+=1
    print("accuracy: ", count/len(ytest)*100)
    print(ytest)
    plt.plot(lst)
    plt.xlabel('iterations')
    plt.ylabel('Error')
    plt.show(block = True)
    return None

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fitModel()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
