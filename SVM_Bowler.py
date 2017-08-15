import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from numpy import array
import numpy as np
from sklearn import datasets, linear_model
import csv
from sklearn.linear_model import LinearRegression

trainingData = []
trainingTarget = []
fileName = "E:\Academia\ML project\MyPyCode\BPLOverallT20Bowling.csv"
file = open(fileName, "r")
data = csv.reader(file)
count=0
for col in data:
    if count==0:
        count+=1
        continue
    if count==34:
        break
    count+=1
    arr={}
    if col[2]=="":
        arr[2]=0
    else:
        arr[2]=int(float(col[2]))
    if col[3]=="":
        arr[3]=0
    else:
        arr[3]=int(float(col[3]))
    if col[4]=="":
        arr[4]=0
    else:
        arr[4]=int(float(col[4]))
    if col[5]=="":
        arr[5]=0
    else:
        arr[5]=int(float(col[5]))
    if col[6]=="":
        arr[6]=0
    else:
        arr[6]=int(float(col[6]))
    if col[7]=="":
        arr[7]=0
    else:
        arr[7]=int(float(col[7]))
    if col[9]=="":
        arr[9]=0
    else:
        arr[9]=int(float(col[9]))
    if col[10]=="":
        arr[10]=0
    else:
        arr[10]=int(float(col[10]))
    if col[11]=="":
        arr[11]=0
    else:
        arr[11]=int(float(col[11]))
    if col[12]=="":
        arr[12]=0
    else:
        arr[12]=int(float(col[12]))
    if col[13]=="":
        arr[13]=0
    else:
        arr[13]=int(float(col[13]))
    
    trainingData.append(array([arr[2],arr[3],arr[4],arr[5],arr[6],arr[7],arr[10],arr[11],arr[12],arr[13]]))
    trainingTarget.append(arr[9])
clf = svm.SVC()
clf.fit(trainingData, trainingTarget)
#print(svc)

predictionData = []
predictionTarget = []
fileName = "E:\Academia\ML project\MyPyCode\BPLBowling2015.csv"
file = open(fileName, "r")
data = csv.reader(file)
count=0
for col in data:
    if count==0:
        count+=1
        continue
    if count==34:
        break
    count+=1
    arr={}
    if col[1]=="":
        arr[1]=0
    else:
        arr[1]=int(float(col[1]))
    if col[2]=="":
        arr[2]=0
    else:
        arr[2]=int(float(col[2]))
    if col[3]=="":
        arr[3]=0
    else:
        arr[3]=int(float(col[3]))
    if col[4]=="":
        arr[4]=0
    else:
        arr[4]=int(float(col[4]))
    if col[5]=="":
        arr[5]=0
    else:
        arr[5]=int(float(col[5]))
    if col[6]=="":
        arr[6]=0
    else:
        arr[6]=int(float(col[6]))
    if col[8]=="":
        arr[8]=0
    else:
        arr[8]=int(float(col[8]))
    if col[9]=="":
        arr[9]=0
    else:
        arr[9]=int(float(col[9]))
    if col[10]=="":
        arr[10]=0
    else:
        arr[10]=int(float(col[10]))
    if col[11]=="":
        arr[11]=0
    else:
        arr[11]=int(float(col[11]))
    if col[12]=="":
        arr[12]=0
    else:
        arr[12]=int(float(col[12]))
    predictionData.append([arr[1],arr[2],arr[3],arr[4],arr[5],arr[6],arr[9],arr[10],arr[11],arr[12]])
    predictionTarget.append(arr[8])
clf.predict(predictionData)
confidence = clf.score(predictionData, predictionTarget)
print('\x1b[6;30;42m',"Confidence:",'\x1b[0m',"\n",confidence)



print(clf.predict(predictionData))
print('\x1b[6;30;42m',"Mean squared error:",'\x1b[0m'," %.2f" % np.mean((clf.predict(predictionData) - predictionTarget) ** 2))
print('\x1b[6;30;42m','Variance score: ','\x1b[0m','%.2f' % clf.score(predictionData, predictionTarget))

X = clf.predict(predictionData)
Y = predictionTarget
# ///Graph start
for xe, ye in zip(X,Y):
#     plt.scatter([xe] * ye, ye)
    plt.scatter(xe, ye)
plt.plot(X, Y , color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())
plt.xticks([1, 200])
plt.axes().set_xticklabels(['cat1', 'cat2'])
plt.xlabel('x', fontsize=13)
plt.ylabel('y', fontsize=13)
plt.show()
# ///Graph end