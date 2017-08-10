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
fileName = "E:\Academia\ML project\MyPyCode\BPLOverallBattingPoints.csv"
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
        arr[2]=float(col[2])
    if col[3]=="":
        arr[3]=0
    else:
        arr[3]=float(col[3])
    if col[4]=="":
        arr[4]=0
    else:
        arr[4]=float(col[4])
    if col[5]=="":
        arr[5]=0
    else:
        arr[5]=float(col[5])
    if col[6]=="":
        arr[6]=0
    else:
        arr[6]=float(col[6])
    if col[7]=="":
        arr[7]=0
    else:
        arr[7]=float(col[7])
    if col[8]=="":
        arr[8]=0
    else:
        arr[8]=float(col[8])
    if col[9]=="":
        arr[9]=0
    else:
        arr[9]=float(col[9])
    if col[10]=="":
        arr[10]=0
    else:
        arr[10]=float(col[10])
    if col[11]=="":
        arr[11]=0
    else:
        arr[11]=float(col[11])
    if col[12]=="":
        arr[12]=0
    else:
        arr[12]=float(col[12])
    if col[13]=="":
        arr[13]=0
    else:
        arr[13]=float(col[13])
    if col[14]=="":
        arr[14]=0
    else:
        arr[14]=float(col[14])
    if col[15]=="":
        arr[15]=0
    else:
        arr[15]=float(col[15])
    if col[16]=="":
        arr[16]=0
    else:
        arr[16]=float(col[16])
    if col[17]=="":
        arr[17]=0
    else:
        arr[17]=float(col[17])
    if col[18]=="":
        arr[18]=0
    else:
        arr[18]=float(col[18])
    if col[19]=="":
        arr[19]=0
    else:
        arr[19]=float(col[19])
    
    trainingData.append(array([arr[2],arr[3],arr[4],arr[5],arr[6],arr[7],arr[8],arr[9],arr[10],arr[11],arr[12],arr[13],arr[14],arr[15],arr[16],arr[17],arr[18]]))
    trainingTarget.append(arr[19])
C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='rbf',gamma=(200.0/700)).fit(trainingData, trainingTarget)

predictionData = []
predictionTarget = []
fileName = "E:\Academia\ML project\MyPyCode\BPLRecentBattingPoints.csv"
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
        arr[1]=float(col[1])
    if col[2]=="":
        arr[2]=0
    else:
        arr[2]=float(col[2])
    if col[3]=="":
        arr[3]=0
    else:
        arr[3]=float(col[3])
    if col[4]=="":
        arr[4]=0
    else:
        arr[4]=float(col[4])
    if col[5]=="":
        arr[5]=0
    else:
        arr[5]=float(col[5])
    if col[6]=="":
        arr[6]=0
    else:
        arr[6]=float(col[6])
    if col[7]=="":
        arr[7]=0
    else:
        arr[7]=float(col[7])
    if col[8]=="":
        arr[8]=0
    else:
        arr[8]=float(col[8])
    if col[9]=="":
        arr[9]=0
    else:
        arr[9]=float(col[9])
    if col[10]=="":
        arr[10]=0
    else:
        arr[10]=float(col[10])
    if col[11]=="":
        arr[11]=0
    else:
        arr[11]=float(col[11])
    if col[12]=="":
        arr[12]=0
    else:
        arr[12]=float(col[12])
    if col[13]=="":
        arr[13]=0
    else:
        arr[13]=float(col[13])
    if col[14]=="":
        arr[14]=0
    else:
        arr[14]=float(col[14])
    if col[15]=="":
        arr[15]=0
    else:
        arr[15]=float(col[15])
    if col[16]=="":
        arr[16]=0
    else:
        arr[16]=float(col[16])
    if col[17]=="":
        arr[17]=0
    else:
        arr[17]=float(col[17])
    if col[18]=="":
        arr[18]=0
    else:
        arr[18]=float(col[18])
    predictionData.append([arr[1],arr[2],arr[3],arr[4],arr[5],arr[6],arr[7],arr[8],arr[9],arr[10],arr[11],arr[12],arr[13],arr[14],arr[15],arr[16],arr[17]])
    predictionTarget.append(arr[18])
    t=(arr[1],arr[2],arr[3],arr[4],arr[5],arr[6],arr[7],arr[8],arr[9],arr[10],arr[11],arr[12],arr[13],arr[14],arr[15],arr[16],arr[17])
    pData.append(t)

confidence = svc.score(predictionData, predictionTarget)
print(confidence)    
# print(clf.predict(predictionData))
# print("Mean squared error: %.2f" % np.mean((clf.predict(predictionData) - predictionTarget) ** 2))
# print('Variance score: %.2f' % clf.score(predictionData, predictionTarget))

# # The coefficients
# print('Coefficients: \n', clf.coef_)
# Y=predictionTarget

# pData = [(1,2,3)]
# t=()
# t+=(7,)
# t+=(8,)
# t+=(9.2,)
# pData.append((4,5.5,6))
# pData.append(t)
# x=[1,2,3]
# print(pData)
# ///Graph start
for xe, ye in zip(Y,pData):
#     plt.scatter([xe] * ye, ye)
    plt.scatter([xe] * len(ye), ye)
plt.plot(pData, Y , color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())
plt.xticks([1, 200])
plt.axes().set_xticklabels(['cat1', 'cat2'])
plt.xlabel('x', fontsize=13)
plt.ylabel('y', fontsize=13)
plt.show()
# ///Graph end
#SVM start
#C = 1.0 # SVM regularization parameter
# svc = svm.SVC(kernel='rbf',gamma=(200.0/700)).fit(trainingData,trainingTarget)
#svc = svm.SVC(kernel='linear',gamma=(200.0/700)).fit(trainingData,trainingTarget)
# svc = svm.SVC(kernel='poly',gamma=(200.0/700)).fit(trainingData,trainingTarget)
# svc = svm.SVC(kernel='sigmoid',gamma=(200.0/700)).fit(trainingData,trainingTarget)
# svc = svm.SVC(kernel='precomputed',gamma=(200.0/700)).fit(trainingData,trainingTarget)
# print(svc)
#SVM end