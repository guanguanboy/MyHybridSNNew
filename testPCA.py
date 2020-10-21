'''
Created on Jun 1, 2011
@author: Peter Harrington
'''
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return mat(datArr)


def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)  # sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # cut off unwanted dimensions
    redEigVects = eigVects[:, eigValInd]  # reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects  # transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def replaceNanWithMean():
    datMat = loadDataSet('secom.data',' ')  # 解析数据
    print(datMat.shape)
    numFeat = shape(datMat)[1]      # 获取特征维度
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) # 利用该维度所有非NaN特征求取均值
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal # 将该维度中所有NaN特征全部用均值替换
    return datMat

datMat = loadDataSet('secom.data',' ')  # 解析数据
print(datMat.shape)

datMat = replaceNanWithMean()
meanVals = mean(datMat, axis=0)
meanRemoved = datMat - meanVals
covMat = cov(meanRemoved, rowvar=0)
eigVals, eigVects = linalg.eig(mat(covMat))
print(sum(eigVals)*0.9)     # 计算90%的主成分方差总和
print(sum(eigVals[:6]))     # 计算前6个主成分所占的方差
plt.plot(eigVals[:20])      # 对前20个画图观察
plt.show()


