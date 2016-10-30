#coding=utf-8
#Created by Chen Cheng from BUPT
#Email:chencheng_bupt@139.com
import numpy as np 
from struct import *

from sklearn.linear_model import LogisticRegression

def readLabel(f):
	with open(f,'rb') as file:
		buf = file.read()
		#获取label数量信息
		init,label_count = unpack_from('>II',buf,0)
		print f,' : label_count=',label_count
		#获取label数据信息
		return getLabel(buf,label_count)

def getLabel(buf,label_count):
	offset = calcsize('II')
	label=[]
	for x in range(label_count):
		la = unpack_from('B',buf,offset)
		offset+=calcsize('B')
		label.append(la[0])
	return label

def readImage(f):
	with open(f,'rb') as file:
		buf = file.read()
		offset = calcsize('I')
		#获取图片信息数据
		img_count,img_x,img_y = unpack_from('>3I',buf,offset)
		print f,':img_count=',img_count,' size=',img_x,'*',img_y
		return getImage(buf,img_count,img_x,img_y);

def getImage(buf,img_count,img_x,img_y):
	images = []
	size = str(img_x * img_y)+'B'
	offset = calcsize('4I')
	for x in range(img_count):
		img = unpack_from(size,buf,offset)
		offset+=calcsize(size)
		list(img)
		images.append(img)
	return images

def useScipyTest():
	print '######program start...'
	print '开始读文件...'
	train_image = readImage('train-images-idx3-ubyte')
	train_label = readLabel('train-labels-idx1-ubyte')
	test_image = readImage('t10k-images-idx3-ubyte')
	test_label = readLabel('t10k-labels-idx1-ubyte')
	print '读文件完成...'
	#使用scipy包做计算，由于计算量大，仅计算1000个样本数据
	#使用logistic regression训练器训练数据
	#将images作为X,label作为y
	classifier = LogisticRegression()
	regression = classifier.fit(train_image[1:1000],train_label[1:1000])
	#result = classifier.score(test_image,test_label)
	result = classifier.score(test_image[1:200],test_label[1:200])
	print '使用1000数据集结果的准确率为:',result
	print '######program end...'

#使用梯度上升算法找到最优化参数
#sigmoid函数
def sigmoid(inX):
	return 1.0/(1+exp(-inX))

def gradAcent(trainSet,labelSet):
	trainMatrix = mat(trainSet)
	labelMatrix = mat(labelSet).transpose()
	m,n = shape(trainMatrix)
	alpha = 0.001
	maxCycle = 100
	weight = ones((n,1))
	for k in range(maxCycle):
		h = sigmoid(trainMatrix*weight)
		error = labelMatrix - h
		weight+=alpha*trainMatrix.transpose()*error
	return weight



def useSelfGradAcent():
	print '开始读文件...'
	train_image = readImage('train-images-idx3-ubyte')
	train_label = readLabel('train-labels-idx1-ubyte')
	test_image = readImage('t10k-images-idx3-ubyte')
	test_label = readLabel('t10k-labels-idx1-ubyte')
	print '读文件完成...'
	#train:60000*784 test:60000*1 weightMatrix:1*784
	weight = gradAcent(train_image,train_label)
	weightMatrix = mat(weight).transpose()
	predict_result = mat(test_image)*weightMatrix
	print predict_result
	print '####program end ...'


def main():
	useScipyTest()

if __name__ == "__main__":
	main()
