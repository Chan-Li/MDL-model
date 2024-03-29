#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln
import random
from numpy import random
import math
from matplotlib.pyplot import plot,savefig
import sys
filePath = '../Utils/nup'
sys.path.append(filePath)
import os
from functions import relu,drelu,softmax,divi_,mini_batch_generate,sigmoid,dsigmoid,turn_2_zero,scale,tanh,dtanh
from optimizers import Adam
from model_save import model_save
from dataset import load
mnist=np.array(load.load_mnist(one_hot=True),dtype="object")
train_data = mnist[0][0][0:60000].T
train_label = mnist[0][1][0:60000].T
test_data = mnist[1][0][:10000].T
test_label = mnist[1][1][:10000].T

#if we want to load CIFAR dataset
# import pickle
# def load_CIFAR_batch(filename):
#     with open(filename, 'rb') as f:
#         datadict = pickle.load(f,encoding='latin1')
#         X = datadict['data']
#         Y = datadict['labels']
#         X = X.reshape(10000, 3, 32,32).transpose(0,2,3,1).astype("float")
#         Y =np.array(Y)
#         return X, Y
# def load_CIFAR10(ROOT):
#     xs = []
#     ys = []
#     for b in range(1,6):
#         f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
#         X, Y = load_CIFAR_batch(f)
#         xs.append(X)
#         ys.append(Y) 
#     xs = np.array(xs)
#     ys = np.array(ys)
#     Xtr = np.concatenate(xs)#使变成行向量
#     Ytr = np.concatenate(ys)
#     del X, Y
#     Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
#     return Xtr, Ytr, Xte, Yte
# def _one_hot(labels, num):
#     size= labels.shape[0]
#     label_one_hot = np.zeros([size, num])
#     for i in range(size):
#         label_one_hot[i, np.squeeze(labels[i])] = 1
#     return label_one_hot
# def data_generate(root):
#     Xtr_ori, Ytr_ori, Xte_ori, Yte_ori=load_CIFAR10(root)
#     Xtr=(Xtr_ori).reshape(50000,3072).astype('float32')
#     Xte=(Xte_ori).reshape(10000,3072).astype('float32') 
#     Xtr = (Xtr.astype('float32') / 255.0).T
#     Xte = (Xte.astype('float32') / 255.0).T
#     Ytr=_one_hot(Ytr_ori, 10).T
#     Yte=_one_hot(Yte_ori, 10).T
#     return Xtr,Ytr,Xte,Yte

# train_data,train_label,test_data,test_label=data_generate('dataset/cifar-10-batches-py')


# In[2]:


def sig_mat(sig):
    lenth = len(sig)
    mat=np.zeros((lenth,lenth))
    for i in range(lenth):
        mat[i][i]=float(sig[i]*1)
    return mat
def clip_n(mat,bound):
    test = mat*1
    test[np.abs(mat)<bound] = 0
    return test


class NeuralNetwork:
    def __init__(self,sizes,p_m):
        self.num_layers = len(sizes)
        self.patterns = p_m
        self.sizes = sizes
        self.Sigma =  [(((1/(self.sizes[l]*ln(self.sizes[l])))**(1/6))*np.random.normal(0,1.0,size = (int(self.patterns[l]),1))) for l in range(self.num_layers-1)]
        self.xi1 = [(((1/(self.sizes[l]*ln(self.sizes[l])))**(1/6))*np.random.normal(0,1.0,size = (self.sizes[l+1],int(self.patterns[l])))) for l in range(self.num_layers-1)]
        self.xi2=[(((1/(self.sizes[l]*ln(self.sizes[l])))**(1/6))*np.random.normal(0,1.0,size = (int(self.patterns[l]),self.sizes[l]))) for l in range(self.num_layers-1)]
        self.w_sy = [np.dot(np.dot(self.xi1[l],sig_mat(self.Sigma[l])),self.xi2[l]) for l in range(self.num_layers-1)]
        self.Adam_s = Adam(self.Sigma*1)
        self.Adam_xi1 = Adam(self.xi1*1)
        self.Adam_xi2=Adam(self.xi2*1)
    def update(self):
        self.w_sy = [np.dot(np.dot(self.xi1[l],sig_mat(self.Sigma[l])),self.xi2[l]) for l in range(self.num_layers-1)]
    def feedforward(self,a1,activate,back=False):
        #x为输入的图片，尺寸为784*mini_batch_size
        a = a1*1
        zm=[]
        process=[a*1]
        for l in range(self.num_layers-1):#0,1,2
            z = np.dot(self.w_sy[l],process[l])
            zm.append(z*1)
            if (l<(self.num_layers-2)):#l=0,1
                a = activate(z)
            if (l>=(self.num_layers-2)):
                a = softmax(z)                       
            process.append(a*1)
        
        if back == False:
            return process[-1]
        if back == True:
            return zm,process
        
    def evaluate(self, testdata1,activate,clip_now = 0):
        # 获得预测结果a:10*batch_size
        #testlabel:10*batch_size
        data1,label1 = mini_batch_generate(500,testdata1*1,test_label*1)
        accuracy=[]
        for j in range(data1.shape[0]):
            a=self.feedforward(data1[j],activate,back=False)
            max0=np.argmax(a,axis=0)
            max1=np.argmax(label1[j],axis=0)
            accuracy.append((np.sum((max0-max1) == 0))/(data1[j].shape[1]))                
        return np.average(accuracy)
    
    def backprop(self,x,y,activate,dactivate,back=True):
        medicine=pow(10,-30)
        #x:输入：784*batch_size
        #y:输入标签：10*batch_size
        tri=[]
        self.update()
        zm,out=self.feedforward(x,activate,back=True)                        
        ## out is attached with sigmoid
        nabla_sig = [np.zeros(b_s.shape) for b_s in self.Sigma]
        nabla_xi1 = [np.zeros(b_s.shape) for b_s in self.xi1]
        nabla_xi2 = [np.zeros(b_s.shape) for b_s in self.xi2]
        for l in range(1, (self.num_layers)):
            self.update()
            if l==1:
                tri_=(out[-1]-y)
                tri.append(tri_)
            else:
                tri_=np.dot(self.w_sy[-l+1].T,tri[-1])*dactivate(zm[-l])
                tri.append(tri_) 
            nabla_sig[-l] = (np.sum((tri_@(np.dot(self.xi2[-l],out[-l-1]).T))*self.xi1[-l],axis=0)/(out[-l-1].shape[1])).reshape(len(self.Sigma[-l]),1)
            nabla_xi1[-l] = ((tri_@((self.xi2[-l]@out[-l-1]).T))*(self.Sigma[-l].T))/out[-l-1].shape[1]
            nabla_xi2[-l] = ((self.xi1[-l]*(self.Sigma[-l].T)).T)@(tri_@out[-l-1].T)/out[-l-1].shape[1]
        return nabla_sig,nabla_xi1,nabla_xi2
    
    def adam_update(self,lr,mini_batch_size,activate,dactivate,train_data_x,train_label_x):
        data_x=train_data_x*1
        label_x=train_label_x*1
        data,label = mini_batch_generate(mini_batch_size,data_x,label_x)
        for j in range(data.shape[0]):
            self.update()
            delta_nabla_sig,delta_nabla_xi1,delta_nabla_xi2= self.backprop(data[j],label[j],activate,dactivate,back=True)
            self.xi1 = (self.Adam_xi1.New_theta(self.xi1,delta_nabla_xi1,lr,1e-3))
            self.xi2 = (self.Adam_xi2.New_theta(self.xi2,delta_nabla_xi2,lr,1e-3))
            self.Sigma = (self.Adam_s.New_theta(self.Sigma,delta_nabla_sig,lr,1e-3))
            #print('\r'+str(j)+'/'+str(int(data.shape[0])),end='')
        
        
    def SGD(self,mini_batch_size,epoch,lr0,activate,dactivate):
        acc1_=[]
        for i in range(epoch):
            train_labelt=train_label*1#改参数
            train_datat=train_data
            lr = divi_(lr0,i,50)
#             print("become sparse",np.sum(np.abs(self.Sigma[0]))/(np.sqrt(np.sum(self.Sigma[0]**2))))
#            print ("epoch %s training complete" % i)
            acc1 = self.evaluate(test_data,activate)
#            print("the test Accuracy for task0 is:{} %".format((acc1)*100))
            self.adam_update(lr,mini_batch_size,activate,dactivate,train_datat,train_labelt)
            acc1_.append(acc1*100)
        return acc1_


# In[3]:


p=30 # define the number of patterns
if __name__ == '__main__':
    import time
    for lr in [0.003,0.004,0.005]:
        for N in [100]:
            print("lr=",lr)
            print("N=", N)
            acc_p=[]   
            for i in range(5):
                time1=time.time()
                net = NeuralNetwork([784,N,N,10],[p,p,p])
                ##if CIFAR10
                #net = NeuralNetwork([32*32*3,N,N,10],[p,p,p])
                acc1=net.SGD(100,200,lr,relu,drelu)
                acc_p.append(acc1*1)
                time2=time.time()
                print(time2-time1)
            print("accuracy = ",np.average(np.array(acc_p)[:,-1]))


# In[ ]:





# In[ ]:




