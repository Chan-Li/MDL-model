#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
train_data0 = mnist[0][0][0:60000].T
train_label0 = mnist[0][1][0:60000].T
test_data0 = mnist[1][0][:10000].T
test_label0 = mnist[1][1][:10000].T


# In[3]:


class Adam:
    def __init__(self,theta):
        self.lr=0.01
        self.beta1=0.9
        self.beta2=0.999
        self.epislon=1e-8
        self.m=[np.zeros(ms.shape) for ms in theta]
        self.s=[np.zeros(ms.shape) for ms in theta]
        self.t=0
    
    def New_theta(self,theta,gradient,eta,decay2,decay3,r,index):
        self.t += 1
        self.lr = eta*1
        self.decay2 = decay2
        self.decay3 = decay3
        g=gradient*1
        theta2 = [np.zeros(ms.shape) for ms in theta]
        for l in range(len(gradient)):
            self.m[l] = self.beta1*self.m[l] + (1-self.beta1)*g[l]
            self.s[l] = self.beta2*self.s[l] + (1-self.beta2)*(g[l]*g[l])
            self.mhat = self.m[l]/(1-self.beta1**self.t)
            self.shat = self.s[l]/(1-self.beta2**self.t)
            if index == 'u':
                theta2[l] = theta[l]-self.lr*((self.mhat/(pow(self.shat,0.5)+self.epislon))+self.decay2*((4/(r[l]**2))*theta[l]@(theta[l].T@theta[l]-np.identity(r[l]))))
            if index == 'v':
                theta2[l] = theta[l]-self.lr*((self.mhat/(pow(self.shat,0.5)+self.epislon))+self.decay2*((4/(r[l]**2))*theta[l].T@(theta[l]@theta[l].T-np.identity(r[l]))).T)
            if index == 's':
                theta2[l] = theta[l]-self.lr*((self.mhat/(pow(self.shat,0.5)+self.epislon))+self.decay3*(np.sign(theta[l])*np.sqrt(np.sum(theta[l]**2))-theta[l]*np.sum(np.abs(theta[l]))/(np.sqrt(np.sum(theta[l]**2))))/(np.sum(theta[l]**2)) )
        return theta2*1


# In[52]:


def sig_mat(sig):
    lenth = len(sig)
    mat=np.zeros((lenth,lenth))
    for i in range(lenth):
        mat[i][i]=np.abs(float(sig[i]*1))
    return mat
def clip_n(mat,bound):
    test = mat*1
    test[np.abs(mat)<bound] = 0
    return test


class NeuralNetwork:
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.r = [min(self.sizes[l+1],self.sizes[l]) for l in range(self.num_layers-1)]
        self.Sigma =  [(((1/(self.sizes[l]*ln(self.sizes[l])))**(1/6))*np.random.uniform(0,1.0,size = (min(self.sizes[l+1],self.sizes[l]),1))) for l in range(self.num_layers-1)]
        self.xi1 = [(((1/(self.sizes[l]*ln(self.sizes[l])))**(1/6))*np.random.normal(0,1.0,size = (self.sizes[l+1],min(self.sizes[l+1],self.sizes[l])))) for l in range(self.num_layers-1)]
        self.xi2=[(((1/(self.sizes[l]*ln(self.sizes[l])))**(1/6))*np.random.normal(0,1.0,size = (min(self.sizes[l+1],self.sizes[l]),self.sizes[l]))) for l in range(self.num_layers-1)]
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
            self.update()
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
            self.update()
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
            nabla_sig[-l] = np.sign(self.Sigma[-l])*(np.sum((tri_@(np.dot(self.xi2[-l],out[-l-1]).T))*self.xi1[-l],axis=0)/(out[-l-1].shape[1])).reshape(len(self.Sigma[-l]),1)
            nabla_xi1[-l] = ((tri_@((self.xi2[-l]@out[-l-1]).T))*(np.abs(self.Sigma[-l].T)))/out[-l-1].shape[1]
            nabla_xi2[-l] = ((self.xi1[-l]*np.abs(self.Sigma[-l].T)).T)@(tri_@out[-l-1].T)/out[-l-1].shape[1]
        return nabla_sig,nabla_xi1,nabla_xi2
    
    def adam_update(self,lr,mini_batch_size,activate,dactivate,train_data_x,train_label_x):
        data_x=train_data_x*1
        label_x=train_label_x*1
        data,label = mini_batch_generate(mini_batch_size,data_x,label_x)
        for j in range(data.shape[0]):
            self.update()
            delta_nabla_sig,delta_nabla_xi1,delta_nabla_xi2= self.backprop(data[j],label[j],activate,dactivate,back=True)
            self.xi1 = (self.Adam_xi1.New_theta(self.xi1,delta_nabla_xi1,lr,30.0,1,self.r,'u'))
            self.xi2 = (self.Adam_xi2.New_theta(self.xi2,delta_nabla_xi2,lr,30.0,1,self.r,'v'))
            self.Sigma = (self.Adam_s.New_theta(self.Sigma,delta_nabla_sig,lr,30.0,1,self.r,'s'))
#             print('\r'+str(j)+'/'+str(int(data.shape[0])),end='')
        
        
    def SGD(self,mini_batch_size,epoch,lr0,activate,dactivate):
        acc1_=[]
        for i in range(epoch):
            train_labelt=train_label*1#改参数
            train_datat=train_data
            lr = divi_(lr0,i,30)
            print("become orthogonal",np.sum(np.dot(self.xi2[1],self.xi2[1].T)-np.identity(N)))
            print("become sparse",np.sum(np.abs(self.Sigma[0]))/(np.sqrt(np.sum(self.Sigma[0]**2))))
            print ("epoch %s training complete" % i)
            acc1 = self.evaluate(test_data,activate)
            print("the test Accuracy for task0 is:{} %".format((acc1)*100))
            self.adam_update(lr,mini_batch_size,activate,dactivate,train_datat,train_labelt)
            acc1_.append(acc1*100)
        return acc1_


# In[53]:


##p=30:0.005,1e-4
##p=30,0.005,1e-3
##p=30:0.005,1e-2
##大N用0.003
if __name__ == '__main__':
    import time
    for lr in [0.005]:
        for N in [100]:
            print("lr=",lr)
            print("N=", N)
            acc_p=[]   
            for i in range(1):
                time1=time.time()
                net = NeuralNetwork([784,N,N,10])
                acc1=net.SGD(100,150,lr,relu,drelu)
                acc_p.append(acc1*1)
                time2=time.time()
                print(time2-time1)
            print("accuracy = ",np.average(np.array(acc_p)[:,-1]))




