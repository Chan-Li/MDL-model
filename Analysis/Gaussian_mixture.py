#!/usr/bin/env python
# coding: utf-8

# ## Mixed Gaussan data 

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln
import random
from numpy import random
import math
from matplotlib.pyplot import plot,savefig
import sys
filePath = 'Utils/nup'
sys.path.append(filePath)
import os
from functions import relu,drelu,softmax,divi_,mini_batch_generate,sigmoid,dsigmoid,turn_2_zero,scale,tanh,dtanh
from optimizers import Adam
from model_save import model_save
## generate mixed Gaussian data
def mean_generate(t,pm):
    if pm==1 or pm==-1:
        return np.array([pm*0.5*t,pm*0.5]).reshape(2)
    else:
        raise Exception("Sorry, no numbers like this")
def data_generating(N):
    N_s = int(N/4)
    label1 = 1.0
    label2 = -1.0
    sigma = 0.05
    data = np.zeros((2,N))
    label = np.zeros((1,N))
    cov = sigma*np.array([[1.0,0],[0,1.0]])
    data[:,0:N_s] = np.random.multivariate_normal(mean_generate(label1,1),cov,N_s).T
    data[:,N_s:2*N_s] = np.random.multivariate_normal(mean_generate(label1,-1),cov,N_s).T
    data[:,2*N_s:3*N_s] = np.random.multivariate_normal(mean_generate(label2,1),cov,N_s).T
    data[:,3*N_s:4*N_s] = np.random.multivariate_normal(mean_generate(label2,-1),cov,N_s).T
    label[:,0:2*N_s] = np.ones((2*N_s,1)).T
    label[:,2*N_s:4*N_s] = -1*np.ones((2*N_s,1)).T

#     label[:,0:2*N_s] = np.tile([1,0],(2*N_s,1)).T
#     label[:,2*N_s:4*N_s] = np.tile([0,1],(2*N_s,1)).T
    return data,label
train_data,train_label = data_generating(100000)
test_data,test_label = data_generating(10000)


# # Basic model

# In[3]:


class BPNetwork:
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.messes=[((np.random.normal(0.0,1.0,(y,x)))) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        self.Adam_theta = Adam(self.messes)
    
    
    def w_feedforward(self,a,activate,back=False):
        process=[]
        flag=0
        zm=[]
        process=[a]
        for mess in self.messes:
            flag=flag+1
            z=(np.dot(mess,a))*(1/(np.sqrt(mess.shape[1])))
            
            if (flag<(self.num_layers-1)):
                a = activate(z)
            if (flag>=(self.num_layers-1)):
                a = (z)
            zm.append(z)
            process.append(a)
        if back == False:
            return process[-1]
        if back == True:
            return process,zm
    
    def evaluate(self, testdata1,test_label0,activate,clip_now = 0):
        # 获得预测结果a:10*batch_size
        #testlabel:10*batch_size
        data1 = testdata1
        label1 = test_label0
        err = 0
        accuracy=0
        a=self.w_feedforward(data1,activate,back=False)
        max0=np.argmax(a,axis=0)
        max1=np.argmax(label1,axis=0)
        accuracy = (np.average((np.sign(a)-label1) == 0)*1)
        err = 0.5*np.average((a-label1)**2)
#         accuracy = (np.average((max0-max1) == 0)*1)
#         err = (np.average(-label1*ln(a+pow(10,-30))))
        return accuracy,err


    
    
    def backprop(self,x,y,activate,dactivate,back=True):
        #x:输入：784*batch_size
        #y:输入标签：10*batch_size
        tri=[]
        nabla_mess = [np.zeros(mess.shape) for mess in self.messes]
        out,zm=self.w_feedforward(x,activate,back=True)
        for l in range(1, (self.num_layers)):
            if l==1:
                tri_=(out[-1]-y)
                tri.append(tri_)
            else:
                tri_=(np.dot(self.messes[-l+1].T, tri[-1]) * dactivate(zm[-l]))*(1/np.sqrt(self.sizes[-l]))
                tri.append(tri_)
            nabla_mess[-l]=(np.dot(tri_,out[-l-1].T)*(1/np.sqrt(self.sizes[-l-1])))/(np.shape(x)[1])
        return nabla_mess
    
    
    
    def adam_update(self,lr,mini_batch_size,activate,dactivate):
        data_x=train_data*1
        label_x=train_label*1
        data,label = mini_batch_generate(mini_batch_size,data_x,label_x)
        for j in range(data.shape[0]):
            delta_nabla_m = self.backprop(data[j],label[j],activate,dactivate,back=True)
            self.messes= self.Adam_theta.New_theta(self.messes,delta_nabla_m,lr)
            print('\r'+str(j)+'/'+str(int(data.shape[0])),end='')


           
        
        
    def SGD(self,mini_batch_size,epoch,activate,dactivate,lr0):
        evaluation_cost, evaluation_error = [], []
        training_cost, training_accuracy = [], []
        learning_rate=[]
        test1,label1=train_data,train_label
        for i in range(epoch):
            lr = divi_(lr0,i,50)
            print ("Epoch %s training complete" % i)
            self.adam_update(lr,mini_batch_size,activate,dactivate)
            accuracy1,cost1 = self.evaluate(test1,label1,activate)
            evaluation_cost.append(cost1)
            evaluation_error.append((1-accuracy1))
#             cost2,accuracy2 = self.evaluate(train_data,train_label,relu)
#             training_cost.append(cost2)
#             training_accuracy.append(accuracy2)
#             print("the training Accuracy is:{} %".format((accuracy2)*100))
#             print("the training cost is ",cost2)
            print("the Test accuracy is:{} %".format((accuracy1)*100))
            print("the cost is ",cost1)
        return evaluation_error
# net1=BPNetwork([2,3,1])
# test_error=net1.SGD(10,10,relu,drelu,0.01)


# In[ ]:





# In[ ]:





# In[ ]:





# ## toymodel

# In[4]:


def sig_mat(sig):
    lenth = len(sig)
    mat=np.zeros((lenth,lenth))
    for i in range(lenth):
        mat[i][i]=sig[i]*1
    return mat



class NeuralNetwork:
    def __init__(self,sizes,p):
        self.num_layers = len(sizes)
        self.patterns = p
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
#     def update2(self,clip_now):
#         self.w_sy  = [np.dot(np.dot(self.xi1[l],sig_mat(clip_n(self.Sigma[l],clip_now))),self.xi2[l]) for l in range(self.num_layers-1)]

    def feedforward(self,a1,activate,back=False):
        #x为输入的图片，尺寸为784*mini_batch_size
        a = a1*1
        zm=[]
        process=[a*1]
        for l in range(self.num_layers-1):#0,1,2
#             self.update()
            z = np.dot(self.w_sy[l],process[l])
            zm.append(z*1)
            if (l<(self.num_layers-2)):#l=0,1
                a = activate(z)
            if (l>=(self.num_layers-2)):
                a = (z)                       
            process.append(a*1)
        
        if back == False:
            return process[-1]
        if back == True:
            return zm,process
        
    def evaluate(self, testdata1,test_label0,activate,clip_now = 0):
        # 获得预测结果a:10*batch_size
        #testlabel:10*batch_size
        data1 = testdata1
        label1 = test_label0
        err = 0
        accuracy=0
        a=self.feedforward(data1,activate,back=False)
        max0=np.argmax(a,axis=0)
        max1=np.argmax(label1,axis=0)
        accuracy = (np.average((np.sign(a)-label1) == 0)*1)
        err = 0.5*np.average((a-label1)**2)
#         accuracy = (np.average((max0-max1) == 0)*1)
#         err = (np.average(-label1*ln(a+pow(10,-30))))
        return accuracy,err
    
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
            self.xi1 = (self.Adam_xi1.New_theta(self.xi1,delta_nabla_xi1,lr,1e-4))
            self.xi2 = (self.Adam_xi2.New_theta(self.xi2,delta_nabla_xi2,lr,1e-4))
            self.Sigma = (self.Adam_s.New_theta(self.Sigma,delta_nabla_sig,lr,1e-4))
            print('\r'+str(j)+'/'+str(int(data.shape[0])),end='')
        
        
    def SGD(self,mini_batch_size,epoch,lr0,activate,dactivate):
        acc1_=[]
        err_all_tes= [ ]
        err_all_train= [ ]
        for i in range(epoch):
            train_labelt=train_label*1#改参数
            train_datat=train_data*1
            lr = divi_(lr0,i,50)
            print ("epoch %s training complete" % i)
            acc1,errx = self.evaluate(test_data,test_label,activate)
            acc2,errx2 = self.evaluate(train_data,train_label,activate)
            print("the test Accuracy for task0 is:{} %".format((acc1)*100))
            print("the test Loss for task0 is:{}".format((errx)))
            self.adam_update(lr,mini_batch_size,activate,dactivate,train_datat,train_labelt)
            acc1_.append(acc1*100)
            err_all_tes.append(errx*1)
            err_all_train.append(errx2*1)
        return err_all_tes,err_all_train


# In[5]:


err_t_all = []
err_tes_all = []
for i in range(5):
    net = NeuralNetwork([2,3,1],[3,3])
    err1,err2=net.SGD(10,200,0.001,relu,drelu)
    model_save('fig1/example network_oneout_lnn'+str(i+1)+'.pickle').model_s(net)
    err_t_all.append(err1*1)
    err_tes_all.append(err2*1)


# In[7]:


## functions to plot the Figure(1d).
def merge(a,b,c,bound1,bound2,bound3,bound4,ax,co,alpha):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
    X = np.arange(bound1, bound2, 0.25)
    Y = np.arange(bound3, bound4, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = a*X+b*Y+c
 
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, color=co,
                       linewidth=0,alpha=alpha, antialiased=False)
def fit_merge(x,y,z):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    num = len(x)

 

    a = 0
    A = np.ones((num, 3))
    for i in range(0, num):
        A[i, 0] = x[a]
        A[i, 1] = y[a]
        a = a + 1

    b = np.zeros((num, 1))
    a = 0
    for i in range(0, num):
        b[i, 0] = z[a]
        a = a + 1

    A_T = A.T
    A1 = np.dot(A_T,A)
    A2 = np.linalg.inv(A1)
    A3 = np.dot(A2,A_T)
    X= np.dot(A3, b)
    
    R=0
    for i in range(0,num):
        R=R+(X[0, 0] * x[i] + X[1, 0] * y[i] + X[2, 0] - z[i])**2
    return X[0,0],X[1,0],X[2,0]


# In[9]:


#Analyze the data

num=100
example_data = data_generating(num)
cla = int(num/4)
test_h1=(net.feedforward(example_data[0],relu,back=False))
# test_h1=(net.feedforward(test_data,relu,back=False))
k=0
zero = 0
up = 0
down = 0
data_new = []


num=100
example_data = data_generating(num)
cla = int(num/4)
test_h1=(net.feedforward(example_data[0],relu,back=False))
# test_h1=(net.feedforward(test_data,relu,back=False))
k=0
zero = 0
up = 0
down = 0
data_new = []
index=[0]
num=0
for k in range(4):
    for i in range(cla):
        if np.sign(test_h1.T[25*k+i]) == np.sign(example_data[1][:,25*k+i]):
            num=num+1
            data_new.append(example_data[0][:,k*25+i]*1)
    index.append(num)
        
data_new = np.array(data_new).T
print(zero)
print(up)
print(down)
example_data1x = net.xi2[0]@example_data[0]
example_data2x = sig_mat(net.Sigma[0])@net.xi2[0]@example_data[0]
##picture c
# example_data3x = relu(net.w_sy[1]@(relu(net.w_sy[0]@example_data[0][:,0:2*cla])))
# example_data3y = relu(net.w_sy[1]@(relu(net.w_sy[0]@example_data[0][:,2*cla:4*cla])))
example_data3x = (net.w_sy[0]@example_data[0])
example_data4x = relu(net.w_sy[0]@data_new)
print(np.shape(data_new))

