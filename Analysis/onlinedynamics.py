import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln
import random
from numba import jit
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

## Mean-field ODE

@jit
def I2_error(P,Q,M):
    #P,Q,M 均为矩阵，尺寸分别为 P: k*k, M:m*k, Q: m*m
    k = P.shape[0]
    m = Q.shape[0]
#     ns = nt = 1
    ns = 1/m
    nt = 1/k
    Q_diag = np.diag(Q).reshape(m,1)+np.ones((m,1))
    P_diag = np.diag(P).reshape(k,1)+np.ones((k,1))
    QQ = np.dot(Q_diag,Q_diag.T)
    PP = np.dot(P_diag,P_diag.T)
    MM = np.dot(Q_diag,P_diag.T)
    err = nt**2/(np.pi)*np.sum(np.arcsin(P/np.sqrt(PP)))+ns**2/(np.pi)*np.sum(np.arcsin(Q/np.sqrt(QQ)))-2*ns*nt/(np.pi)*np.sum(np.arcsin(M/np.sqrt(MM)))
    return err
@jit
def I_4(P,Q,M,xi,sigma,d,eta):
    #P,Q,M 均为矩阵，尺寸分别为 P: k*k, M:m*k, Q: m*m
    dt = 1/d
    k = P.shape[0]
    m = Q.shape[0]
#     ns = nt = 1
    ns = 1/m
    nt = 1/k
    I41 = np.zeros((m,m))
    I42 = np.zeros((m,m))
    I43 = np.zeros((m,m))
    Ex1 = (xi@(sigma**2))@xi.T
    for a in range(m):
        for i in range(a+1):
            for p in range(m):
                for q in range(m):
                    C11 = Q[a][a]
                    C12 = Q[a][i]
                    C13 = Q[a][p]
                    C14 = Q[a][q]
                    C22 = Q[i][i]
                    C23 = Q[i][p]
                    C24 = Q[i][q]
                    C33 = Q[p][p]
                    C34 = Q[p][q]
                    C44 = Q[q][q]
                    L4 = (1+C11)*(1+C22) - np.square(C12)
                    L0 = L4*C34 - C23*C24*(1+C11) - C13*C14*(1+C22) + C12*C13*C24 + C12*C14*C23
                    L1 = L4*(1+C33) - np.square(C23)*(1+C11) - np.square(C13)*(1+C22) + 2*C12*C13*C23
                    L2 = L4*(1+C44) - np.square(C24)*(1+C11) - np.square(C14)*(1+C22) + 2*C12*C14*C24
                    I41[a][i] += (4.0/(np.square(np.pi)*np.sqrt(L4)))*np.arcsin(L0/np.sqrt(L1*L2))
            I41[i][a] = I41[a][i]
            for p in range(k):
                for q in range(k):
                    C11 = Q[a][a]
                    C12 = Q[a][i]
                    C13 = M[a][p]
                    C14 = M[a][q]
                    C22 = Q[i][i]
                    C23 = M[i][p]
                    C24 = M[i][q]
                    C33 = P[p][p]
                    C34 = P[p][q]
                    C44 = P[q][q]
                    L4 = (1+C11)*(1+C22) - np.square(C12)
                    L0 = L4*C34 - C23*C24*(1+C11) - C13*C14*(1+C22) + C12*C13*C24 + C12*C14*C23
                    L1 = L4*(1+C33) - np.square(C23)*(1+C11) - np.square(C13)*(1+C22) + 2*C12*C13*C23
                    L2 = L4*(1+C44) - np.square(C24)*(1+C11) - np.square(C14)*(1+C22) + 2*C12*C14*C24
                    I42[a][i] += (4.0/(np.square(np.pi)*np.sqrt(L4)))*np.arcsin(L0/np.sqrt(L1*L2))
            I42[i][a] = I42[a][i]
            for p in range(m):
                for q in range(k):
                    C11 = Q[a][a]
                    C12 = Q[a][i]
                    C13 = Q[a][p]
                    C14 = M[a][q]
                    C22 = Q[i][i]
                    C23 = Q[i][p]
                    C24 = M[i][q]
                    C33 = Q[p][p]
                    C34 = M[p][q]
                    C44 = P[q][q]
                    L4 = (1+C11)*(1+C22) - np.square(C12)
                    L0 = L4*C34 - C23*C24*(1+C11) - C13*C14*(1+C22) + C12*C13*C24 + C12*C14*C23
                    L1 = L4*(1+C33) - np.square(C23)*(1+C11) - np.square(C13)*(1+C22) + 2*C12*C13*C23
                    L2 = L4*(1+C44) - np.square(C24)*(1+C11) - np.square(C14)*(1+C22) + 2*C12*C14*C24
                    I43[a][i] += (4.0/(np.square(np.pi)*np.sqrt(L4)))*np.arcsin(L0/np.sqrt(L1*L2))
            I43[i][a] = I43[a][i]
    dQ2 =  dt*np.square(eta*ns)*np.square(ns)*(Ex1@(I41.T)@Ex1)    + dt*np.square(eta*ns)*np.square(nt)*(Ex1@(I42.T)@Ex1)    -2*dt*np.square(eta*ns)*(ns*nt)*(Ex1@(I43.T)@Ex1)
    
    return dQ2



@jit
def I_3(P,Q,M,xi,sigma,d,eta):
    dt = 1/d
    k = P.shape[0]
    m = Q.shape[0]
#     ns = nt=1
    ns = 1/m
    nt = 1/k
    dQ = np.zeros((Q.shape))
    dP = np.zeros((P.shape))
    dM = np.zeros((M.shape))
    I3_qs = np.zeros((m,m))
    I3_qt = np.zeros((m,m))
    #xi: k*p, sigma:p*p
    ##lambda all students
    Ex1 = (xi@(sigma**2))@xi.T #k*k
    for i in range(m):
        for t in range(m):
            for p in range(m):
                C11 = Q[i][i]
                C12 = Q[i][t]
                C13 = Q[i][p]
                C22 = Q[t][t]
                C23 = Q[t][p]
                C33 = Q[p][p]
                L3 = (1+C11)*(1+C33)-np.square(C13)
                I3_qs[i][t] += 2*(C23*(1+C11)-C12*C13)/(np.pi*(1+C11)*np.sqrt(L3))
            for p in range(k):
                C11 = Q[i][i]
                C12 = Q[i][t]
                C13 = M[i][p]
                C22 = Q[t][t]
                C23 = M[t][p]
                C33 = P[p][p]
                L3 = (1+C11)*(1+C33)-np.square(C13)
                I3_qt[i][t] += 2*(C23*(1+C11)-C12*C13)/(np.pi*(1+C11)*np.sqrt(L3))
    dQ = -eta*dt*ns*ns*(((I3_qs).T)@Ex1+Ex1@(I3_qs))    + eta*dt*ns*nt*(((I3_qt).T)@Ex1+Ex1@(I3_qt))
    
    I3_m1 = np.zeros((m,k))
    I3_m2 = np.zeros((m,k))
    for i in range(m):
        for t in range(k):
            for p in range(m):
                C11 = Q[i][i]
                C12 = M[i][t]
                C13 = Q[i][p]
                C22 = P[t][t]
                C23 = M[p][t]
                C33 = Q[p][p]
                L3 = (1+C11)*(1+C33)-np.square(C13)
                I3_m1[i][t] += 2*(C23*(1+C11)-C12*C13)/(np.pi*(1+C11)*np.sqrt(L3))
            for p in range(k):
                C11 = Q[i][i]
                C12 = M[i][t]
                C13 = M[i][p]
                C22 = P[t][t]
                C23 = P[t][p]
                C33 = P[p][p]
                L3 = (1+C11)*(1+C33)-np.square(C13)
                I3_m2[i][t] += 2*(C23*(1+C11)-C12*C13)/(np.pi*(1+C11)*np.sqrt(L3))
    dM = -dt*(eta*ns)*ns*(Ex1@((I3_m1))) + dt*(eta*nt)*ns*(Ex1@((I3_m2)))
    
    return dQ,dM
def update(P1,Q1,M1,xi,sigma,d,eta,epoch):
    P = P1*1
    Q = Q1*1
    M = M1*1
    err_all = []
    for i in range(epoch):
#         if (i%100==0):
        err = I2_error(P,Q,M)
        err_all.append(err*1)
        dQ1,dM1 = I_3(P,Q,M,xi,sigma,d,eta)
        dQ2 = I_4(P,Q,M,xi,sigma,d,eta)
        dQ = dQ1+dQ2
        dM = dM1
        Q +=dQ
        M +=dM
        
        
    return err_all,P,M,Q


# fix teacher
from scipy import special
from sklearn.preprocessing import normalize
from scipy.linalg import orth
def set_orth_teacher(sizes,patterns,indi):
    sigma = 1.0
    if indi == 0:
        test= [(np.sqrt(sizes[l]))*normalize(np.random.randn(int(patterns[l]),sizes[l]), axis=1, norm='l2') for l in range(1)]
        return [(np.sqrt(sizes[l]))*orth(test[l].T,rcond=None).T for l in range(1)]
    if indi == 1:
        random.seed(5)
        return [(np.sqrt(sizes[l])**(0.5))*np.random.normal(0,sigma,size = (sizes[l+1], int(patterns[l]))) for l in range(1)]
    if indi == 2:
        random.seed(6)
        return  [((np.sqrt(sizes[l])**(0.5))*np.random.normal(0,sigma,size = (int(patterns[l]),1))) for l in range(1)]


# ## Toy model -- Simulation



from scipy import special
from sklearn.preprocessing import normalize
from scipy.linalg import orth
def sig_mat(sig):
    lenth = len(sig)
    mat=np.zeros((lenth,lenth))
    for i in range(lenth):
        mat[i][i]=sig[i]*1
    return mat
def clip_n(mat,bound):
    test = mat*1
    test[np.abs(mat)<bound] = 0
    return test
def erf(x):
    return special.erf(x/(np.sqrt(2)))
def derf(x):
    return (np.sqrt(2.0)/(np.sqrt(np.pi)))*pow(np.e,(-x**2)/2)

class Teacher:
    def __init__(self,size,p,d,scale):
        self.sizes = size
        self.num_layers = len(self.sizes)
        self.sigma=1.0
        self.patterns = p*np.ones((self.num_layers-1,1))
        if scale == True:
            self.xi1 = [(1/((ln(self.sizes[l]))**(1/6)))*np.random.normal(0,self.sigma,size = (self.sizes[l+1], int(self.patterns[l]))) for l in range(self.num_layers-1)]
            self.xi2 = [(1/((ln(self.sizes[l]))**(1/6)))*np.random.normal(0,self.sigma,size = (int(self.patterns[l]),self.sizes[l])) for l in range(self.num_layers-1)]
            self.Sigma =  [(1/((ln(self.sizes[l]))**(1/6)))*np.random.normal(0,self.sigma,size = (int(self.patterns[l]),1)) for l in range(self.num_layers-1)]
        if scale == False:
            self.xi1 = [(1/((self.sizes[l])**(1/6)))*np.random.normal(0,self.sigma,size = (self.sizes[l+1], int(self.patterns[l]))) for l in range(self.num_layers-1)]
            self.xi2 = [(1/((self.sizes[l])**(1/6)))*np.random.normal(0,self.sigma,size = (int(self.patterns[l]),self.sizes[l])) for l in range(self.num_layers-1)]
            self.Sigma =  [((1/((self.sizes[l])**(1/6)))*np.random.normal(0,self.sigma,size = (int(self.patterns[l]),1))) for l in range(self.num_layers-1)]
        self.w_sy = [np.dot(np.dot(self.xi1[l],sig_mat(self.Sigma[l])),self.xi2[l]) for l in range(self.num_layers-1)]
        self.P = [(1/d)*np.dot(self.w_sy[l],self.w_sy[l].T) for l in range(self.num_layers-1)]
    def update(self):
        self.w_sy = [np.dot(np.dot(self.xi1[l],sig_mat(self.Sigma[l])),self.xi2[l]) for l in range(self.num_layers-1)]
    def feedforward(self,a1,activate,back=False):
        self.update()
        #x为输入的图片，尺寸为784*mini_batch_size
        a = a1*1
        z = (1/np.sqrt(self.sizes[0]))*np.dot(self.w_sy[0],a)
        a_out = activate(z)
        h_out = np.average(a_out,axis=0).reshape(1,a.shape[1])
        if back == False:
            return h_out
        if back == True:
            return a,z,a_out,h_out
    
    def train_generate(self,activate):
        train_sample = np.random.normal(0,1,(self.sizes[0],1))
        train_label =self.feedforward(train_sample,activate)
        return train_sample,train_label
    def test_generate(self,M2,activate):
#         Train_set = np.random.normal(0,1,(self.sizes[0],M1))
#         Train_label =self.feedforward(Train_set,activate)
        Test_set = np.random.normal(0,1,(self.sizes[0],M2))
        Test_label = self.feedforward(Test_set,activate)
        return Test_set,Test_label




def lin_generate(p):
    random.seed(2)
    lin = [normalize(np.random.randn(int(p),int(p)), axis=1, norm='l2') for l in range(1)]
    return lin
def xlog_scale(log_x_max, scale, log_base=10):
        '''Logaritmic scale up to log_alpha_max'''

        bd_block = np.arange(0, log_base**2, log_base) + log_base
        bd_block = bd_block[0:-1]
        xlog = np.tile(bd_block, log_x_max)

        xlog[(log_base-1) : 2*(log_base-1)] = log_base*xlog[(log_base-1) : 2*(log_base-1)]

        for j in range(1, log_x_max - 1):
            xlog[(j+1)*(log_base-1) : (j+2)*(log_base-1)] = log_base*xlog[  j*(log_base-1) :  (j+1)*(log_base-1)  ]

        xlog = np.insert(xlog, 0,  np.arange(1,log_base), axis=0)
        xlog = np.insert(xlog, len(xlog),log_base**(log_x_max+1), axis=0)

        jlog = (xlog*scale).astype(int)

        return jlog
def savelog_list(log_x_max, scale):
        '''Logaritmic scale up to log_alpha_max'''
        xlog = np.logspace(0, log_x_max, log_x_max+1, endpoint=True).astype(int)
        save_xlog = (xlog*scale).astype(int)
        return save_xlog
class Toymodel:
    def __init__(self,size,p_ini,scale):
        self.sizes = size
        self.num_layers = len(self.sizes)
        self.patterns = p_ini*np.ones((self.num_layers-1,1))
        self.Sigma = T1.Sigma
        self.xi1 = T1.xi1
        if scale == True:
            self.xi2 = [(1/((ln(self.sizes[l]))**(1/6)))*np.random.normal(0,1.0,size = (int(self.patterns[l]),self.sizes[l])) for l in range(self.num_layers-1)]
        if scale == False:
            self.xi2 = [(1/((self.sizes[l])**(1/6)))*np.random.normal(0,1.0,size = (int(self.patterns[l]),self.sizes[l])) for l in range(self.num_layers-1)]
#         self.xi2 = [(np.sqrt(self.sizes[l]))*normalize(np.random.randn(int(self.patterns[l]),self.sizes[l]), axis=1, norm='l2') for l in range(self.num_layers-1)]
#         self.xi2 = [np.sqrt(self.sizes[l])*orth(self.xi2[l].T,rcond=None).T for l in range(self.num_layers-1)]
        self.w_sy = [np.dot(np.dot(self.xi1[l],sig_mat(self.Sigma[l])),self.xi2[l]) for l in range(self.num_layers-1)]
        self.Q = [(1/d)*np.dot(self.w_sy[l],self.w_sy[l].T) for l in range(self.num_layers-1)]
        self.M = [(1/d)*np.dot(self.w_sy[l],T1.w_sy[l].T) for l in range(self.num_layers-1)]
        self.P = [(1/d)*np.dot(T1.w_sy[l],T1.w_sy[l].T) for l in range(self.num_layers-1)]
    def update(self):
        self.w_sy = [np.dot(np.dot(self.xi1[0],sig_mat(self.Sigma[0])),self.xi2[0])]
        self.Q = [(1/d)*np.dot(self.w_sy[l],self.w_sy[l].T) for l in range(self.num_layers-1)]
        self.M = [(1/d)*np.dot(self.w_sy[l],T1.w_sy[l].T) for l in range(self.num_layers-1)]
        self.P = [(1/d)*np.dot(T1.w_sy[l],T1.w_sy[l].T) for l in range(self.num_layers-1)]
    def feedforward(self,a1,activate,back=False):
        self.update()
        #x为输入的图片，尺寸为784*mini_batch_size
        a = a1*1
        z = (1/np.sqrt(self.sizes[0]))*np.dot(self.w_sy[0],a)
        a_out = activate(z)
        h_out = (1/(self.sizes[1]))*np.sum(a_out,axis=0).reshape(1,a.shape[1])
        if back == False:
            return h_out
        if back == True:
            return a,z,a_out,h_out
        
    def evaluate(self, testdata1,testlabel1,activate):
        # 获得预测结果a:10*batch_size
        #testlabel:10*batch_size
        data1,label1 = mini_batch_generate(100000,testdata1*1,testlabel1*1)
        accuracy=[]
        for j in range(data1.shape[0]):
            self.update()
            a=self.feedforward(data1[j],activate,back=False)
            error = 0.5*np.average((a - label1[j])**2)
            accuracy.append(error*1)                
        return np.average(accuracy)
    
    def backprop(self,x,y,activate,dactivate,back=True):
        medicine=pow(10,-30)
        #x:输入：784*batch_size
        #y:输入标签：10*batch_size
        tri=[]
        self.update()
        a,z,a_out,h_out=self.feedforward(x,activate,back=True) #2000*60000,1000*60000                       
        ## out is attached with sigmoid
        nabla_sig = [np.zeros(b_s.shape) for b_s in self.Sigma]
        nabla_xi1 = [np.zeros(b_s.shape) for b_s in self.xi1]
        nabla_xi2 = [np.zeros(b_s.shape) for b_s in self.xi2]
        a,z,a_out,h_out=self.feedforward(x,activate,back=True) #2000*60000,1000*60000                       
        ## out is attached with sigmoid
        tri_ = (1/self.sizes[1])*(h_out - y)*dactivate(z)         
#         nabla_sig[0] = (1/np.sqrt(self.sizes[0]))*(np.sum((tri_@(np.dot(self.xi2[0],a).T))*self.xi1[0],axis=0)/(a.shape[1])).reshape(len(self.Sigma[0]),1)
#         nabla_xi1[0] = (1/np.sqrt(self.sizes[0]))*((tri_@((self.xi2[0]@a).T))*(self.Sigma[0].T))/a.shape[1]
        nabla_xi2[0] = (1/np.sqrt(self.sizes[0]))*np.dot(sig_mat(net.Sigma[0]),net.xi1[0].T)@(tri_@a.T)/a.shape[1]
        return nabla_sig,nabla_xi1,nabla_xi2
    
    def adam_update(self,lr,mini_batch_size,activate,dactivate,train_data_x,train_label_x):
        data_x=train_data_x*1
        label_x=train_label_x*1
        data,label = mini_batch_generate(mini_batch_size,data_x,label_x)
        for j in range(data.shape[0]):
            self.update()
            delta_nabla_sig,delta_nabla_xi1,delta_nabla_xi2= self.backprop(data[j],label[j],activate,dactivate,back=True)
#             self.xi1[0] = self.xi1[0] - lr*delta_nabla_xi1[0]
            self.xi2[0] = self.xi2[0] - lr*delta_nabla_xi2[0]
#             self.Sigma[0] = self.Sigma[0]- lr*delta_nabla_sig[0]
#             print('\r'+str(j)+'/'+str(int(data.shape[0])),end='')
        

        

    def SGD(self,mini_batch_size,lr0,activate,dactivate,num):
        acc1_=[]
        numb = []
#         err1=0
        err1,px,mx,qx= update(self.P[0],self.Q[0],self.M[0],self.xi1[0],sig_mat(self.Sigma[0]),d,5.0,num)
        print("ODE result",err1[-1])
        print(mx)
        m_all = []
        for i in range(num):
            self.update()
            m_all.append(self.M)
            lr = lr0
            sample,lab = T1.train_generate(activate)
            self.adam_update(lr,mini_batch_size,activate,dactivate,sample.reshape(self.sizes[0],1),lab.reshape(1,1))
            if i in (xlog_scale(round(np.log10(num)), 10, log_base=10)):
                acc1 = self.evaluate(Test_set,Test_label,activate)
#                 print ("epoch %s training complete" % i)
#                 print("the test Error for task0 is:{} ".format((acc1)))
                acc1_.append(acc1*1)
                numb.append(i)
        return err1,acc1_,numb,m_all




if __name__ == '__main__':
    err_ODE = []
    err_SGD = []
    num_all = []
    alpha = 1.0
    for d in [30,30,30,30]:
        p = int(alpha*ln(d))
        total = d*100000
        print("d = ",d)
        Size = [d,8]
        T1 = Teacher(Size,p,d,scale=True)
        print(np.average(T1.P[0]))
        Test_set,Test_label = T1.test_generate(100000,erf)
        net = Toymodel(Size,p,scale=True)
        print(np.average(net.Q[0]))
        if 0.5*(np.average(net.Q[0])-np.average(T1.P[0]))**2>0.01:
            net = Toymodel(Size,p,scale=True)
        err2,err1,num,m_all = net.SGD(1,5.0,erf,derf,total)
        err_ODE.append(err2*1)
        err_SGD.append(err1*1)
        num_all.append(num*1)
        print(err1[-1])







# In[ ]:




