import numpy as np
import math
from scipy.integrate import quad
import statistics
from scipy import integrate
import random
from sklearn.base import BaseEstimator
import pandas as pd
from numpy.linalg import norm 

class SVM_Hinge_sgd(BaseEstimator):

    def __init__(self, C = 1, max_epochs = 1, n_batches = 2
                ):
        assert C > 0
        self.C = C
        assert max_epochs > 0 and isinstance(max_epochs, int)
        self.max_epochs=max_epochs
        assert n_batches > 0 and isinstance(n_batches, int)
        self.n_batches=n_batches

    '''gradient of uncertainHingeloss'''

    def sigma(self,x): 
        d = []
        for i in range(len(x[0])):
            m = max(x[:,i]) 
            n = min(x[:,i])
            v = m - n
            d.append(v)

        sig = np.diag(d)
        return sig
        #return sigma

    def kvtest(self,x,t):

        v = []

        for n in range(len(x)):
                
          if t[n] == 1:
                k_0 = random.uniform(0.95*0.05,1.05*0.05)
          elif t[n] == -1:
                k_0 = random.uniform(0.95*0.07,1.05*0.07)

          v.append(k_0)

        return v

    def gradient(self,w, b, x, t, sigma,kv): 
        
        gradient = [] 
        

        for k in range(len(t)):

            dx = 1 - t[k]*(np.dot(w.T,x[k]) + b) 
            sigma_new = 0.25*kv[k]*sigma
            dsigma =  math.sqrt(2*np.dot(np.dot(w.T,sigma_new),w))
            g1 = (math.exp(-(dx**2)/(dsigma**2))/ (math.sqrt(math.pi)*dsigma)) * np.dot(sigma_new,w)      

            f = lambda y : math.exp(-y**2)
            integ = integrate.quad(f, 0, (dx/dsigma))
            g2 = 1/2 * ((2/math.sqrt(math.pi))*integ[0] + 1) * x[k]

            g = g1 - g2
            gradient.append(g) 
          
       
        return gradient


    #gradient on b
    def bgradient(self, w, b, x, t, sigma,kv): 
        
        bgradient = [] 
        
              
        for k in range(len(t)):

            dx = 1 - t[k]*(np.dot(w.T,x[k]) + b) 
            sigma_new = 0.25*kv[k]*sigma
            dsigma =  math.sqrt(2*np.dot(np.dot(w.T,sigma_new),w))
            f = lambda y : math.exp(-y**2)
            integ = integrate.quad(f, 0, (dx/dsigma))
            bg =  (2/math.sqrt(math.pi))*integ[0] + 1
            #bg =  (2/math.sqrt(math.pi))*integ[0] - 1

            bgradient.append(bg)

        return bgradient
        
    def fit(self, x, t):
        # checks for labels
        self.classes_ = np.unique(t)
        #t[t==0] = -1

        # initail variables k, w_0
        it = 0
        w = np.ones(len(x[0])) 
        b = 1               
        #w = np.zeros(len(x[0]))

        obj_batchsgd = []
        iter_batchsgd = []
        #sigma = np.identity(len(x[0]))

        for epoch in range(self.max_epochs):
            idx = np.random.permutation(len(t))
            print("Epoch: %d" %(epoch+1), idx)
            #print("Epoch: %d" %(epoch+1))
            for i in range(len(t)):
                
                r = idx[i*self.n_batches:(i+1)*self.n_batches]
                if r.size==0: break

                it = it + 1
                print("----it: ", it)
                iter_batchsgd.append(it)


                print("----Iteration: %d" %(i+1), r)

                X = x[r,:]
                T = t[r]
                print('t=', T)
                

                kv = self.kvtest(X,T)  
                print('kv=', kv)
                sigma = self.sigma(X)
                
                # compute gradient of loss depend on w 
                gradloss = self.gradient(w,b,X,T,sigma,kv)
                gradloss = np.vstack(gradloss)
                gloss = np.mean(gradloss, axis = 0)

                # compute gradient depend on w 
                grad = self.C* w - gloss

                
                # step size
                eta =1/it
                
                # update weight
                w -= eta*grad
                l2 = norm(w,2)
                w_ = min(1, (1/math.sqrt(self.C))/l2)*w
                print("----w: ",w_)
                

                # compute gradient depend on b 
                bgrad1 = self.bgradient(w,b,X,T,sigma,kv)
                bgrad2 = np.vstack(bgrad1)
                bgrad = (-1)*np.mean(bgrad2)
                #bgrad = np.mean(bgrad2)
                         
                # update bias
                b -= eta*bgrad
                print("----b: ",b)
                        
        self.final_iter = it
        self._coef = w_
        self._intercept = b
        self.obj_batchsgd = obj_batchsgd
        self.iter_batchsgd = iter_batchsgd

        return self

    def predict(self, x):
        p = np.sign(np.matmul(x,self._coef)+self._intercept)
        p[p==0] = 1
        return p.astype(int)
