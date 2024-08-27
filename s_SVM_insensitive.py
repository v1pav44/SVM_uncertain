import numpy as np
import math
from scipy.integrate import quad
import statistics
from scipy import integrate
import random
from sklearn.base import BaseEstimator
import pandas as pd


class SVM_Insensitive_BFGS_2(BaseEstimator):

    def __init__(self, C = 1, max_epochs = 1, n_batches = 2, 
                 tau = 1, epsilon = 1
                ):
        assert C > 0
        self.C = C
        assert tau > 0
        self.tau = tau
        assert epsilon > 0
        self.epsilon = epsilon
        assert max_epochs > 0 and isinstance(max_epochs, int)
        self.max_epochs=max_epochs
        assert n_batches > 0 and isinstance(n_batches, int)
        self.n_batches=n_batches

    '''gradient of uncertaininsensitive'''


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

    #gradient on w 
    def gradient(self, w, b, x, t, sigma, kv): 
        
        gradient = [] 
               
        for k in range(len(t)):

            dx_1 = (1 - t[k]*(np.dot(w.T,x[k]) + b)) - self.epsilon
            dx_2 = - self.tau*(1 - t[k]*(np.dot(w.T,x[k]) + b)) - self.epsilon

            sigma_new = 0.25*kv[k]*sigma
            dsigma =  math.sqrt(2*np.dot(np.dot(w.T,sigma_new),w))
            
            g1 = ((math.exp(-(dx_2**2)/(dsigma**2)) + math.exp(-(dx_1**2)/(dsigma**2)))/(math.sqrt(math.pi)*dsigma)) * np.dot(sigma,w)
            

            f = lambda y : math.exp(-y**2)
            integ_1 = integrate.quad(f, 0, (-(dx_1)/dsigma))
            integ_2 = integrate.quad(f, 0, (-(dx_2)/dsigma))


            g2 =  (self.tau*(1- (2/math.sqrt(math.pi))*integ_2[0]) - ((1- (2/math.sqrt(math.pi))*integ_1[0])))/2 * (x[k])*t[k]

            g = g1 + g2
            gradient.append(g) 
  
        return gradient

    #gradient on b 
    def bgradient(self, w, b, x, t, sigma, kv): 
        
        bgradient = [] 
                
        for k in range(len(t)):
            dx_1 = (1 - t[k]*(np.dot(w.T,x[k]) + b)) - self.epsilon
            dx_2 = - self.tau*(1 - t[k]*(np.dot(w.T,x[k]) + b)) - self.epsilon
            sigma_new = 0.25*kv[k]*sigma
            dsigma =  math.sqrt(2*np.dot(np.dot(w.T,sigma_new),w))
            
            f = lambda y : math.exp(-y**2)
            integ_1 = integrate.quad(f, 0, (-(dx_1)/dsigma))
            integ_2 = integrate.quad(f, 0, (-(dx_2)/dsigma))
            bg =  (((self.tau*(1- ((2/math.sqrt(math.pi))*integ_2[0]))) - (1- ((2/math.sqrt(math.pi))*integ_1[0])))*t[k])/2

            bgradient.append(bg) 

        return bgradient

    def cost_function(self,w,b,x,t):
            
      loss = []
      
      kv = self.kvtest(x,t)

      sigma = self.sigma(x)
      #sigma = np.identity(len(x[0]))

      for k in range(len(t)):

        dx_1 = (1 - t[k]*(np.dot(w.T,x[k]) + b)) - self.epsilon
        dx_2 = - self.tau*(1 - t[k]*(np.dot(w.T,x[k]) + b)) - self.epsilon

        sigma_new = 0.25*kv[k]*sigma
        dsigma =  math.sqrt(2*np.dot(np.dot(w.T,sigma_new),w))    # dsigma = e
            
        f = lambda y : math.exp(-y**2)
        integ_1 = integrate.quad(f, 0, (-(dx_1)/dsigma))
        integ_2 = integrate.quad(f, 0, (-(dx_2)/dsigma))

        g1 = ((dx_1)/2)*(1- (2/math.sqrt(math.pi))*integ_1[0]) + dsigma/(2*math.sqrt(math.pi))*(math.exp(-(dx_1**2/dsigma**2)))
        
        g2 = (dx_2/2)*(1- (2/math.sqrt(math.pi))*integ_2[0]) + dsigma/(2*math.sqrt(math.pi))*(math.exp(-(dx_2**2/dsigma**2)))  

        loss_ = g1+g2 
        loss.append(loss_)
      
      loss = np.array(loss)
      cost = (1/2)*np.linalg.norm(w)**2  + (1/2)*(b**2)+ self.C*np.mean(loss)
      return loss, cost

     
    def construct_x_uncertain(self,x,t):
        x_new = []

        kv = self.kvtest(x,t)
        #print('kv=', kv)
        sigma = self.sigma(x)
        #sigma = np.identity(len(x[0]))
        
        for i in range(len(x)):
            
            sigma_new = 0.25*kv[i]*sigma
  
            cov = sigma_new 
            mean = x[i]

            x_uncertain = np.random.multivariate_normal(mean, cov)
            x_new.append(x_uncertain)
        
        return np.array(x_new)   

    #approximate RES_BFGS matrix 
    def Update_RES_BFGS(self,B, dw, dg):
        dg_t =  dg[:, np.newaxis]
        Bdw = np.dot(B, dw)
        dw_t_B = np.dot(dw, B)
        dwBdw = np.dot(np.dot(dw, B), dw)

        p = dg_t*dg
        u = Bdw[:, np.newaxis] * dw_t_B

        B_new = B + p / np.dot(dg, dw) - u / dwBdw 
        return B_new 

        #approximate RES_BFGS matrix 
    def Update_RES_BFGS_b(self,B, dw, dg):
        dg_t =  dg
        Bdw = np.dot(B, dw)
        dw_t_B = np.dot(dw, B)
        dwBdw = np.dot(np.dot(dw, B), dw)

        p = dg_t*dg
        u = Bdw * dw_t_B

        B_new = B + p / np.dot(dg, dw) - u / dwBdw 
        return B_new 
        
    def fit(self, x, t):

        x_new = self.construct_x_uncertain(x,t)

        # checks for labels
        self.classes_ = np.unique(t)
        #t[t==0] = -1

        # initail variables k, w_0
        k = 0
        w = np.ones(len(x[0])) 
        w_ = np.ones(len(x[0])+1)
        b = 1               
        #w = np.zeros(len(x[0]))
        obj_batchsgd = []
        iter_batchsgd = []
        obj_func = []

        H_w = np.identity(len(x[0]))
        H_b = 1


        for i in range(self.max_epochs):
            
                k = k + 1
                print("----k: ",k)
                iter_batchsgd.append(k)

                # compute cost function
                loss, cost = self.cost_function(w,b,x,t)
                obj_func.append(cost)  

                X = x
                T = t
                #sigma = self.sigma(X)
                sigma = np.identity(len(X[0]))
                kv = self.kvtest(X,T)
                #print('t=', T)

                X_new = x_new
                T_new = t

                X = x
                T = t

                # step size
                eta = 1/k
             
                
                # compute gradient of loss depend on w 
                gradloss = self.gradient(w,b,X_new,T_new,sigma, kv)
                gradloss = np.vstack(gradloss)
                gloss = np.mean(gradloss, axis = 0)

                # compute gradient depend on w 
                grad =  w + self.C*gloss


                # update inv hessian matrix of w  
                Hw_inv = np.linalg.inv(H_w)


                # update w 
                w_new = w - eta * np.matmul(Hw_inv,grad)
                                

                # compute gradient of loss depend on b 
                bgradloss1 = self.bgradient(w,b,X_new,T_new,sigma, kv)
                bgradloss2 = np.vstack(bgradloss1)
                bgradloss = np.mean(bgradloss2)

                # compute gradient depend on w
                bgrad = b + self.C*bgradloss

                # update inv hessian matrix of b
                Hb_inv = 1/H_b

                # update b
                b_new = b - eta * Hb_inv * bgrad

                #gradient  ที่จุดใหม่
                # compute gradient of loss depend on w ที่จุดใหม่
                gradloss_ = self.gradient(w_new,b_new,X_new,T_new,sigma, kv)
                gradloss_ = np.vstack(gradloss_)
                gloss_ = np.mean(gradloss_, axis = 0)

                # compute gradient depend on w ที่จุดใหม่
                grad_new =  w_new + self.C*gloss_

                # compute gradient of loss depend on b ที่จุดใหม่
                bgradloss1_ = self.bgradient(w_new,b_new,X_new,T_new,sigma, kv)
                bgradloss2_ = np.vstack(bgradloss1_)
                bgradloss_ = np.mean(bgradloss2_)

                # compute gradient depend on b ที่จุดใหม่
                bgrad_new = b_new + self.C*bgradloss_


                #Hessian
                p_w = w_new - w
                g_w = grad_new - grad 

                p_b = b_new - b
                g_b = bgrad_new - bgrad 

                H_w_new = self.Update_RES_BFGS(H_w, p_w, g_w)

                H_w = H_w_new

                H_b_new = self.Update_RES_BFGS_b(H_b, p_b, g_b)

                H_b = H_b_new

                # update step
                w = w_new
                b = b_new 

                

              
        self.final_iter = k
        self._coef = w
        self._intercept = b
        self.obj_func = obj_func
        self.obj_batchsgd = obj_batchsgd
        self.iter_batchsgd = iter_batchsgd

        return self

    def predict(self, x):
        p = np.sign(np.matmul(x,self._coef)+self._intercept)
        p[p==0] = 1
        return p.astype(int)
