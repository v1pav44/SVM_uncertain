import numpy as np
import pandas as pd
import os
#from RBF_Svm_generalized_pinball import RBF_SVM_GP_sgd

from SVM_generalized_pinball import SVM_GP_sgd
from SVM_Hingeloss import SVM_Hinge_sgd
from SVM_insensitive import SVM_Insensitive_BFGS

from s_SVM_Hingeloss import SVM_Hinge_sgd_2
from s_SVM_insensitive import SVM_Insensitive_BFGS_2
from s_SVM_generalized_pinball import SVM_GP_sgd_2

data = np.genfromtxt(r'C:\Users\Kook\Desktop\SVM_Uncertain_2\data\monk-2.dat',
                      skip_header=11,
                      skip_footer=0,
                      names=None,
                      dtype=float,
                      delimiter=',')

#data = data[:,:-1]

x = data[:,:-1]
t = data[:,-1]
t[t == 0] = -1


from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)


import noiseADD as na
#x = na.withnoise(x, r=0.05)
x = na.withnoise(x, r=0.1)
#x = na.withnoise(x, r=0.15)
#x = na.withnoise(x, r=0.20)
#x  = na.withnoise(x, r=0.30)
#x = na.withnoise(x, r=0.50)
#x = na.withnoise(x, r=0.70)

#x = np.array(x)
#t = np.array(t)

'''
                            10-fold cross validation
'''
from sklearn.model_selection import KFold
n_folds = 10
cv = KFold(n_splits=n_folds, shuffle=True, random_state=60031073)

acc_train = np.empty((n_folds))
acc_test = np.empty((n_folds))
processingtime = np.empty((n_folds))
conf_M_train = np.empty((n_folds,2,2))
conf_M_test = np.empty((n_folds,2,2))
print('#'*60)
      
'''
                            Import evaluation package
'''
import time
from sklearn.metrics import accuracy_score, confusion_matrix
#############################################
#start Cross Validation
for i, (train, test) in enumerate(cv.split(x)):
    #choose model
    
    ##############################     Linear Case
    
    #model = SVM_Hinge_sgd(max_epochs = 10, n_batches = 64, C=100)
    #model = SVM_Insensitive_BFGS(max_epochs = 3, n_batches = 64, C=0.5, tau =0.2, epsilon = 0.4)
    #model = SVM_GP_sgd(C = 4, max_epochs = 50, n_batches = 64, tau_1 = 0.8, tau_2 = 0.2, epsilon_1 = 0.4, epsilon_2 = 0.2)
    
    #model = SVM_Hinge_sgd_2(max_epochs = 50, n_batches = 64, C=10)
    model = SVM_Insensitive_BFGS_2(max_epochs = 50, n_batches = 64, C=0.5, tau =0.2, epsilon = 0.4)
    #model = SVM_GP_sgd_2(C = 0.25, max_epochs = 50, n_batches = 64, tau_1 = 0.8, tau_2 = 0.2, epsilon_1 = 0.8, epsilon_2 = 1)
    
    #############################     Nonlinear Case   
    
    #C in linear case and lamba are the same but using different alphabet in Hingeloss_sgd
    #model = RBF_Hingeloss(max_epochs = 50, n_batches = 32, lamba= 0.125   )
    #model = RBF_SVM_Insensitive_BFGS(max_epochs = 50, n_batches = 64, C=0.125, tau =1  , epsilon = 0.4)
    #model = RBF_SVM_GP_sgd(C = 2, max_epochs = 50, n_batches = 32, tau_1 = 0.25, tau_2 = 0.2, epsilon_1 = 0.25, epsilon_2 = 0.25)
    
    start = time.time()

    #modeling
    #model.fit(x[train], t[train])
    model.fit(x[train], t[train]) 

    #cpu time used
    processingtime[i] = time.time() - start

    # x[train] = model.construct_x_uncertain(x[train], t[train])
    # x[test] = model.construct_x_uncertain(x[test], t[test])


    #prediction
    y_train = model.predict(x[train])
    y_test = model.predict(x[test])
    
    #evaluation model
    acc_train[i]= accuracy_score(t[train], y_train)
    acc_test[i] = accuracy_score(t[test], y_test)

    # confusion matrix
    conf_M_train[i] = confusion_matrix(t[train], y_train)
    conf_M_test[i] = confusion_matrix(t[test], y_test)
    #w_b = model.w_b
    #df = pd.DataFrame(w_b)

#display result
#df.to_csv(os.path.join(r'w_b.csv'))
import collections
#print(x[train].shape)
#print(t[train].shape)
print('CPU time = %.4f and its S.D. = %.4f' % (np.mean(processingtime), np.std(processingtime)))
print('TRUTH : ',collections.Counter(t[train]))
print('PREDICT : ',collections.Counter(y_train))
print("Training accuracy = %.4f and its S.D. = %.4f" % (np.mean(acc_train), np.std(acc_train)))
print('Training confusion matirx: \n', np.mean(conf_M_train, axis=0))
print('TRUTH : ',collections.Counter(t[test]))
print('PREDICT : ',collections.Counter(y_test))
print("Test accuracy     = %.4f and its S.D. = %.4f" % (np.mean(acc_test), np.std(acc_test)))
print('Test confusion matirx: \n', np.mean(conf_M_test, axis=0))
print('#'*60)