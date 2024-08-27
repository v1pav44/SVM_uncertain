from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_moons, make_blobs
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from SVM_generalized_pinball import SVM_GP_sgd
from SVM_Hingeloss import SVM_Hinge_sgd
from SVM_insensitive import SVM_Insensitive_BFGS
import math
import numpy as np
import pandas as pd
import random

# from RBF_Svm_insensitive import *  
# from RBF_Svm_generalized_pinball import RBF_SVM_GP_sgd
# from RBF_Svm_Hingeloss import *

from s_SVM_Hingeloss import SVM_Hinge_sgd_2

######################            generate data           ##################

#X, y = make_classification(n_samples=500, n_features=2,n_redundant=0, random_state= 7)
#X,y= make_moons(n_samples=700, random_state=200, noise = 0.1)
#X, y = make_blobs(n_samples=3000, centers=2, n_features=2, random_state=1, cluster_std=3)
#y[y==0] = -1

######################            UCI data           ##################
data = np.genfromtxt(r'C:\Users\Kook\Desktop\SVM_Uncertain_2\data\monk-2.dat',
                      skip_header= 11,
                      skip_footer=0,
                      names=None,
                      dtype=float,
                      delimiter=',')

np.random.shuffle(data)

x = data[:, :-1]
t = data[:, -1]
t[t ==  0 ] = -1

 
x = data[0:math.floor(0.1*len(data)), :-1]
t = data[0:math.floor(0.1*len(data)), -1]


from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler((0,1))
x = norm.fit_transform(x)

'''
noise
'''
#import noiseADD as na
#X = na.withnoise(X, r=0.02)
#X = na.withnoise(X, r=0.05)
#X = na.withnoise(X, r=0.09)

'''
Get data for training and testing
'''
#x_train, x_test, y_train, y_test = train_test_split(positive_data, y_positive_data, test_size=0.35, random_state= 1, stratify=y_positive_data)
#M, p, A, B, u, u_pos, u_neg, test_data, p_test = Processed_data_1(X,y)


##########################             Grid search          #####################
from sklearn.model_selection import GridSearchCV
#self, C = 1, max_epochs = 3, n_batches = 32, 
#tau_1 = 1, tau_2 = 1, epsilon_1 = 1, epsilon_2 = 1

_params = [{'C': [0.125, 0.25, 0.5, 2, 4, 8],
#    #'tau': [0.2, 0.4, 0.8, 1],
    'tau_1': [0.2, 0.4, 0.8, 1],
    'tau_2': [0.2, 0.4, 0.8,1], 
#    #'epsilon': [0.2, 0.4, 0.8,1],
    'epsilon_1': [0.2, 0.4, 0.8,1],
    'epsilon_2': [0.2, 0.4, 0.8,1]
    },
    ]

#_params = [{'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
   #'tau': [0.2, 0.4, 0.8, 1],
   # 'tau_1': [0.0625, 0.125, 0.25, 1, 2],
   # 'tau_2': [0.0625, 0.125, 0.25, 1, 2], 
   # #'epsilon': [0.2, 0.4, 0.8,1],
   # 'epsilon_1': [0.0625, 0.125, 0.25, 1, 2],
   # 'epsilon_2': [0.0625, 0.125, 0.25, 1, 2]
#   },
#   ]

_cv = 5

#using_model = SVM_Hinge_sgd(max_epochs = 10, n_batches = 64)
#using_model = SVM_Hinge_sgd_2(max_epochs = 50, n_batches = 32)
using_model = SVM_GP_sgd(max_epochs = 10, n_batches = 64)
#using_model = RBF_SVM_Insensitive_BFGS(max_epochs = 3, n_batches = 32)
#using_model = RBF_SVM_GP_sgd(max_epochs = 3 , n_batches = 32)

grid = GridSearchCV(using_model, param_grid=_params, scoring='accuracy', cv=_cv, n_jobs=-1)
grid.fit(x, t)

print(grid.best_params_, grid.best_score_)