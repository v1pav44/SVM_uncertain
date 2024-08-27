import numpy as np

data = np.genfromtxt('australian.dat',
                      skip_header=19,
                      skip_footer=0,
                      names=None,
                      dtype=float,
                      delimiter=',')

x = data[:,:-1]
t = data[:,-1]
t[t == 0] = -1
  

data = np.genfromtxt('bupa.dat',
                      skip_header=11,
                      skip_footer=0,
                      names=None,
                      dtype=float,
                      delimiter=',')
        
x = data[:,:-1]
t = data[:,-1]
t[t == 2] = -1

data = np.genfromtxt('heart.dat',
                      skip_header=18,
                      skip_footer=0,
                      names=None,
                      dtype=float,
                      delimiter=',')

x = data[:,:-1]
t = data[:,-1]
t[t == 2] = -1
        
    

data = np.genfromtxt('spectfheart.dat',
                      skip_header=49,
                      skip_footer=0,
                      names=None,
                      dtype=float,
                      delimiter=',')

x = data[:,:-1]
t = data[:,-1]
t[t == 0] = -1
  
    
data = np.genfromtxt('ionosphere.dat',
                      skip_header=38,
                      skip_footer=0,
                      names=None,
                      dtype=float,
                      delimiter=',')
    
x = data[:,:-1]
t = data[:,-1]
    

data = np.genfromtxt('cmc.csv',
                      skip_header=1,
                      skip_footer=0,
                      names=None,
                      dtype=float,
                      delimiter=',')
    
x = data[:,:-1]
t = data[:,-1]
t[t == 1] = 1
t[t!=1] = -1
        
data = np.genfromtxt('spect.dat',
                      skip_header=0,
                      skip_footer=0,
                      names=None,
                      dtype=float,
                      delimiter=',')
x = data[:,1:]
t = data[:,:1]
t[t != 1] = -1
 