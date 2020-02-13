# import
import numpy as np

# parameters
m = 16
n = 32
num_sample = 100

# simulation data
sigma = np.exp(np.random.normal(size=[num_sample,m]))
rnasq = np.random.poisson(sigma)
A = 0.01*np.random.normal(size=[m,n])
atac = 0.5*np.random.normal(size=[num_sample,n])+np.matmul(rnasq,A)
atac[atac>0.5] = 1
atac[atac<=0.5] = 0
data = {"x":rnasq, "y":atac}
