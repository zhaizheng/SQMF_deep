import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.parametrizations import orthogonal
import matplotlib.pyplot as plt


# Parameters
input_dim = 3  # Dimension of input data (X)
latent_dim = 2  # Dimension of latent representation (tau)
normal_dim = 1  # Dimension of normal subspace (related to V)
sample_n = 1000  # Number of samples
learning_rate = 0.002

#X = torch.randn(input_dim, sample_n)  # Example data (100 samples)
import numpy as np
X = np.random.randn(input_dim, sample_n)
nX = np.sqrt(np.sum(X*X,0))
Y = X/nX




n = 100
#X = torch.tensor(X/nX).float()


import matplotlib.pyplot as plt


#a = np.arange(0,0.3,0.01)
#X = torch.tensor(np.array([np.sin(a),np.cos(a)])).float()


class SQMF(nn.Module):
    def __init__(self, D, d, s, n, Xs):
        super().__init__()
        self.c = nn.Parameter(torch.mean(Xs,dim=1,keepdim=True))
        init_U, init_S, init_V = torch.linalg.svd(Xs-self.c)
        orth_linear = nn.Linear(d+s,D)
        orth_linear.weight.data = init_U
        self.Q = orthogonal(orth_linear)#,orthogonal_map="cayley")
        self.phi = nn.Parameter(torch.randn(d, n))
        #self.psi = self.count_psi(self.phi)
        self.theta = nn.Parameter(torch.randn(d * (d + 1) // 2, s))

    def calculate_psi(self, phi):
        d, n = phi.shape
        output_dim = d * (d + 1) // 2#+d
        output = torch.zeros((output_dim, n))

        idx = 0
        for i in range(d):
            for j in range(i, d):
                output[idx,:] = phi[i,:] * phi[j,:]
                idx += 1
        #for i in range(d):
        #    output[idx,:] = phi[i,:] * phi[i,:]* phi[i,:]
        #    idx += 1
        return output

    def forward(self):
        return self.pred(self.phi)

    def regularization(self):
        return torch.norm(self.theta.T @ self.calculate_psi(self.phi),p="fro")

    def pred(self, phi):
        return self.c+self.Q.weight @ torch.cat((phi, self.theta.T @ self.calculate_psi(phi)),dim=0)



import time
use_gpu = False
device = torch.device("cuda")



#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

for i in range(1): 

    #if np.random.random()>0.1:
        center = Y[:,i]
        t = np.array([[center[0]]*sample_n,[center[1]]*sample_n,[center[2]]*sample_n])
        idd = np.argsort(np.sum((Y-t)*(Y-t),0))
        #print(idx)
        Xs = torch.tensor(Y[:,idd[0:n]]+0.001*np.random.randn(input_dim,n)).float()
        use_gpu = "False"
        # Initialize model, optimizer

        model = SQMF(input_dim, latent_dim, normal_dim, n, Xs)

        # Use Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        num_epochs = 20000
        if use_gpu == True:
            loss_func = nn.MSELoss().to(device)
        else:
            loss_func = nn.L1Loss()#nn.MSELoss()#

        init_time = time.time()
        for epoch in range(num_epochs):
            time_start = time.time()
            optimizer.zero_grad()
            if use_gpu == True:
                output = model().to(device)
                loss = loss_func(output, Xs.to(device)).to(device)
            else:
                output = model()
                #print(output)
                loss = loss_func(output, Xs)#+ 0.01*model.regularization()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f},time: {time.time()-time_start}")

        print("Training complete. Final loss:{0},time:{1}, Use_GPU:{2}".format(loss.item(),time.time()-init_time,use_gpu))

        #print(model.theta.data)
        x = np.linspace(-0.5,0.5,20)
        y = np.linspace(-0.5,0.5,20)
        [xx,yy] = np.meshgrid(x,y)
        aa = np.zeros(np.shape(xx))
        bb = np.zeros(np.shape(xx))
        cc = np.zeros(np.shape(xx))
        for i in range(np.shape(xx)[0]):
            for j in range(np.shape(xx)[1]):
                data = np.squeeze(model.pred(torch.tensor(np.array([[xx[i,j],yy[i,j]]]).T).float()).detach().numpy())
                aa[i,j] = data[0]
                bb[i,j] = data[1]
                cc[i,j] = data[2]
        fig = plt.figure()
        axis = plt.axes(projection="3d")
        
        axis.scatter(aa,bb,cc,marker='o')
        axis.scatter(Xs[0,:],Xs[1,:],Xs[2,:],marker='*')
        #surf = ax.scatter(aa,bb,cc)

plt.show()
print(model.Q.weight)
print(model.c)
    #print(model.predict(torch.tensor(np.array([[1],[0]]))))
