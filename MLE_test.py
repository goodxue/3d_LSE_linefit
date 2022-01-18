import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    a , b, c = -1, -1, 30
    x_t = np.arange(10).reshape(1,10)
    y_t = np.arange(10).reshape(1,10)
    z_t = a* x_t + b * y_t + c
    fall_point = np.array([12,12,0])
    X = np.column_stack([x_t.transpose(),y_t.transpose(),z_t.transpose()])#.reshape(10,3)
    X = np.row_stack([X,fall_point])
    N = X.shape[0]

    X_ave = np.mean(X,0)
    dx = X - X_ave.reshape(1,3)
    w = np.array([1,1,1,1,1,1,1,1,1,1,10]).reshape(11,1)
    dx = np.multiply(w,dx)
    C = dx.transpose().dot(dx) /(N-1)
    U,S,V = np.linalg.svd(C)
    x = dx.dot(U[:,0])
    x_min = np.min(x)
    x_max = np.max(x)
    difx = x_max - x_min
    Xa = (x_max - 0.19*difx) * U[:,0].transpose() + X_ave
    Xb = (x_max + 0.05 * difx) * U[:,0].transpose() + X_ave
    end_x = np.row_stack([Xa,Xb])

    #print(X)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(X[:,0],X[:,1],X[:,2])
    ax.plot(end_x[:,0],end_x[:,1],end_x[:,2],'r')
    plt.show()

    