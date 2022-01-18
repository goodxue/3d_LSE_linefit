#-*-  coding: utf-8  -*-
# Brief: LSE estimiation for radar trace.

__author__  =  'Weipeng Xue'
__email__  =  'weipengxue@sjtu.edu.cn'

import numpy as np
import math

# Keys of dictionaries.
XYZ_NAME = ('target_coordinatesx','target_coordinatesy','target_coordinatesz')
XYZ_VELOC = ('target_motion_vectorx','target_motion_vectory','target_motion_vectorz')

# Parameters:
#
# # radar_trace: List of List [[timestamp, x, y, z, vx, vy, vz, speed], ...]
# # location: List of [x, y, z] 
# # loc_weight_param: A parameter of weighted LSE. Only set for the weight of the location point.
# #
# # return: List of List [[timestamp, x, y, z, vx, vy, vz, speed], ...]

def fusion(radar_trace, location, loc_weight_param=10):
    num_train = len(radar_trace)
    # Weighted LSE parameters.
    LSE_WEIGHT = np.column_stack([np.ones((1,num_train)),np.array([loc_weight_param])]).reshape(-1,1)

    # Limit the process range of the function.
    if radar_trace[-1][3] < location[2]:
        radar_trace = [trace for trace in radar_trace if trace[3] > location[2]] #select trace[z] > falling_location as trainning data

    start_z = radar_trace[0][3]
    end_z = radar_trace[-1][3]
    loc_x, loc_y, loc_z = location[0], location[1], location[2]
    delta_z = np.abs(start_z-end_z)/num_train # Output listed with fixed delta z.

    # Convert data from python list to numpy ndarray.
    radar_trace.append([0]+location+[0,0,0,0])
    trace2X = []
    for trace in radar_trace:
        temp_loc = []
        for i in range(3):
            temp_loc.append(trace[1+i])
        trace2X.append(temp_loc)
    X = np.array(trace2X).reshape(-1,3)

    # Do LSE
    X_ave = np.mean(X,0)
    dx = X - X_ave.reshape(1,3)
    dx = np.multiply(LSE_WEIGHT,dx)
    C = dx.transpose().dot(dx) /(N-1)
    U,S,V = np.linalg.svd(C)

    # Generate data grid.
    i = 0
    zave_left = []
    while X_ave[2] + delta_z * i < start_z :
        zave_left.append(delta_z * i / U[:,0][2] )
        i += 1
    zave_right = []
    i = 1
    while X_ave[2] - delta_z * i > loc_z :
        zave_right.append(-delta_z * i / U[:,0][2] )
        i += 1

    zave = (zave_left + zave_right)
    zave.sort()
    result = []
    for det in zave:
        result.append([det * U[:,0].transpose() + X_ave])
    
    # Data format.
    result = np.array(result).reshape(-1,3)
    ret = [[0 for j in range(8)] for i in range(result.shape[0])]
    for i,point in enumerate(result):
        for j in range(3):
            ret[i][j+1] = point[j]
            ret[i][j+4] = delta_z / U[:,0][2]*U[:,0][j] #Unit direction vector.
        ret[i][7] = delta_z / U[:,0][2]
        ret[i][0] = 0
    for i,(track,res) in enumerate(zip(radar_trace,ret)) :
            res[0] = track[0]
            
    return ret


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    a , b, c = -1, -1, 30
    x_t = np.arange(10).reshape(1,10)
    y_t = np.arange(10).reshape(1,10)
    z_t = a* x_t + b * y_t + c
    fall_point = np.array([12,12,0])
    X = np.column_stack([x_t.transpose(),y_t.transpose(),z_t.transpose()])#.reshape(10,3)
    N = X.shape[0]
    radar_trace = [[0 for j in range(8)] for i in range(10)]
    for i in range(x_t.shape[1]):
        radar_trace[i][1] = x_t[0][i]
        radar_trace[i][2] = y_t[0][i]
        radar_trace[i][3] = z_t[0][i]

    location = [0 for i in range(3)]
    location[0] = fall_point[0]
    location[1] = fall_point[1]
    location[2] = fall_point[2]

    # Test the function.
    res = fusion(radar_trace,location)
    res = np.array(res)
    print(res)

    # Plot 
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(X[:,0],X[:,1],X[:,2],'g')
    ax.scatter(fall_point[0],fall_point[1],fall_point[2],'k')
    ax.scatter(res[:,1],res[:,2],res[:,3],'r')
    plt.show()