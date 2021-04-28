import numpy as np

K = [[580.6161339,    0.,         268.94026972, 0],
 [  0.,         577.98851373, 363.36070951, 0],
 [  0. ,          0.,           1.        , 0]]
#30 degrees
theta = np.radians(25) #degrees
#height = 2185 #mm
height = 1314.45
costheta = np.cos(theta)
sintheta = np.sin(theta)
h_div_sin = -height / sintheta

R = [[1, 0, 0, 0],
    [0, costheta, -sintheta, 0],
    [0, sintheta, costheta, 0],
    [0, 0, 0, 1]
]

T = [[1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, h_div_sin],
    [0, 0, 0, 1]
]

np_k = np.array(K)
np_r = np.array(R)
np_t = np.array(T)

kFin = np.matmul(np_k, np.matmul(np_r, np_t))
print(kFin)