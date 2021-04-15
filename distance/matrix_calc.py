import numpy as np

K = [[506.33875218,   0.0,         275.41040741, 0],
    [  0.0,         504.50223021, 351.27749683, 0],
    [  0.0,           0.0,           1.0, 0      ]]

theta = np.radians(30) #degrees
height = 2185 #mm
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