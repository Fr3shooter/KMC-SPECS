import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('msd_CM_0.1_N2.dat')
times = data[0]
new_time = data[1]
D_self = data[2]
D_jump = data[3]
N = 409

new_time = new_time[..., np.newaxis]
slope = np.linalg.lstsq(new_time[:20000], D_self[:20000], rcond=None)[0][0]
d = slope / (6*N)

print(d)

plt.plot(D_self)
plt.plot(D_jump)
plt.show()