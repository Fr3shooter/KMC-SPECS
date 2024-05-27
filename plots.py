import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(__file__)

A = np.loadtxt(f'{script_dir}/output_175.dat')
B = np.loadtxt(f'{script_dir}/output_200.dat')

plt.plot(A[:,0], A[:,1], marker='o', color='blue', label='225K')
plt.plot(A[:,0], A[:,2], marker='o', color='blue')
plt.plot(B[:,0], B[:,1], marker='o', color='red', label='250K')
plt.plot(B[:,0], B[:,2], marker='o', color='red')
plt.grid()
plt.xlabel('Total occupancy')
plt.ylabel('Fractional occupancies')
plt.legend()
plt.title('Evolution of fractional occupancies')
plt.ylim(0,1)
plt.xlim(0,1)
plt.show()

# C = np.loadtxt(f'{script_dir}/hoppinh.txt')
# plt.plot(C[:,0], C[:,1], marker='o', label='S5L')
# plt.plot(C[:,0], C[:,2], marker='o', label='L5S')
# plt.plot(C[:,0], C[:,3], marker='o', label='L5L')
# plt.plot(C[:,0], C[:,4], marker='o', label='L6L')
# plt.ylabel('Events')
# plt.xlabel('Total occupancy')
# plt.legend()
# plt.title('Evolution of hopping events')
# plt.grid()
# plt.show()