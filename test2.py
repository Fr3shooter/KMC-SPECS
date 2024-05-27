import numpy as np
import matplotlib.pyplot as plt


# Faire jusqu'à 270K

A = np.loadtxt('output_150_II.dat')
B = np.loadtxt('output_.dat')
C = np.loadtxt('output_200.dat')
interval1 = np.array(range(5,100,10))/100
interval = np.insert(interval1, 0, 0)
interval = np.append(interval, 1)

# k = 0
# for i in range(0,len(A),2):
#     plt.plot(A[i,:]/(0.25*4096), label='Small')
#     plt.plot(A[i+1,:]/(0.75*4096), label='Large')
#     plt.title(f'Fractional occupancies at T=250K, for d={interval1[k]}')
#     plt.grid()
#     plt.show()
#     k += 1
S, L = [0], [0]
S2, L2 = [0], [0]
S3, L3 = [0], [0]
for i in range(0,len(A),2):
    S.append(np.mean(A[i,:])/(2*12288/3))
    L.append(np.mean(A[i+1,:])/(12288/3))
    # S2.append(np.mean(B[i,:])/(0.25*4096))
    # L2.append(np.mean(B[i+1,:])/(0.75*4096))
    # S3.append(np.mean(C[i,:])/(0.25*4096))
    # L3.append(np.mean(C[i+1,:])/(0.75*4096))
S.append(1)
L.append(1)
S2.append(1)
L2.append(1)
S3.append(1)
L3.append(1)

# plt.plot(interval, L2, marker='o', label='150K', color='orange')
# plt.plot(interval, S2, marker='+', color='orange')
# plt.plot(interval, L3, marker='o', label='200K', color='grey')
# plt.plot(interval, S3, marker='+', color='grey')
plt.plot(interval, L, marker='o', label='175K', color='red')
plt.plot(interval, S, marker='+', color='red')
plt.xlabel('Total occupancy (%)')
plt.ylabel('Fractional occupancy (% cages)')
plt.legend()
plt.grid()
plt.title('Fractional occupancies for various T, + are for small, o for large')
plt.show()