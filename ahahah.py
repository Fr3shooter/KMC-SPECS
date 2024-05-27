import numpy as np
import matplotlib.pyplot as plt

# connection between macroscopic kinetic measurables and the degree of rate control jorgensen
# limiter temps pour fit et voir différence

data = np.loadtxt('msd_CM_0.1_N2.dat')
print(data.shape)
msd = data[1,:]
time = data[0,:]

time = time[..., np.newaxis]
slope = np.linalg.lstsq(time, msd, rcond=None)[0][0]/(6*0.1*4096)
print(slope)

idx = np.where(time<1e-7)[0][-1]
time = time[:idx]
msd = msd[:idx]
print(np.linalg.lstsq(time, msd, rcond=None)[0][0]/(6*0.1*4096))
print(time[-1])

plt.plot(msd)
plt.show()

# print(time[-1])
# # times = np.loadtxt('times.dat')
# slope = np.linalg.lstsq(time[:1000], msd[:1000], rcond=None)[0][0]
# d = slope / (6*int(0.6*4096))
# L = []
# densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# N = [int(i*4096) for i in densities]
# for i in range(len(densities)):
#     data = np.loadtxt(f'msd_{densities[i]}.dat')
#     msd = data[1,:]
#     time = data[0,:]
#     idx = np.where(time<1e-6)[0][-1]
#     time = time[:idx]
#     time = time[..., np.newaxis]
#     msd = msd[:idx]
#     print(densities[i],np.linalg.lstsq(time, msd, rcond=None)[0][0]/(6*N[i]))
#     L.append(np.linalg.lstsq(time, msd, rcond=None)[0][0]/(6*N[i]))

# plt.plot(densities, L, marker='+')
# plt.xlabel('Xt')
# plt.ylabel('Self Diffusion (m²/s)')
# plt.grid()
# plt.show()