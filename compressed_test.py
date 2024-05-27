import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
from alive_progress import alive_bar
from numba import njit
import cProfile
import scipy.sparse as sp
import multiprocessing as mp
from tqdm import tqdm


centers = np.loadtxt('centers.dat')*1e-10
occupancies = np.loadtxt('occupancies_250_0.1.dat')

# plt.plot(occupancies[0,:])
# plt.plot(occupancies[1,:])
# plt.plot(occupancies[2,:])
# plt.show()
times = np.loadtxt('times_250_0.1.dat')
sp_pos = sp.load_npz('sp_diff_250_0.1.npz')
coord = 0
N = sp_pos.shape[1]
print(N)

new_time = np.zeros(len(times))
dt = times[-1]/len(times)

for i in range(len(new_time)):
    new_time[i] = i*dt

def unwrap_trajectories(positions):
    criteria = [9.1637e-09, 9.1652e-09, 9.1652e-09]
    for coord in range(3):
        criterion = criteria[coord]
        diffs = np.diff(positions[:,coord])
        places = np.where(np.abs(diffs) >= criterion*0.9)[0]
        corrections = np.zeros_like(diffs)
        corrections[places] = - np.sign(diffs[places])*(11.7*8e-10)
        positions[1:,coord] += np.cumsum(corrections)
        interpolated_function = interp1d(times, positions[:,coord], kind='linear')
        positions[:,coord] = interpolated_function(new_time)
    return positions

def autocorrFFT(x):
  N=len(x)
  
  F = np.fft.fft(x, n=2*N)
  PSD = F * F.conjugate()
  res = np.fft.ifft(PSD)
  res = (res[:N]).real   
  n = N*np.ones(N)-np.arange(0,N) 
  return res/n 

def msd_fft(r):
  N=len(r)
  D=np.square(r).sum(axis=1) 
  D=np.append(D,0) 
  S2=sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
  Q=2*D.sum()
  S1=np.zeros(N)
  for m in range(N):
      Q=Q-D[m-1]-D[N-m]
      S1[m]=Q/(N-m)
  return S1-2*S2


def compute_msd():
    msd = 0
    r_tot = 0
    with alive_bar(N) as bar:
        for i in range(N):
            r = centers[np.cumsum(sp_pos.tocsr()[:,i].toarray()).astype(int)]
            r = unwrap_trajectories(r)
            r_tot += r

            test = msd_fft(r)
            msd += test
            bar()
    return msd

if __name__ == '__main__':
    def update(*a):
        pbar.update()
    
    pbar = tqdm(total=N)
    pool = mp.Pool()
    results = 0
    r_tot = 0
    for i in range(N):
        r = centers[np.cumsum(sp_pos.tocsr()[:,i].toarray()).astype(int)]
        r = unwrap_trajectories(r)
        r_tot += r
        result = pool.apply_async(msd_fft, args=([r]), callback=update)
        results += result.get()
    pool.close()
    pool.join()
    pbar.close()

# # msd = msd_fft(r_tot)
# msd = compute_msd()
# # cProfile.run('compute_msd()')
# new_time = new_time[..., np.newaxis]
# slope = np.linalg.lstsq(new_time[:20000], msd[:20000], rcond=None)[0][0]
# d = slope / (6*N)
# print(d)
# # slope = np.linalg.lstsq(new_time[:20000], msd_test[:20000], rcond=None)[0][0]
# d = slope / (6*N)
# print(d)
# plt.plot(msd)
# # plt.plot(msd_test)
# plt.show()

# d = 0.1
# np.savetxt(f'msd_CM_{d}_N2.dat', np.array([new_time[:,0], msd, times]))