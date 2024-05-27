import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from alive_progress import alive_bar
import scipy.sparse as sp

centers = np.loadtxt('centers.dat')*1e-10

occupancies = np.loadtxt('occupancies.dat')
print(occupancies.shape)
plt.plot(occupancies[0,:])
plt.plot(occupancies[1,:])
plt.plot(occupancies[2,:])
plt.show()
# positions = np.load('np_pos.npz', allow_pickle=True)['arr_0'].astype(int)
# positions = np.load('np_pos.npz', allow_pickle=True)['arr_0']
# positions = centers[positions.astype(int)]
times = np.loadtxt('times.dat')
sp_pos = sp.load_npz('sp_diff.npz')
print(times[:10])
print(sp_pos.shape)
# print(positions.shape)

# positions = positions*1e-10
coord = 0
N = sp_pos.shape[1]
print(N)
# l_param = 17.05
l_param = 11.7
new_time = np.zeros(len(times))
dt = times[-1]/len(times)

for i in range(len(new_time)):
    new_time[i] = i*dt

criteria = np.amax(centers, axis=0)
def unwrap_trajectories(positions):
    for coord in range(3):
        criterion = criteria[coord]
        diffs = np.diff(positions[:,coord])

        places = np.where(np.abs(diffs) >= criterion*0.5)[0]
        corrections = np.zeros_like(diffs)
        corrections[places] = - np.sign(diffs[places])*(l_param*8e-10)
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
            if np.sum(np.diff(r, axis=0)) > 0:
                r = unwrap_trajectories(r)  
                r_tot += r
                test = msd_fft(r)
                msd += test
            else:
                r = unwrap_trajectories(r)
                r_tot += r


            bar()
    return msd, r_tot

# msd = msd_fft(r_tot)
D_self, r_tot = compute_msd()
D_jump = msd_fft(r_tot)
# cProfile.run('compute_msd()')
t_fit = new_time[..., np.newaxis]
slope = np.linalg.lstsq(t_fit[:20000], D_self[:20000], rcond=None)[0][0]
d = slope / (6*N)
print(d)
slope = np.linalg.lstsq(t_fit[:20000], D_jump[:20000], rcond=None)[0][0]
d = slope / (6*N)
print(d)
plt.plot(D_self)
plt.plot(D_jump)
plt.show()

d = 0.1
np.savetxt(f'msd.dat', np.array([new_time, times, D_self, D_jump]))