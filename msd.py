import numpy as np
from scipy.interpolate import interp1d
import time
import scipy.sparse as sp

T = 150
structure = 'sI'
mode = 'DO'
densities = np.round(np.arange(1,10,1)*0.1,2)

if structure == 'sI':
    small = 2/8 * 8 * 8 * 8 * 8
    large = 6/8 * 8 * 8 * 8 * 8
    l_param = 11.7
elif structure == 'sII':
    small = 16/24 * 24 * 8 * 8 * 8
    large = 8/24 * 24 * 8 * 8 * 8
    l_param = 17.05

l_occ, s_occ, d_occ, occ = [], [], [], []

for i in densities:
    arr = np.loadtxt(f'{i}/occupancies_{T}_{i}.dat')
    s_occ.append(np.mean(arr[0,:])/small)
    l_occ.append(np.mean(arr[1,:])/large)
    d_occ.append(np.mean(arr[2,:])/large)
    if mode == 'DO':
        occ.append((np.mean(arr[0,:])+np.mean(arr[1,:])+np.mean(arr[2,:]))/(small+2*large))
    else:
        occ.append((np.mean(arr[0,:])+np.mean(arr[1,:]))/(small+large))

def unwrap_trajectories(positions, times, criteria):
    # criteria = [9.1637e-09, 9.1652e-09, 9.1652e-09]
    
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


def compute_msd(centers, sp_pos):
    msd = 0
    r_tot = 0
    for i in range(N):
        r = centers[np.cumsum(sp_pos.tocsr()[:,i].toarray()).astype(int)]
        r = unwrap_trajectories(r, times, criteria)
        r_tot += r
        test = msd_fft(r)
        msd += test

    return msd, r_tot

np.savetxt('occupancies.dat', [s_occ, l_occ, occ])
densities = np.round(np.arange(1,10,1)*0.1,2)
for i in densities:
    print(50*'_')
    print(f'Computing MSD for {i}')
    t_init = time.time()
    centers = np.loadtxt(f'{i}/centers.dat')*1e-10
    criteria = np.max(centers, axis=0)
    times = np.loadtxt(f'{i}/times_{T}_{i}.dat')
    new_time = np.zeros(len(times))
    dt = times[-1]/len(times)
    # print(np.max(centers, axis=0)*1e-10)
    for j in range(len(new_time)):
        new_time[j] = j*dt
    sp_pos = sp.load_npz(f'{i}/trajectories_{T}_{i}.npz')
    N = sp_pos.shape[1]
    D_self, D_jump = compute_msd(centers, sp_pos)
    D_jump = msd_fft(D_jump)
    np.savetxt(f'msd_{i}.dat', np.array([new_time, times, D_self, D_jump]))
    print(f'Done in {np.round(time.time()-t_init,3)}s.')

