#____________________________________________________________________________________#
#                                                                                    #
#      Welcome, traveller. At the time I coded this, I knew what I was doing,        #
#      now only god knows. Feel free to ask him for any recommendations.             #
#                                                                                    #
#       Time spent making it work: 213 hours (actually my top played game).          #
#                                                                                    #
#____________________________________________________________________________________#


from engine_backup import *
from parameters import *
import scipy.sparse as sp
import gc

# tracemalloc.start()

def run():
    def update(*a):
        pbar.update()

    if __name__ == '__main__':
        densities, temperatures = get_params()
        tinit = time.time()
        if MP:
            pbar = tqdm(total=len(densities))
            pool = mp.Pool()
            results = []
            for i in range(len(densities)):
                result = pool.apply_async(mp_run, args=([densities[i], temperatures[i]]), callback=update)
                results.append([result,densities[i]])

            pool.close()
            pool.join()
            pbar.close()
            gc.collect()
            occupancies = [result[0].get()[0] for result in results]
            probe_stats = [result[0].get()[1] for result in results]
            times = [result[0].get()[2] for result in results]
            traj = [result[0].get()[3] for result in results]
            probe_stats = np.concatenate(probe_stats, axis=0)
            occupancies = np.concatenate(occupancies, axis=0)
            np.savetxt(f'occupancies_{T}.dat', occupancies)
            np.savetxt(f'probe_{T}.dat', probe_stats)
            np.savetxt(f'times_{T}.dat', times)
            np.savez(f'trajectories_{T}', traj)

        else:
            densities = [0]
            data = mp_run(occupancy_density, T)
            # sp.save_npz(f'{occupation_density}/trajectories_{T}_{occupation_density}', data[3])
            # np.savetxt(f'{occupation_density}/times_{T}_{occupation_density}.dat', data[2])
            # np.savetxt(f'{occupation_density}/prob_{T}_{occupation_density}.dat',data[1])
            # np.savetxt(f'{occupation_density}/occupancies_{T}_{occupation_density}.dat', data[0])
            # np.savetxt(f'{occupation_density}/events_{T}_{occupation_density}.dat', data[4])
            sp.save_npz(f'sI_SO_{T}/trajectories', data[3])
            np.savetxt(f'sI_SO_{T}/times.dat', data[2])
            np.savetxt(f'sI_SO_{T}/probe.dat',data[1])
            np.savetxt(f'sI_SO_{T}/occupancies.dat', data[0])
            np.savetxt(f'sI_SO_{T}/events.dat', data[4])

        final_time = np.round(time.time()-tinit,1)
        return print(f'Computational time of {final_time}s for {iterations*len(densities)} iterations.')

# cProfile.run('run()')
run()
# snapshot = tracemalloc.take_snapshot()
# top_stats = snapshot.statistics('lineno')

# print("[ Top 10 ]")
# for stat in top_stats[:10]:
#     print(stat)