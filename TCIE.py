import numpy
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3


################################################################################
# generate the swiss roll

n_samples = 2000
n_turns, radius = 1.2, 1.0
rng = numpy.random.RandomState(0)
data_int = numpy.zeros((n_samples, 2))
data_obs = numpy.zeros((n_samples, 3))

n_samples_gen=0
while n_samples_gen < n_samples:
    ok=False
    test_sample = rng.uniform(low=-0.5, high=0.5, size=2)
    if abs(test_sample[0])>0.25 or abs(test_sample[1])>0.25:
        data_int[n_samples_gen, :] = test_sample
        n_samples_gen = n_samples_gen + 1

# generate the 2D spiral data driven by a 1d parameter t
max_rot = n_turns * 2 * numpy.pi
data_obs[:, 0]  = ((data_int[:, 0]-0.5) + radius)* numpy.cos((data_int[:, 0]-0.5) * max_rot)
data_obs[:, 1]  = ((data_int[:, 0]-0.5) + radius) * numpy.sin((data_int[:, 0]-0.5) * max_rot)
data_obs[:, 2] = data_int[:, 1]
#manifold = np.vstack((t * 2 - 1, data[:, 2])).T.copy()
colors = data_int[:, 0]


fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
ax.scatter(data_int[:, 0], data_int[:, 1], c=colors)
ax.set_title("Intrinsic Manifold")

# rotate and plot original data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d',  aspect='equal')
ax.scatter(data_obs[:, 0], data_obs[:, 1], data_obs[:, 2], c=colors)
ax.set_title("Observed Manifold")

plt.show(block=True)
