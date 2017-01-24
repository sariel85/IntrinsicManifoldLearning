import numpy
from sklearn import preprocessing


def linear(intrinsic_process):
    assert intrinsic_process.shape[0] == 2
    observed_process = numpy.empty((3, intrinsic_process.shape[1]), dtype=numpy.float64)
    observed_process[0] = intrinsic_process[0]
    observed_process[1] = intrinsic_process[1]
    observed_process[2] = 0
    return observed_process


def parabola2d2d(intrinsic_process, k=2):
    assert intrinsic_process.shape[0] == 2
    scale_x = numpy.max(intrinsic_process[0]) - numpy.min(intrinsic_process[0])
    scale_y = numpy.max(intrinsic_process[1]) - numpy.min(intrinsic_process[1])
    scale = max(scale_x, scale_y)
    origin = numpy.mean(intrinsic_process, axis=1)
    intrinsic_process_temp = (intrinsic_process.T-origin.T).T/scale
    observed_process = numpy.empty((2, intrinsic_process.shape[1]), dtype=numpy.float64)
    observed_process[0, :] = intrinsic_process_temp[0, :]
    observed_process[1, :] = intrinsic_process_temp[1, :] - k * intrinsic_process_temp[0, :] ** 2
    return observed_process


def parabola2d3d(intrinsic_process, k=3):
    assert intrinsic_process.shape[0] == 2
    observed_process = numpy.empty((3, intrinsic_process.shape[1]), dtype=numpy.float64)
    intrinsic_process = intrinsic_process - 0.5
    observed_process[0, :] = intrinsic_process[0, :]
    observed_process[1, :] = intrinsic_process[1, :]
    observed_process[2, :] = k * numpy.sum(intrinsic_process ** 2, axis=0)
    return observed_process


def singers_mushroom(intrinsic_process):
    assert intrinsic_process.shape[0] == 2
    #scale_x = numpy.max(intrinsic_process[0]) - numpy.min(intrinsic_process[0])
    #scale_y = numpy.max(intrinsic_process[1]) - numpy.min(intrinsic_process[1])
    #scale = max(scale_x, scale_y)
    #origin = numpy.min(intrinsic_process, axis=1)
    #intrinsic_process_temp = (intrinsic_process.T-origin.T).T/scale
    intrinsic_process_temp = intrinsic_process
    observed_process = numpy.empty((2, intrinsic_process_temp.shape[1]), dtype=numpy.float64)
    observed_process[0] = intrinsic_process_temp[0]+numpy.power(intrinsic_process_temp[1], 3)
    observed_process[1] = intrinsic_process_temp[1]-numpy.power(intrinsic_process_temp[0], 3)
    return observed_process

def singers_sphere(intrinsic_process):
    assert intrinsic_process.shape[0] == 2
    intrinsic_process_temp = intrinsic_process
    observed_process = numpy.empty((3, intrinsic_process_temp.shape[1]), dtype=numpy.float64)
    radius = numpy.sqrt(intrinsic_process_temp[0]**2+intrinsic_process_temp[1]**2+1)
    observed_process[0] = intrinsic_process_temp[0]/radius
    observed_process[1] = intrinsic_process_temp[1]/radius
    observed_process[2] = 1/radius
    return observed_process


def whole_sphere(intrinsic_process):
    assert intrinsic_process.shape[0] == 2
    intrinsic_process_temp = intrinsic_process
    observed_process = numpy.empty((3, intrinsic_process_temp.shape[1]), dtype=numpy.float64)
    radius = numpy.sqrt(intrinsic_process_temp[0]**2+intrinsic_process_temp[1]**2)
    theta = numpy.arctan(intrinsic_process[1, :] / intrinsic_process[0, :])
    theta[numpy.where(intrinsic_process[0, :] < 0)] = theta[numpy.where(intrinsic_process[0, :] < 0)]+numpy.pi

    observed_process[0] = numpy.sin(4.7*radius)*numpy.sin(theta)
    observed_process[1] = numpy.sin(4.7*radius)*numpy.cos(theta)
    observed_process[2] = -numpy.cos(4.7*radius)
    return observed_process




def photo_dist(intrinsic_process, k=1.5):
    assert intrinsic_process.shape[0] == 2
    observed_process = numpy.empty((2, intrinsic_process.shape[1]), dtype=numpy.float64)
    intrinsic_process = intrinsic_process - 0.5
    r = numpy.sqrt(intrinsic_process[0, :] ** 2 + intrinsic_process[1, :] ** 2)
    observed_process[0, :] = intrinsic_process[0, :]*(1 + k * r ** 2)
    observed_process[1, :] = intrinsic_process[1, :]*(1 + k * r ** 2)
    observed_process = observed_process + 0.5
    return observed_process


def twirl(intrinsic_process, k=6):
    assert intrinsic_process.shape[0] == 2
    observed_process = numpy.empty((2, intrinsic_process.shape[1]), dtype=numpy.float64)
    temp_mean = numpy.mean(intrinsic_process, 1)
    intrinsic_process = (intrinsic_process.T - temp_mean.T).T
    r = numpy.sqrt(intrinsic_process[0, :]**2 + intrinsic_process[1, :]**2)
    theta = numpy.arctan(intrinsic_process[1, :] / intrinsic_process[0, :])
    theta[numpy.where(intrinsic_process[0, :] < 0)] = theta[intrinsic_process[0, :] < 0]+numpy.pi
    newr = r
    newtheta = theta + newr * k
    newtheta = -newtheta
    observed_process[0, :] = newr * numpy.cos(newtheta)
    observed_process[1, :] = newr * numpy.sin(newtheta)
    observed_process = (observed_process.T + temp_mean.T).T
    return observed_process


def bend(intrinsic_process, k=45):
    assert intrinsic_process.shape[0] == 2
    deg = 2*numpy.pi*(k/360)
    observed_process = numpy.empty((3, intrinsic_process.shape[1]), dtype=numpy.float64)
    for x in range(0, intrinsic_process.shape[1]):
        if intrinsic_process[0, x] < 0.5:
            observed_process[0, x] = intrinsic_process[0, x]
            observed_process[1, x] = intrinsic_process[1, x]
            observed_process[2, x] = 0
        else:
            observed_process[0, x] = 0.5 + numpy.cos(deg)*(intrinsic_process[0, x] - 0.5)
            observed_process[1, x] = intrinsic_process[1, x]
            observed_process[2, x] = numpy.sin(deg) * (intrinsic_process[0, x] - 0.5)
    return observed_process


def swissroll(intrinsic_process, k=8):
    assert intrinsic_process.shape[0] == 2
    observed_process = numpy.empty((3, intrinsic_process.shape[1]), dtype=numpy.float64)
    observed_process[0] = intrinsic_process[0] * numpy.cos(k * intrinsic_process[0])
    observed_process[1] = intrinsic_process[1]
    observed_process[2] = intrinsic_process[0] * numpy.sin(k * intrinsic_process[0])
    return observed_process


def tube(intrinsic_process, k=160):
    assert intrinsic_process.shape[0] == 2
    scale = numpy.max(intrinsic_process[0]) - numpy.min(intrinsic_process[0])
    radius = (360/k)/(2*numpy.pi)
    observed_process = numpy.empty((3, intrinsic_process.shape[1]), dtype=numpy.float64)
    observed_process[0] = radius * numpy.cos(2*numpy.pi*(k/360) * (intrinsic_process[0]/scale))
    observed_process[1] = intrinsic_process[1]
    observed_process[2] = radius * numpy.sin(2*numpy.pi*(k/360) * (intrinsic_process[0]/scale))
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    observed_process = min_max_scaler.fit_transform(observed_process.T).T
    return observed_process


def helix(intrinsic_process, k=2):
    assert intrinsic_process.shape[0] == 2
    observed_process = numpy.empty((3, intrinsic_process.shape[1]), dtype=numpy.float64)
    intrinsic_process = intrinsic_process - 0.5
    observed_process[0, :] = intrinsic_process[0, :]
    observed_process[1, :] = intrinsic_process[1, :] * numpy.cos(k * numpy.pi * (intrinsic_process[0, :]))
    observed_process[2, :] = intrinsic_process[1, :] * numpy.sin(k * numpy.pi * (intrinsic_process[0, :]))
    observed_process = observed_process + 0.5
    return observed_process

def papillon(intrinsic_process, k=8):
    assert intrinsic_process.shape[0] == 2
    observed_process = numpy.empty((2, intrinsic_process.shape[1]), dtype=numpy.float64)
    intrinsic_process = intrinsic_process - 0.5
    observed_process[0, :] = intrinsic_process[0, :] + 0.5
    observed_process[1, :] = intrinsic_process[1, :] + k * intrinsic_process[1, :] * intrinsic_process[0, :] ** 2 + 0.5
    return observed_process

def twist(intrinsic_process, k=6):

    assert intrinsic_process.shape[0] == 3
    intrinsic_process = intrinsic_process - 0.5
    r = numpy.sqrt(intrinsic_process[0, :]**2 + intrinsic_process[1, :]**2)
    theta = numpy.arctan(intrinsic_process[1, :] / intrinsic_process[0, :])
    theta[numpy.where(intrinsic_process[0, :] < 0)] = theta[numpy.where(intrinsic_process[0, :] < 0)]+numpy.pi
    observed_process = numpy.empty([3, intrinsic_process.shape[1]])
    observed_process[0, :] = r*numpy.cos(theta + intrinsic_process[2, :]*k)
    observed_process[1, :] = r*numpy.sin(theta + intrinsic_process[2, :]*k)
    observed_process[2, :] = intrinsic_process[2, :]
    observed_process = observed_process + 0.5
    return observed_process

def antena(intrinsic_process, centers, amplitudes, range_factor):

    n_antenas = centers.shape[1]

    observed_process = numpy.empty([n_antenas, intrinsic_process.shape[1]])

    assert intrinsic_process.shape[0] == centers.shape[0]

    for i_antena in range(0, n_antenas):
        dists = (intrinsic_process.T - centers[:, i_antena].T).T
        dists = dists * dists
        dists = numpy.sqrt(numpy.sum(dists, axis=0))
        observed_process[i_antena, :] = amplitudes[i_antena]*(1/(range_factor[i_antena]+dists))
        #observed_process[i_antena, :] = dists
    return 10*observed_process


