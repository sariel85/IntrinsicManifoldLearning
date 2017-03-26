from scipy.io import loadmat
import h5py
import numpy
import cv2
from cv2 import *
from sklearn.decomposition.pca import PCA
import matplotlib.pyplot as plt
from sklearn import preprocessing
import cv2
from data_generation import print_process, create_color_map

sim_dir_name = "2D Apartment - Static - Depth"
sim_dir = './' + sim_dir_name
video_file = sim_dir + '/' + 'video_input.avi'

cap = cv2.VideoCapture(video_file)

n_frames = numpy.int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_hight = numpy.int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = numpy.int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
n_pixels = numpy.int(frame_hight * frame_width)

movie_frames = None

while not cap.isOpened():
    cap = cv2.VideoCapture(video_file)
    cv2.waitKey(1000)
    print("Wait for the header")

pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

smoothing_kernel_size = 17
kernel = numpy.ones((smoothing_kernel_size, smoothing_kernel_size), numpy.float32)/(smoothing_kernel_size * smoothing_kernel_size)

while True:
    flag, frame = cap.read()
    if flag:
        # The frame is ready and already captured
        frame_3 = numpy.concatenate((frame, frame, frame), axis=1)
        dst = cv2.GaussianBlur(frame_3, (smoothing_kernel_size, smoothing_kernel_size), sigmaX=2, sigmaY=2, borderType=BORDER_REPLICATE)
        dst = dst[:, frame_width:2*frame_width, :]
        dst = dst[::2, :, :][:, ::2, :]
        #plt.subplot(121), plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), interpolation='none'), plt.title('Original')
        #plt.subplot(122), plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB), interpolation='none'), plt.title('Averaging')
        #plt.show(block=False)

        if movie_frames is None:
            movie_frames = numpy.empty([n_frames, dst.shape[0]*dst.shape[1]*dst.shape[2]])
        else:
            movie_frames[pos_frame, :] = numpy.asarray(dst).reshape([dst.shape[0]*dst.shape[1]*dst.shape[2]])
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print(str(pos_frame) + " frames")
    else:
        # The next frame is not ready, so we try to read it again
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
        print("frame is not ready")
        # It is better to wait for a while for the next frame to be ready
        cv2.waitKey(1000)

    if cv2.waitKey(10) == 27:
        break
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == n_frames:
        # If the number of captured frames is equal to the total number of frames,
        # we stop
        cv2.CAP_PROP_POS_FRAMES
        break


#numpy.savetxt(sim_dir + '/' + 'movie_mat.txt', movie_frames, delimiter=',')
#movie_frames = numpy.loadtxt(sim_dir + '/' + 'movie_mat.txt', delimiter=',')

n_pca = 4000
pca = PCA(n_components=8, whiten=False)
pca.fit(movie_frames[0:n_pca, :]-numpy.mean(movie_frames[0:n_pca, :], 0))
pca_base = pca.components_
explained_variance = pca.explained_variance_

plt.plot(explained_variance)
plt.show(block=False)

numpy.savetxt(sim_dir + '/' + 'pca_vects.txt', pca_base, delimiter=',')
pca_base = numpy.loadtxt(sim_dir + '/' + 'pca_vects.txt', delimiter=',')

movie_pca = numpy.dot(numpy.linalg.pinv(pca_base.T), (movie_frames-numpy.mean(movie_frames.T, 1)).T)

intrinsic_process = numpy.loadtxt(sim_dir + '/' + 'intrinsic_process_to_measure.txt', delimiter=',').T
color_map = create_color_map(intrinsic_process)
print_process(movie_pca, bounding_shape=None, color_map=color_map, titleStr="Feature Space")
plt.show(block=False)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
sensor_noisy = min_max_scaler.fit_transform(movie_pca.T)

numpy.savetxt(sim_dir + '/' + 'sensor_noisy.txt', sensor_noisy, delimiter=',')