from scipy.io import loadmat
import h5py
import numpy
import cv2
from cv2 import *
from sklearn.decomposition.pca import PCA
import matplotlib.pyplot as plt
from sklearn import preprocessing
import cv2

sim_dir_name = "2D Non Convex"
sim_dir = './' + sim_dir_name
video_file = sim_dir + '/' + 'new_vid.avi'

cap = cv2.VideoCapture(video_file)

n_frames = numpy.int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_hight = numpy.int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = numpy.int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
n_pixels = numpy.int(3 * frame_hight * frame_width)

movie_frames = numpy.empty([n_frames, 3*frame_hight*frame_width])

while not cap.isOpened():
    cap = cv2.VideoCapture(video_file)
    cv2.waitKey(1000)
    print("Wait for the header")

pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

kernel = numpy.ones((2,2),numpy.float32)/(2*2)

while True:
    flag, frame = cap.read()
    if flag:
        # The frame is ready and already captured
        #cv2.imshow('video', frame)

        dst = cv2.filter2D(numpy.asarray(frame[:, :, :]), -1, kernel)
        #dst = dst[::4, :, :][:, ::4, :]
        #plt.subplot(121), plt.imshow(numpy.asarray(frame[:, :, :])), plt.title('Original')
        #plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
        #plt.show(block=False)

        movie_frames[pos_frame, :] = numpy.asarray(dst).reshape([n_pixels])
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print(str(pos_frame)+" frames")
    else:
        # The next frame is not ready, so we try to read it again
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
        print ("frame is not ready")
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

n_pca = 7000
pca = PCA(n_components=3, whiten=False)
pca.fit(movie_frames[0:n_pca, :])
pca_base = pca.components_
explained_variance = pca.explained_variance_

plt.plot(explained_variance)
plt.show(block=False)

numpy.savetxt(sim_dir + '/' + 'pca_vects.txt', pca_base, delimiter=',')
pca_base = numpy.loadtxt(sim_dir + '/' + 'pca_vects.txt', delimiter=',')

movie_pca = numpy.dot(numpy.linalg.pinv(pca_base.T), (movie_frames-numpy.mean(movie_frames.T, 1)).T)

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
movie_pca = min_max_scaler.fit_transform(movie_pca.T).T


sensor_noisy = movie_pca[:, numpy.arange(0, n_frames, 2)]
numpy.savetxt(sim_dir + '/' + 'sensor_noisy.txt', sensor_noisy, delimiter=',')


