from __future__ import print_function
from __future__ import absolute_import
from __future__ import print_function
from __future__ import absolute_import
from DataGeneration import print_process, create_color_map
import matplotlib.pyplot as plt
import numpy, numpy.random
from DataGeneration import print_process, create_color_map, print_dynamics
from Util import *

n_points = 1000
x_width = 1
y_width = 4
hole_start_x = 0.25
hole_start_y = 1.5
hole_width_x = 0.5
hole_width_y = 2

tube_radius = 0.8

noise_std = 0.05

n_neighbors_isomaps = 10

#Generate Data Set
intrinsic_points = numpy.zeros((2, n_points))
intrinsic_points = intrinsic_points.T
for i_point in range(n_points):
    while 1:
        test_point = numpy.asarray([numpy.random.rand()*x_width, numpy.random.rand()*y_width])
        if not((test_point[0] > hole_start_x) and (test_point[0] < (hole_start_x + hole_width_x)) and (test_point[1] > hole_start_y) and (test_point[1] < hole_start_y + hole_width_y)):
            break
    intrinsic_points[i_point, :] = test_point
intrinsic_points = intrinsic_points.T

#Display Intrinsic Data Set
color_map = create_color_map(intrinsic_points)
print_process(intrinsic_points, color_map=color_map, titleStr="Intrinsic Space")

#Isometric Deformed Manifold
observed_points_clean = numpy.zeros((3, n_points))
observed_points_clean[0, :] = intrinsic_points[0, :]
observed_points_clean[1, :] = tube_radius*numpy.cos(intrinsic_points[1, :]/tube_radius)
observed_points_clean[2, :] = tube_radius*numpy.sin(intrinsic_points[1, :]/tube_radius)

print_process(observed_points_clean, color_map=color_map, titleStr="Observed Space - Clean Data")

observed_points = observed_points_clean + noise_std*numpy.random.randn(observed_points_clean.shape[0], observed_points_clean.shape[1])

print_process(observed_points, color_map=color_map, titleStr="Observed Space")

dist_mat_ground_truth = numpy.sqrt(calc_dist(intrinsic_points))
dist_mat_observed_clean = numpy.sqrt(calc_dist(observed_points_clean))
dist_mat_observed = numpy.sqrt(calc_dist(observed_points))

dist_mat_ground_truth_trimmed = trim_distances(dist_mat_ground_truth, n_neighbors=n_neighbors_isomaps)
dist_mat_observed_clean_trimmed = trim_distances(dist_mat_observed_clean, n_neighbors=n_neighbors_isomaps)
dist_mat_observed_trimmed = trim_distances(dist_mat_observed, n_neighbors=n_neighbors_isomaps)

dist_mat_ground_truth_geo_fill = scipy.sparse.csgraph.shortest_path(dist_mat_ground_truth_trimmed, directed=False)
dist_mat_observed_clean_geo_fill = scipy.sparse.csgraph.shortest_path(dist_mat_observed_clean_trimmed, directed=False)
dist_mat_observed_geo_fill = scipy.sparse.csgraph.shortest_path(dist_mat_observed_trimmed, directed=False)

mds = manifold.MDS(n_components=2, max_iter=100, eps=1e-5, dissimilarity="precomputed", n_jobs=1, n_init=1)

iso_embedding_observed_clean = mds.fit(dist_mat_observed_clean_geo_fill).embedding_.T
iso_embedding_observed = mds.fit(dist_mat_observed_geo_fill).embedding_.T

print_process(iso_embedding_observed_clean, bounding_shape=None, color_map=color_map, titleStr="Isomap with Locally Learned Intrinsic Metric", align_points=intrinsic_points)
stress, stress_normlized = embbeding_score(intrinsic_points, iso_embedding_observed_clean, titleStr="Isomap with Locally Learned Intrinsic Metric", n_points=100)
print('iso_embedding_observed_clean:', stress_normlized)

print_process(iso_embedding_observed, bounding_shape=None, color_map=color_map, titleStr="Isomap with Locally Learned Intrinsic Metric", align_points=intrinsic_points)
stress, stress_normlized = embbeding_score(intrinsic_points, iso_embedding_observed, titleStr="Isomap with Locally Learned Intrinsic Metric", n_points=100)
print('iso_embedding_observed:', stress_normlized)

plt.show(block=True)



