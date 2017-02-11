import numpy
import matplotlib.pyplot as plt
import scipy
from sklearn import manifold

def multiscale_isomaps(dist_mat, dist_mat_trimmed, intrinsic_points, dim_intrinsic=2):

    n_points = dist_mat.shape[0]

    n_clusters = 5
    sigma = 0.2
    t = 10

    dist_mat_intrinsic = numpy.copy(dist_mat)

    dist_mat_rbf = numpy.exp(-(dist_mat**2)/(2*sigma**2))
    row_sum = numpy.sum(dist_mat_rbf, axis=1)
    col_sum = numpy.sum(dist_mat_rbf, axis=0)
    transition_prob = numpy.dot(dist_mat_rbf, numpy.diag(1/row_sum))
    transition_prob = numpy.dot(numpy.diag(1/col_sum), transition_prob)
    row_sum = numpy.sum(transition_prob, axis=1)
    transition_prob = numpy.dot(numpy.diag(1/row_sum), transition_prob)
    U, S, V = numpy.linalg.svd(transition_prob)
    diff_embedding = numpy.dot(U[:, 1:10], numpy.sqrt(numpy.diag(S[1:10]**t)))
    diff_embedding = diff_embedding.T
    (mu, clusters) = find_centers(diff_embedding, n_clusters)

    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(intrinsic_points[0, :], intrinsic_points[1, :], c="k")
    for i_point in range(mu.shape[0]):
        ax.scatter(intrinsic_points[0, clusters[i_point]], intrinsic_points[1, clusters[i_point]], color=numpy.random.rand(3,1))
        ax.scatter(intrinsic_points[0, mu[i_point]], intrinsic_points[1, mu[i_point]], c='r')
    plt.axis('equal')
    plt.show(block=False)

    for i_point in range(mu.shape[0]):

        clusters_ind = clusters[i_point]

        knn_indexes = numpy.argsort(dist_mat[mu[i_point], clusters_ind], kind='quicksort')

        n_neighbors_start = clusters_ind.__len__()
        n_neighbors_step = 50

        n_neighbors = n_neighbors_start

        rank_check_list = []
        n_neighbors_list = []

        while n_neighbors <= clusters_ind.__len__():

            dist_mat_sub = dist_mat[clusters_ind, :][:, clusters_ind]
            dist_mat_trimmed_sub = dist_mat_trimmed[clusters_ind, :][:, clusters_ind]

            dist_mat_sub_squared = dist_mat_sub ** 2

            # centering matrix
            J_c = 1. / n_neighbors * (numpy.eye(n_neighbors) - 1 + (n_neighbors - 1) * numpy.eye(n_neighbors))

            # perform double centering
            B = -0.5 * (J_c.dot(dist_mat_sub_squared)).dot(J_c)

            # find eigenvalues and eigenvectors
            U, eigen_val, V = numpy.linalg.svd(B)
            eigen_vect = V
            eigen_val_sort_ind = numpy.argsort(-numpy.abs(eigen_val))
            eigen_val = numpy.abs(eigen_val[eigen_val_sort_ind])

            eigen_vect = eigen_vect[:, eigen_val_sort_ind]
            eigen_vect = eigen_vect[:dim_intrinsic].T
            eigen_vect = numpy.dot(eigen_vect, numpy.diag(numpy.sqrt(eigen_val[:dim_intrinsic]))).T

            mds = manifold.MDS(n_components=2, max_iter=2000, eps=1e-5, dissimilarity="precomputed", n_jobs=1, n_init=1)
            iso_embedding = mds.fit(dist_mat_sub, init=eigen_vect.T).embedding_.T

            mds = manifold.MDS(n_components=2, max_iter=2000, eps=1e-5, dissimilarity="precomputed", n_jobs=1, n_init=1)
            iso_embedding_no_geo = mds.fit(dist_mat_trimmed_sub, init=eigen_vect.T, weight=(dist_mat_trimmed_sub!=0)).embedding_.T

            expl = numpy.sum(eigen_val[:dim_intrinsic])
            res = numpy.sum(eigen_val[dim_intrinsic:])
            # dis = (D_sub_trust_original*wgt).sum()
            # check = (stress2/dis)
            rank_check_list.append(dim_intrinsic * eigen_val[dim_intrinsic] / expl)

            n_neighbors_list.append(n_neighbors)

            fig = plt.figure()
            ax = fig.gca()
            ax.scatter(eigen_vect[0, :], eigen_vect[1, :], c="k")
            ax.scatter(eigen_vect[0, :], eigen_vect[1, :], c='r')
            ax.scatter(eigen_vect[0, 0], eigen_vect[1, 0], c='g')
            plt.axis('equal')

            fig = plt.figure()
            ax = fig.gca()
            ax.scatter(iso_embedding[0, :], iso_embedding[1, :], c="k")
            ax.scatter(iso_embedding[0, :], iso_embedding[1, :], c='r')
            ax.scatter(iso_embedding[0, 0], iso_embedding[1, 0], c='g')
            plt.axis('equal')

            fig = plt.figure()
            ax = fig.gca()
            ax.scatter(iso_embedding_no_geo[0, :], iso_embedding_no_geo[1, :], c="k")
            ax.scatter(iso_embedding_no_geo[0, :], iso_embedding_no_geo[1, :], c='r')
            ax.scatter(iso_embedding_no_geo[0, 0], iso_embedding_no_geo[1, 0], c='g')
            plt.axis('equal')

            fig = plt.figure()
            ax = fig.gca()
            ax.scatter(intrinsic_points[0, :], intrinsic_points[1, :], c="k")
            for j_point in clusters_ind:
                ax.scatter(intrinsic_points[0, j_point], intrinsic_points[1, j_point], c='r')
            ax.scatter(intrinsic_points[0, mu[i_point]],
                       intrinsic_points[1, mu[i_point]], c='g')
            plt.axis('equal')

            plt.figure()
            plt.plot(numpy.asarray(n_neighbors_list), numpy.asarray(rank_check_list))

            n_neighbors = n_neighbors + n_neighbors_step

            dist_new_sub = scipy.spatial.distance.cdist(iso_embedding_no_geo.T, iso_embedding_no_geo.T, metric='euclidean', p=2, V=None, VI=None, w=None)
            dist_mat_intrinsic[clusters_ind, :][:, clusters_ind] = dist_new_sub[:, :]

    dist_mat_sub_squared = dist_mat_intrinsic ** 2

    n_neighbors = dist_mat_sub_squared.shape[0
    ]
    # centering matrix
    J_c = 1. / n_neighbors * (numpy.eye(n_neighbors) - 1 + (n_neighbors - 1) * numpy.eye(n_neighbors))

    # perform double centering
    B = -0.5 * (J_c.dot(dist_mat_sub_squared)).dot(J_c)

    # find eigenvalues and eigenvectors
    U, eigen_val, V = numpy.linalg.svd(B)
    eigen_vect = V
    eigen_val_sort_ind = numpy.argsort(-numpy.abs(eigen_val))
    eigen_val = numpy.abs(eigen_val[eigen_val_sort_ind])

    eigen_vect = eigen_vect[:, eigen_val_sort_ind]
    eigen_vect = eigen_vect[:dim_intrinsic].T
    eigen_vect = numpy.dot(eigen_vect, numpy.diag(numpy.sqrt(eigen_val[:dim_intrinsic]))).T

    mds = manifold.MDS(n_components=2, max_iter=2000, eps=1e-5, dissimilarity="precomputed", n_jobs=1, n_init=1)
    iso_embedding_no_geo = mds.fit(dist_mat_trimmed, init=eigen_vect.T,
                                   weight=(dist_mat_trimmed != 0)).embedding_.T

    expl = numpy.sum(eigen_val[:dim_intrinsic])
    res = numpy.sum(eigen_val[dim_intrinsic:])
    # dis = (D_sub_trust_original*wgt).sum()
    # check = (stress2/dis)
    rank_check_list.append(dim_intrinsic * eigen_val[dim_intrinsic] / expl)

    n_neighbors_list.append(n_neighbors)

    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(eigen_vect[0, :], eigen_vect[1, :], c="k")
    ax.scatter(eigen_vect[0, :], eigen_vect[1, :], c='r')
    ax.scatter(eigen_vect[0, 0], eigen_vect[1, 0], c='g')
    plt.axis('equal')

    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(iso_embedding[0, :], iso_embedding[1, :], c="k")
    ax.scatter(iso_embedding[0, :], iso_embedding[1, :], c='r')
    ax.scatter(iso_embedding[0, 0], iso_embedding[1, 0], c='g')
    plt.axis('equal')

    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(iso_embedding_no_geo[0, :], iso_embedding_no_geo[1, :], c="k")
    ax.scatter(iso_embedding_no_geo[0, :], iso_embedding_no_geo[1, :], c='r')
    ax.scatter(iso_embedding_no_geo[0, 0], iso_embedding_no_geo[1, 0], c='g')
    plt.axis('equal')

    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(intrinsic_points[0, :], intrinsic_points[1, :], c="k")
    for j_point in clusters_ind:
        ax.scatter(intrinsic_points[0, j_point], intrinsic_points[1, j_point], c='r')
    ax.scatter(intrinsic_points[0, mu[i_point]],
               intrinsic_points[1, mu[i_point]], c='g')
    plt.axis('equal')

    plt.show(block=False)

def cluster_points(X, mu_ind):
    cluster_sizes = numpy.zeros(mu_ind.shape)
    clusters = {}
    dist = scipy.spatial.distance.cdist(X, X[mu_ind, :], metric='euclidean', p=2, V=None, VI=None, w=None)
    knn_indexes = numpy.argsort(dist, kind='quicksort')

    for i_x in range(X.shape[0]):
        bestmukey = knn_indexes[i_x, 0]

        try:
            clusters[bestmukey].append(i_x)
        except KeyError:
            clusters[bestmukey] = [i_x]

        cluster_sizes[bestmukey] = cluster_sizes[bestmukey] + 1

    return clusters, cluster_sizes

def reevaluate_centers(mu_ind, clusters_ind, X):
    center = numpy.ones((mu_ind.shape[0], X.shape[1]))
    for i_center in range(mu_ind.size):
        center[i_center, :] = numpy.mean(X[clusters_ind[i_center], :], axis=0)

    dist = scipy.spatial.distance.cdist(center, X, metric='euclidean', p=2, V=None, VI=None, w=None)
    knn_indexes = numpy.argsort(dist, kind='quicksort')

    return knn_indexes[:, 0]

def has_converged(mu_ind, oldmu_ind, X):
    oldmu = X[oldmu_ind, :]
    mu = X[mu_ind, :]
    return (set(mu_ind) == set(oldmu_ind))

def find_centers(X, K_Final):
    K = K_Final*4

    X=X.T
    # Initialize to K random centers

    mu_ind = numpy.random.choice(X.shape[0], size=K, replace=False)

    while mu_ind.shape[0]>K_Final:

        oldmu_ind = numpy.random.choice(X.shape[0], size=mu_ind.shape[0], replace=False)

        while not has_converged(mu_ind, oldmu_ind, X):
            oldmu_ind = mu_ind
            # Assign all points in X to clusters
            clusters, clusters_sizes = cluster_points(X, mu_ind)
            # Reevaluate centers
            mu_ind = reevaluate_centers(mu_ind, clusters, X)

        clusters, clusters_sizes = cluster_points(X, mu_ind)
        cluster_to_remove = numpy.argmin(clusters_sizes)
        mu_ind = mu_ind[numpy.where(numpy.arange(mu_ind.shape[0])!=cluster_to_remove)]

        clusters, clusters_sizes = cluster_points(X, mu_ind)
        mu_ind = reevaluate_centers(mu_ind, clusters, X)

        '''
        X = X.T
        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(X[0, :], X[1, :], c="k")
        for i_point in range(mu_ind.shape[0]):
            ax.scatter(X[0, clusters[i_point]], X[1, clusters[i_point]],
                       color=numpy.random.rand(3, 1))
            ax.scatter(X[0, mu_ind[i_point]], X[1, mu_ind[i_point]], c='r')
        plt.axis('equal')
        plt.show(block=False)
        X = X.T
        '''


    oldmu_ind = numpy.random.choice(X.shape[0], size=mu_ind.shape[0], replace=False)

    while not has_converged(mu_ind, oldmu_ind, X):
        oldmu_ind = mu_ind
        # Assign all points in X to clusters
        clusters, clusters_sizes = cluster_points(X, mu_ind)
        # Reevaluate centers
        mu_ind = reevaluate_centers(mu_ind, clusters, X)

    '''
    X = X.T
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(X[0, :], X[1, :], c="k")
    for i_point in range(mu_ind.shape[0]):
        ax.scatter(X[0, clusters[i_point]], X[1, clusters[i_point]],
                   color=numpy.random.rand(3, 1))
        ax.scatter(X[0, mu_ind[i_point]], X[1, mu_ind[i_point]], c='r')
    plt.axis('equal')
    plt.show(block=False)
    X = X.T
    '''

    X = X.T
    return mu_ind, clusters

def init_board(N):
    X = numpy.array([(numpy.random.uniform(-1, 1), numpy.uniform(-1, 1)) for i in range(N)])
    return X