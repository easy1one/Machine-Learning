import numpy as np

class KMeans():
    '''
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Number of iterations: number of time I update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # Initialize means by picking self.n_cluster from N data points
        means = np.zeros((self.n_cluster, D)) 
        membership = np.zeros(N) 
        number_of_updates = 0
        J_sum_prev = float('inf')

        # Get random means
        means_idx = np.random.choice(N, size=self.n_cluster, replace=False)
        for i,m_idx in enumerate(means_idx):
            means[i] = x[m_idx]

        for _ in range(self.max_iter):
            number_of_updates += 1

            J = 0
            means_sum = np.zeros((self.n_cluster, D))
            means_cnt = np.zeros(self.n_cluster)

            for x_idx in range(N):

                sub = x[x_idx] - means
                sub_mul = np.multiply(sub,sub)
                dis = np.sum(sub_mul, axis=1)
                my_class = int(np.argmin(dis))
                membership[x_idx] = my_class
                means_cnt[my_class] += 1
                means_sum[my_class] += x[x_idx]
                J += dis[my_class]

            # Check conversion
            J /= N
            if abs(J-J_sum_prev) <= self.e:
                break
            J_sum_prev = J

            # Update means
            for m in range(self.n_cluster):
                means[m] = means_sum[m] / means_cnt[m]

        res = (means, membership, number_of_updates)
        return res


class KMeansClassifier():
    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''
    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape

        k_means = KMeans(self.n_cluster, self.max_iter, self.e)
        centroids, membership, i = k_means.fit(x)

        centroid_labels = np.zeros(self.n_cluster)
        vote = np.zeros((self.n_cluster, self.n_cluster))

        for i in range(N):
            mem_class = int(membership[i])
            y_class = int(y[i])
            vote[mem_class][y_class] += 1

        for v_idx, v_row in enumerate(vote):
            centroid_labels[v_idx] = np.argmax(v_row)

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape

        predicted_labels = np.zeros(N)

        for x_idx in range(N):
            sub = x[x_idx] - self.centroids
            sub_mul = np.multiply(sub,sub)
            dis = np.sum(sub_mul, axis=1)
            predicted_labels[x_idx] = self.centroid_labels[int(np.argmin(dis))]

        return predicted_labels