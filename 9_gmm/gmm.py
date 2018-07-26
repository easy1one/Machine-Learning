import numpy as np
from kmeans import KMeans


class GMM():
    '''
        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):

            # 1. initialize means using k-means clustering
            get_k_means = KMeans(self.n_cluster, self.max_iter, self.e)
            centroids, membership, i = get_k_means.fit(x)

            self.means = centroids
            self.variances = np.zeros((self.n_cluster, D, D))
            self.pi_k = np.zeros(self.n_cluster)
            means_cnt = np.zeros(self.n_cluster)

            # 2. compute variance and pi_k
            for x_idx, x_row in enumerate(x):
                my_class = int(membership[x_idx])
                means_cnt[my_class] += 1
                val = np.matrix(x_row - self.means[my_class])
                self.variances[my_class] += np.dot(val.T, val)

            for m_idx in range(self.n_cluster):
                self.variances[m_idx] /= means_cnt[m_idx]
            self.pi_k = means_cnt/N

        elif (self.init == 'random'):

            self.means = np.zeros((self.n_cluster, D))
            for i in range(self.n_cluster):
                for j in range(D):
                    self.means[i][j] = np.random.rand()

            self.variances = np.zeros((self.n_cluster, D, D))
            self.pi_k = np.zeros(self.n_cluster)

            for v_idx in range(self.n_cluster):
                self.variances[v_idx] = np.identity(D)
            self.pi_k = np.ones(self.n_cluster)/self.n_cluster

        else:
            raise Exception('Invalid initialization provided')
        
        # 0. Initialize Loglikelihood
        numOfItr = 0
        log = self.compute_log_likelihood(x)
        cur_means = self.means
        cur_vari = self.variances
        cur_pi_k = self.pi_k

        for _ in range(self.max_iter): #self.max_iter
            numOfItr += 1

            # 1. E Step: Compute responsibilities
            gam = np.zeros((N, self.n_cluster))
            gam_x_sum = np.zeros((self.n_cluster, D))
            tmp_var = np.zeros((self.n_cluster, D, D))

            for x_idx, x_row in enumerate(x):
                gam_sum = 0
                for k_idx, k_val in enumerate(cur_pi_k):
                    tmp_norm = np.exp((-1/2)*np.matrix(x[x_idx]-self.means[k_idx])*np.linalg.inv(self.variances[k_idx]+0.001*np.identity(D))*np.matrix(x[x_idx]-self.means[k_idx]).T).tolist()[0][0]
                    normal = (1 / np.sqrt(np.power((2*np.pi),D)*np.linalg.det(self.variances[k_idx]+0.001*np.identity(D)))) * tmp_norm

                    gam[x_idx][k_idx] = k_val * normal
                    gam_sum += gam[x_idx][k_idx]
                gam[x_idx] /= gam_sum

                for k_idx in range(self.n_cluster):
                    gam_x_sum[k_idx] += x_row * gam[x_idx][k_idx]
                    val = np.matrix(x_row - cur_means[k_idx])
                    tmp_var[k_idx] += gam[x_idx][k_idx] * np.dot(val.T, val)

            # 2. M Step:
            # 2-1. Estimate means
            N_k = gam.sum(axis=0)
            for c_idx in range(self.n_cluster):
                cur_means[c_idx] = gam_x_sum[c_idx]/N_k[c_idx]
                cur_vari[c_idx] = tmp_var[c_idx]/N_k[c_idx] 
            cur_pi_k = N_k/N

            self.means = cur_means
            self.variances = cur_vari
            self.pi_k = cur_pi_k

            log_new = self.compute_log_likelihood(x)
            if(abs(log_new - log) <= self.e):
                break
            log = log_new

        return numOfItr


    def sample(self, N):

        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        R, D = self.means.shape
        x_sample = np.zeros((N, D))

        for n_idx in range(N):
            cur_class = int(np.argmax(np.random.multinomial(1, self.pi_k)))
            row_sample = np.random.multivariate_normal(self.means[cur_class], self.variances[cur_class])
            x_sample[n_idx] = row_sample

        return x_sample



    def compute_log_likelihood(self, x):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'

        N, D = x.shape

        log_likelihood = 0
        for i in range(N):
            prob = 0
            for j in range(self.n_cluster):
                tmp_norm = np.exp((-1/2)*np.matrix(x[i]-self.means[j])*np.linalg.inv(self.variances[j]+0.001*np.identity(D))*np.matrix(x[i]-self.means[j]).T).tolist()[0][0]
                normal = (1 / np.sqrt(np.power((2*np.pi),D)*np.linalg.det(self.variances[j]+0.001*np.identity(D))))*tmp_norm
                prob += self.pi_k[j]*normal
            log_likelihood += np.log(prob)

        return float(log_likelihood)





