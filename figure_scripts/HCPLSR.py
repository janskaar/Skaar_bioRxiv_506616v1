import numpy as np
from sklearn.cross_decomposition import PLSRegression
from fcmeans import FCM

class PLSR():
    def __init__(self, X, Y, n_components):
        self.X = X
        self.Y = Y
        self.n_components = n_components

        self.X_m, self.X_std = self.moments(X)
        self.X_exp = self.polyn_expansion(self.X)
        self.X_exp_m, self.X_exp_std = self.moments(self.X_exp)
        self.X_exp = self.normalize(self.X_exp, 'X')

        self.Y_m, self.Y_std = self.moments(self.Y)
        self.Y = (self.Y - self.Y_m) / self.Y_std

        self.setup()

    def polyn_expansion(self, X):
        dims = X.shape[1]
        X = X - self.X_m
        return np.concatenate([X] + [np.expand_dims(X[:,i]*X[:,j], 1) for i in range(dims) for j in range(i, dims)], axis=1)

    def moments(self, X):
        return X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True)

    def normalize(self, X, data='X'):
        if data.lower() == 'x':
            return (X - self.X_exp_m) / self.X_exp_std
        elif data.lower() == 'y':
            return (X - self.Y_m) / self.Y_std

    def unnormalize(self, X, data='X'):
        if data.lower() == 'x':
            return X * self.X_exp_std + self.X_exp_m
        elif data.lower() == 'y':
            return X * self.Y_std + self.Y_m

    def setup(self):
        self.PLS = PLSRegression(n_components=self.n_components, scale=True)
        self.PLS.fit(self.X_exp, self.Y)

    def predict(self, X):
        X_exp = self.normalize(self.polyn_expansion(X), 'X')
        return self.unnormalize(self.PLS.predict(X_exp), 'Y')

class HCPLSR():
    def __init__(self,
                 X,
                 Y,
                 global_components=8,
                 local_components=8,
                 n_clusters=10,
                 fuzzy_m=2.0,
                 cluster_min=20):

        self.global_components = global_components
        self.local_components = local_components
        self.X = X
        self.Y = Y
        self.n_clusters = n_clusters
        self.cluster_min = cluster_min
        self.fuzzy_m = fuzzy_m
        self.setup()

    def setup(self):
        self.global_PLS = PLSR(self.X,
                               self.Y,
                               self.global_components)
        self.fcm_clustering()
        clusters = self.fcm.u.argmax(axis=1)
        self.local_PLS = []
        for i in range(self.n_clusters):
            if (clusters == i).sum() < self.cluster_min:
                self.local_PLS.append(self.global_PLS)
            else:
                X = self.X[clusters == i]
                Y = self.Y[clusters == i]
                PLS = PLSR(X, Y, self.local_components)
                self.local_PLS.append(PLS)

    def fcm_clustering(self):
        self.fcm = FCM(n_clusters=self.n_clusters, m=self.fuzzy_m)
        self.fcm.fit(self.global_PLS.PLS.y_scores_)

    def predict(self, X):
        Xn = self.global_PLS.normalize(self.global_PLS.polyn_expansion(X), 'X')
        x_scores = self.global_PLS.PLS.transform(Xn)
        probs = self.fcm.soft_predict(x_scores)
        psds = []
        for i, x in enumerate(X):
            psd = None
            for j, prob in enumerate(probs[i]):
                pred = self.local_PLS[j].predict(x[None,:])
                if psd is None:
                    psd = prob * pred
                else:
                    psd += prob * pred
            psds.append(psd.squeeze())
        return np.array(psds)
