import torch
import numpy as np
from sklearn.decomposition import PCA

class TransferModel():
    def __init__(self, enc_model, clf_model, seg_count=8, embedding_dims=2048,
                 reduce_dims=None):
        self.enc_model = enc_model
        self.clf_model = clf_model
        self.seg_count = seg_count
        self.embedding_dims = embedding_dims
        self.reduce_dims = reduce_dims
        self.pca = None

    def encode(self, X):
        features = np.zeros((X.shape[0], self.seg_count,
                                self.embedding_dims))
        for i, x in enumerate(X):
            features[i] = self.enc_model.features(x).detach().numpy()
        if self.reduce_dims != None:
            features_length = features.shape[0]
            features = features.reshape(-1, self.embedding_dims)
            if self.pca == None:
                self.pca = PCA(self.reduce_dims)
                features = self.pca.fit_transform(features)
            else:
                features = self.pca.transform(features)
            try:
                features = features.reshape(features_length, self.seg_count, self.reduce_dims)
            except:
                import ipdb; ipdb.set_trace()
        return torch.FloatTensor(features)

    def fit(self, X, y, epochs=10, batch_size=4):
        self.features = self.encode(X)
        self.clf_model.fit(self.features, y, epochs, batch_size)

    def predict(self, X):
        self.features_pred = self.encode(X)
        return self.clf_model.predict(self.features_pred)
