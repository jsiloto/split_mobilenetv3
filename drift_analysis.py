import argparse
import os

import h5py
import numpy as np
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')


class BaselineTransformation:
    def __init__(self, baseline_window):
        d, l, p = baseline_window
        self.ss = StandardScaler()
        self.ss.fit(d)
        self.pca = PCA(n_components=50)
        self.pca.fit(d)

    def transform(self, data):
        return self.pca.transform(self.ss.transform(data))

    def make_statistics(self, samples):
        d, l, p = samples
        data = self.transform(d)
        cov = np.cov(data.T)
        mean = np.mean(data, axis=0)
        return mean, cov


class Embeddings:
    def __init__(self, dataset_file):
        self.pca = {}
        self.ss = {}
        with h5py.File(dataset_file, "r") as hf:
            print(hf['data'].shape)
            print(hf['predicted'])
            self.shape = hf['data'].shape
            self.data = np.array(hf['data'])
            self.labels = np.array(hf['class_id'])
            self.pred_labels = np.array(hf['predicted'])

    def get_sample(self, n=100):
        index = np.random.choice(len(self.labels), size=n, replace=False, p=None)
        d, l, p = self.data[index], self.labels[index], self.pred_labels[index]
        shape1 = self.shape[1] * self.shape[2] * self.shape[3]
        d = np.reshape(d, (d.shape[0], shape1))
        return d, l, p


def get_mixed_samples(embeddings1: Embeddings, embeddings2: Embeddings, n=100, p=0.5):
    d1, l1, p1 = embeddings1.get_sample(int(n * p))
    d2, l2, p2 = embeddings2.get_sample(int(n * (1 - p)))
    d = np.concatenate([d1, d2])
    l = np.concatenate([l1, l2])
    p = np.concatenate([p1, p2])
    index = np.random.choice(len(d), size=len(d), replace=False, p=None)
    d = d[index]
    l = l[index]
    p = p[index]
    return d, l, p


def fid_score(mean_a, cov_a, mean_b, cov_b):
    m = sum((mean_a - mean_b) ** 2)
    n = cov_a + cov_b - 2 * sqrtm(cov_a @ cov_b)
    c = m + np.trace(n)
    return c


def main():
    args = parser.parse_args()

    # Read Embeddings
    golden_embeddings = Embeddings(os.path.join("./", "golden_embeddings.h5"))
    perturbed_embeddings = Embeddings(os.path.join("./", "f_0_embeddings.h5"))

    windows = []
    for prob in np.arange(0.0, 1.0, 0.05):
        d, l, p = get_mixed_samples(golden_embeddings, perturbed_embeddings, n=1000, p=prob)
        windows.append((d, l, p))

    # Train PCA on first window
    baseline_pca = BaselineTransformation(windows[0])
    mean0, cov0 = baseline_pca.make_statistics(windows[0])

    distances = []
    acc = []
    for w in windows[1:]:
        d, l, p = w
        accw = sum(l==p)/len(l)
        meanw, covw = baseline_pca.make_statistics(w)
        dd = fid_score(meanw, covw, mean0, cov0)
        distances.append(dd)
        acc.append(accw)

    print(distances)
    print(acc)


if __name__ == '__main__':
    main()
