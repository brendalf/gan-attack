import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from scipy import linalg
from torch.autograd import Variable
from torch.nn import functional as F

from torchvision.models.inception import inception_v3

def read_stats_file(filepath):
    """read mu, sigma from .npz"""
    if filepath.endswith('.npz'):
        f = np.load(filepath)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        raise Exception('ERROR! pls pass in correct npz file %s' % filepath)
    return m, s

class FID:
    def __init__(self):
        self.__image_size = (299,299,3)
        self.dtype = torch.cuda.FloatTensor
        self.__model = inception_v3(pretrained=True, transform_input=False).type(self.dtype)
        self.__model.eval()
        self.__fc = self.__model.fc
        self.__model.fc = nn.Sequential()

        # wrap with nn.DataParallel
        self.__model = nn.DataParallel(self.__model)
        self.__fc = nn.DataParallel(self.__fc)

    def __forward(self, x):
        """
        x should be N x 3 x 299 x 299
        and should be in range [-1, 1]
        """
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = self.__model(x)
        pool3_ft = x.data.cpu().numpy()

        x = self.__fc(x)
        preds = F.softmax(x, 1).data.cpu().numpy()
        return pool3_ft, preds

    def __calc_stats(self, pool3_ft):
        # calculate mean and covariance statistics
        mu = np.mean(pool3_ft, axis=0)
        sigma = np.cov(pool3_ft, rowvar=False)
        return mu, sigma

    def calculate_statistics(self, images, batch_size=16):
        # calculate activations
        n_img = images.shape[0]

        pool3_ft, preds = np.zeros((n_img, 2048)), np.zeros((n_img, 1000))

        for i in range(np.int32(np.ceil(1.0 * n_img / batch_size))):
            batch_size_i = min((i+1) * batch_size, n_img) - i * batch_size

            batchv = Variable(images[i * batch_size:i * batch_size + batch_size_i, ...].type(self.dtype))
            pool3_ft[i * batch_size:i * batch_size + batch_size_i], preds[i * batch_size:i * batch_size + batch_size_i] = self.__forward(batchv)

        mu, sigma = self.__calc_stats(pool3_ft)

        return mu, sigma

    def calculate_fid(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths %s, %s' % (mu1.shape, mu2.shape)
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions %s, %s' % (sigma1.shape, sigma2.shape)
        diff = mu1 - mu2

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            covmean = covmean.real
        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean