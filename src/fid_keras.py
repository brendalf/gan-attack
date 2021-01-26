import numpy as np
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class FID:
    def __init__(self):
        self.__image_size = (299,299,3)
        self.__model = InceptionV3(
            include_top=False, 
            pooling='avg', 
            input_shape=self.__image_size
        )

    def calculate_fid(self, images1, images2):
        # scale images
        images1 = self.__scale_images(images1)
        images2 = self.__scale_images(images2)

        # pre-process images
        images1 = preprocess_input(images1)
        images2 = preprocess_input(images2)

        # calculate activations
        act1 = self.__model.predict(images1)
        act2 = self.__model.predict(images2)

        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2)**2.0)

        # calculate sqrt of product between np.cov
        covmean = sqrtm(sigma1.dot(sigma2))

        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def __scale_images(self, images):
        images_list = list()
        for image in images:
            # resize with nearest neighbor interpolation
            new_image = resize(image, self.__image_size, 0)
            # store
            images_list.append(new_image)
        return np.asarray(images_list)