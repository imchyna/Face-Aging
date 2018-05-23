import caffe
import cv2
import numpy as np
from config import *
from scipy.misc import imread, imresize, imsave

class learn_age_feature(object):

    def __init__(self, config_path):
        config.read(config_path);
        if config.get('app').get('mode') == 'gpu':
            caffe.set_mode_gpu()
        self.net = caffe.Net(config.get('app').get('age').get('model_decl'),
            config.get('app').get('age').get('model_weights'),
            caffe.TEST)
        self.size = config.get('app').get('Rothe').get('size')
        self.margin = config.get('app').get('Rothe').get('margin')
        self.means = config.get('app').get('Rothe').get('means')

    def get_feature(self, batch_images,batch_age):
        for i, i_img in enumerate(batch_images):
            try:
                img = cv2.imread(i_img)
            except:
                print(i_img)

            face = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32);
            for c_i in range(3):
                face[:, :, c_i] -= self.means[c_i]

            face = cv2.resize(face, tuple(self.size))
            data = np.expand_dims(face.transpose(2, 0, 1), 0)

            #self.net.forward_all(data=data);
            pred = np.argmax(self.net.forward_all(data=data)['prob'])
            batch_age[i]= self.net.blobs['fc8-101'].data
            print '[INFO] Age: {} '.format(pred)

        return batch_age
