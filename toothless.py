## Toothless - Radio Galaxy Classifier
# Arun Aniyan, Kshitij Thorat
# SKA SA / RATT
# arun@ska.ac.za
# 27th March 2018
# http://doi.org/10.5281/zenodo.579637


## Base class for toothless


import os
from collections import Counter
import numpy as np
import PIL.Image

os.environ['GLOG_minloglevel'] = '2'  # Suppress most caffe output
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import scipy.misc
from scipy.misc import imsave
from skimage.transform import resize


class Toothless(object):
    """Baseclass for toothless. Provides function to classify an in put image"""

    caffemodels = ['include/Models/fr1vsfr2.caffemodel', 'include/Models/fr1vsbent.caffemodel',
                        'include/Models/fr2vsbent.caffemodel']
    deployfiles = ['include/Prototxt/fr1vsfr2.prototxt', 'include/Prototxt/fr1vsbent.prototxt',
                        'include/Prototxt/fr2vsbent.prototxt']
    labelfiles = ['include/Labels/fr1vsfr2-label.txt', 'include/Labels/fr1vsbent-label.txt',
                       'include/Labels/fr2vsbent-label.txt']

    fr1fr2net = caffe.Net(deployfiles[0], caffemodels[0], caffe.TEST)
    fr1bentnet = caffe.Net(deployfiles[1], caffemodels[1], caffe.TEST)
    fr2bentnet = caffe.Net(deployfiles[2], caffemodels[2], caffe.TEST)

    fr1fr2lb = ['FRI','FRII']
    fr1bentlb = ['Bent-Tail','FRI']
    fr2bentlb = ['Bent-Tail','FRII']


    '''Read individual label files'''

    def __read_label(self, labels_file):
        """
            Returns a list of strings

            Arguments:
            labels_file -- path to a .txt file
            """
        if not labels_file:
            print 'WARNING: No labels file provided. Results will be difficult to interpret.'
            return None

        labels = []
        with open(labels_file) as infile:
            for line in infile:
                label = line.strip()
                if label:
                    labels.append(label)
        assert len(labels), 'No labels found'
        return labels

    '''Decide label based on threshold'''

    def __decide(self, classification):
        lbl = []
        conf = []
        for label, confidence in classification:
            lbl.append(label)
            conf.append(confidence)
        idx = np.argmax(conf)

        return lbl[idx], conf[idx]

    # Load image to caffe
    def __load_image(self,path, height, width, mode='RGB'):
        """
        Load an image from disk

        Returns an np.ndarray (channels x width x height)

        Arguments:
        path -- path to an image on disk
        width -- resize dimension
        height -- resize dimension

        Keyword arguments:
        mode -- the PIL mode that the image should be converted to
            (RGB for color or L for grayscale)
        """

        image = PIL.Image.open(path)
        image = image.convert(mode)
        image = np.array(image)
        # squash
        image = scipy.misc.imresize(image, (height, width), 'bilinear')
        return image

    # Transformer function to perform image transformation
    def __get_transformer(self,deploy_file):
        """
        Returns an instance of caffe.io.Transformer

        Arguments:
        deploy_file -- path to a .prototxt file

        Keyword arguments:
        mean_file -- path to a .binaryproto file (optional)
        """
        network = caffe_pb2.NetParameter()

        with open(deploy_file) as infile:
            text_format.Merge(infile.read(), network)

        if network.input_shape:

            dims = network.input_shape[0].dim
        else:
            dims = network.input_dim[:4]

        t = caffe.io.Transformer(
            inputs={'data': dims}
        )
        t.set_transpose('data', (2, 0, 1))  # transpose to (channels, height, width)


        return t

    # Forward pass of input through the network
    def __forward_pass(self,images, net, transformer, batch_size=1):
        """
        Returns scores for each image as an np.ndarray (nImages x nClasses)

        Arguments:
        images -- a list of np.ndarrays
        net -- a caffe.Net
        transformer -- a caffe.io.Transformer

        Keyword arguments:
        batch_size -- how many images can be processed at once
            (a high value may result in out-of-memory errors)
        """
        caffe_images = []
        for image in images:
            if image.ndim == 2:
                caffe_images.append(image[:, :, np.newaxis])
            else:
                caffe_images.append(image)

        caffe_images = np.array(caffe_images)

        dims = transformer.inputs['data'][1:]

        scores = None
        for chunk in [caffe_images[x:x + batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
            new_shape = (len(chunk),) + tuple(dims)
            if net.blobs['data'].data.shape != new_shape:
                net.blobs['data'].reshape(*new_shape)
            for index, image in enumerate(chunk):
                image_data = transformer.preprocess('data', image)
                net.blobs['data'].data[index] = image_data
            output = net.forward()[net.outputs[-1]]
            if scores is None:
                scores = output
            else:
                scores = np.vstack((scores, output))
            # print 'Processed %s/%s images ...' % (len(scores), len(caffe_images))

        return scores

    # Perform Single classification
    def __classify(self, net, labels, image_files,
                   deploy_file):
        """
        Classify some images against a Caffe model and print the results

        """
        # Load the model and images

        transformer = self.__get_transformer(deploy_file)
        _, channels, height, width = transformer.inputs['data']
        if channels == 3:
            mode = 'RGB'
        elif channels == 1:
            mode = 'L'
        else:
            raise ValueError('Invalid number for channels: %s' % channels)
        images = [self.__load_image(image_file, height, width, mode) for image_file in image_files]

        # Classify the image
        scores = self.__forward_pass(images, net, transformer)

        # Process the results

        indices = (-scores).argsort()[:, :5]  # take top 5 results
        classifications = []
        for image_index, index_list in enumerate(indices):
            result = []
            for i in index_list:
                # 'i' is a category in labels and also an index into scores
                if labels is None:
                    label = 'Class #%s' % i
                else:
                    label = labels[i]
                result.append((label, round(100.0 * scores[image_index, i], 4)))
            classifications.append(result)

        for index, classification in enumerate(classifications):
            lbl, conf = self.__decide(classification)

        return lbl, conf

    # Fusion decision model
    def __vote(self, ypreds, probs, thresh):
        # Find the repeating class among the three models
        high_vote = [item for item, count in Counter(ypreds).items() if count > 1]

        if high_vote != []:
            final_class = high_vote[0]
            # Check if their probability is greater than 60
            idx = np.where(np.array(ypreds) == final_class)[0]

            if float(probs[idx[0]]) > thresh or float(probs[idx[1]]) > thresh:
                final_classification = final_class
                final_probability = (float(probs[idx[0]]) + float(probs[idx[1]])) / 2.0
            else:
                final_classification = final_class + '?'
                final_probability = min(float(probs[idx[0]]), float(probs[idx[1]]))
        else:
            final_classification = 'Strange'
            final_probability = 0

        return final_classification, final_probability

    def __clip(self, data, lim):
        data[data < lim] = 0.0
        return data

    # Convert fits image to png
    def __fits2jpg(self, fname):
        hdu_list = fits.open(fname)
        image = hdu_list[0].data
        image = np.squeeze(image)
        img = np.copy(image)
        idx = np.isnan(img)
        img[idx] = 0
        img_clip = np.flipud(img)
        sigma = 3.0
        # Estimate stats
        mean, median, std = sigma_clipped_stats(img_clip, sigma=sigma, iters=10)
        # Clip off n sigma points
        img_clip = self.__clip(img_clip, std * sigma)
        if img_clip.shape[0] != 150 or img_clip.shape[1] != 150:
            img_clip = resize(img_clip, (150, 150))

        outfile = fname[0:-5] + '.png'
        imsave(outfile, img_clip)
        return outfile

    # Exposed function
    def fusion_classify(self, fitsfile, thresh=90, png_keep=False):

        ypreds = []
        probs = []

        # Check fits else raise error
        if fitsfile.rsplit('.', 1)[1] == 'fits':
            image_file = self.__fits2jpg(fitsfile)

        else:
            print 'Input file not in fits format'
            exit(0)

        # Stupid but works
        lbl, conf = self.__classify(self.fr1fr2net, self.fr1fr2lb, [image_file],self.deployfiles[0])
        ypreds.append(lbl)
        probs.append(conf)

        lbl, conf = self.__classify(self.fr1bentnet, self.fr1bentlb, [image_file],self.deployfiles[1])
        ypreds.append(lbl)
        probs.append(conf)

        lbl, conf = self.__classify(self.fr2bentnet, self.fr2bentlb, [image_file],self.deployfiles[2])
        ypreds.append(lbl)
        probs.append(conf)

        prediction, probability = self.__vote(ypreds, probs, thresh)

        # Deletes the png file created in between
        if png_keep == False:
            os.remove(image_file)

        return prediction, probability
