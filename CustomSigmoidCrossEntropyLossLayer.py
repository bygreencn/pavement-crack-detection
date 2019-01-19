import caffe
import scipy
import numpy as np
class CustomSigmoidCrossEntropyLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check for all inputs
        if len(bottom) != 2:
            raise Exception("Need two inputs (scores and labels) to compute sigmoid crossentropy loss.")

    def reshape(self, bottom, top):
        # check input dimensions match between the scores and labels
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference would be the same shape as any input
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # layer output would be an averaged scalar loss
        top[0].reshape(1)

    def forward(self, bottom, top):
        score=bottom[0].data
        label=bottom[1].data

        first_term=np.maximum(score,0)
        second_term=-1*score*label
        third_term=np.log(1+np.exp(-1*np.absolute(score)))

        top[0].data[...]=np.sum(first_term+second_term+third_term)
        sig=scipy.special.expit(score)
        self.diff=(sig-label)
        if np.isnan(top[0].data):
                exit()

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...]=self.diff