import numpy as np

def mix_feature(feature1,feature2):
    result = np.hstack((feature1,feature2))
    return result