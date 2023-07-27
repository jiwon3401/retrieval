import random
import numpy as np


def mixgen(image, text, num, lam=0.5):
    # default MixGen
    for i in range(num):
        # image mixup
        image[i,:] = lam * image[i,:] + (1 - lam) * image[i+num,:]
        # text concat
        text[i] = text[i] + " " + text[i+num]
    return image, text


def mixgen_batch(image, text, num, lam=0.5):
    batch_size = image.size()[0]
    index = np.random.permutation(batch_size)
    for i in range(batch_size):
        # image mixup
        image[i,:] = lam * image[i,:] + (1 - lam) * image[index[i],:]
        # text concat
        text[i] = text[i] + " " + text[index[i]]
    return image, text


# how to use mixgen
# import mixgen as mg
# for i, (image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#     image, text = mg.mixgen(image, text, num=16)
#     optimizer.zero_grad()
    


### change image to audio
