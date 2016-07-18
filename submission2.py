import numpy as np
import scipy.misc

def submission():
    from data2 import load_imgs_id
    from data2 import load_test_data
    imgs_id = load_imgs_id()
    imgs_test = np.load('imgs_output_test2.npy')

    imgs_test *= 255.  # scale outputs to [0, 255]
    imgs_test = imgs_test.astype('int32')

    inp, imgs_testg = load_test_data()


    i = 0
    for img in imgs_test:
        scipy.misc.imsave('{0}.png'.format(imgs_id[i]), img[0])
        i+=1

#   for img, img2,img3,img4 in zip(imgs_test,imgs_testg,inp[:,0],inp[:,1]):
#       scipy.misc.imsave('a{0}.png'.format(imgs_id[i]), np.hstack((img3, img4, img2[0], img[0], (img2[0]-img[0]))))
#        scipy.misc.imsave('a{0}.png'.format(imgs_id[i]), img2[0]-img[0])
#       i+=1

if __name__ == '__main__':
    submission()
