import os
import numpy
import SimpleITK
import matplotlib.pyplot as plt
import cv2


####Simple ITK reads the image as (x, y, z) order
####When converting to numpy array it reads as (z, y, x) 


def sitk_show(img, title=None, margin=0.05, dpi=40 ):
    nda = SimpleITK.GetArrayFromImage(img)
    print(nda.shape)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,extent=extent,interpolation=None)
    
    if title:
        plt.title(title)
    
    plt.show()
    

#Change the file path to the desire mha file
filenameT1 = "/Users/Fercho/Desktop/Tomographic_Slice_Reconstruction/Data_BRATS_2015/BRATS2015_Training/LGG/brats_tcia_pat109_0001/VSD.Brain.XX.O.MR_T1.40826/VSD.Brain.XX.O.MR_T1.40826.mha"

imgT1Original = SimpleITK.ReadImage(filenameT1)

volume = SimpleITK.GetArrayFromImage(imgT1Original)

#print(nda.shape)
for idxSlice in range(volume.shape[0]):
    print("Slide", idxSlice)
    sitk_show(imgT1Original[:, :, idxSlice])


