import matplotlib.pyplot as plt
import os
import DFFT

def ShowDFFT(img,fft,name,save=False):
    magnitude=DFFT.GistDFFT(fft)

    plt.figure(figsize=(15,9),dpi=80)

    plt.subplot(121)
    plt.imshow(img,cmap="gist_gray")
    plt.title('Input:'+name)
   
    plt.subplot(122)
    plt.imshow(magnitude,cmap='gray', interpolation="none")
    plt.title("Magnitude Spectrum")
    if save:
        path=os.path.join('DFFT_Compare',name)
        print(path)
        plt.savefig(path)
    plt.show()

def CompareImages(img1,img2,save=False,dir_path=None,names=['','']):
    plt.figure(figsize=(15,9))
    plt.title(names[0])

    plt.subplot(121)
    plt.imshow(img1,cmap="gray",vmax=255,vmin=0)

    plt.subplot(122)
    plt.imshow(img2,cmap='gray',vmax=255,vmin=0)
    plt.title(names[1])
    if save:
        path=os.path.join(dir_path,names[0]+'.png')
        print(path)
        plt.savefig(path)
    plt.show()
