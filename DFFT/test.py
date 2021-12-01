from scipy import fftpack
import FileWorker as FW
import matplotlib.pyplot as plt
import numpy as np
import DFFT
fw=FW.FileWorker()
im = fw.ReadImg('halftone.png')
F1 = fftpack.fft2((im).astype(float))
F2 = fftpack.fftshift(F1)
w,h=im.shape
for i in range(60, w, 135):
    for j in range(100, h, 200):
        if not (i == 330 and j == 500):
            F2[i-10:i+10, j-10:j+10] = 0

for i in range(0, w, 135):
    for j in range(200, h, 200):
        if not (i == 330 and j == 500):
            F2[max(0,i-15):min(w,i+15), max(0,j-15):min(h,j+15)] = 0

plt.figure(figsize=(6.66,10))
plt.imshow( (20*np.log10( 0.1 + F2)).astype(int), cmap=plt.cm.gray)
plt.show()
im1 = DFFT.ReverseDFFT(F2)
plt.figure(figsize=(10,10))
plt.imshow(im1, cmap='gray')
plt.axis('off')
plt.show()