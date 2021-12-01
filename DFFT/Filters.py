from DFFT import * 
import matplotlib.pyplot as plt
import Visualisation as vs
import FileWorker as FW
import cv2 as cv
from scipy import fftpack
fw=FW.FileWorker()

def Sobel(img):
    f_shift=ApplyDFFT(img)
    ker_size=3
    kernel=np.zeros(img.shape)
    sobel_v=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    sobel_h=sobel_v.T

    #Horizontal
    kernel[0:ker_size,0:ker_size]=sobel_h
    f_ker_shift=ApplyDFFT(kernel)
    multi_h=np.multiply(f_shift,f_ker_shift)
    reverse_h=ReverseDFFT(multi_h)
    #Vertical
    kernel[0:ker_size,0:ker_size]=sobel_v
    f_ker_shift=ApplyDFFT(kernel)
    multi_v=np.multiply(f_shift,f_ker_shift)

    reverse_v=ReverseDFFT(multi_v)

    reverse_vh=np.sqrt(reverse_v**2+reverse_h**2)

   
    return reverse_vh

def Gauss(img,k_size=3):
    kernel=np.zeros(img.shape)

    blur_kernel=cv.getGaussianKernel(k_size,-1)
    blur_kernel=np.matmul(blur_kernel,blur_kernel.T)
   
    kernel[:k_size,:k_size]=blur_kernel

    img_fshift,ker_fshift=ApplyDFFT(img),ApplyDFFT(kernel)
    
    multi=np.multiply(img_fshift,ker_fshift)
    vs.CompareImages(img,ReverseDFFT(multi))
    return ReverseDFFT(multi)

def FillWithZeros(img,x,y):
    img[x,:]=0
    img[:,y]=0

def ApplyQuantile(img,q):
    img_fshift=ApplyDFFT(img)
    arch=np.copy(img_fshift)

    mag=np.abs(img_fshift)
    print('q=',np.quantile(mag,q),q)
    mag_q=np.quantile(mag,q)
    x_len,y_len=img_fshift.shape

    sqx=[int(img.shape[0]/2 - img.shape[0]/50),int(img.shape[0]/2 + img.shape[0]/50)]
    sqy=[int(img.shape[1]/2 - img.shape[1]/50),int(img.shape[1]/2 +img.shape[1]/50)]
    for x in range(0,x_len):
        for y in range(0,y_len):
            if np.abs(img_fshift[x][y])>mag_q:
                if x<sqx[0] or x>sqx[1]:
                    if y<sqy[0] or y>sqy[1]:
                        img_fshift[max(0,x-6):min(x+6,x_len),max(0,y-6):min(y+6,y_len)]=0
                        FillWithZeros(img_fshift,x,y)
                    #img_fshift[x][y]=np.median(img_fshift)

   # img_fshift[mag>mag_q]=np.median(img_fshift)

    vs.CompareImages(GistDFFT(arch),GistDFFT(img_fshift) )
    return img_fshift

def Quantile(img,q):
    q_fshift=ApplyQuantile(img,q)
    reverse_img=ReverseDFFT(q_fshift)
    return reverse_img


#plt.imshow(Gauss(fw.ReadImg('test_img.png'),21))
#Quantile(fw.ReadImg('test_img.png'),q=0.9992)

'''
from scipy import fftpack
im = imread('images/halftone.png')
F1 = fftpack.fft2((im).astype(float))
F2 = fftpack.fftshift(F1)
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
im1 = fp.ifft2(fftpack.ifftshift(F2)).real
plt.figure(figsize=(10,10))
plt.imshow(im1, cmap='gray')
plt.axis('off')
plt.show()
'''