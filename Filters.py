from DFFT import * 
import Visualisation as vs
import FileWorker as FW
import cv2 as cv

fw=FW.FileWorker()

def Gauss(img,k_size=3):
    kernel=np.zeros(img.shape)

    blur_kernel=cv.getGaussianKernel(k_size,-1)
    blur_kernel=np.matmul(blur_kernel,blur_kernel.T)
   
    kernel[:k_size,:k_size]=blur_kernel

    img_fshift,ker_fshift=ApplyDFFT(img),ApplyDFFT(kernel)
    
    multi=np.multiply(img_fshift,ker_fshift)
    return ReverseDFFT(multi)



def ApplyQuantile(img,q,dmz=25,hazard_zone=5):
    img_fshift=ApplyDFFT(img)
    arch=np.copy(img_fshift)

    mag=np.abs(img_fshift)
    print('q=',np.quantile(mag,q),q)
    mag_q=np.quantile(mag,q)
   

    dmz_x=[int(img.shape[0]/2 - img.shape[0]/dmz +1 ),int(img.shape[0]/2 + img.shape[0]/dmz + 1)]
    dmz_y=[int(img.shape[1]/2 - img.shape[1]/dmz +1 ),int(img.shape[1]/2 +img.shape[1]/dmz + 1)]
    x_len,y_len=img_fshift.shape
    for x in range(0,x_len):
        for y in range(0,y_len):
            if np.abs(img_fshift[x][y])>mag_q:
                if  dmz_x[0]< x <dmz_x[1] and dmz_y[0]< y <dmz_y[1]:
                    continue
                else:
                    img_fshift[max(0,x-hazard_zone):min(x+hazard_zone,x_len),max(0,y-hazard_zone):min(y+hazard_zone,y_len)]=0
                    
    vs.CompareImages( GistDFFT(arch),GistDFFT(img_fshift) )
    return img_fshift

def Notch(img,q,dmz=25,hazard_zone=5):
    q_fshift=ApplyQuantile(img,q,dmz,hazard_zone)
    reverse_img=ReverseDFFT(q_fshift)
    return reverse_img
