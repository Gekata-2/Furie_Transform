import numpy as np

def ApplyDFFT(image):
    fft_image = np.fft.fft2(image)
    fft_image = np.fft.fftshift(fft_image)
    return fft_image

def GistDFFT(fft_image):
    mag=np.abs(fft_image)
    mag=np.log(mag+1)
    mag=( mag-mag.min() )*255 / ( mag.max()-mag.min())
    return mag


def ReverseDFFT(fft_image):
    f_inv_shift=np.fft.ifftshift(fft_image)
    reverse_image=np.fft.ifft2(f_inv_shift)
    reverse_image=np.real(reverse_image)
    return reverse_image




