import numpy as np
    
def ILD(Xl, Xr):
    ILD = 20*np.log10(abs(np.divide(Xl, Xr)))
    return ILD

def IPD(Xl, Xr):
    IPD = np.angle(np.divide(Xl,Xr),deg=0)
    return IPD

def LPS(Xl, Xr):
    LPS_l = np.log(np.power(np.abs(Xl),2))
    LPS_r = np.log(np.power(np.abs(Xr),2))    
    LPS = (LPS_l+LPS_r)/2
    return LPS

def MV(X):
    v=X_all/np.linalg.norm(X, ord='fro')
    u,W,n = np.linalg.svd(np.cov(v))
    W = np.reshape(W,[len(X),1])
    MV = W*v
    MV = MV / np.linalg.norm(MV,ord='fro');
    MV_real = MV.real
    MV_imag = MV.imag
    return MV_real, MV_imag

def IMD(Xl, Xr):
    IMD = Xl-Xr
    return IMD

