import math, copy, random, time, string
#import cv2 as cv2
import numpy as np

class Convolution():

    def __init__(self, nc_in, nc_out, kernel_size, stride=2,padding=1):
        self.kernel_size = kernel_size
        self.weights = np.random.randn(nc_in * kernel_size[0] * kernel_size[1] ,nc_out) * np.sqrt(2/nc_in)
        self.biases = np.zeros(nc_out)
        self.stride = stride
        self.padding = padding

    def forward(self,x):
        mb, ch, n, p = x.shape
        self.old_size = (n,p)
        self.old_x = arr2vec(x,self.kernel_size,self.stride,self.padding)
        y = np.matmul(self.old_x, self.weights) + self.biases
        y = np.transpose(y,(0,2,1))
        n1 = (n-self.kernel_size[0]+ 2 * self.padding) //self.stride + 1
        p1 = (p-self.kernel_size[1]+2 * self.padding )//self.stride + 1
        return y.reshape(mb,self.biases.shape[0],n1,p1)

    def backward(self,grad):
        mb, ch_out, n1, p1 = grad.shape
        grad = np.transpose(grad.reshape(mb,ch_out,n1*p1),(0,2,1))
        self.grad_b = grad.sum(axis=1).mean(axis=0)
        self.grad_w = (np.matmul(self.old_x[:,:,:,None],grad[:,:,None,:])).sum(axis=1).mean(axis=0)
        new_grad = np.matmul(grad,self.weights.transpose())
        return vec2arr(new_grad, self.kernel_size, self.old_size, self.stride, self.padding)

def arr2vec(x, kernel_size, stride=1,padding=0):
    k1,k2 = kernel_size
    mb, ch, n1, n2 = x.shape
    y = np.zeros((mb,ch,n1+2*padding,n2+2*padding))
    y[:,:,padding:n1+padding,padding:n2+padding] = x
    start_idx = np.array([j + (n2+2*padding)*i for i in range(0,n1-k1+1+2*padding,stride) for j in range(0,n2-k2+1+2*padding,stride) ])
    grid = np.array([j + (n2+2*padding)*i + (n1+2*padding) * (n2+2*padding) * k for k in range(0,ch) for i in range(k1) for j in range(k2)])
    to_take = start_idx[:,None] + grid[None,:]
    batch = np.array(range(0,mb)) * ch * (n1+2*padding) * (n2+2*padding)
    return y.take(batch[:,None,None] + to_take[None,:,:])

def vec2arr(x, kernel_size, old_shape, stride=1,padding=0):
    k1,k2 = kernel_size
    n,p = old_shape
    mb, md, ftrs = x.shape
    ch = ftrs // (k1*k2)
    idx = np.array([[[i-k1i, j-k2j] for k1i in range(k1) for k2j in range(k2)] for i in range(n) for j in range(p)])
    in_bounds = (idx[:,:,0] >= -padding) * (idx[:,:,0] <= n-k1+padding)
    in_bounds *= (idx[:,:,1] >= -padding) * (idx[:,:,1] <= p-k2+padding)
    in_strides = ((idx[:,:,0]+padding)%stride==0) * ((idx[:,:,1]+padding)%stride==0)
    to_take = np.concatenate([idx[:,:,0] * k2 + idx[:,:,1] + k1*k2*c for c in range(ch)], axis=0)
    to_take = to_take + np.array([ftrs * i for i in range(k1*k2)])
    to_take = np.concatenate([to_take + md*ftrs*m for m in range(mb)], axis=0)
    in_bounds = np.tile(in_bounds * in_strides,(ch * mb,1))
    return np.where(in_bounds, np.take(x,to_take), 0).sum(axis=1).reshape(mb,ch,n,p)