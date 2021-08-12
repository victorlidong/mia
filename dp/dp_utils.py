# coding=utf-8

import sys
import numpy as np
import random
import math
import torch
import tensorflow as tf

#梯度裁剪功能
def clip_func(clip_bound,clip_type,input):
    if(clip_bound<=0):
        return input
    if(clip_type=="norm1"):
        return tf.clip_by_value(input,-1*clip_bound,clip_bound)
    elif(clip_type=="norm2"):
        norm2=float(tf.norm(input))
        tmp=max(norm2/clip_bound,1)
        return input/tmp
    else:
        print("no such clip-type")
        return input

#拉普拉斯噪声,高斯噪声
def laplace_function(beta,size):
    return np.random.laplace(0,beta,size=size)

def gauss_function(sigma):
    return random.gauss(0,sigma)

def get_tensor_size(param_size):
    tmp=1
    for i in param_size:
        tmp*=i
    return tmp.value

#计算所有参数梯度的敏感度
def calculate_l1_sensitivity(clip_bound,param_size):
    return 2*clip_bound*param_size

def calculate_l2_sensitivity(clip_bound):
    return 2*clip_bound

def calculate_l1_sensitivity_sample(grad_data,param_size,sample_num):
    # sample
    grad_data_1D=tf.reshape(grad_data,[-1])
    if(sample_num<=param_size):
        sample_index=random.sample(range(param_size),sample_num)
        sample_grad = tf.gather(grad_data_1D,sample_index)
    else:
        sample_grad=grad_data_1D

    #计算标准差
    mean, var = tf.nn.moments(sample_grad, axes=0)
    # array=sample_grad.numpy()
    # std=array.std()
    std_deviation=math.sqrt(float(var))

    return (1.13*param_size+2.56*math.sqrt(param_size))*std_deviation


def calculate_l2_sensitivity_sample(grad_data,param_size,sample_num):
    # sample
    grad_data_1D = tf.reshape(grad_data, [-1])
    if (sample_num <= param_size):
        sample_index = random.sample(range(param_size), sample_num)
        sample_grad = tf.gather(grad_data_1D, sample_index)
    else:
        sample_grad = grad_data_1D

    # 计算标准差
    mean, var = tf.nn.moments(sample_grad, axes=0)
    # array=sample_grad.numpy()
    # std=array.std()
    std_deviation = math.sqrt(float(var))

    return 1.45*math.sqrt(param_size)*std_deviation


def gen_laplace_beta(batchsize,Parallelnum,sensitivity,privacy_budget):
    scaledEpsilon=privacy_budget*float(batchsize)/Parallelnum
    beta=sensitivity/scaledEpsilon
    return beta

def gen_gaussian_sigma(batchsize,Parallelnum,sensitivity,privacy_budget,privacyDelta):
    scaledEpsilon = privacy_budget * float(batchsize) / Parallelnum
    sigma= (2.0 * math.log(1.25 / privacyDelta)) * sensitivity / scaledEpsilon
    return sigma
