
# coding: utf-8

import matplotlib as mpl
mpl.use('Agg')
import pickle
import numpy as np
from PIL import Image
import os
import math
import pylab
import glob
import time
import scipy
from math import exp
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
import random
import csv
import numpy

from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from pylab import *

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer.utils import type_check
from chainer import function
import chainer.functions as F
import chainer.links as L


#HRFの生成関数
def hrf(nt=0.3,
        peak_delay=6,
        under_delay=10,
        p_u_ratio = 6,):
    t = np.arange(0,30+nt,nt)
    peak_disp=1
    under_disp=1
    normalize=True    
                  
    hrf = np.zeros(t.shape, dtype=np.float)
    
    pos_t = t[t > 0]
    peak = sps.gamma.pdf(pos_t,
                         peak_delay/peak_disp,
                         loc=0,
                         scale=peak_disp)
    UD = under_delay + peak_delay
    undershoot = sps.gamma.pdf(pos_t,
                               UD / under_disp,
                               loc=0,
                               scale=under_disp)
    hrf = peak - undershoot / p_u_ratio
    if not normalize:
        return hrf
    return hrf / np.max(hrf)

#学習データ作成
x_train = []
hrf_random = []
for i in range(100):
    #for random
    p_random = randint(1,20)
    w_random = randint(1,7)
    h_random = random()
    p_n = choice(range(-1, 2, 2))
    h_rand =  h_random*p_n
    
    hrf_random=hrf(0.3,p_random,10,w_random)
    hrf_random=list(hrf_random)
    for i in range(100):
        hrf_random[i] = hrf_random[i]+h_rand
    x_train.append(hrf_random)

x_train = np.array(x_train)
print (x_train.shape)
print x_train

#パラメータの設定

#乱数の長さ
nz = 10
#乱数の範囲
n_n_rand = -1
n_p_rand = 1
batchsize=1
#エポック数
n_epoch=10000
#1epochごとの学習回数
n_train = 100


#generatorのクラス
class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            lstm1 = L.LSTM(10,10),
        )

    def __call__(self, z):
        l1 = F.reshape(z,(1,1,1,10))
        ls = self.lstm_forward_all_step(l1)
        return ls

    def reset_state(self):
        self.lstm1.reset_state()

    def lstm_forward_all_step(self,x_seq):
        x_seq = F.reshape(x_seq,(10,1))
        h = Variable(np.zeros((1, 10),dtype=np.float32))
        state_output = Variable(np.zeros((1, 0),dtype=np.float32))
        for i in range(10):
            if i == 0:
                h = self.lstm1(F.reshape(x_seq[0:10],(1,10)))
                state_output = h
            else:
                h = self.lstm1(np.zeros((1, 10),dtype=np.float32))
                state_output = F.concat((state_output,h), axis=1)
        return state_output

#discriminatorのクラス
class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            lstm1 = L.LSTM(10,10),
            l1 = L.Linear(10,2),
        )

    def __call__(self, x):
        ls = F.reshape(x,(100,1))
        ls = self.lstm_forward_all_step(ls)
        l1 = F.reshape(ls,(1,1,1,10))
        l1 = self.l1(l1)
        return l1

    def reset_state(self):
        self.lstm1.reset_state()

    def lstm_forward_all_step(self,x_seq):
        h = Variable(np.zeros((1, 10),dtype=np.float32))
        for i in range(10):
            h = self.lstm1(F.reshape(x_seq[i*10:i*10+10],(1,10)))
        return h

init_gen = []
loss_gen = []
loss_dis = []

#実際の学習を行う関数
def train_dcgan_labeled(gen, dis, epoch0=0):
    o_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_gen.setup(gen)
    o_dis.setup(dis)
    o_gen.add_hook(chainer.optimizer.WeightDecay(0.0001))
    o_gen.add_hook(chainer.optimizer.GradientClipping(5.0))

    o_dis.add_hook(chainer.optimizer.WeightDecay(0.0001))
    o_dis.add_hook(chainer.optimizer.GradientClipping(5.0))

    x_gen = []
    print "Epoch Start!"

    #学習回数分まわす
    for epoch in xrange(epoch0,n_epoch):
        sum_l_dis = np.float32(0)
        sum_l_gen = np.float32(0)
        #一回の学習で教師何個使うか
        for i in xrange(0, n_train, batchsize):
            # discriminator
            # 0: from dataset
            # 1: from noise

            # generator学習
            # 一様ランダム生成
            z_np = np.random.randint(n_n_rand, n_p_rand, nz).reshape(1,1,1,nz)
            z = z_np.astype(np.float32)
            gen.reset_state()
            x = gen(z)
            dis.reset_state()
            yl = dis(x)

            L_gen = F.softmax_cross_entropy(yl, Variable(np.zeros(batchsize, dtype=np.int32)))
            L_dis = F.softmax_cross_entropy(yl, Variable(np.ones(batchsize, dtype=np.int32)))

            # discriminator学習
            dis.reset_state()
            x2 = Variable(x_train[i].astype(np.float32))
            yl2 = dis(x2)

            L_dis += F.softmax_cross_entropy(yl2, Variable(np.zeros(batchsize, dtype=np.int32)))

            o_gen.zero_grads()
            L_gen.backward()
            o_gen.update()

            o_dis.zero_grads()
            L_dis.backward()
            o_dis.update()

            sum_l_gen += L_gen.data
            sum_l_dis += L_dis.data
        x_gen.append(x.data[0])
        print ('epoch end:'), epoch+1,('generator loss:'), sum_l_gen/n_train, ('discriminator loss:'),sum_l_dis/n_train
        loss_gen.append(sum_l_gen/n_train)
        loss_dis.append(sum_l_dis/n_train)
    return x_gen


file_name = "gen_data_rand"+str(nz)+"_int_"+str(n_n_rand)+str(n_p_rand)
pic_path = "gen_data_rand_hrf/"+str(file_name)+"/pic"
model_data_path = "gen_data_rand_hrf/"+str(file_name)
gen = Generator()
dis = Discriminator()
x_gen_out = []


#学習開始
x_gen_out = train_dcgan_labeled(gen, dis)

#generatorが生成したhrfを画像として保存
file_name = "gen_data_rand"+str(nz)+"_int_"+str(n_n_rand)+str(n_p_rand)
pic_path = "gen_data_rand_hrf/"+str(file_name)+"/pic"
model_data_path = "gen_data_rand_hrf/"+str(file_name)
for i in range(n_epoch):
    matplot_name = pic_path+"/gen_epoch"+str(i+1)+".jpg"
    matplot_title = "Epoch "+ str(i+1)
    plt.figure(figsize=(8,6))
    plt.ylim([-0.2,1.4])
    plt.plot(x_train[0],'--b',label = "Real HRF",linewidth=2)
    plt.plot(x_gen_out[i],'r',label = "Generated HRF",linewidth=2)
    plt.legend()
    plt.title(matplot_title)
    plt.xlabel("Time[s]")
    plt.ylabel("Oxy-Hb[a.u.]")
    plt.savefig(matplot_name)
    plt.clf()

print "pic save done!"

#modelの保存およびgeneratorが生成した配列を保存
x_gen_out = np.array(x_gen_out)
serializers.save_npz(model_data_path+"/gen_"+str(file_name)+".model", gen)
serializers.save_npz(model_data_path+"/dis_"+str(file_name)+".model", dis)
np.save(model_data_path+"/x_gen_out_"+str(file_name)+".npy",x_gen_out)

#epochごとの損失関数グラフを画像で保存
loss_gen = np.array(loss_gen)
loss_dis = np.array(loss_dis)
plt.figure(figsize=(8,6))
plt.plot(loss_dis,'b',label = "dis",linewidth=2)
plt.plot(loss_gen,'r',label = "gen",linewidth=2)
plt.legend()
plt.savefig(model_data_path+"/loss"+str(file_name)+".jpg")
plt.clf()
print "model save done!"

#コサイン類似度を計算する関数
def cosine_similarity(v1, v2):
    return sum([a*b for a, b in zip(v1, v2)])/(sum(map(lambda x: x*x, v1))**0.5 * sum(map(lambda x: x*x, v2))**0.5)

#コサイン類似度計算
train_ave = np.array([])
train_ave = x_train[0]
x_gen_out = np.load(model_data_path+"/x_gen_out_"+str(file_name)+".npy")
cos_h = []
for i in range(n_epoch):
    cos_h_v = 0
    cos_h_v = cosine_similarity(list(train_ave),list(x_gen_out[i]))
    cos_h.append(cos_h_v)
cos_h = np.array(cos_h)

#コサイン類似度をnumpy形式で保存
np.save(model_data_path+"/cos_sim_"+str(file_name)+".npy",cos_h)

#コサイン類似度をグラフ化して保存
plt.clf()
plt.figure(figsize=(8,6))
plt.title("cos similarity")
plt.ylim([-1.0,1.0])
plt.xlabel("epoch")
plt.ylabel("cosine similarity")
plt.plot(cos_h,'b',label = "cosine similarity")
plt.savefig(model_data_path+"/cos_sim"+str(file_name)+".jpg")

print "cos_sim save done!"
