# Proejekt na ISS
# Made by Adam Fabo - xfaboa00
# subor obsahuje pomocne funkcie prie riesenie ulohy


import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import math

from  scipy import signal
from scipy.io.wavfile import write


#nacita wav
def load_wav(name):
    spf = wave.open(name, "r")
    # print("Number of channels", spf.getnchannels())
    # print("Sample width", spf.getsampwidth())
    # print("Frame rate.", spf.getframerate())
    # print("Number of frames", spf.getnframes())
    # print("parameters:", spf.getparams())
    signal = spf.readframes(-1)

    signal = np.frombuffer(signal, "Int16")

    arr = np.array(signal).astype("float32")
    return arr


# Znormalizuje signal
def normalizacia(arr):
    mean = np.mean(arr)
    absmax = np.abs(arr).max()

    counter = 0
    for stuff in arr:
        arr[counter] = (arr[counter] - mean) / absmax
        counter = counter + 1
    return arr

# prahuje signal na 1 0 -1
def prahovanie(arr,prah):

    arr = normalizacia(arr)
    counter = 0
    for stuff in arr:
        if (arr[counter] > prah):
            arr[counter] = 1
        elif arr[counter] < -prah:
            arr[counter] = -1
        else:
            arr[counter] = 0
        counter = counter + 1
    return arr

#vytvori zo signalu 98 ramcov (prvy a posledny zahadzuje)
def ramcovanie(arr):
    #ak je jedna sekunda 16000 vzorkov a ramec ma mat dlzku 20ms tak jeden ramec ma 320 vorkov
    krok = 160
    n = 320
    pole = []
    for i in range(98):
        pole.append(arr[krok*(i+1):krok*(i+1)+320])
    return pole


# spravi korelaciu ramca so samym sebou
# vrati pole hodnot pre jeden ramec
def korelacia(arr):
    arr1 = arr
    x=[]
    for i in range(320):
        cor = np.correlate(arr1, arr)
        x = np.append(x, cor / (320))

        arr1 = np.append(arr1, 0)
        arr = np.insert(arr, 0, 0)
    return x


def my_dft(arr,N):

    if len(arr)< N:
        arr = np.pad(arr, (0,N-len(arr)),"constant")

    vys = []
    for i in range(0,N):
        vys.append((0))
        for j in range(0,N):
            vys[i] += arr[j] * np.exp(np.complex(0,-2*np.pi*i*j/N))

    return np.array(vys)



def my_idft(arr,N):

    if len(arr)< N:
        arr = np.pad(arr, (0,N-len(arr)),"constant")


    vys = []
    for i in range(0,N):
        vys.append((0))
        for j in range(0,N):
            vys[i] += arr[j] * np.exp(np.complex(0,2*np.pi*i*j/N))

    return np.array(vys)/N
