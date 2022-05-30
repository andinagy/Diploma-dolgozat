# from PyEMD import EMD
import pywt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pyhht.visualization import plot_imfs


def integrated_spectrum(x, fs):
    Xf = np.fft.fft(x)
    n = len(Xf)
    Xf = Xf[:int(n / 2)]
    ReXf = np.abs(Xf)
    p = np.power(ReXf, 2)
    cs = np.cumsum(p)
    cs /= cs[-1]
    f = np.linspace(0, int(fs / 2), len(cs))
    return cs, f

def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    # for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
    #     ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('C:/Users/Asus/Desktop/allamviszga/programok/erzelmek/emotions.csv')
    plotScatterMatrix(df, 10, 10)
    # fft_data=df.loc[:,'fft_0_b':'fft_100_b']
    #
    # wc1,wc2 = pywt.dwt(fft_data,"haar")
    # plt.plot(wc1,wc2)
    # plt.show()

    # plt.figure(1)
    # plt.subplot(211)
    # fft_data.iloc[0,:].plot(figsize=(10,5), label="power spectrum")

    # plt.figure(2)
    # t = np.linspace(0, 1, 1000)
    # emd=EMD()
    # imfs=emd(fft_data)
    # plot_imfs(fft_data, imfs, t)
    # plt.show()

#power spectrum
    # plt.subplot(212)
    # ps = np.abs(np.fft.fft(fft_data)) ** 2
    #
    # time_step = 1 / 30
    # freqs = np.fft.fftfreq(fft_data.size, time_step)
    # idx = np.argsort(freqs)
    #
    # ps = np.reshape(ps,215332)
    # plt.plot(freqs[idx], ps[idx])
    # plt.show()
