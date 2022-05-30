import numpy as np
import pandas as pd
from numpy.fft import fft
from matplotlib import pyplot as plt

import kivy
kivy.require('2.1.0')
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.utils import get_color_from_hex
import pywt
import scaleogram as sc
from ssqueezepy import cwt
from ssqueezepy.visuals import plot, imshow

Builder.load_file('transform.kv')

class MyLayout(Widget):

    checks =[]
    Names = ""
    def radiobox_click(self,instance,value,name):
        if value == True:
            MyLayout.checks.append(name)
            names = ''
            for x in MyLayout.checks:
                names = f'{names} {x}'
            if name == "FFT":
                self.ids.output_label.text = f"A választásod: {names}"
                MyLayout.Names = name
            if name == "Haar":
                self.ids.output_label.text = f"A választásod: {names}"
                MyLayout.Names = name
            if name == "Morlet":
                self.ids.output_label.text = f"A választásod: {names}"
                MyLayout.Names = name
        else:
            MyLayout.checks.remove(name)
            names = ''
            for x in MyLayout.checks:
                names = f'{names} {x}'
            self.ids.output_label.text = f"A választásod: {names}"

    def selected(self, filename):
        print("OK")

    def plot_spectrum(self):
        #itt lesz megoldva hogy elvegezze a megfelo kirajzolast
        FilePath = self.ids.filechooser.selection[0]

        if MyLayout.Names == "FFT":
            Name_tr = self.ids.FFT.text
            transform(FilePath, Name_tr)
        if MyLayout.Names == "Haar":
            Name_tr = self.ids.Haar.text
            transform(FilePath, Name_tr)
        if MyLayout.Names == "Morlet":
            Name_tr = self.ids.Morlet.text
            transform(FilePath, Name_tr)
            printing(FilePath, MyLayout.Names)
        #printing(FilePath,Name_tr)


def printing(FilePath,Name_tr):
    print(FilePath)
    print(Name_tr)

def transform(filepath,name):
    #print(filepath)
    print(name)
    if name == "FFT":
        df = pd.read_csv(filepath)
        df.drop('label', axis=1, inplace=True)
        df = df.loc[:,'RAW_AF8']
        #print(df)
        np_array = df.to_numpy()
        X_f = np.fft.fft(np_array)
        # sig_fft_filtered = X_f.copy()
        # freq = np.fft.fftfreq(len(np_array), d=1. / 2000)
        # cut_off = 5
        # sig_fft_filtered[np.abs(freq) < cut_off] = 0
        N = len(X_f)
        n = np.arange(N)
        sr = 1/(N*N)
        T = N/sr
        fs = n/T
        n_oneside = N//2
        f_oneside = fs[:n_oneside]
        t = 1 / f_oneside /(N*N)
        plt.figure(figsize=(12, 6))
        #plt.plot(freq,np.abs(sig_fft_filtered))
        #filtered = np.fft.ifft(sig_fft_filtered)

        #plt.plot(t,filtered)
        plt.plot(t, np.abs(X_f[:n_oneside])/n_oneside)
        #plt.xticks([12, 24, 84, 168])
        plt.xlim(0, 25)
        plt.xlabel('time(s)')
        plt.ylabel('FFT Amplitude |X(freq)|')

        #plt.plot(fs, np.abs(X_f), 'b')
        #plt.xlabel('Freq (Hz)')
        #plt.ylabel('FFT Amplitude |X(freq)|')
        plt.savefig('spectrum.png')
        plt.show()

    # if name == "Haar":
    #     df = pd.read_csv(filepath)
    #     df.drop('label', axis=1, inplace=True)
    #
    #     np_array = df.to_numpy()
    #     widths = np.arange(1, 10)
    #     cwtmatr, freqs = pycwt.cwt(np_array, widths, 'haar')
    #     #megcsinalni
    #     # plt.plot(cwtmatr,freqs)
    #     plt.imshow(cwtmatr.astype('uint8'), extent=[0, 1, 1, 10], cmap='PRGn', aspect='auto',
    #                vmax = abs(cwtmatr).max(), vmin = -abs(cwtmatr).max())
    #     plt.savefig('spectrum.png')

    if name == "Morlet":
        df = pd.read_csv(filepath)
        df.drop('label', axis=1, inplace=True)
        df = df.loc[:, 'RAW_AF8']
        np_array = df.to_numpy()
        widths = np.arange(4,64)
        Wx, scales = cwt(np_array, 'morlet')
        res = scales[::-1]
        signal = Wx[::-1]
        imshow(signal,yticks=res, abs=1,
               title="Morlet wavelet (neutral valence)",
               ylabel="scales", xlabel="samples")
        # cwtmatr,freqs  = pywt.cwt(np_array,widths,'morl',sampling_period=0.00001)
        #
        # plt.ylim(0, 60)
        # plt.ylabel("Frequency in [Hz]")
        # plt.xlabel("Time in [s]")
        # plt.title("Scaleogram using wavelet Morlet")
        # #plt.plot(cwtmatr,freqs)
        #
        # #plt.specgram(cwtmatr, NFFT=128, Fs=1000, noverlap= 90, cmap='seismic')
        #
        # c = plt.imshow(cwtmatr, cmap='Blues', aspect='auto',
        #         vmax = 0.2, vmin = 0.02, interpolation='kaiser', origin='lower')
        # plt.xlabel('Time')
        # plt.ylabel('Frequency')
        #plt.colorbar()
        # plt.xlabel('time(s)')
        # plt.ylabel('scale')
        plt.savefig('spectrum.png')
        #plt.show()

def fft_own(xt):
    x = np.asarray(xt, dtype=float)
    N = x.shape[0]

    N_min = min(N, 2)

    n = np.arange(N_min)
    k = n[:, None]
    W = np.exp(-2j*np.pi*n*k/N_min)
    X = np.dot(W, x.reshape((N_min, -1)))

    while X.shape[0] < N:
        X_even = X[:, :X.shape[1]//2]
        X_odd = X[:, X.shape[1] // 2 :]
        factor = np.exp(-1j*np.pi*np.arange(X.shape[0])/X.shape[0])[:, None]

        X = np.vstack([X_even + factor*X_odd, X_even - factor*X_odd ])

    return X.ravel()

def call_fft_own(filepath):
    df = pd.read_csv(filepath)
    df.drop('label', axis=1, inplace=True)
    df = df.loc[:,'RAW_AF7']
    np_array = df.to_numpy()

    X_f = fft_own(np_array)
    A_f = np.zeros_like(np_array)
    N = len(X_f)
    n = np.arange(N)
    sr = 1 / (N * N)
    T = N / sr
    fs = n / T
    for k in range(N):
        A_f[k] = np.sqrt(np.power(X_f[k].imag, 2) + np.power(X_f[k].real, 2))

    A_f = A_f / N * 2.0
    f = np.linspace(0, fs / 2, N // 2)
    n_oneside = N // 2
    f_oneside = fs[:n_oneside]
    #plt.plot(f_oneside, A_f[:N // 2])
    plt.plot(X_f)
    plt.xlabel("Freq (Hz)")
    plt.ylabel("FFT Amplitude |X(freq)|")
    plt.show()

class FirstApp(App):

    def build(self):
        Window.clearcolor = get_color_from_hex('#32321a')
        return MyLayout()


if __name__ == '__main__':
    FirstApp().run()
    #transform('C:/Users/Asus/Desktop/allamviszga/programok/jelek/neutral_data.csv',"Morlet")
    #call_fft_own('C:/Users/Asus/Desktop/allamviszga/programok/jelek/my_data2.csv')
