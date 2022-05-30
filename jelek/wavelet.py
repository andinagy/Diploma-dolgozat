import pywt
import numpy as np
import matplotlib.pyplot as plt


# x = np.arange(512)
# y = np.sin(2*np.pi*x/32)
# coef, freqs=pywt.cwt(y,np.arange(1,129),'gaus1')
#
# plt.matshow(coef)
# plt.title("Spektrum gauss wavelet transzformációval")
# plt.show()

t = np.linspace(-1, 1, 200, endpoint=False)
sig = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7 * (t - 0.4) ** 2) * np.exp(1j * 2 * np.pi * 2 * (t - 0.4)))
widths = np.arange(1, 31)
cwtmatr, freqs = pywt.dwt(sig, widths, 'haar')
plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
           vmax = abs(cwtmatr).max(), vmin = -abs(cwtmatr).max())
plt.title("Spektrum morlet transzformációval")
plt.show()

wavlist = pywt.wavelist(kind='discrete')
print(wavlist)