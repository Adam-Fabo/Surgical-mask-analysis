# Proejekt na ISS
# Made by Adam Fabo - xfaboa00
# subor obsahuje riesenia pre ulohy 14


from pomoc_funkcie import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io.wavfile import write
import wave

from scipy.signal import butter, lfilter
from pomoc_funkcie import *
from scipy import integrate


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


prah = 0.30
# pre znazornenie ulohy 12 nastavit prah na 0.67

maskoff_arr = load_wav("audio/maskoff_tone.wav")  # nacitam audio
maskon_arr = load_wav("audio/maskon_tone.wav")

# výber sekundy v nahrávkach
x = 2
y = 4.45
maskoff_arr = np.array(maskoff_arr)

ramec_plot_maskoff = np.array(maskoff_arr[int(16000 * x) + 160 * 80:int(16000 * (x)) + 160 * 80 + 320])
ramec_plot_maskoff = normalizacia(ramec_plot_maskoff)

maskoff_arr = prahovanie(maskoff_arr[int(16000 * x):int(16000 * (x + 1))], prah)
maskoff_ramce = ramcovanie(maskoff_arr)

ramec_plot_maskon = np.array(maskon_arr[int(16000 * y) + 160:int(16000 * (y)) + 160 + 320])
ramec_plot_maskon = normalizacia(ramec_plot_maskon)

maskon_arr = np.array(maskon_arr)
maskon_arr = prahovanie(maskon_arr[int(16000 * y):int(16000 * (y + 1))], prah)

maskon_ramce = ramcovanie(maskon_arr)
ramec_plot_prah = maskon_ramce[80]

# tuna mam uz ramce nasekane a znormalizovane

maskoff_zakladnef_ramcov = []
maskon_zakladnef_ramcov = []
spektrum_maskoff = []
spektrum_maskon = []
dft_maskoff = []
dft_maskon = []

for i in range(98):

    maskoff_korelacia = korelacia(maskoff_ramce[i])

    # kniznicna funkcia
    tmp = np.fft.fft(maskoff_ramce[i], 1024)[0:512]

    # moja funkcia
    # tmp = np.array(my_dft(maskoff_ramce[i],1024)[0:512])

    dft_maskoff.append((np.array(tmp)))

    tmp = pow(np.abs(tmp), 2)

    # skontroluje ci tam nie je 0 (lebo inak inf v logaritme)
    counter = 0
    for stuff in tmp:
        if (stuff < 0.00000001):
            tmp[counter] = 1
        counter = counter + 1

    # pridanie jedneho stlpca do spektra
    spektrum_maskoff.append((10 * (np.log10(np.abs(tmp)))))

    # najde lag v autokorelacii, prah = 10
    for j in range(10):
        maskoff_korelacia = np.delete(maskoff_korelacia, 0)

    # ziska index najvacsej hodnoty
    maskoff_zakladnef_ramcov.append(np.argmax(maskoff_korelacia) + 10)

#############################################################################################################
# to iste co hore len pre maskon

for i in range(98):
    maskon_korelacia = korelacia(maskon_ramce[i])

    # kniznicna func
    tmp = np.fft.fft(maskon_ramce[i], 1024)[0:512]

    # moja func - treba pridat zero padding do 1024
    # tmp = np.array(my_dft(maskon_ramce[i], 1024)[0:512])

    dft_maskon.append(tmp)

    tmp = pow(abs(tmp), 2)

    # skontroluje ci tam nie je 0 (lebo inak inf v logaritme)
    counter = 0
    for stuff in tmp:
        if (stuff < 0.00000001):
            tmp[counter] = 1
        counter = counter + 1

    # pridanie jedneho stlpca do spektra
    spektrum_maskon.append((10 * (np.log10(np.abs(tmp)))))

    # najde lag v autokorelacii, prah = 10
    for j in range(10):
        maskon_korelacia = np.delete(maskon_korelacia, 0)

    # ziska index najvacsej hodnoty
    maskon_zakladnef_ramcov.append(np.argmax(maskon_korelacia) + 10)

# tuna uz mam vsetko po ulohu 5 - po spektogram


sirka = 100
koef = np.full(sirka, 1 / sirka)
freq_char = np.zeros(512)

############################################################################ toto je dobre - delenie ramcov a nasledne predelenie

for i in range(98):
    # print(spektrum_maskoff[i])
    if (signal.lfilter(koef, [1], spektrum_maskoff[i])[0]) != 0:
        freq_char += signal.lfilter(koef, [1], spektrum_maskon[i]) / (signal.lfilter(koef, [1], spektrum_maskoff[i]))
freq_char /= 100
plt.plot(np.arange(0, 8000, 8000 / 512), freq_char,label = "Pomocou hlavnej metódy")
plt.title("Frekvenčná charakteristika rúška")
plt.xlabel("frequency")
plt.ylabel("Spektrálna hustota výkonu [db]")

maskoff_arr = load_wav("audio/maskoff_sentence.wav")  # nacitam audio
maskon_arr = load_wav("audio/maskon_sentence.wav")
x = 2
y = 4.45
maskon_arr = np.array(maskon_arr)
maskon_arr = (maskon_arr[int(16000 * y):int(16000 * (y + 1))])

maskoff_arr = np.array(maskoff_arr)
maskoff_arr = (maskoff_arr[int(16000 * y):int(16000 * (y + 1))])

maskoff_integ = []
maskon_integ = []

for i in range(790):  # vypocet charakteristiky pomocou energe v signale
    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 16000.0
    # print(i)
    lowcut = 20 + 10 * i  # 5220.0
    highcut = 40 + 10 * i  # 5240.0

    maskoff_sig = butter_bandpass_filter(maskoff_arr, lowcut, highcut, fs, order=3)
    maskoff_integ.append(integrate.simps(np.abs(maskoff_sig)))

    maskon_sig = butter_bandpass_filter(maskon_arr, lowcut, highcut, fs, order=3)
    maskon_integ.append(integrate.simps(np.abs(maskon_sig)))

sirka = 200
koef = np.full(sirka, 1 / sirka)
maskoff_integ = signal.lfilter(koef, [1], maskoff_integ)
maskon_integ = signal.lfilter(koef, [1], maskon_integ)

plt.plot(np.arange(0, 8000, 8000 / 790), np.array(maskon_integ) / np.array(maskoff_integ),label = "Pomocou pomeru výkonov")

plt.legend()
plt.show()

