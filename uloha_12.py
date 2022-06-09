# Proejekt na ISS
# Made by Adam Fabo - xfaboa00
# subor obsahuje riesenia pre ulohy 12

from pomoc_funkcie import *
import matplotlib.pyplot as plt
import numpy as np
from  scipy import signal
from scipy.io.wavfile import write
import wave
import statistics

from scipy.signal import butter, lfilter
from pomoc_funkcie import  *
from scipy import integrate

def korekcia(arr):
    counter = 0
    for prvok in arr:
        if prvok > np.mean(arr)*1.25 or prvok < np.mean(arr)*0.75:
            arr[counter] = statistics.mode(np.array(arr[int(counter-10) : int(counter+10)]))
        counter +=1

    return arr


prah = 0.67
#pre znazornenie ulohy 12 nastavit prah na 0.67

maskoff_arr = load_wav("audio/maskoff_tone.wav")#nacitam audio
maskon_arr = load_wav("audio/maskon_tone.wav")


# výber sekundy v nahrávkach
x = 2
y = 4.45
maskoff_arr = np.array(maskoff_arr)



ramec_plot_maskoff = np.array(maskoff_arr[int(16000*x)+160*80:int(16000*(x))+160*80+320])
ramec_plot_maskoff = normalizacia(ramec_plot_maskoff)

maskoff_arr = prahovanie(maskoff_arr[int(16000*x):int(16000*(x+1))],prah)
maskoff_ramce = ramcovanie(maskoff_arr)


ramec_plot_maskon = np.array(maskon_arr[int(16000*y)+160:int(16000*(y))+160+320])
ramec_plot_maskon = normalizacia(ramec_plot_maskon)

maskon_arr = np.array(maskon_arr)
maskon_arr = prahovanie(maskon_arr[int(16000*y):int(16000*(y+1))],prah)


maskon_ramce = ramcovanie(maskon_arr)
ramec_plot_prah = maskon_ramce[80]


#tuna mam uz ramce nasekane a znormalizovane

maskoff_zakladnef_ramcov = []
maskon_zakladnef_ramcov = []
spektrum_maskoff = []
spektrum_maskon  = []
dft_maskoff = []
dft_maskon = []



for i in range(98):

    maskoff_korelacia = korelacia(maskoff_ramce[i])

    # kniznicna funkcia
    tmp = np.fft.fft(maskoff_ramce[i], 1024)[0:512]

    # moja funkcia
    #tmp = np.array(my_dft(maskoff_ramce[i],1024)[0:512])

    dft_maskoff.append((np.array(tmp)))

    tmp =  pow( np.abs(tmp) ,2)


    # skontroluje ci tam nie je 0 (lebo inak inf v logaritme)
    counter = 0
    for stuff in tmp:
        if(stuff<0.00000001):
            tmp[counter] = 1
        counter = counter +1

    # pridanie jedneho stlpca do spektra
    spektrum_maskoff.append( (10 * (np.log10(np.abs(tmp)))))

    # najde lag v autokorelacii, prah = 10
    for j in range(10):
        maskoff_korelacia = np.delete(maskoff_korelacia,0)

    # ziska index najvacsej hodnoty
    maskoff_zakladnef_ramcov.append(np.argmax(maskoff_korelacia)+10)

#############################################################################################################
# to iste co hore len pre maskon

for i in range(98):
    maskon_korelacia = korelacia(maskon_ramce[i])

    #kniznicna func
    tmp = np.fft.fft(maskon_ramce[i], 1024)[0:512]

    # moja func - treba pridat zero padding do 1024
    # tmp = np.array(my_dft(maskon_ramce[i], 1024)[0:512])

    dft_maskon.append(tmp)

    tmp =  pow( abs(tmp) ,2)

    # skontroluje ci tam nie je 0 (lebo inak inf v logaritme)
    counter = 0
    for stuff in tmp:
        if(stuff<0.00000001):
            tmp[counter] = 1
        counter = counter +1

    # pridanie jedneho stlpca do spektra
    spektrum_maskon.append( (10 * (np.log10(np.abs(tmp)))))


    # najde lag v autokorelacii, prah = 10
    for j in range(10):
        maskon_korelacia = np.delete(maskon_korelacia,0)

    # ziska index najvacsej hodnoty
    maskon_zakladnef_ramcov.append(np.argmax(maskon_korelacia)+10)



#tuna uz mam vsetko po ulohu 5 - po spektogram


sirka = 100
koef = np.full(sirka,1/sirka)
freq_char = np.zeros(512)



ramec = korelacia(maskon_ramce[80])
fig, axs = plt.subplots(4)


axs[0].set_title("Rámec")
axs[0].set_ylabel("y")
axs[0].set_xlabel("time")
axs[0].plot(np.arange(0,0.02,0.02/320),ramec_plot_maskoff)

axs[1].set_title("Centrálne klipovanie s 67%")
axs[1].set_ylabel("y")
axs[1].set_xlabel("time")
axs[1].plot(np.arange(0,0.02,0.02/320),ramec_plot_prah)

axs[2].set_title("Autokorelácia s chybou")
axs[2].set_ylabel("y")
axs[2].set_xlabel("vzorky")
axs[2].axvline(10,c='k',label = "Prah")
#axs[2].plot([],'r-',label='Prah')
axs[2].stem([np.argmax(ramec[10:])+10],[np.max(ramec[10:])], linefmt="C3-",label="Lag")
#axs[2].plot(131,0.025,'ro')
axs[2].plot(ramec)
axs[2].legend()


axs[3].set_title("Základne frekvencie rámcov s chybou")
axs[3].set_ylabel("f0")
axs[3].set_xlabel("rámce")

maskoff_zakladnef_ramcov = 16000/np.array(maskoff_zakladnef_ramcov)         #opravenie chyb
maskon_zakladnef_ramcov = 16000/np.array(maskon_zakladnef_ramcov)

axs[3].plot(maskoff_zakladnef_ramcov)
axs[3].plot(maskon_zakladnef_ramcov)
axs[3].legend(["maskoff", "maskon"])


plt.tight_layout()
#plt.figure()


maskoff_zakladnef_ramcov = korekcia(maskoff_zakladnef_ramcov)
maskon_zakladnef_ramcov = korekcia(maskon_zakladnef_ramcov)


ramec = korelacia(maskon_ramce[80])
figg, axs = plt.subplots(4)


axs[0].set_title("Rámec")
axs[0].set_ylabel("y")
axs[0].set_xlabel("time")
axs[0].plot(np.arange(0,0.02,0.02/320),ramec_plot_maskoff)

axs[1].set_title("Centrálne klipovanie s 67%")
axs[1].set_ylabel("y")
axs[1].set_xlabel("time")
axs[1].plot(np.arange(0,0.02,0.02/320),ramec_plot_prah)

axs[2].set_title("Autokorelácia s opravou")
axs[2].set_ylabel("y")
axs[2].set_xlabel("vzorky")
axs[2].axvline(10,c='k',label = "Prah")
#axs[2].plot([],'r-',label='Prah')
axs[2].stem([np.argmax(ramec[10:])+10],[np.max(ramec[10:])], linefmt="C3-",label="Lag")
#axs[2].plot(131,0.025,'ro')
axs[2].plot(ramec)
axs[2].legend()


axs[3].set_title("Základne frekvencie rámcov s opravou")
axs[3].set_ylabel("f0")
axs[3].set_xlabel("rámce")

maskoff_zakladnef_ramcov = 16000/np.array(maskoff_zakladnef_ramcov)
maskon_zakladnef_ramcov = 16000/np.array(maskon_zakladnef_ramcov)

axs[3].plot(maskoff_zakladnef_ramcov)
axs[3].plot(maskon_zakladnef_ramcov)
axs[3].legend(["maskoff", "maskon"])


plt.tight_layout()

plt.show()