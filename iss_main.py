# Proejekt na ISS
# Made by Adam Fabo - xfaboa00
# subor obsahuje riesenia pre ulohy 1-9

from pomoc_funkcie import *
import matplotlib.pyplot as plt
import numpy as np
from  scipy import signal
from scipy.io.wavfile import write
import wave

prah = 0.30

maskoff_arr = load_wav("audio/maskoff_tone.wav")#nacitam audio
maskon_arr = load_wav("audio/maskon_tone.wav")


# výber sekundy v nahrávkach
x = 2
y = 4.45

maskoff_arr = np.array(maskoff_arr)

ramec_plot_maskoff = np.array(maskoff_arr[int(16000*x)+160:int(16000*(x))+160+320])     #výber rámcu pre vykreslenie
ramec_plot_maskoff = normalizacia(ramec_plot_maskoff)

maskoff_arr = prahovanie(maskoff_arr[int(16000*x):int(16000*(x+1))],prah)
maskoff_ramce = ramcovanie(maskoff_arr)


ramec_plot_maskon = np.array(maskon_arr[int(16000*y)+160:int(16000*(y))+160+320])       #výber rámcu pre vykreslenie
ramec_plot_maskon = normalizacia(ramec_plot_maskon)

maskon_arr = np.array(maskon_arr)
maskon_arr = prahovanie(maskon_arr[int(16000*y):int(16000*(y+1))],prah)


maskon_ramce = ramcovanie(maskon_arr)
ramec_plot_prah = maskon_ramce[24]


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

    # moja funkcia - dost pomala
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


    # ziska index najvacsej hodnoty
    maskoff_zakladnef_ramcov.append(np.argmax(maskoff_korelacia[10:])+10)

#############################################################################################################
# to iste co hore len pre maskon

for i in range(98):
    maskon_korelacia = korelacia(maskon_ramce[i])

    #kniznicna func
    tmp = np.fft.fft(maskon_ramce[i], 1024)[0:512]

    # moja func -  - dost pomala
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



    # ziska index najvacsej hodnoty
    maskon_zakladnef_ramcov.append(np.argmax(maskon_korelacia[10:])+10)



#tuna uz mam vsetko po ulohu 5 - po spektogram


sirka = 100
koef = np.full(sirka,1/sirka)
freq_char = np.zeros(512)


############################################################################ toto je dobre - delenie ramcov a nasledne predelenie

for i in range(98):
    if(signal.lfilter(koef,[1],spektrum_maskoff[i])[0]) != 0:

        freq_char += signal.lfilter(koef,[1],spektrum_maskon[i])/(signal.lfilter(koef,[1],spektrum_maskoff[i]))
freq_char /= 98
plt.plot(np.arange(0,8000,8000/512),freq_char)
plt.title("Frekvenčná charakteristika rúška")
plt.xlabel("frequency")
plt.ylabel("Spektrálna hustota výkonu [db]")

plt.figure()


# kniznicna impulzna odozva
# impulz_odozva = np.fft.ifft(freq_char,1024)
# plt.plot(np.arange(0,8000,8000/1024),(impulz_odozva))


# moja impulzna odozva
impulz_odozva = my_idft(freq_char,1024)
plt.plot(np.arange(0,0.5,0.5/1024),impulz_odozva)

plt.title("Impluzná odozva")
plt.xlabel("Time")
plt.ylabel("Amplitude")

#print(np.allclose(impulz_odozva,aaa))
plt.figure()

vysledna_nahrav = np.array(load_wav("audio/maskoff_sentence.wav"))
plt.plot(vysledna_nahrav, label = "maskoff")

vysledna_nahrav = signal.lfilter(impulz_odozva,[1],np.array(vysledna_nahrav))
#write("sim_maskon_tone.wav", 16000, np.array(vysledna_nahrav).astype("int16"))

maskon = np.array(load_wav("audio/maskon_sentence.wav"))
plt.plot((maskon), label = "maskon")
plt.plot((vysledna_nahrav), label = "maskon - filtered")
plt.legend()
plt.show()


##############################
# tuna je uz iba vykreslovanie grafov


plt.imshow(np.transpose(spektrum_maskoff),origin  = 'lower', aspect='auto',extent = [0,1,0,8000])
plt.title("Spectogram maskoff")
plt.xlabel("time")
plt.ylabel("frequency")
plt.colorbar()
plt.figure()
plt.title("Spectogram maskon")
plt.xlabel("time")
plt.ylabel("frequency")
plt.imshow(np.transpose(spektrum_maskon),origin  = 'lower', aspect='auto',extent=[0,1,0,8000])
plt.colorbar()

hodnota = np.sum(np.abs(np.array(maskoff_zakladnef_ramcov)-np.array(maskon_zakladnef_ramcov)))
print("hodota ", hodnota)


ramec = korelacia(maskon_ramce[24])
fig, axs = plt.subplots(4)


axs[0].set_title("Rámec")
axs[0].set_ylabel("y")
axs[0].set_xlabel("time")
axs[0].plot(np.arange(0,0.02,0.02/320),ramec_plot_maskoff)

axs[1].set_title("Centrálne klipovanie s 30%")
axs[1].set_ylabel("y")
axs[1].set_xlabel("time")
axs[1].plot(np.arange(0,0.02,0.02/320),ramec_plot_prah)

axs[2].set_title("Autokorelácia")
axs[2].set_ylabel("y")
axs[2].set_xlabel("vzorky")
axs[2].axvline(10,c='k',label = "Prah")
#axs[2].plot([],'r-',label='Prah')
axs[2].stem([np.argmax(ramec[10:])+10],[np.max(ramec[10:])], linefmt="C3-",label="Lag")
#axs[2].plot(131,0.025,'ro')
axs[2].plot(ramec)
axs[2].legend()


axs[3].set_title("Základne frekvencie rámcov")
axs[3].set_ylabel("f0")
axs[3].set_xlabel("rámce")

maskoff_zakladnef_ramcov = 16000/np.array(maskoff_zakladnef_ramcov)
maskon_zakladnef_ramcov = 16000/np.array(maskon_zakladnef_ramcov)

axs[3].plot(maskoff_zakladnef_ramcov)
axs[3].plot(maskon_zakladnef_ramcov)
axs[3].legend(["maskoff", "maskon"])


plt.tight_layout()
plt.figure()

plt.title("Rámce")

plt.plot(np.arange(0,0.02,0.02/320),ramec_plot_maskoff)
plt.plot(np.arange(0,0.02,0.02/320),ramec_plot_maskon)
plt.xlabel("time")
plt.ylabel("y")
plt.legend(["maskoff", "maskon"])

stred_hodnota_off = 0
stred_hodnota_on = 0

roz_off = 0
roz_on = 0
for i in range(98):
    stred_hodnota_off += 1 / 98 * maskoff_zakladnef_ramcov[i]
    stred_hodnota_on  += 1 / 98 * maskon_zakladnef_ramcov[i]

    roz_off += 1 / 98 * pow(maskoff_zakladnef_ramcov[i],2)
    roz_on += 1 / 98 * pow(maskon_zakladnef_ramcov[i],2)

print(stred_hodnota_off,stred_hodnota_on)

rozptyl_off = roz_off - pow(stred_hodnota_off,2)
rozptyl_on = roz_on - pow(stred_hodnota_on,2)

print(rozptyl_off,rozptyl_on)

print(np.mean(maskoff_zakladnef_ramcov),np.var(maskoff_zakladnef_ramcov))
plt.show()