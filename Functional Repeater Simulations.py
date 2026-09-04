# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:00:16 2022

@author: dlago & sgrandi

Script to estimate the coincidence rates between two photon events in a functional repeater link

"""

import numpy as np
import matplotlib.pyplot as plt
import numba as nb

# plt.close('all')

# This just assigns colours for the plots later
signal_col = (193/255, 51/255, 255/255)
back_col = (34/255, 0/255, 193/255)
noise_col = (0/255, 135/255, 255/255)

in_col = (193/255, 51/255, 255/255)
in_sel_col = (103/255, 31/255, 0/255)
sws_col = (0/255, 135/255, 255/255)
sws_sel_col = (255/255, 159/255, 14/255)
noise_col = (0/255, 0/255, 0/255)
afc_col = (187/255, 59/255, 14/255)



@nb.njit()
def generate_photons(span, p, delta_t, rep):
    # WHAT IS THE POINT OF THIS FUNCTION? SPDC SOURCES CREATE BOTH IDLER AND SIGNAL SIMULATNEOUSLY SO THIS MAKES NO SENSE NO?
    """
    

    Parameters
    ----------
    span : the time of simulation of one repetition. (total simulation time of one repetition)
    p : probability per dt.
    delta_t : time interval.
    rep : simulation repetition.

    Returns
    -------
    idler_list : TYPE
        DESCRIPTION.
    signal_list : TYPE
        DESCRIPTION.

    """
    
    signal_list = [] # List of timestamps of signal photons
    idler_list = [] # List of timestamps of idler photons
    
    time_mark = 0
    for j in range(rep):
        list_idler = np.random.rand(int(span/delta_t)) # for every time stamp we create random number between 0 and 1
        index_idler = np.where((list_idler[0] - p)<0) # If random number is less than p we have a idler photon (the [0] makes no sense to me here)
        idler_list.append(index_idler + time_mark)
        
        list_signal = np.random.rand(int(span/delta_t))
        index_signal = np.where((list_signal[0] - p)<0)
        signal_list.append(index_signal + time_mark)
        
        time_mark += int(span/delta_t)
    
    idler_list = np.array(idler_list)[0]*delta_t # Gives us the actual timestamps of when the photons are generated
    signal_list = np.array(signal_list)[0]*delta_t
    
    return idler_list, signal_list


@nb.njit()
def correlate_next(trig, sig, size, bsize, dmax):
    # This function takes two lists of chronological timestamps (trig for heralding/trigger events, and sig for signal events) and builds a histogram of the time delays between them.
    """

    Parameters
    ----------
    trig : The list of heralding events.
    sig : The list of signals.
    dmax : The maximum delay in the g2 histogram.
    size : The size of the g2 histogram.
    binsize : The binning size of the histogram.

    Returns
    -------
    hist : array
        Return the heralded correlation histogram between trig and sig.

    """
    
    hist = np.zeros(2*size + 1) # Creates blank histogram
    marker = 0
    counter = []
    for j in range(len(trig)): # For each trigger
        m = marker # Remembers where it left of since it's in chronological order, so we don't look at older stuff
        t_ref = trig[j]
        while (m < len(sig)):
            if(sig[m] > t_ref + dmax): # Signal photon is detected too late
                break
            elif(sig[m] < t_ref):
                marker = m
                m = m + 1
            else: # Signal photon arrived after the trigger but before dmax (time limit). This is a succes
                t_diff = sig[m] - t_ref
                bin_pos = np.floor(t_diff/(bsize))
                hist[int(size + bin_pos)] += 1
                m = m + 1
                counter.append(bin_pos)
                break
    return counter, hist


@nb.njit()
def learn_statistics(trig, sig, tout, lat):
    """

    Parameters
    ----------
    trig : The list of heralding events.
    sig : The list of signals.
    tout : The maximum delay that the second memory can wait. (maximum storage time)
    lat : The latency period that is introduced after any successfull event. (time it takes to restore the memory)

    Returns
    -------
    hist : the list of good heralding events.
    del_list : the list containing the storage durations.

    """
    
    her = []
    del_list = []
    marker = 0
    t_mark = 0
    
    c_mark = 0
    c_tout = 0
    c_ok = 0
    for j in range(len(trig)):
        
        if trig[j] < t_mark: # The system ignores this time stamp because it is busy either processing another event or dead time
            c_mark += 1
        else:
            m = marker
            t_ref = trig[j]
            while (m < len(sig)):
                if(sig[m] >= t_ref + tout): # If it arrives after a certain amount of time is fails
                    t_mark = t_ref + tout # busy for this time
                    c_tout += 1
                    break
                elif(sig[m] <= t_ref):
                    marker = m
                    m += 1
                else: # Success
                    her.append(t_ref)
                    del_list.append(sig[m] - t_ref) # Storage time
                    m += 1
                    t_mark = sig[m] + lat # Busy for the reset time
                    c_ok += 1
                    break
    return her, del_list, c_mark, c_tout, c_ok, j


#%%
"""
SPECS

details for both source and memory
"""

# Quantum Memory
tau_AFC = 20*1e-06 # AFC storagte time [s]
tau_SW = np.linspace(0, 1000, 20)*1e-06 # time in the spin state [s]
eta_AFC0 = 0.6 # zero-time AFC efficiency
eta_CP = 0.8 # control pulses 

eta_duty_mem = (303/707)
eta_duty_chopper = (20/33) # Downtime of SPDC
eta_duty = eta_duty_mem*eta_duty_chopper # final duty cycle

NF = 8.2*1e-04 # noise floor
T2 = 2e-03 # effective coherence time of the memory [s]
gamma_inhom = 5e02 # spin inhomogeneity [Hz]
T_eff = 2*1e-03 # effective time for DD [s]


# SPDC
etaH = 0.4 # heralding efficiency
P = 4 # power of the SPDC pump [mW]
a = 195 # g2 model coefficient
Rid0 = 2*1513 # [Hz/mW] the factor of 2 comes from using two outputs

g2 = 1 + a/(P) # idler-signal cross-correlation


# Sync capabilities:
dt = 100*1e-09 # mode size [s]
t_out = tau_AFC + tau_SW # maximum time that one QM can wait after that link detects one idler. First option is for time-bin


# Transmission efficiencies
eta_T1 = 0.95 # source-to-memory transmission
eta_T2 = 0.45 # memory-to-filter transmission
eta_T3 = 0.8 # filter-to-fibre transmission


# Detector specs
etaD = 0.8
etaDi = etaD
etaDs = 0.8 # 606 detector efficiency


# Fibre
L = 0 # length of idler fibre [km]
alpha = 0.3
fib = 10**(-alpha*L/10)


#%%
"""
2-FOLD HERALDING RATE

with a SW memory able to store conditionally during a fixed amount of time in 
the spin state. The model for the dead-time is likely a worst-case scenario, as 
it heavily reduces the heralding rate - while we believe for infinite t_out it
should go to simply half the standard heralding rate.

with this formula, if t_out = 1/R = dt/pdt, then pH2 = pdt**2, as expected for one QM with a storage time matching the heralding rate.
"""

"""
MONTECARLO

just for funsies, until I learn statistics. Or Stackexchange people reply
"""
# Given the random firing of our lasers and the strict time limits of our quantum memories, how many times per second do we successfully get both Mode 1 and Mode 2 loaded into the memories simultaneously?
time_span = 20 # [s]
repetition = 100

Rid = Rid0*P*fib*etaDi*eta_duty_chopper # raw rate of getting a single successful heralding click at the beam splitter for one of the modes
pdt = Rid*dt # probability of getting a click for a mode during a time window dt

print(Rid)

# Ctrigger, Csignal = generate_photons(time_span, pdt, dt, repetition)

signal_list = [] # Model one (Rail-1)
idler_list = [] # Mode two (Rail-2)

# We ARE ATTEMPTING BOTH AT THE SAME TIME?

time_mark = 0
for j in range(repetition):
    # This still makes no sense to me bc in an SPDC source these two events aren't independent. THINKING ABOUT IT I THINK IT ACTUALLY SIMULATES A DOUBLE RAIL. THEY AREN'T SIGNAL AND IDLERS THEY ARE TWO RAILS AT THE SAME TIME (PARALLEL NOT SEQUENTIAL). This is the code to create the elementary link A-R1
    list_idler = np.random.rand(int(time_span/dt))
    index_idler = np.where((list_idler - pdt) < 0)
    idler_list.append(index_idler[0] + time_mark)
    
    list_signal = np.random.rand(int(time_span/dt))
    index_signal = np.where((list_signal - pdt) < 0)
    signal_list.append(index_signal[0] + time_mark)
    
    time_mark += int(time_span/dt)
    
    print(j)
    
    del list_idler
    del list_signal

Ctrigger = np.array([item for sublist in idler_list for item in sublist])*dt
Csignal = np.array([item for sublist in signal_list for item in sublist])*dt

# This is to make sure one isn't measuring for a lot longer than the other, we just delete those numbers
tmin = max(min(Csignal), min(Ctrigger))
Csignal = Csignal[Csignal >= tmin]
Ctrigger = Ctrigger[Ctrigger >= tmin]

tmax = min(max(Csignal), max(Ctrigger))
Csignal = Csignal[Csignal <= tmax]
Ctrigger = Ctrigger[Ctrigger <= tmax]

tmeasure = (tmax-tmin) # [s] Total measurement time of data
length_t = len(Ctrigger)
length_s = len(Csignal)

#Ctrigger = Ctrigger(Ctrigger > Ctrigger(1)/2) # this line is to remove one Hydra bug, in case the time stamps go back in time

sing606 = len(Csignal)/tmeasure #acquisition rates
sing1436 = len(Ctrigger)/tmeasure

histo_size = 2000
bin_size = 800*1e-09
del_max = histo_size*bin_size
delay = np.linspace(-histo_size, histo_size, 2*histo_size + 1)*bin_size

count, start_stop = correlate_next(Ctrigger, Csignal, histo_size, bin_size, del_max)
count_hist = np.histogram(count, bins = 200, range = (0,1500))

# fig, ax = plt.subplots()
# # ax.bar(count_hist[1][:-1], count_hist[0])
# ax.plot(delay*1e6, start_stop/(time_span*repetition), color = back_col)
# ax.set(xlabel = r'Delay [$\mu$s]', ylabel = 'Coincidences/s', xlim = (0, 1000))
# plt.show()


# tout_list = np.linspace(17.5, 1000, 25)*1e-06
lat = [0, 100*1e-06, 400*1e-06, 1e-03]

fig, ax = plt.subplots()

rate_lat_list = []
for lt in lat:
    rate_list = []
    for to in t_out:
        
        success, delay_list, counter_mark, counter_tout, counter_ok, mm = learn_statistics(Ctrigger, Csignal, to, lt)
        delay_list_hist = np.histogram(delay_list, bins = int(max(delay_list)/dt + 1))
        binsize = delay_list_hist[1][1] - delay_list_hist[1][0]
        
        rate = len(success)/tmeasure
        rate_list.append(rate)
        
    ax.plot(t_out*1e06, rate_list)
    rate_lat_list.append(rate_list)
    
ax.set(xlabel = r't_{out}', ylabel = 'Heralding rate [Hz]')
plt.tight_layout()
plt.show()

RH = np.array(rate_lat_list[0]) # This is the rate at which we can produce the elementary link (very useful for our simulation)
print (RH)
fig, ax = plt.subplots()
ax.bar(1e06*delay_list_hist[1][:-1], delay_list_hist[0], width = binsize*1e06)
# ax.plot(range(len(delay_list)), delay_list, color = back_col)
ax.set(xlabel = r'Delay [$\mu$s]', ylabel = 'Coincidences/s')
plt.show()



# Rid0 = Rid0*P*fib*etaDi*eta_duty_chopper # heralding rate at the idler
# Rid0_dead = Rid0/(1 + Rid0*(t_out*1e-06)) # this includes the dead-time
# Rid0_dead = Rid0

# pdt = Rid0_dead*dt*1e-06 # probability of detecting an idler mode during a time window dt
# pH2 = pdt*(1-(1-pdt)**(t_out/dt)) # double-click probability
# RH = pH2/(dt*1e-06) # heralding rate for the functional repeater

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(6, 6))
ax1.plot(1e06*t_out, RH)
ax1.axhline(RH[0], c = 'k', ls = '--')
ax1.axhline(Rid0, c = 'r', ls = '--')
ax1.set(ylabel = 'Heralding rate [Hz]')


#%%
"""
COINCIDENCES

Final rate, after mapping back to light and into polarisation.
"""

eta_AFC = eta_AFC0*np.exp(-4*tau_AFC/T2) # Efficiency of the AFC comb (shouldn't it also effect the quality?)
eta_coh = np.array([np.exp(-(tau_SW*np.pi*gamma_inhom)**2/(2*np.log(2))),
           np.exp(-2*(tau_SW/T_eff)**0.5)], dtype = object) # Efficiency in going to spin wave and back (we have an array of two models for some reason)
eta_QM = eta_AFC * eta_CP**2 * eta_coh # QM efficiency (efficiency of AFC, efficiency if spin wave state and efficiency of the two pulses)

eta_QN = etaH * eta_T1 * eta_QM # transmission of the signal photon at each QN (probability of the photon existing when idler clicked, efficiency of the transimission to the memory and memory efficiency). Survival probs of the signal photon at the node
eta_map = etaDs * eta_T3 * eta_T2 # efficiency of mapping the excitation to a photon (from memeory to filter, from filter to fibre and then detection which I don't get because we don't want to detect it no?). IN OUR CASE WE WOULDN'T NEED THE DETECTION EFFICIENCY BECAUSE WE WOULDN'T YET

Rcoinc = eta_duty_mem*np.array([1/2 * RH * eta_QN[0]**2 * eta_map**2,
                                1/2 * RH * eta_QN[1]**2 * eta_map**2], dtype = object) # detected coincidence rate (2 fold-rate times two memories and mapping succesess * 1/2 for when we don't have one photon at A and one at R1 * deadtime efficicency for memory)

[ax2.plot(1e06*t_out, r*3600) for r in Rcoinc] # counts per hour
# ax2.axhline(Rcoinc[0], c = 'k', ls = '--')
ax2.set(ylabel='Coinc rate [counts/h]')


#%%
"""
FIDELITY

This is the fidelity of the state, as it is in the memory
"""

# g2
g2sw = (etaH * eta_T1 * eta_AFC * eta_CP**2 * eta_coh) * 1/( (etaH * eta_T1 * np.sqrt(eta_AFC))/g2 + NF) + 1 # Signal to noise ratio


#diag elements
p10 = 1/2 * eta_QN # one photon on one side none on the other (what we want at each mode/rail)
p01 = p10
p11 = 4 * p10*p01/(g2sw)**2 * (1 + g2sw)

#Vis
Vphase = 0.95
etaOv = 0.95
V = Vphase * etaOv * (g2sw - 1)/(g2sw + 1)

#Fid
Feff = 1/2 * (1 + V) * (p10 + p01)/(p10 + p01 + p11) # Fidelity of one mode
F = Feff**2 # Fidelity of the final double rail mode where we have one photon at one side and one photon at the other in polaristion entanglement. This could be put as the fidelity of the link. And witht he rate calculate the time. Amaze, amaze, amaze. STILL NEED TO THINK IF WE CATCH THE TWO PHOTON ERRORS AND IF IT SHOULD BE SQUARED

[ax3.plot(1e06*t_out, f*100) for f in F]
ax3.set(ylim = (0, 100), ylabel = 'Fidelity [%]')

plt.xlabel('Total storage time ($\mu$s)')
plt.tight_layout()
plt.show()

