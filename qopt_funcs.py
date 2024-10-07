import numpy as np
import numpy.linalg as LA
from scipy.linalg import block_diag

# This file contains the functions needed to compute the key rate bounds for Continuous-Variable QKD
# and for BB84 protocol (DVQKD). Also here are stored the state-of-the-art parameters used for the simulation
# REFERENCES: (T) https://arxiv.org/abs/quant-ph/0407149 ; (W) https://arxiv.org/abs/1110.3234

Z = np.array([[1,0],[0,-1]])     # Z Pauli matrix
I2 = np.array([[1,0],[0,1]])     # identity
om = np.array([[0,1],[-1,0]])    # symplectic form
Pi_q = np.array([[1,0],[0,0]])   # projector over q

T_A_dict = {'homodyne': 1, 'heterodyne': 0.5}
ixs_dict = {'Alice': [0,1], 'Bob': [2,3]}

cov_vacuum = block_diag(I2,I2)
Omega = block_diag(om, om)

def squeeze_symplectic(param):
    return np.block([[np.dot(np.cosh(param), I2), np.dot(np.sinh(param), Z)], [np.dot(np.sinh(param), Z), np.dot(np.cosh(param), I2)]])

def beamspl_symplectic(transm):   # np.kron returns tensor product
    return np.kron(np.array([[np.sqrt(transm), np.sqrt(1-transm)], [np.sqrt(1-transm), np.sqrt(transm)]]), I2) 

def apply_symplectic_transform(S, cov):
    return LA.multi_dot([S,cov,S.T])

def squeezed_cov(param):
    return apply_symplectic_transform(squeeze_symplectic(param), cov_vacuum)

def g(x):     # following notation in (W) ; different notations in other papers (eg: Pirandola; Eisert, Holevo1999)
    if x>1:
        return (x+1)/2 * np.log2((x+1)/2) - (x-1)/2 * np.log2((x-1)/2)
    elif np.isclose(x,1,rtol=1e-03):
        return 0
    else:
        print('Error: symplectic eigvals are always >= 1.')
        return None

def binary_entropy(x):    # often called h(x) , but again notation is not unanimous (h(x) may identify g(x) defined above)
    if np.isclose(x,0) or np.isclose(x,1):
        return 0
    elif 0<=x<=1:
        return -x*np.log2(x)-(1-x)*np.log2(1-x)
    else:
        return None

def transmittance_v_distance(d):
    alpha = state_of_the_art_params.alpha
    return 10**(-alpha/10 * d)    # 1/10 factor in the exponent due to dB def: https://en.wikipedia.org/wiki/Decibel

def rE_noisy(T, eps):                 # (T)
    return np.arccosh(1+eps*T/(1-T))

def conditional_cov_mtx(V, detection_mode = 'homodyne'):
    # Schur's complement: V = covariance matrix of Alice and Bob's states. reference for theory: Weedbrock etal "Gaussian quantum info"
    # A is the a priori covariance matrix of the party who decides the quadrature (ie the one who measures): Alice if direct, Bob if reverse
    A = V[0:2, 0:2]
    B = V[2:4, 2:4]
    C = V[0:2, 2:4]
    if detection_mode == 'homodyne':
        pseu_inv = LA.pinv( LA.multi_dot([Pi_q, A, Pi_q]) )
    return B - LA.multi_dot([C, pseu_inv, C.T])      # the returned mtx is the conditional cov. matrix of the other party

def mutual_information(V, Vb_alpha):    # V --> full 4x4 cov. mtx ;  Vb_alpha --> 2x2 conditional cov. mtx of Bob
    B = V[2:4, 2:4]                     # 1st 2 cols --> prepare party; 2nd 2 cols --> measure party
    first_term = B[0,0]/Vb_alpha[0,0]
    second_term = B[1,1]/Vb_alpha[1,1]
    return 0.5 * ( np.log2(first_term) + np.log2(second_term) )

def holevo_bound(gamma):
    if gamma.shape[0]==4:
        Omega = block_diag(om, om)
    elif gamma.shape[0]==2:
        Omega = om
    else:
        print('Error: input expected to be 2x2 or 4x4 array.')
        return
    symplectic_eigvals = LA.eigvals( 1j * np.dot(Omega, gamma) )
    g_nus = [g(np.real(nu)) for nu in symplectic_eigvals if np.real(nu) > 0]  # np.real to discard infinitesimal imag. parts, >0 to select positive eigvals
    return sum(g_nus)

def entangling_cloner_covariance_mtx(r_E, r_A, T, T_A):
    cov_sq = squeezed_cov(r_A/2)
    cov_sq_w_vac = block_diag(I2, cov_sq)
    S_bsA = block_diag(beamspl_symplectic(T_A),I2)
    cov_sq_bs = apply_symplectic_transform(S_bsA,cov_sq_w_vac)[2:,2:]  # removing the vacuum mode
    cov_w_Eve = block_diag(cov_sq_bs, squeezed_cov(r_E/2))
    S_bsE = block_diag(I2, beamspl_symplectic(T), I2)
    cov_final = apply_symplectic_transform(S_bsE, cov_w_Eve)
    return cov_final

def CV_keyrate(r_A, T, eps, detection_mode='homodyne', reconciliation='reverse'):
    r_E = rE_noisy(T, eps)
    T_A = T_A_dict[detection_mode]
    cov_final = entangling_cloner_covariance_mtx(r_E, r_A, T, T_A)
    if reconciliation == 'direct':
        ixs = ixs_dict['Alice'] + ixs_dict['Bob']
    elif reconciliation == 'reverse':
        ixs = ixs_dict['Bob'] + ixs_dict['Alice']
    ixgrid = np.ix_(ixs, ixs)
    sigma_AB = cov_final[ixgrid]    # at this point A = Alice/Bob depending if reconciliation is direct/reverse
    sigma_AB_beta = conditional_cov_mtx(sigma_AB, detection_mode=detection_mode)
    info_AB = mutual_information(sigma_AB, sigma_AB_beta)
    key_rate = info_AB - holevo_bound(sigma_AB) + holevo_bound(sigma_AB_beta)
    return np.real( key_rate )

def bisection_solver(f, x1, x2, rel_tol=0.000001):
    # a basic bisection algo: it works properly with pieces of functions that have a single zero in the (x1,x2) interval
    y1 = f(x1)
    y2 = f(x2)
    while np.abs(x2-x1) > rel_tol*x1:
        xm = (x1+x2)/2
        ym = f(xm)
        if y1*ym < 0:
            x2=xm
            y2=ym
        elif y2*ym < 0:
            x1=xm
            y1=ym
        else:
            print('No zeros found. Maybe bad starting points?')
            return
    return xm

def DV_keyrate(q, p_signal, p_darkcount):           # number of bits per pulse detected
    q_tilde = (0.5*p_darkcount + p_signal*q)/(p_signal + p_darkcount)
    return 1 - 2 * binary_entropy(q_tilde)

def hybrid_keyrate_bitpersec(pars, d, d_hybrid):           # in bit/s! assuming homodyne detection and reverse reconciliation
    T = transmittance_v_distance(d)
    if d <= d_hybrid:   # use CV
        eps_A = pars.eps_B/pars.eta_det_CV/T
        rate_per_pulse = CV_keyrate(pars.r_A, T, eps_A, detection_mode='homodyne', reconciliation='reverse') # bits per state
        return pars.freq * pars.eta_source_CV * pars.eta_det_CV * rate_per_pulse
    if d > d_hybrid:    # use DV
        rate_per_pulse = DV_keyrate(pars.q, pars.eta_source_DV * pars.eta_det_DV * T, pars.p_darkcount) # bits per state
        return pars.eta_source_DV * pars.eta_det_DV * T * pars.freq * rate_per_pulse

########## It is convenient to define an object containing all the params needed to compute the rates ##########

class param_set(object):
    def __init__(self):
        self.alpha = None          # the exponential decaying factor in T(d); units: dB/km
        self.freq = None           # source repetition rate [Hz]: same for CV and DV
        # CV-specific
        self.eps_B = None          # excess noise on Bob's side
        self.eps_critical = None   # approx estimate for critical value for excess noise (see Navascues, Acin)
        self.eta_source_CV = None  # (see raja's mail 26th jan) im adding it just for completeness
        self.eta_det_CV = None     # detector efficiency
        self.T_A = None            # homodyne: 1; heterodyne: 0.5
        self.r_A = None            # squeezing parameter (numerically gives results pretty close to r_A=\infty approx)
        # DV-specific
        self.q = None              # the QBER
        self.eta_source_DV = None  # source efficiency
        self.eta_det_DV = None     # detector efficiency
        self.R_dark = None         # dark count rate: 100 Hz
        self.deltat_det = None     # time gate duration: 100 ps
        self.p_darkcount = None    # probability of having a dark count (per pulse)

    def crossover_distance(self):  # to find the crossover distance beyond which CV rate > DV rate
        f = lambda d: hybrid_keyrate_bitpersec(state_of_the_art_params, d, 0) - hybrid_keyrate_bitpersec(state_of_the_art_params, d, 1000)
        return bisection_solver(f, 0.001, 1000)


state_of_the_art_params = param_set()
state_of_the_art_params.alpha = 0.18          # the exponential decaying factor in T(d); units: dB/km
state_of_the_art_params.freq = 1E9          # source repetition rate [Hz]: same for CV and DV (source: meeting w Luis)
# CV-specific
state_of_the_art_params.eps_B = 0.005          # excess noise on Bob's side ### VALUE UPDATED FROM 0.1
state_of_the_art_params.eps_critical = 0.75   # approx estimate for critical value for excess noise (see Navascues, Acin)
state_of_the_art_params.eta_source_CV = 1     # (see raja's mail 26th jan) im adding it just for completeness
state_of_the_art_params.eta_det_CV = 0.8      # detector efficiency (luis: eta ranges from 0.2 to 0.8); would be interesting to sweep it
state_of_the_art_params.T_A = 1               # homodyne: 1; heterodyne: 0.5
state_of_the_art_params.r_A = 10              # squeezing parameter (numerically gives results pretty close to r_A=\infty approx)
# DV-specific
state_of_the_art_params.q = 0.01                       # the QBER
state_of_the_art_params.eta_source_DV = 0.1            # source efficiency
state_of_the_art_params.eta_det_DV = 0.95              # detector efficiency
state_of_the_art_params.R_dark = 100                   # dark count rate: 100 Hz
state_of_the_art_params.deltat_det = 100.E-12          # time gate duration: 100 ps
state_of_the_art_params.p_darkcount = state_of_the_art_params.R_dark*state_of_the_art_params.deltat_det    # probability of having a dark count (per pulse)
