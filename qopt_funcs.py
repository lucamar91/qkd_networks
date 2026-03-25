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

T_A_dict = {'homodyne': 1., 'heterodyne': 0.5}    # Alice's measures in EB scheme. Corresponds to squeezed/coherent state P&M protocols respectively. change strings accordingly?
ixs_dict = {'Alice': [0,1], 'Bob': [2,3]}           # DO WE REALLY NEED THIS

cov_vacuum = block_diag(I2,I2)
# Omega = block_diag(om, om)

def squeeze_symplectic(param):     # this is correct according to convention in (T-->V=cosh r), but it's not the sympl transform a name suggests: it's the TMSS cov mtx
    return np.block([[np.dot(np.cosh(param), I2), np.dot(np.sinh(param), Z)], [np.dot(np.sinh(param), Z), np.dot(np.cosh(param), I2)]])

def beamspl_symplectic(transm):   # now THIS is a simplectic transform
    return np.kron(np.array([[np.sqrt(transm), np.sqrt(1-transm)], [np.sqrt(1-transm), np.sqrt(transm)]]), I2) # np.kron returns tensor product

def apply_symplectic_transform(S, cov):
    return LA.multi_dot([S,cov,S.T])

def squeezed_cov(param):
    return apply_symplectic_transform(squeeze_symplectic(param), cov_vacuum)

def g(xs, atol=1e-04):     # following notation in (W) ; different notations in other papers (eg: Pirandola; Eisert, Holevo1999)
    # to allow vector inputs but also return outputs with the same (scalar or array) type as the input
    is_scalar = np.isscalar(xs)     # BUT float casting otherwise int inputs --> truncated results
    if is_scalar:
        xs = np.array([xs]) 
    xs = np.asarray(xs, dtype=float)
    gs = np.zeros_like(xs)
    # now to manage possible (small) numerical errors:
    for i, x in enumerate(xs):
        if x < 1 - atol:        # if symplectic eigvals are less than 1 outside a certain small tolerance for numerical errors, raise an error 
            raise ValueError(f"Unphysical (less than 1 with tolerance {atol}) symplectic eigenvalue detected: nu = {x.min()}.")
        elif x < 1 + atol:      # if symplectic eigvals are close to 1 within the small tolerance, set them to the limit of g(x) for x--> 1 = 0
            gs[i] = 0
        else:                   # otherwise, usual formula for VN entropy of thermal states
            gs[i] = (x+1)/2. * np.log2((x+1)/2.) - (x-1)/2. * np.log2((x-1)/2.)
    return gs[0] if is_scalar else gs

def binary_entropy(x):    # often called h(x), but again notation is not unanimous (h(x) may identify g(x) defined above)
    if np.isclose(x,0) or np.isclose(x,1):
        return 0
    elif 0<=x<=1:
        return -x*np.log2(x)-(1-x)*np.log2(1-x)
    else:
        return None

def transmittance_v_distance(d):
    alpha = state_of_the_art_params.alpha
    return 10**(-alpha/10 * d)    # 1/10 factor in the exponent due to dB def: https://en.wikipedia.org/wiki/Decibel


def conditional_cov_mtx(cov_mtx, detection_mode = 'homodyne', reconciliation = 'reverse'):    # QUESTA è GIUSTA, MA NON PUO ESSERE USATA SULLA entangling_cloner_covariance_mtx  
    # Schur's complement: V = covariance matrix of Alice and Bob's states. reference for theory: (W); Pirandola etal, ""
    A, B, C = cov_mtx[:2,:2], cov_mtx[2:,2:], cov_mtx[:2,2:]

    # X is the a priori cov. mtx of the party sharing the measurement outcomes: Alice if direct reconciliation, Bob if reverse. Y refers to the other party
    if reconciliation == 'direct':
        X, Y = A, B
    elif reconciliation == 'reverse':
        X, Y = B, A
        C = C.T    # following from the "swap" between Alice and Bob (actually C is often symmetric)
    else:
        raise ValueError("Invalid detection mode. Expected 'direct' or 'reverse'.")
    print(cov_mtx)#################################
    if detection_mode == 'homodyne':
        pseu_inv = LA.pinv( LA.multi_dot([Pi_q, X, Pi_q]) )
    elif detection_mode == 'heterodyne':
        pseu_inv = LA.inv( X + I2 )    # WARNING: T=1 breaks the SVD in the pseudoinverse
    else:
        raise ValueError("Invalid detection mode. Expected 'homodyne' or 'heterodyne'.")
    return Y - LA.multi_dot([C, pseu_inv, C.T])      # the returned mtx is the conditional cov. matrix of the other party

def mutual_information(V, Vb_alpha, detection_mode='homodyne'):    # V --> full 4x4 cov. mtx ;  Vb_alpha --> 2x2 conditional cov. mtx of Bob
    # ANCHE QUI,CAMBIARE NOMI, CHE SI RIFERISCONO A SIMBOLI PER CASO DIRECT REC
    B = V[2:4, 2:4]                     # A shares info about the measurements, B adapts its key accordingly
    if detection_mode=='homodyne':
        first_term = B[0,0]/Vb_alpha[0,0]
        second_term = B[1,1]/Vb_alpha[1,1]
        return 0.5 * ( np.log2(first_term) + np.log2(second_term) )
    elif detection_mode=='heterodyne':
        first_term = (B[0,0] + 1)/(Vb_alpha[0,0] + 1)
        second_term = (B[1,1] + 1)/(Vb_alpha[1,1] + 1)
        return np.log2(first_term) + np.log2(second_term)
    else:
        raise ValueError("Invalid detection mode. Expected 'homodyne' or 'heterodyne'.")

def mutual_information_zhang(V, T, epsilon, detection_mode='homodyne'):
    chi_line = (1 - T) / T + epsilon
    if detection_mode == 'homodyne':
        chi_tot = chi_line
        I_AB = 0.5 * np.log2((V + chi_tot) / (1 + chi_tot))
    elif detection_mode == 'heterodyne':
        chi_tot = chi_line + 1
        I_AB = np.log2((T*(V + chi_tot) + 1) / (T*(1 + chi_tot) + 1))
    else:
        raise ValueError("Invalid detection mode.")
    return I_AB

def symplectic_eigvals(gamma):
    # we will only diagonalize 2x2 or 4x4 cov. mtxs, so we'll only be considering those cases
    if gamma.shape[0]==4:
        Omega = block_diag(om, om)
    elif gamma.shape[0]==2:
        Omega = om
    else:
        raise ValueError("Error: input expected to be 2x2 or 4x4 array.")   
    symplectic_eigvals = [np.real(nu) for nu in LA.eigvals( 1j * np.dot(Omega, gamma) ) if np.real(nu) > 0] # np.real to discard infinitesimal imag. parts, >0 to select positive eigvals
    return symplectic_eigvals

def rE_noisy(T, eps):                 # (T)
    return np.arccosh(1+eps*T/(1-T))

def TMSS_through_lossy_noisy_channel_cov_mtx(r, T, eps):
    cov_mtx = np.block([[np.cosh(r)*I2, np.sqrt(T)*np.sinh(r)*Z], [np.sqrt(T)*np.sinh(r)*Z, (T*np.cosh(r) + 1 - T + eps*T)*I2]])
    return cov_mtx


def CV_keyrate_multi_protocol(r_A, T, eps, alice_detection_mode = 'homodyne', bob_detection_mode='homodyne', reconciliation='reverse'):        # DEBUGGING
    sigma_AB = TMSS_through_lossy_noisy_channel_cov_mtx(r_A, T, eps)
    if reconciliation == 'direct':
        detection_mode = alice_detection_mode
    elif reconciliation == 'reverse':
        detection_mode = bob_detection_mode
    else:
        raise ValueError("Invalid detection mode. Expected 'direct' or 'reverse'.")
    sigma_AB_cond = conditional_cov_mtx(sigma_AB, detection_mode=detection_mode, reconciliation=reconciliation)
    print("sigma_AB_cond=", sigma_AB_cond)#################################
    I_AB = mutual_information(sigma_AB, sigma_AB_cond, detection_mode=detection_mode)

    nus_12 = symplectic_eigvals(sigma_AB)
    S_E = np.sum(np.array([g(nu) for nu in nus_12]))       # Eve's entropy before the measurement
    nu_prime = symplectic_eigvals(sigma_AB_cond)
    print("nu1, nu2, nu_prime=", nus_12, nu_prime)
    S_E_cond = np.sum(np.array([g(nu) for nu in nu_prime])) # Eve's entropy after the measurement 
    holevo_bound = S_E - S_E_cond
    
    key_rate = I_AB - holevo_bound
    return np.real( key_rate )



def entangling_cloner_covariance_mtx(eps, r_A, T, T_A):
    r_E = rE_noisy(T, eps)       ############# MI DA ANCORA ERRORE CHIARAMENTE SE T=1 PER LA DIVISIONE PER T-1 CHE PERò NON è ESSENZIALE
    cov_sq = squeezed_cov(r_A/2)
    cov_sq_w_vac = block_diag(I2, cov_sq)
    S_bsA = block_diag(beamspl_symplectic(T_A),I2)
    cov_sq_bs = apply_symplectic_transform(S_bsA, cov_sq_w_vac)[2:,2:]  # removing the vacuum mode
    cov_w_Eve = block_diag(cov_sq_bs, squeezed_cov(r_E/2))
    S_bsE = block_diag(I2, beamspl_symplectic(T), I2)
    cov_final = apply_symplectic_transform(S_bsE, cov_w_Eve)
    return cov_final

def CV_keyrate(r_A, T, eps, alice_detection_mode = 'homodyne', detection_mode='homodyne', reconciliation='reverse'):
    # r_E = rE_noisy(T, eps)
    
    T_A = T_A_dict[alice_detection_mode]
    cov_final = entangling_cloner_covariance_mtx(eps, r_A, T, T_A)
    sigma_AB = cov_final[:4, :4]    # selecting only Alice and Bob's modes (in this order) from the final covariance matrix of the whole system (Alice, Bob, Eve)
    sigma_AB_cond = conditional_cov_mtx(sigma_AB, detection_mode=detection_mode, reconciliation=reconciliation)
    I_AB = mutual_information(sigma_AB, sigma_AB_cond, detection_mode=detection_mode)
    # I_AB = mutual_information_zhang(sigma_AB[0,0], T, eps, detection_mode=detection_mode) ###################################

    nus_12 = symplectic_eigvals(sigma_AB)
    S_E = np.sum(np.array([g(nu) for nu in nus_12]))       # Eve's entropy before the measurement
    nu_prime = symplectic_eigvals(sigma_AB_cond)
    S_E_cond = np.sum(np.array([g(nu) for nu in nu_prime])) # Eve's entropy after the measurement 
    holevo_bound = S_E - S_E_cond
    
    key_rate = I_AB - holevo_bound
    # PRINT QUI PER FARE CHECK 
    return np.real( key_rate )

def bisection_solver(f, x1, x2, rel_tol=0.000001):
    # basic bisection algo: it works properly with pieces of functions that have a single zero in the (x1,x2) interval
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

def hybrid_keyrate_bitpersec(pars, d, d_hybrid):    # in bit/s! assuming homodyne detection and reverse reconciliation
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
        self.alpha = None          # the exponential decaying factor in T(d) [dB/km]
        self.freq = None           # source repetition rate [Hz]: same for CV and DV
        # CV-specific
        self.eps_B = None          # excess noise on Bob's side [SNU] (Shot Noise Units)
        self.eps_critical = None   # approx estimate for critical value for excess noise (see Navascues, Acin)
        self.eta_source_CV = None  # source efficiency, added for completeness, always set to 1
        self.eta_det_CV = None     # detector efficiency
        self.T_A = None            # homodyne detection: T_A=1; heterodyne det.: T_A=0.5
        self.r_A = None            # squeezing parameter (must be less than 10 approx. to avoid numerical errors)
        # DV-specific
        self.q = None              # the QBER
        self.eta_source_DV = None  # source efficiency
        self.eta_det_DV = None     # detector efficiency
        self.R_dark = None         # dark count rate [Hz]
        self.deltat_det = None     # time gate duration: [s]
        self.p_darkcount = None    # probability of having a dark count (per pulse)

    def crossover_distance(self):  # to find the crossover distance beyond which CV rate > DV rate
        f = lambda d: hybrid_keyrate_bitpersec(state_of_the_art_params, d, 0) - hybrid_keyrate_bitpersec(state_of_the_art_params, d, 1000)
        return bisection_solver(f, 0.001, 1000)

# Values from table I
state_of_the_art_params = param_set()
state_of_the_art_params.alpha = 0.18
state_of_the_art_params.freq = 1E9
# CV-specific
state_of_the_art_params.eps_B = 0.005
state_of_the_art_params.eta_source_CV = 1
state_of_the_art_params.eta_det_CV = 0.8
state_of_the_art_params.T_A = 1
state_of_the_art_params.r_A = 10
# DV-specific
state_of_the_art_params.q = 0.01
state_of_the_art_params.eta_source_DV = 0.1
state_of_the_art_params.eta_det_DV = 0.95
state_of_the_art_params.R_dark = 100
state_of_the_art_params.deltat_det = 100.E-12
state_of_the_art_params.p_darkcount = state_of_the_art_params.R_dark*state_of_the_art_params.deltat_det
