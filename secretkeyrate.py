
import qutip
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import scissor

def tms(r, ph, dimension):
    z = r*np.exp(1j*ph)
    unity = qutip.qeye(dimension)
    a = qutip.destroy(dimension)
    a1 = qutip.tensor(a, unity)
    a2 = qutip.tensor(unity, a)
    sqz = qutip.squeezing(a1, a2, z)
    vac2 = qutip.tensor(qutip.fock_dm(dimension, 0), qutip.fock_dm(dimension, 0))
    return sqz.dag()*vac2*sqz

def corr(op1, op2):
    return (1/2)*(op1*op2 + op2*op1)

def covariance_matrix_2mode(quadratures, rho, V_mod = None):
    '''
    Calculate the two mode covariance matrix based on a density matrix.
    Parameters:
        quadratures: list of Qobj
            quadratures used for calculating the variance and covariances
        rho: Qobj
            density matrix of the state
        V_mod: float
            a value different than None will simulate the EB version of the QKD protocol
    Return:
        cov_ab: nparray
            covariance matrix of the two-mode state
    '''
    # Preparing quadrature operators
    if V_mod == None:
        q = quadratures[0]
        p = quadratures[1]
    else:
        factor = np.sqrt(2*V_mod/(V_mod + 2))
        q = factor*quadratures[0]
        p = factor*quadratures[1]

    dimension = q.dims[0][0]    
    unity = qutip.qeye(dimension)
    q_a = qutip.tensor(q, unity)
    p_a = qutip.tensor(p, unity)
    q_b = qutip.tensor(unity, q)
    p_b = qutip.tensor(unity, p)

    q2_a = qutip.tensor(q**2, unity)
    p2_a = qutip.tensor(p**2, unity)
    q2_b = qutip.tensor(unity, q**2)
    p2_b = qutip.tensor(unity, p**2)

    # Calculating expected values
    cov_ab = np.zeros((4,4), dtype = complex)

    alice_p = np.trace(rho*p_a)
    alice_q = np.trace(rho*q_a)

    alice_2p = np.trace(rho*p2_a)
    alice_2q = np.trace(rho*q2_a)

    var_alice_p = alice_2p - alice_p**2
    var_alice_q = alice_2q - alice_q**2

    cov_ab[0,0] = var_alice_q
    cov_ab[1,1] = var_alice_p

    bob_p = np.trace(rho*p_b)
    bob_q = np.trace(rho*q_b)

    bob_2p = np.trace(rho*p2_b)
    bob_2q = np.trace(rho*q2_b)

    var_bob_p = bob_2p - bob_p**2
    var_bob_q = bob_2q - bob_q**2

    cov_ab[2,2] = var_bob_q
    cov_ab[3,3] = var_bob_p

    alice_corr = np.trace(rho*corr(q_a,p_a))
    bob_corr = np.trace(rho*corr(q_b,p_b))

    cov_alice = alice_corr - alice_p*alice_q
    cov_bob = bob_corr - bob_p*bob_q

    cov_ab[0,1] = cov_alice
    cov_ab[1,0] = np.conjugate(cov_alice)
    cov_ab[2,3] = cov_bob
    cov_ab[3,2] = np.conjugate(cov_bob)

    cov_ab[0,2] = np.trace(rho*corr(q_a, q_b)) - alice_q*bob_q
    cov_ab[2,0] = np.conjugate(cov_ab[0,2])
    cov_ab[0,3] = np.trace(rho*corr(q_a, p_b)) - alice_q*bob_p
    cov_ab[3,0] = np.conjugate(cov_ab[0,3])

    cov_ab[1,2] = np.trace(rho*corr(p_a, q_b)) - alice_p*bob_q
    cov_ab[2,1] = np.conjugate(cov_ab[1,2])
    cov_ab[1,3] = np.trace(rho*corr(p_a, p_b)) - alice_p*bob_p
    cov_ab[3,1] = np.conjugate(cov_ab[1,3])

    '''
    for i, row in enumerate(cov_ab):
        for j, element in enumerate(row):
            if np.abs(element) < 10**(-10) and np.isreal(element) == False:
                cov_ab[i,j] = 0
    '''
    return cov_ab

def mutual_information(cov):
    '''
    Calculates the mutual information given the covariance matrix of the EB version of the protocol
    Parameters
        cov: ndarray
            Covariance matrix
    Return
        I: float
            Mutual information
    '''
    a = (1/2)*(cov[0][0] + cov[1][1])
    b = (1/2)*(cov[2][2] + cov[3][3])
    c = (1/4)*(cov[0][2] + cov[2][0] - cov[1][3] - cov[3][1])
    I = np.log2((1+a)/(1+a-c**2/(a+b)))
    return I

def sympletic_eigenvalues(cov):
    '''
    Calculates the positive sympletic eigenvalue of a single-mode or two-mode covariance matrix
    Parameters:
        cov: ndarray
            Covariance matrix
    Return:
        e or e1, e2: float
            Sympletic eigenvalues
    '''
    Omega = 1j*qutip.sigmay()
    zero2 = qutip.qeye(2)-qutip.qeye(2)
    Omega2 = np.block([[Omega, zero2],[zero2, Omega]])
    
    if len(cov) == 2:
        # Covariance matrix describes a single-mode system
        sigma = qutip.Qobj(1j*Omega*cov)
        _, e = sigma.eigenenergies()
        return e
    if len(cov) == 4:
        # Covariance matrix describes a two-mode system
        sigma = qutip.Qobj(1j*Omega2*cov)
        _, _, e1, e2 = sigma.eigenenergies()
        return e1, e2

def cov_block(cov):
    '''
    Take the blocks on 2 mode covariance matrix related to mode 1, mode 2 and correlations.
    Parameters
        cov: ndarray
            2mode covariance matrix
    Return
        block_A: ndarray
            1mode covariance matrix of first mode
        block_B: ndarray
            1mode covariance matrix of second mode
        block_corr: ndarray
            correlations between modes
    '''
    block_A = [[cov[0,0], cov[0,1]],[cov[1,0], cov[1,1]]]
    block_B = [[cov[2,2], cov[2,3]],[cov[3,2], cov[3,3]]]
    block_corr = [[cov[0,2], cov[0,3]],[cov[1,2], cov[1,3]]]
    
    return block_A, block_B, block_corr

def homodyne_measurement(cov, quadrature, mode):
    '''
    Perform a homodyne measurement in one mode of a 2mode system described by a covariance matrix.
    Parameters
        cov: ndarray
            2-mode covariance matrix
        quadrature: string 
            quadrature being measured (possibilities: 'q', 'p', 'heterodyne')
        mode: int
            mode on which measurements are being performed (possibilities: 0 or 1)
    Return
        reduced_cov: ndarray
            1-mode covariance matrix of the remaining system
    '''
    block_A, block_B, block_corr = cov_block(cov)
    
    if quadrature == 'q':
        meas = [[1,0],[0,0]]
        var = cov[0 + 2*mode][0 + 2*mode]
    elif quadrature == 'p':
        meas = [[0,0],[0,1]]
        var = cov[1 + 2*mode][1 + 2*mode]
    elif quadrature == 'heterodyne':
        meas = np.array(qutip.Qobj(block_B) + qutip.Qobj([[1,0],[0,1]]))
        var = 1
    else:
        print('Quadrature not recognized')
        return
    
    if mode == 0:
        reduced_cov = np.array(qutip.Qobj(block_A) - (1/var)*qutip.Qobj(block_corr)*qutip.Qobj(meas)*(qutip.Qobj(block_corr).conj()))
    elif mode == 1:
        reduced_cov = np.array(qutip.Qobj(block_B) - (1/var)*qutip.Qobj(block_corr)*qutip.Qobj(meas)*(qutip.Qobj(block_corr).conj()))
        
    return reduced_cov

def von_neumann_entropy(cov):
    '''
    Calculate the Von Neumann entropy of a covariance matrix in terms of the sympletic eigenvalues
    Parameters
        cov: ndarray
            Covariance matrix
    Return
        S: float
            Von Neumann entropy
    '''
    if len(cov) == 4:
        # Two-mode system
        e1, e2 = sympletic_eigenvalues(cov)
        eigenvalues = [e1, e2]
    elif len(cov) == 2:
        # Single-mode system
        e = sympletic_eigenvalues(cov)
        eigenvalues = [e]
        
    S = 0
    for e in eigenvalues:
        plus = (e+1)/2
        minus = (e-1)/2
        S += plus*np.log2(plus) - minus*np.log2(minus)
    
    return S

def holevo_information(cov, quadrature = 'heterodyne', mode = 1):
    '''
    Calculate Eve's Holevo information assuming the Gaussian optimality.
    Parameters
        cov: ndarray
            2-mode covariance matrix
        quadrature: string
            quadrature being measured (possibilities: 'q', 'p', 'heterodyne')
        mode: int
            mode on which measurements are being performed (possibilities: 0 or 1)
    Return
        chi: float
            Eve's Holevo information
    '''
    S_AB = von_neumann_entropy(cov)
    
    reduced_cov = homodyne_measurement(cov, quadrature, mode)
    S_A = von_neumann_entropy(reduced_cov)
    
    chi = S_AB - S_A
    
    return chi
