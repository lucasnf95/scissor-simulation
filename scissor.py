
import qutip
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import secretkeyrate as skr

# Initial functions

def multimode_operator(operator, amount_of_modes, mode):
    '''
    Apply a tensor product between a certain operator and identities.
    Parameters
        operator: Qobj
            Quantum operator
        amount_of_modes: int
            Total amount of modes in the system
        mode: int
            mode in which operator should be
    Return
        out_op: Qobj
            Final quantum operator NOT NORMALIZED
    '''
    dimension = operator.dims[0][0]
    unity = qutip.qeye(dimension)
    op_list = []
    for n in range(amount_of_modes):
        if n == mode:
            op_list.append(operator)
        else:
            op_list.append(unity)
    out_op = op_list[0]
    for n in range(amount_of_modes - 1):
        out_op = qutip.tensor(out_op, op_list[n+1])
    return out_op

def beamsplitter(dimension, amount_of_modes = 1, modes= (0,0), T=1/2):
    '''
    Generate beam-splitter unitary operator for specified dimension and transmission
    Parameters
        dimension: int
            Fock state dimension
        amount_of_modes: int
            Amount of modes on the input with the highest amount of modes
        modes: tuple
            Modes on which beam-splitter transformation should be applied
        T: float
            Beam-splitter transmission
    Return: Qobj
        Beam splitter unitary operator
    '''
    a_operator = qutip.destroy(dimension)
    a1 = multimode_operator(a_operator, amount_of_modes, modes[0])
    a2 = multimode_operator(a_operator, amount_of_modes, modes[1])
#     unity = qutip.qeye(dimension)
#     a1_list = []
#     a2_list = []
#     for n in range(amount_of_modes):
#         if n == modes[0]:
#             a1_list.append(qutip.create(dimension))
#         if n != modes[0]:
#             a1_list.append(unity)
#         if n == modes[1]:
#             a2_list.append(qutip.destroy(dimension))
#         if n != modes[1]:
#             a2_list.append(unity)
#     a1 = a1_list[0]
#     a2 = a2_list[0]
#     print('Amount of modes', amount_of_modes)
#     for n in range(amount_of_modes - 1):
#         a1 = qutip.tensor(a1, a1_list[n+1])
#         a2 = qutip.tensor(a2, a2_list[n+1])
    theta = np.arccos(np.sqrt(T))
    return (1j*theta * (qutip.tensor(a1.dag(), a2) + qutip.tensor(a1, a2.dag()))).expm()

def apply_beamsplitter(state1, state2, modes = (0,0), T=1/2, verbose = False):
    '''
    Apply beam splitter transformation to the inputs state1 and state 2
    Parameters
        state1: Qobj
            Either ket vector or density matrix
        state2: Qobj
            Either ket vector or density matrix
        modes: tuple
            Modes on which beam-splitter transformation should be applied
        T: float
            Beam splitter transmission
    Return
        out: Qobj
            Either ket vector or density matrix describing the two-mode output of the beam splitter
    '''
    # Check dimensions
    dimension = state1.dims[0][0]
    unity = qutip.qeye(dimension)
    if state2.dims[0][0] != dimension:
        print("Dimensions don't match!")
        return
    if state2.type != state1.type:
        print("Inputs are different objects")
        return
    
    # Calculate number of modes on each state
    modes1 = len(state1.dims[0])
    modes2 = len(state2.dims[1])
    amount_of_modes = max(modes1, modes2)
    
    # Generate the proper beam-splitter transformation
    bs = beamsplitter(dimension, amount_of_modes, modes, T)

    # Insert extra identity modes where needed
    extra_modes = []
    if modes1 > modes2:
        for n in range(modes1-modes2):
            state2 = qutip.tensor(state2, unity)
            extra_modes.append(modes1+modes2+n)
    if modes2 > modes1:
        for n in range(modes2 - modes1):
            state1 = qutip.tensor(state1, unity)
            extra_modes.append(modes1+n)
    
    remaining_modes = []
    if len(extra_modes) > 0:
        for n in range(2*amount_of_modes):
            if n not in extra_modes:
                remaining_modes.append(n)
    
    # Calculate global density matrix of the input state
    inputs = qutip.tensor(state1, state2)
    
    # Apply beam-splitter transformation    
    if state1.type == 'oper':
        out = (bs.dag()*inputs*bs)
        if len(remaining_modes) > 0:
            out = out.ptrace(tuple(remaining_modes))
    elif state1.type == 'ket':
        print('Input is a ket')
        out = (bs*inputs)
        if len(remaining_modes) > 0:
            out = out.ptrace(tuple(remaining_modes))
    else:
        print('Input is not recognized')
        return
    out /= np.trace(out)
    if verbose:
        print('Output modes are ', modes[0], 'and', modes1 + modes[1])
    return out    

def n_measurement(state, n):
    '''
    Calculate probability of measuring n photons in a state
    Parameters:
        state: Qobj
            Either ket vector or density matrix
        n: int
            Number state
    Return
        m: float
            Probability of measuring n photons
    '''
    dimension = state.shape[0]
    if n > dimension:
        print("Dimension of the state is too small")
        return
    if state.type == 'oper':
        prob = np.trace(state*qutip.fock_dm(dimension, n))
    if state.type == 'ket':
        m = qutip.fock(dimension, n).dag()*state
        prob = (m.dag()*m)[0][0][0]
    return prob

def measurement_operator(n, mode, amount_of_modes, dimension):
    '''
    Calculate measurement operator for a state of a certain Fock dimension when a n-photon
    measurement is applied on a certain mode.
    Parameters
            n: int
                Fock basis being measured
            mode: int
                mode in which measurement is being performed
            amount_of_modes: int
                total amount of modes in the systen
            dimension: int
                Hilbert space dimension
    Return
        meas_op: Qobj
            Tensor product measurement operator on specified mode and identities on the others
    '''
    unity = qutip.qeye(dimension)
    meas = qutip.fock_dm(dimension, n)
    meas_list = []
    for n in range(amount_of_modes):
        if n == mode:
            meas_list.append(meas)
        else:
            meas_list.append(unity)
    meas_op = meas_list[0]
    for n in range(amount_of_modes - 1):
        meas_op = qutip.tensor(meas_op, meas_list[n+1])
    return meas_op

def herald_teleportation(state, modes, dead_time = 1e-7, snr = 10):
    '''
    Calculate success rate and output density matrix when heralding teleportation measurement is obtained
    Parameters
        state: Qobj
            multi-mode state before Charlie's measurement
        modes: int
            modes in which Charlie's measurement are performed (single-photon on the first and vacuum on the second)
    Return
        prob: float
            success rate
        resulting_state: Qobj
            density matrix of state at the homodyne station
    '''
    amount_of_modes = np.shape(state.dims)[1]
    dimension = state.dims[0][0]
    if type(modes) == tuple:
        if max(modes) > amount_of_modes:
            print("There are not that many modes in the system!")
            return
#         Probabilities
#         sp = n_measurement(state.ptrace(modes[0]), 1)
#         vac = n_measurement(state.ptrace(modes[1]), 0)
#         prob = 2*sp*vac
        # Measurement operators
        sp_op = multimode_operator(qutip.fock_dm(dimension, 1), amount_of_modes, modes[0])
        vac_op = multimode_operator(qutip.fock_dm(dimension, 0), amount_of_modes, modes[1])
        ## Click operator
        click = qutip.qeye(dimension) - qutip.qeye(dimension)
        for n in range(dimension-1):
            click += qutip.fock_dm(dimension, n+1)
        click_op = multimode_operator(click, amount_of_modes, modes[0])
        
        # Probability
        measurement_operator = click_op*vac_op
        if np.trace(state) != 1:
            state /= np.trace(state)
        
        remaining_modes = []
        for n in np.arange(amount_of_modes):
            if n not in modes:
                remaining_modes.append(n)
        #remaining_state = (vac_op.dag()*sp_op.dag()*state*sp_op*vac_op)
        remaining_state = measurement_operator.dag()*state*measurement_operator
        prob = 2*np.trace(remaining_state)
        
        # Adding SSPD imperfections
        if snr != 0:
            real_prob = prob
            fake_prob = prob/snr
            prob = real_prob + fake_prob
                               
        remaining_state /= np.trace(remaining_state)
        if len(remaining_modes) == 1:
            d = remaining_modes[0]
            if snr != 0:
                resulting_state_real = remaining_state.ptrace(2) # For some reason tracing d doesn't work and I need to use numbers
                resulting_state_fake = state.ptrace(2)
                resulting_state = (real_prob/prob)*resulting_state_real + (fake_prob/prob)*resulting_state_fake
            else:
                resulting_state = remaining_state.ptrace(2)
        elif len(remaining_modes) == 2:
            if snr != 0:
                resulting_state_real = remaining_state.ptrace((0,3)) # For some reason tracing d doesn't work and I need to use numbers
                resulting_state_fake = state.ptrace((0,3))
                resulting_state = (real_prob/prob)*resulting_state_real + (fake_prob/prob)*resulting_state_fake
            else:
                resulting_state = remaining_state.ptrace((0,3))
        else:
            print('There are extra modes in the systems')
            return
        
        return prob, resulting_state
    if type(modes) == int:
        prob = n_measurement(state.ptrace(modes), 1)
        return prob
    
def quantum_fidelity(rho1, rho2):
    '''
    Fidelity between two density matrices
    Pararameters
        rho1: Qobj
            Density matrix
        rho2: Qobj
            Density matrix
    Return
        fidelity: float
            fidelity between states
    '''
    dms = rho1.sqrtm()*rho2*rho1.sqrtm()
    trace = np.trace(dms.sqrtm())
    return np.conjugate(trace)*trace

def alice_bob_interference(alpha, dimension, single_photon_efficiency = 1, alice_transmission = 1, bob_transmission = 1, Tb = 1/2):
    '''
    Generate Alice and Bob's imperfect states and interfere them on a beam-splitter.
    Parameters
        alpha: float
            Input state amplitude
        dimension: int
            Hilbert space dimension
        single_photon_efficiency: float
            Efficiency for generation of single-photon for Bob
        alice_transmission: float
            Alice's transmission efficiency to Charlie
        bob_transmission: float
            Bob's transmission efficiency to Charlie
        Tb: float
            Bob's beam-splitter transmission
    Return
        rho_alice: Qobj
            Ideal Alice's input density matrix
        charlie_bs: Qobj
            Density matrix for global state of the system after interference on Charlie
    '''
    # Lossy single-photon
    single_photon_rho = (1-single_photon_efficiency)*qutip.fock_dm(dimension, 0) + single_photon_efficiency*qutip.fock_dm(dimension, 1)

    # Bob's beam splitter
    rho_bob = apply_beamsplitter(single_photon_rho, qutip.fock_dm(dimension, 0), T = Tb)
    rho_bob_loss = apply_beamsplitter(rho_bob, qutip.fock_dm(dimension, 0), T = bob_transmission).ptrace((0, 1))
    ##  Mode 0 will interfer with Alice, mode 1 will be sent to homodyne

    # Alice's state
    alpha_eff = np.sqrt(alice_transmission)*alpha
    rho_alice = qutip.coherent_dm(dimension, alpha)
    rho_alice_eff = qutip.coherent_dm(dimension, alpha_eff)
    rho_alice_eff /= np.trace(rho_alice_eff)

    ## Calculations without losses
    # Charlie's beam splitter
    charlie_bs = apply_beamsplitter(rho_alice_eff, rho_bob_loss)
    
    return rho_alice, charlie_bs

def quantum_scissor(alpha, dimension, single_photon_efficiency = 1, alice_transmission = 1, bob_transmission = 1, Tb = 1/2, snr = 0):
    '''
    Calculates the expected result for teleportation from a quantum scissor.
    Parameters
        alpha: float
            Input state amplitude
        dimension: int
            Hilbert space dimension
        single_photon_efficiency: float
            Efficiency for generation of single-photon for Bob
        alice_transmission: float
            Alice's transmission efficiency to Charlie
        bob_transmission: float
            Bob's transmission efficiency to Charlie
        Tb: float
            Bob's beam-splitter transmission
        snr: float
            Signal-to-Noise Ratio on Charlie's SSPD
    Return
        fidelity: float
            Fidelity between input and output states
        success_rate: float
            Success rate of heralding teleportation
        Purity: float
            Purity of output state
    '''
    print_state = False
    verbose = False
    
    # Create Alice and Bob's state and interfere on a beam splitter
    rho_alice, charlie_bs = alice_bob_interference(alpha, dimension, single_photon_efficiency=single_photon_efficiency, alice_transmission=alice_transmission, bob_transmission=bob_transmission, Tb = Tb)
    
    ## Modes 0 and 1 go to single-photon detecors, mode 2 is sent to homodyne
    success_rate, output_rho = herald_teleportation(charlie_bs, (0,1), snr=snr)

    fidelity = quantum_fidelity(rho_alice, output_rho)
    
    purity = np.trace(output_rho**2)
    
    # Should I calculate what is the density matrix in the EB version?
    #rho_ab = qutip.tensor(rho_alice, output_rho)

    if verbose:
        print('Fidelity:', fidelity)
        print('Success rate:', success_rate)
    
    if print_state:
        print('Input state:', rho_alice)
        print('Output state:', output_rho)
        
    return fidelity, success_rate, purity

def eb_quantum_scissor(r, dimension, single_photon_efficiency=1, alice_transmission=1, bob_transmission=1):
    # Lossy single-photon
    single_photon_rho = (1-single_photon_efficiency)*qutip.fock_dm(dimension, 0) + single_photon_efficiency*qutip.fock_dm(dimension, 1)

    # Bob's beam splitter
    rho_bob = apply_beamsplitter(single_photon_rho, qutip.fock_dm(dimension, 0))
    rho_bob_loss = apply_beamsplitter(rho_bob, qutip.fock_dm(dimension, 0), T = bob_transmission).ptrace((0, 1))
    ##  Mode 0 will interfer with Alice, mode 1 will be sent to homodyne

    # Alice's state
    rho_alice =  skr.tms(r, 0, dimension)
    rho_alice_loss = apply_beamsplitter(rho_alice, qutip.fock_dm(dimension, 0), modes = (1,0), T = alice_transmission).ptrace((0,1))

    ## Calculations without losses
    # Charlie's beam splitter
    charlie_bs = apply_beamsplitter(rho_alice_loss, rho_bob_loss, modes=(1,0))
    ## Modes 0 and 1 go to single-photon detecors, mode 2 is sent to homodyne
    success_rate, output_rho = herald_teleportation(charlie_bs, (1,2))
    
    return output_rho
        
def fid_loss(alpha, dimension, transmission):
    '''
    Plots expected fidelity results for scissor experiment when subjected to losses on Alice, Bob and single-photon generation.
    Parameters
        alpha: float
            Input state amplitude
        transmission: list
            Transmission efficiency for the parameters
    '''
    vector_scissor = np.vectorize(quantum_scissor)
    fidelity_single_photon_loss, success_rate_single_photon_loss, purity_single_photon_loss = vector_scissor(alpha, dimension, single_photon_efficiency = transmission)#, alice_transmission = alice_transmission, bob_transmission = bob_transmission)
    fidelity_alice_loss, success_rate_alice_loss, purity_alice_loss = vector_scissor(alpha, dimension, alice_transmission = transmission)
    fidelity_bob_loss, success_rate_bob_loss, purity_bob_loss = vector_scissor(alpha, dimension, bob_transmission = transmission)
    fid_perf, _, _ = quantum_scissor(alpha, dimension)
    
    loss = [1 - t for t in transmission]

    size = (10, 8)
    plt.figure(figsize = size)
    plt.plot(loss[::-5], fidelity_alice_loss[::-5], label = 'Losses on Alice')
    plt.plot(loss[::-5], fidelity_bob_loss[::-5], label = 'Losses on Bob')
    #plt.plot(loss[::-5], fidelity_imperfect_loss[::-5], label = 'Single photons: ' + str(single_photon_efficiency))
    plt.plot(loss[::-5], fidelity_single_photon_loss[::-5], label = 'Losses on single-photon')#'Alice transmission: ' + str(alice_transmission) + '\nBob transmission: ' + str(bob_transmission))
    plt.axhline(y = fid_perf, color = 'red', linestyle = '--', linewidth = 2)
    plt.grid()
    plt.xlabel('Losses on the channel')
    plt.ylabel('Fidelity')
    plt.title('Fidelity as a function of losses for $|\\alpha| =$' + str(alpha))
    plt.legend()
    plt.show()
