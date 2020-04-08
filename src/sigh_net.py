import pandas as pd
import os
import click
from brian2 import *
import pickle


def create_ns():
    '''
    Initialize parameters and store in dicts
    to be passed as namespace to NeuronGroups and SynapseGroups
    :return ns: namespace for sigh_net model
    '''
    ns = {
        "Cm": 21.0*pF,
        "g_na": 28 * nsiemens,
        "g_nap": 3.3 * nsiemens,
        "g_k": 11.2 * nsiemens,
        "g_cas": 0.5 * nsiemens,
        "gsyn_max_val" : 70*nsiemens,
        "E_na": 50.0*mV,
        "E_k": -85.0*mV,
        "E_l": -59.0*mV,
        "E_ca": 0.0*mV,
        "E_syn": 0.0*mV,
        'vm' : -34. * mV,
        'vn' : -29. * mV,
        'vmp' : -40. * mV,
        'vh' : -48. * mV,
        'sm' : -5. * mV,
        'sn' : -4. * mV,
        'smp' : -6. * mV,
        'sh' : 5. * mV,
        "tau_syn" : 15 * ms,
        "Qs" : 15* mV,
        "ko" : 0.4,
        "II": 1.0*umolar,
        "k1" : 0.27,
        "k2" : 2.5,
        "k3" : 1.*Hz,
        "n1": 2,
        "Ct": 1.25*umolar,
        "fi": 0.006/pliter,
        "L": 0.35*pliter/second,
        "P": 2.9e4*pliter/second,
        "kI": 1.0*umolar,
        "Ve": 400e-6*pmole/second,
        "ke": 0.2 * umolar,
        "A": 0.5/umolar/second,
        "ka": 0.4*umolar,
        "kd": 0.4*umolar,
        "sigma": 0.185,
        "ssyn": -3.0*mV,
        "fo" : 0.0 * umolar,
        "x1" : 0.50 * umolar,
        "x2" : 0.51 * umolar,
    }
    return(ns)


def set_ics(neuron_group,ca_source):
    '''
    Inplace assignment of initial conditions of state variables
    :param neuron_group:
    :return:
    '''
    N = neuron_group.N
    neuron_group.v = -58 * mV + np.random.randn(N) * mV
    neuron_group.h = 0.3
    neuron_group.n = 0.7
    neuron_group.h_cas = 0.02
    neuron_group.m_cas = 0.05
    neuron_group.Ca = .6 * umolar
    ca_source.C = 0.3 * umolar
    ca_source.l = 0.1


def gen_eqs():
    '''
    Generate set of equations for the sighnet simulation
    :return: neuron_eqs, syn_eqs, ca_eqs
    '''
    neuron_eqs = '''
        # ============== #
        # Currents
        # ============== #

        I_na =    g_na * (m_inf**3) * (1-n) * (v-E_na) : amp
        I_k =     g_k * (n**4) * (v-E_k) : amp
        I_nap =   g_nap * m_infp * h *(v-E_na) : amp
        I_cas =   g_cas * m_cas**2 * h_cas * (v-E_ca) : amp
        I_syn =   g_syn * (v - E_syn) : amp
        I_l =     g_l * (v - E_l) : amp


        m_inf  = 1 / (1 + exp((v - vm)/sm)) : 1
        m_infp = 1 / (1 + exp((v - vmp)/smp)) : 1
        n_inf  = 1 / (1 + exp((v - vn)/sn)) : 1
        h_inf  = 1 / (1 + exp((v - vh)/sh)) : 1
        h_casinf  = 1 / (1 + exp(420/umolar *(Ca - x2))) : 1
        m_casinf  = 1 / (1 + exp(-360/umolar *(Ca - x1))) : 1

        # ============== #
        # Steady state  
        # ============== #

        C_mod = C*C_bool(t) : mmolar
        dCa/dt = k3*(fo + (ko+k1)*l*C_mod + (k2)*umolar * m_infp * h  - Ca) : mmolar

        # ============== #
        # Time Constants
        # ============== #

        tau_n = 10*ms     / (cosh((v - vn)/(2*sn))) : second
        tau_h = 10*second / (cosh((v - vh)/(2*sh))) : second
        tau_hcas = 50*ms +   5.25 * second / (1+exp(-250/umolar* (Ca-x2))) : second 
        tau_mcas = 2*ms  +  0.134 * second / (1+exp(-400/umolar* (Ca-x1))) : second

        # ============== #
        # derivatives
        # ============== #

        dn/dt = (n_inf - n)/tau_n : 1
        dh/dt = (h_inf - h)/tau_h : 1
        dv/dt = 1./Cm * -(I_na + I_k + I_nap + I_syn + I_l + I_cas) : volt
        dm_cas/dt = (m_casinf - m_cas) / tau_mcas : 1 
        dh_cas/dt = (h_casinf - h_cas) / tau_hcas : 1 

        g_syn : siemens
        g_l : siemens (constant)
        C : mmolar (linked)
        l : 1 (linked)
        '''

    ca_eqs = '''
        J_ER_in = (L + P * ((II * C * l)/((II+kI)*(C+ka)))**3) * ((Ct-C)/sigma - C) : mole/second
        J_ER_out = Ve*(C**2)/(ke**2+C**2) : mole/second
        dC/dt = ko * fi * (J_ER_in - J_ER_out) : mmolar
        dl/dt = ko * A * (kd -(C+kd)*l) : 1
        '''

    syn_eqs = '''
        ds/dt = ((1-s) * msyninf - s)/tau_syn : 1 (clock-driven)
        msyninf = 1 / (1+exp((v_pre - Qs)/ssyn)) : 1
        g_syn_post = gsyn_max*s : siemens (summed)
        gsyn_max = gsyn_max_t(t) : siemens
        '''
    return (neuron_eqs, syn_eqs, ca_eqs)


def run_sigh_net(edge_csv,f_save,ns):
    '''
    Run the network and save dictionaries to a pickled dict
    :param f_save: File to save the states to
    :param edge_csv: Load in an edgelist csv
    :return:
    '''
    ## ========================= ##
    ## Set run parameters
    ## ========================= ##
    runtime = 1000 * second
    disconnect_time = 200 * second  # how much time to run without synaptic connections
    defaultclock.dt = 0.25 * ms
    is_parallel = True # Set to false to skip optimization for SCRI HPC
    p_TS = 0.1  # Percent of neurons tonic spiking
    p_Q = 0.3  # Percent of neurons quiescent
    p_B = (1-p_TS-p_Q)  # Percent of neurons bursting

    if is_parallel:
        set_device('cpp_standalone', directory=f'{np.random.random():0.5f}', clean=True)
        prefs.devices.cpp_standalone.openmp_threads = int(os.environ['OMP_NUM_THREADS'])

    # Format the output filename and check for overwrite
    base_name = os.path.splitext(os.path.split(f_save)[1])[0] +'.dat'
    f_save = os.path.join(os.path.split(f_save)[0], base_name)
    print(f'Saving to {f_save}')
    edge_list = pd.read_csv(edge_csv).values
    if not os.path.isdir(os.path.split(f_save)[0]):
        print("Save destination does not exist. Exiting")
        return (-1)
    if os.path.isfile(f_save):
        print('Target save file exists. Exiting.')
        return (-1)

    N = np.max(edge_list[:,0])+1 # Compute the number of neurons from the edge list
    N_TS = np.round(N * p_TS).astype('int')
    N_B = np.round(N * p_B).astype('int')
    N_Q = np.round(N * p_Q).astype('int')

    neuron_eqs, syn_eqs, ca_eqs = gen_eqs()

    # Create Brian2 Groups
    ca_source = NeuronGroup(1, ca_eqs, name='glia', namespace=ns, method='rk4')
    neurons = NeuronGroup(N, neuron_eqs, name='prebot_pop', namespace=ns, method='rk4', threshold='v>-20*mV')
    Q_pop = neurons[:N_Q]
    B_pop = neurons[N_Q:N_Q + N_B]
    TS_pop = neurons[N_B + N_Q:]
    synapses = Synapses(neurons, neurons, syn_eqs, namespace=ns, method='rk4')

    # Mirror external Calcium source into neurons
    neurons.l = linked_var(ca_source, 'l')
    neurons.C = linked_var(ca_source, 'C')

    # Set up synapses
    synapses.connect(i=edge_list[:, 0], j=edge_list[:, 1])

    # Initialize
    set_ics(neurons, ca_source)

    # Set maximum leak conductance for each cell type
    TS_pop.g_l = 1.0 * nsiemens + np.random.uniform(-0.5, 0.5, N_TS) * nsiemens
    B_pop.g_l = 3.0 * nsiemens + +np.random.uniform(-0.5, 0.5, N_B) * nsiemens
    Q_pop.g_l = 4.5 * nsiemens + np.random.uniform(-0.5, 0.5, N_Q) * nsiemens

    # Create TimedArrays to disconnect synapses and external calcium after runtime
    gsyn_max_t = TimedArray([ns['gsyn_max_val'], 0 * nsiemens], dt=runtime)
    C_bool = TimedArray([1, 0], dt=runtime)

    M = StateMonitor(neurons, variables=['v', 'Ca'], record=np.arange(0, N, N / 15), dt=.25 * ms, name='neuron_state')
    Mc = StateMonitor(ca_source, variables=['C'], record=True, dt=5 * ms, name='glia_state')
    S = SpikeMonitor(neurons, name='spikes')
    R = PopulationRateMonitor(neurons, name='pop_rate')
    net = Network(collect())

    # Run connected network for runtime, then additional time in disconnected mode
    net.run(runtime + disconnect_time, report='text', report_period=30 * second)

    # save
    states = net.get_states()
    states['eqs'] = [neuron_eqs, ca_eqs, syn_eqs]
    states['params'] = ns
    with open(f_save, 'wb') as fid:
        pickle.dump(states, fid)


    device.delete()
    print('Simulation Complete!')
    return (0)


@click.command()
@click.argument('edgelist')
@click.argument('fsave')
@click.option('--ko',default=0.7,type=float)
def main(edgelist,fsave,ko):
    """
    Run the sigh net from the graph in edgelist and save to fsave
    """
    ns = create_ns()
    ns['ko'] = ko # set the ko value for this run
    run_sigh_net(fsave,edgelist, ns)

    return(0)

if __name__ == '__main__':
    main()
