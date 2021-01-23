"""
Partially Permutational Invariant Quantum Solver (PIQS extension)

This module calculates the Liouvillian for the dynamics of ensembles
of two-level systems (TLS), consisting of two or more species, in the
presence of local and collective processes by exploiting permutational
symmetry within ensables of the same species and using the Dicke basis.
"""

# Authors: Marcello Negri
# Contact: mnegri@student.ethz.ch

import pandas as pd
import numpy as np
from functools import partial
import timeit

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

from qutip import *
from qutip import piqs

#==============================================================================#
# SINGLE SPECIES TWO LEVEL SYSTEM

def one_species_piqs_mesolve (N, initial_state, nphot=None, w0=1., wx=0.1,
                             gce=0.1, gcd=0.1, gcp=0.1,
                             ge=0.1, gd=0.1, gp=0.1,
                             wc=1, k=0.1, wp=0.01, g=1,
                             plot=False):
    # Ensemble of N TLS
    system = piqs.Dicke(N = N)
    [jx, jy, jz] = piqs.jspin(N)
    jp = piqs.jspin(N, op='+')
    jm = piqs.jspin(N, op='-')

    h_tls = w0 * jz + wx * jx
    system.hamiltonian = h_tls
    system.emission = ge
    system.dephasing = gd
    system.pumping = gp
    system.collective_emission = gce
    system.collective_dephasing = gcd
    system.collective_pumping = gcp

    L_tls = system.liouvillian()
    L_tot = L_tls
    rho0_tls = initial_state
    rho0 = rho0_tls

    if nphot:
        a = destroy(nphot)
        h_int = g * tensor(jx, a + a.dag())

        # Photonic Liouvillian
        c_ops_phot = [np.sqrt(k) * a, np.sqrt(wp) * a.dag()]
        L_phot = liouvillian(wc * a.dag()*a , c_ops_phot)

        nds = piqs.num_dicke_states(N)
        id_tls = to_super(qeye(nds))
        id_phot = to_super(qeye(nphot))

        # Define the total Liouvillian
        L_int = -1j* spre(h_int) + 1j* spost(h_int)
        L_tot = L_int + super_tensor(id_tls, L_phot) + super_tensor(L_tls, id_phot)

        # Total operators
        jz = tensor(qeye(nphot), jz)
        jp = tensor(qeye(nphot), jp)
        jm = tensor(qeye(nphot), jm)

        ground_phot = ket2dm(basis(nphot,0))
        rho0 = tensor(rho0_tls, ground_phot)

    # Time integration (use 'mesolve()'in Dicke basis Liouvillian space)
    t = np.linspace(0, 20, 1000)
    result = mesolve(L_tot, rho0, t, [], options=Options(atol=1e-15,rtol=1e-15))
    rhot = result.states

    return rhot


def one_species_qutip_mesolve (N, initial_state, nphot=None, w0=1., wx=0.1,
                             gce=0.1, gcd=0.1, gcp=0.1,
                             ge=0.1, gd=0.1, gp=0.1,
                             wc=1, k=0.1, wp=0.01, g=1,
                             plot=False):
    # Ensemble of N TLS
    [jx, jy, jz] = piqs.jspin(N, basis="uncoupled")
    jm = piqs.jspin(N, op='-', basis="uncoupled")
    jp = piqs.jspin(N, op='+', basis="uncoupled")
    h_tls = w0 * jz + wx * jx

    cops = piqs.collapse_uncoupled(N = N, emission = ge, pumping = gp, dephasing = gd,
                             collective_emission = gce, collective_pumping = gcp,
                             collective_dephasing = gcd)

    L_tls = liouvillian(h_tls, cops)
    L_tot = L_tls
    rho0_tls = initial_state
    rho0 = rho0_tls

    # Light-matter coupling parameters
    if nphot:
        a = destroy(nphot)
        h_int = g * tensor(jx, a + a.dag())

        # Photonic Liouvillian
        c_ops_phot = [np.sqrt(k) * a, np.sqrt(wp) * a.dag()]
        L_phot = liouvillian(wc * a.dag()*a , c_ops_phot)

        id = qeye(jz.dims[0])
        id_tls = to_super(id)
        id_phot = to_super(qeye(nphot))

        # Define the total Liouvillian
        L_int = -1j* spre(h_int) + 1j* spost(h_int)
        L_tot = L_int + super_tensor(id_tls, L_phot) + super_tensor(L_tls, id_phot)

        ground_phot = ket2dm(basis(nphot,0))
        rho0 = tensor(ground_phot, rho0_tls)

    t = np.linspace(0, 20, 1000)
    result = mesolve(L_tot, rho0, t, [], options=Options(atol=1e-15,rtol=1e-15))
    rhot = result.states

    return rhot


#==============================================================================#
# TWO SPECIES TWO LEVEL SYSTEMS

def two_species_piqs_mesolve (N1, N2, initial_state_1, initial_state_2,
                              nphot=None, gce=0.5, gcd=0.5, gcp=0.5,
                              w01=1., wx1=0.1, ge1=0.5, gd1=0.5, gp1=0.5,
                              w02=1., wx2=0.1, ge2=0.5, gd2=0.5, gp2=0.5,
                              wc=1, k=1, wp=0.1, g1=10, g2=20, plot=False):
    # Ensemble of N1 TLSs
    system1 = piqs.Dicke(N = N1)

    [jx1, jy1, jz1] = piqs.jspin(N1)
    jp1 = piqs.jspin(N1, op='+')
    jm1 = piqs.jspin(N1, op='-')

    h_tls1 = w01 * jz1 + wx1 * jx1
    system1.hamiltonian = h_tls1
    system1.emission = ge1
    system1.dephasing = gd1
    system1.pumping = gp1

    L_tls1 = system1.liouvillian()

    # Ensemble of N2 TLSs
    system2 = piqs.Dicke(N = N2)

    [jx2, jy2, jz2] = piqs.jspin(N2)
    jp2 = piqs.jspin(N2, op='+')
    jm2 = piqs.jspin(N2, op='-')

    h_tls2 = w02 * jz2 + wx2 * jx2
    system2.hamiltonian = h_tls2
    system2.emission = ge2
    system2.dephasing = gd2
    system2.pumping = gp2

    L_tls2 = system2.liouvillian()

    # Identity super-operators
    nds1 = piqs.num_dicke_states(N1)
    nds2 = piqs.num_dicke_states(N2)
    id_1 = qeye(nds1)
    id_2 = qeye(nds2)
    id_tls1 = to_super(qeye(nds1))
    id_tls2 = to_super(qeye(nds2))

    # Total hamiltonian
    jz = tensor(jz1, id_2) + tensor(id_1, jz2)
    jp = tensor(jp1, id_2) + tensor(id_1, jp2)
    jm = tensor(jm1, id_2) + tensor(id_1, jm2)
    jx = tensor(jx1, id_2) + tensor(id_1, jx2)
    jz1_tot = tensor(jz1, id_2)
    jz2_tot = tensor(id_1, jz2)
    jp1_tot = tensor(jp1, id_2)
    jp2_tot = tensor(id_1, jp2)
    jm1_tot = tensor(jm1, id_2)
    jm2_tot = tensor(id_1, jm2)

    h_loc = tensor(h_tls1, id_2) + tensor(id_1, h_tls2)

    # Total Liouvillian for local operators
    L_loc = super_tensor(L_tls1, id_tls2) + super_tensor(id_tls1, L_tls2)
    L_col = liouvillian(h_loc, [np.sqrt(gce)*jm, np.sqrt(gcd)*jz, np.sqrt(gcp)*jp])
    L_tls = L_col + L_loc
    L_tot = L_tls

    # Initial state and time evolution
    rho0_tls1 = initial_state_1
    rho0_tls2 = initial_state_2
    rho0_tls = tensor(rho0_tls1, rho0_tls2)
    rho0 = rho0_tls


    if nphot:
        #a = destroy(nphot)
        #h_int = g * tensor(a + a.dag(), jx)

        a = destroy(nphot)
        h_int1 = g1 * (tensor(jp1_tot, a.dag()) + tensor(jm1_tot, a))
        h_int2 = g2 * (tensor(jm2_tot, a.dag()) + tensor(jp2_tot, a))
        h_int = h_int1 + h_int2

        # Photonic Liouvillian
        c_ops_phot = [np.sqrt(k) * a, np.sqrt(wp) * a.dag()]
        L_phot = liouvillian(wc * a.dag()*a , c_ops_phot)

        id_tls = to_super(tensor(id_1, id_2))
        id_phot = to_super(qeye(nphot))

        # Define the total Liouvillian
        L_int = -1j* spre(h_int) + 1j* spost(h_int)
        L_tot = L_int + super_tensor(id_tls, L_phot) + super_tensor(L_tls, id_phot)

        ground_phot = ket2dm(basis(nphot,0))
        rho0 = tensor(rho0_tls, ground_phot)

    t = np.linspace(0, 5, 1000)
    result = mesolve(L_tot, rho0, t, [], options=Options(atol=1e-15,rtol=1e-15))
    rhot = result.states

    return rhot


def two_species_piqs_expected (N1, N2, initial_state_1, initial_state_2,
                              nphot=None, gce=0.5, gcd=0.5, gcp=0.5,
                              w01=1., wx1=0.1, ge1=0.5, gd1=0.5, gp1=0.5,
                              w02=1., wx2=0.1, ge2=0.5, gd2=0.5, gp2=0.5,
                              wc=1, k=1, wp=0.1, g1=10, g2=20, plot=False):

    # Ensemble of N1 TLSs
    system1 = piqs.Dicke(N = N1)

    [jx1, jy1, jz1] = piqs.jspin(N1)
    jp1 = piqs.jspin(N1, op='+')
    jm1 = piqs.jspin(N1, op='-')

    h_tls1 = w01 * jz1 + wx1 * jx1
    system1.hamiltonian = h_tls1
    system1.emission = ge1
    system1.dephasing = gd1
    system1.pumping = gp1

    L_tls1 = system1.liouvillian()

    # Ensemble of N2 TLSs
    system2 = piqs.Dicke(N = N2)

    [jx2, jy2, jz2] = piqs.jspin(N2)
    jp2 = piqs.jspin(N2, op='+')
    jm2 = piqs.jspin(N2, op='-')

    h_tls2 = w02 * jz2 + wx2 * jx2
    system2.hamiltonian = h_tls2
    system2.emission = ge2
    system2.dephasing = gd2
    system2.pumping = gp2

    L_tls2 = system2.liouvillian()

    # Identity super-operators
    nds1 = piqs.num_dicke_states(N1)
    nds2 = piqs.num_dicke_states(N2)
    id_1 = qeye(nds1)
    id_2 = qeye(nds2)
    id_tls1 = to_super(qeye(nds1))
    id_tls2 = to_super(qeye(nds2))

    # Total hamiltonian
    jz_1_jz_2 = tensor(jz1, id_2) + tensor(id_1, jz2)
    jp_1_jp_2 = tensor(jp1, id_2) + tensor(id_1, jp2)
    jm_1_jm_2 = tensor(jm1, id_2) + tensor(id_1, jm2)
    jx_1_jx_2 = tensor(jx1, id_2) + tensor(id_1, jx2)
    jy_1_jy_2 = tensor(jy1, id_2) + tensor(id_1, jy2)

    jz1_tot = tensor(jz1, id_2)
    jz2_tot = tensor(id_1, jz2)
    jp1_tot = tensor(jp1, id_2)
    jp2_tot = tensor(id_1, jp2)
    jm1_tot = tensor(jm1, id_2)
    jm2_tot = tensor(id_1, jm2)
    jx1_tot = tensor(jx1, id_2)
    jx2_tot = tensor(id_1, jx2)
    jy1_tot = tensor(jy1, id_2)
    jy2_tot = tensor(id_1, jy2)

    h_loc = tensor(h_tls1, id_2) + tensor(id_1, h_tls2)

    # Total Liouvillian for local operators
    L_loc = super_tensor(L_tls1, id_tls2) + super_tensor(id_tls1, L_tls2)
    L_col = liouvillian(h_loc, [np.sqrt(gce)*jm_1_jm_2, np.sqrt(gcd)*jz_1_jz_2, np.sqrt(gcp)*jp_1_jp_2])
    #L_col = liouvillian(h_loc, [np.sqrt(gce)*jp, np.sqrt(gcd)*jm1_tot, np.sqrt(gcp)*jm2_tot])
    L_tls = L_col + L_loc
    L_tot = L_tls

    # Initial state and time evolution
    rho0_tls1 = initial_state_1
    rho0_tls2 = initial_state_2
    rho0_tls = tensor(rho0_tls1, rho0_tls2)
    rho0 = rho0_tls

    if nphot:
        #a = destroy(nphot)
        #h_int = g1 * tensor(jx, a + a.dag())

        a = destroy(nphot)
        h_int1 = g1 * (tensor(jp1_tot, a.dag()) + tensor(jm1_tot, a))
        h_int2 = g2 * (tensor(jm2_tot, a.dag()) + tensor(jp2_tot, a))
        h_int = h_int1 + h_int2

        # Photonic Liouvillian
        c_ops_phot = [np.sqrt(k) * a, np.sqrt(wp) * a.dag()]
        L_phot = liouvillian(wc * a.dag()*a , c_ops_phot)

        id_tls = to_super(tensor(id_1, id_2))
        id_phot = to_super(qeye(nphot))

        # Define the total Liouvillian
        L_int = -1j* spre(h_int) + 1j* spost(h_int)
        L_tot = L_int + super_tensor(id_tls, L_phot) + super_tensor(L_tls, id_phot)

        ground_phot = ket2dm(basis(nphot,0))
        rho0 = tensor(rho0_tls, ground_phot)

        jz_1_jz_2 = tensor(jz_1_jz_2, qeye(nphot))
        jm_1_jm_2 = tensor(jm_1_jm_2, qeye(nphot))
        jp_1_jp_2 = tensor(jp_1_jp_2, qeye(nphot))
        jx_1_jx_2 = tensor(jx_1_jx_2, qeye(nphot))
        jy_1_jy_2 = tensor(jy_1_jy_2, qeye(nphot))

        jz1_tot = tensor(jz1_tot, qeye(nphot))
        jz2_tot = tensor(jz2_tot, qeye(nphot))
        jx1_tot = tensor(jx1_tot, qeye(nphot))
        jx2_tot = tensor(jx2_tot, qeye(nphot))
        jy1_tot = tensor(jy1_tot, qeye(nphot))
        jy2_tot = tensor(jy2_tot, qeye(nphot))

    t = np.linspace(0, 5, 1000)
    jops = [jz_1_jz_2,jx_1_jx_2,jy_1_jy_2,jz1_tot,jz2_tot,jx1_tot,jx2_tot,jy1_tot,jy2_tot]

    result = mesolve(L_tot, rho0, t, [], e_ops=jops, options=Options(atol=1e-15,rtol=1e-15))
    rhot = result.states
    exp = result.expect

    return exp


def two_species_qutip_mesolve (N1, N2, initial_state_1, initial_state_2,
                              nphot=None, gce=0.5, gcd=0.5, gcp=0.5,
                              w01=1., wx1=0.1, ge1=0.5, gd1=0.5, gp1=0.5,
                              w02=1., wx2=0.1, ge2=0.5, gd2=0.5, gp2=0.5,
                              wc=1, k=1, wp=0.1, g1=10, g2=20, plot=False):

    # Ensamble of N1 TLSs
    [jx1, jy1, jz1] = piqs.jspin(N1, basis="uncoupled")
    jm1 = piqs.jspin(N1, op='-', basis="uncoupled")
    jp1 = piqs.jspin(N1, op='+', basis="uncoupled")

    h_tls1 = w01 * jz1 + wx1 * jx1

    cops_local1 = piqs.collapse_uncoupled(N = N1, emission = ge1,
                                     pumping = gp1, dephasing = gd1)

    L_tls1 = liouvillian(h_tls1, cops_local1)

    # Ensamble of N2 TLSs
    [jx2, jy2, jz2] = piqs.jspin(N2, basis="uncoupled")
    jm2 = piqs.jspin(N2, op='-', basis="uncoupled")
    jp2 = piqs.jspin(N2, op='+', basis="uncoupled")

    h_tls2 = w02 * jz2 + wx2 * jx2

    cops_local2 = piqs.collapse_uncoupled(N = N2, emission = ge2,
                                     pumping = gp2, dephasing = gd2)

    L_tls2 = liouvillian(h_tls2, cops_local2)

    # Identity super-operators
    id_1 = qeye(jz1.dims[0]) # qeye(2**N1) has incompatbile dimensions: e.g.
    id_2 = qeye(jz2.dims[0]) # dims = [[2],[2]] vs dims = [[2,2],[2,2]]
    id_tls1 = to_super(id_1)
    id_tls2 = to_super(id_2)

    jz = tensor(jz1, id_2) + tensor(id_1, jz2)
    jp = tensor(jp1, id_2) + tensor(id_1, jp2)
    jm = tensor(jm1, id_2) + tensor(id_1, jm2)
    jx = tensor(jx1, id_2) + tensor(id_1, jx2)
    jz1_tot = tensor(jz1, id_2)
    jz2_tot = tensor(id_1, jz2)
    jp1_tot = tensor(jp1, id_2)
    jp2_tot = tensor(id_1, jp2)
    jm1_tot = tensor(jm1, id_2)
    jm2_tot = tensor(id_1, jm2)

    h_loc = tensor(h_tls1, id_2) + tensor(id_1, h_tls2)

    # Total Liouvillian for local operators
    L_loc = super_tensor(L_tls1, id_tls2) + super_tensor(id_tls1, L_tls2)
    L_col = liouvillian(h_loc, [np.sqrt(gce)*jm, np.sqrt(gcd)*jz, np.sqrt(gcp)*jp])
    L_tls = L_col + L_loc
    L_tot = L_tls

    # Initial state and time evolution
    rho0_tls1 = initial_state_1
    rho0_tls2 = initial_state_2
    rho0_tls = tensor(rho0_tls1, rho0_tls2)
    rho0 = rho0_tls


    if nphot:
        #a = destroy(nphot)
        #h_int = g * tensor(a + a.dag(), jx)

        a = destroy(nphot)
        h_int1 = g1 * (tensor(jp1_tot, a.dag()) + tensor(jm1_tot, a))
        h_int2 = g2 * (tensor(jm2_tot, a.dag()) + tensor(jp2_tot, a))
        h_int = h_int1 + h_int2

        # Photonic Liouvillian
        c_ops_phot = [np.sqrt(k) * a, np.sqrt(wp) * a.dag()]
        L_phot = liouvillian(wc * a.dag()*a , c_ops_phot)

        id_tls = to_super(tensor(id_1, id_2))
        id_phot = to_super(qeye(nphot))

        # Define the total Liouvillian
        L_int = -1j* spre(h_int) + 1j* spost(h_int)
        L_tot = L_int + super_tensor(id_tls, L_phot) + super_tensor(L_tls, id_phot)

        # Total operators
        jz = tensor(qeye(nphot), jz)
        jp = tensor(qeye(nphot), jp)
        jm = tensor(qeye(nphot), jm)
        jz1_tot = tensor(qeye(nphot), jz1_tot)
        jz2_tot = tensor(qeye(nphot), jz2_tot)
        #nphot_tot = tensor(a.dag()*a, qeye(nds))
        #adag_tot = tensor(a.dag(), qeye(nds))
        #a_tot = tensor(a, qeye(nds))

        ground_phot = ket2dm(basis(nphot,0))
        rho0 = tensor(rho0_tls, ground_phot)

    t = np.linspace(0, 5, 1000)
    result = mesolve(L_tot, rho0, t, [], options=Options(atol=1e-15,rtol=1e-15))
    rhot = result.states

    return rhot



def two_species_qutip_expected (N1, N2, initial_state_1, initial_state_2,
                              nphot=None, gce=0.5, gcd=0.5, gcp=0.5,
                              w01=1., wx1=0.1, ge1=0.5, gd1=0.5, gp1=0.5,
                              w02=1., wx2=0.1, ge2=0.5, gd2=0.5, gp2=0.5,
                              wc=1, k=1, wp=0.1, g1=10, g2=20, plot=False):

    # Ensamble of N1 TLSs
    [jx1, jy1, jz1] = piqs.jspin(N1, basis="uncoupled")
    jm1 = piqs.jspin(N1, op='-', basis="uncoupled")
    jp1 = piqs.jspin(N1, op='+', basis="uncoupled")

    h_tls1 = w01 * jz1 + wx1 * jx1

    cops_local1 = piqs.collapse_uncoupled(N = N1, emission = ge1,
                                     pumping = gp1, dephasing = gd1)

    L_tls1 = liouvillian(h_tls1, cops_local1)

    # Ensamble of N2 TLSs
    [jx2, jy2, jz2] = piqs.jspin(N2, basis="uncoupled")
    jm2 = piqs.jspin(N2, op='-', basis="uncoupled")
    jp2 = piqs.jspin(N2, op='+', basis="uncoupled")

    h_tls2 = w02 * jz2 + wx2 * jx2

    cops_local2 = piqs.collapse_uncoupled(N = N2, emission = ge2,
                                     pumping = gp2, dephasing = gd2)

    L_tls2 = liouvillian(h_tls2, cops_local2)

    # Identity super-operators
    id_1 = qeye(jz1.dims[0]) # qeye(2**N1) has incompatbile dimensions: e.g.
    id_2 = qeye(jz2.dims[0]) # dims = [[2],[2]] vs dims = [[2,2],[2,2]]
    id_tls1 = to_super(id_1)
    id_tls2 = to_super(id_2)

    jz_1_jz_2 = tensor(jz1, id_2) + tensor(id_1, jz2)
    jp_1_jp_2 = tensor(jp1, id_2) + tensor(id_1, jp2)
    jm_1_jm_2 = tensor(jm1, id_2) + tensor(id_1, jm2)
    jx_1_jx_2 = tensor(jx1, id_2) + tensor(id_1, jx2)
    jy_1_jy_2 = tensor(jy1, id_2) + tensor(id_1, jy2)

    jz1_tot = tensor(jz1, id_2)
    jz2_tot = tensor(id_1, jz2)
    jp1_tot = tensor(jp1, id_2)
    jp2_tot = tensor(id_1, jp2)
    jm1_tot = tensor(jm1, id_2)
    jm2_tot = tensor(id_1, jm2)
    jx1_tot = tensor(jx1, id_2)
    jx2_tot = tensor(id_1, jx2)
    jy1_tot = tensor(jy1, id_2)
    jy2_tot = tensor(id_1, jy2)

    h_loc = tensor(h_tls1, id_2) + tensor(id_1, h_tls2)

    # Total Liouvillian for local operators
    L_loc = super_tensor(L_tls1, id_tls2) + super_tensor(id_tls1, L_tls2)
    L_col = liouvillian(h_loc, [np.sqrt(gce)*jm_1_jm_2, np.sqrt(gcd)*jz_1_jz_2, np.sqrt(gcp)*jp_1_jp_2])
    L_tls = L_col + L_loc
    L_tot = L_tls

    # Time integration (use 'mesolve()' in full 4^N Liouvillian space)
    rho0_tls1 = initial_state_1
    rho0_tls2 = initial_state_2
    rho0_tls = tensor(rho0_tls1, rho0_tls2)
    rho0 = rho0_tls

    if nphot:
        #a = destroy(nphot)
        #h_int = g1 * tensor(jx, a + a.dag())

        a = destroy(nphot)
        h_int1 = g1 * (tensor(jp1_tot, a.dag()) + tensor(jm1_tot, a))
        h_int2 = g2 * (tensor(jm2_tot, a.dag()) + tensor(jp2_tot, a))
        h_int = h_int1 + h_int2

        # Photonic Liouvillian
        c_ops_phot = [np.sqrt(k) * a, np.sqrt(wp) * a.dag()]
        L_phot = liouvillian(wc * a.dag()*a , c_ops_phot)

        id_tls = to_super(tensor(id_1, id_2))
        id_phot = to_super(qeye(nphot))

        # Define the total Liouvillian
        L_int = -1j* spre(h_int) + 1j* spost(h_int)
        L_tot = L_int + super_tensor(id_tls, L_phot) + super_tensor(L_tls, id_phot)

        ground_phot = ket2dm(basis(nphot,0))
        rho0 = tensor(rho0_tls, ground_phot)

        jz_1_jz_2 = tensor(jz_1_jz_2, qeye(nphot))
        jx_1_jx_2 = tensor(jx_1_jx_2, qeye(nphot))
        jy_1_jy_2 = tensor(jy_1_jy_2, qeye(nphot))

        jz1_tot = tensor(jz1_tot, qeye(nphot))
        jz2_tot = tensor(jz2_tot, qeye(nphot))
        jx1_tot = tensor(jx1_tot, qeye(nphot))
        jx2_tot = tensor(jx2_tot, qeye(nphot))
        jy1_tot = tensor(jy1_tot, qeye(nphot))
        jy2_tot = tensor(jy2_tot, qeye(nphot))

    t = np.linspace(0, 5, 1000)

    jops = [jz_1_jz_2,jx_1_jx_2,jy_1_jy_2,jz1_tot,jz2_tot,jx1_tot,jx2_tot,jy1_tot,jy2_tot]

    result = mesolve(L_tot, rho0, t, [], e_ops=jops, options=Options(atol=1e-15,rtol=1e-15))
    rhot = result.states
    exp = result.expect

    return exp



def plot_times(functions_main, inputs_main, functions_sec=None, inputs_sec=None, repeats=3,
               n_tests=1, file_name=None, x_label_main=r'$[N_1,N_2]$', x_label_sec=None):
    if functions_sec:
        timings_main = get_timings(functions_main, inputs_main, repeats, n_tests=1)
        timings_sec = get_timings(functions_sec, inputs_sec, repeats, n_tests=1)
        results_main = aggregate_results(timings_main, inputs_main)
        results_sec = aggregate_results(timings_sec, inputs_sec)
        fig, ax = plot_results(results_main, inputs_main, x_label_main, results_sec, inputs_sec, x_label_sec)
        if file_name: fig.savefig(file_name, dpi=200)

        return fig, ax, results_main, results_sec

    else:
        timings_main = get_timings(functions_main, inputs_main, repeats, n_tests=1)
        results_main = aggregate_results(timings_main, inputs_main)
        fig, ax = plot_results(results_main, inputs_main, x_label_main)
        if file_name: fig.savefig(file_name, dpi=200)

        return fig, ax, results_main


def get_timings(functions, inputs, repeats, n_tests):
    values = list(inputs.values())
    for ind in range(len(functions)):
        result = pd.DataFrame(index = [str(i) for i in values[ind]],
                 columns = range(repeats),
                 data=(timeit.Timer(partial(functions[ind], *i)).repeat(repeat=repeats, number=n_tests) for i in values[ind]))
        yield functions[ind], result

def aggregate_results(timings, inputs):
    empty_multiindex = pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=['func', 'result'])
    aggregated_results = pd.DataFrame(columns=empty_multiindex)

    labels = list(inputs.keys())
    index = 0
    for func, timing in timings:
        for measurement in timing:
            aggregated_results[labels[index], measurement] = timing[measurement]
        aggregated_results[labels[index], 'avg'] = timing.mean(axis=1)
        aggregated_results[labels[index], 'yerr'] = timing.std(axis=1)
        index += 1
    return aggregated_results

def plot_results(results_main, inputs_main, x_label_main, results_sec=None, inputs_sec=None, x_label_sec=None):
    fig, ax = plt.subplots()

    x = results_main.index
    ax.set_prop_cycle(color=['b','r','g'])
    for func in results_main.columns.levels[0]:
        y = results_main[func, 'avg']
        yerr = results_main[func, 'yerr']
        ax.errorbar(x, y, yerr=yerr, fmt='-o', label=func)

    ax.set_xlabel(x_label_main)
    ax.set_ylabel('Time [s]')
    ax.set_yscale('log')

    if x_label_sec:
        x = results_sec.index
        sec_ax = ax.twiny()
        sec_ax.set_prop_cycle(color=['b','r','g'])
        sec_ax.set_frame_on(True)
        sec_ax.patch.set_visible(False)
        sec_ax.xaxis.set_ticks_position('top')
        sec_ax.xaxis.set_label_position('top')
        sec_ax.set_xlabel(x_label_sec)
        sec_ax.spines['top'].set_position(('outward', 20))
        for func in results_sec.columns.levels[0]:
            y = results_sec[func, 'avg']
            yerr = results_sec[func, 'yerr']
            sec_ax.errorbar(x, y, yerr=yerr, fmt='--o', label=func)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = sec_ax.get_legend_handles_labels()
        sec_ax.legend(h1+h2, l1+l2, loc='upper left')
    else:
        ax.legend()

    return fig, ax
