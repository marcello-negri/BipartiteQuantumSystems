# Performance comparison between QuTiP and PIQS
# Physical system: two species with bosonic coupling (variable number of TLSs)

# Estimated simulation time
# 1: 5min, 2: 40min, 3: 5min, 4: 40min

import euler_library

import os
import shutil

import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import matplotlib as mpl
#from matplotlib import cm
#from importlib import reload

from qutip import *
from qutip import piqs

plt.rcParams["figure.figsize"] = (12,8)

# One species without bosonic coupling
parameters = {'piqs_mesolve':  [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]],
              'qutip_mesolve': [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]}
functions = [euler_library.one_species_piqs_mesolve,
             euler_library.one_species_qutip_mesolve]
fig, ax, results = euler_library.plot_times(functions, parameters, repeats=2,
                                      file_name='./mesolve_plot/one_species_mesolve.jpg',
                                      x_label_main=r'$[N]$')

pd.DataFrame(results).to_csv('./mesolve_plot/one_species_mesolve.csv')


# One species with bosonic coupling
parameters_main = {'piqs matrix': [[1],[2],[3],[4],[5],[6],[7],[8],[9]],
                  'qutip matrix': [[1],[2],[3],[4],[5],[6],[7],[8],[9]]}
parameters_sec = {'piqs photon matrix':  [[1,5],[2,5],[3,5],[4,5],[5,5],[6,5],[7,5],[8,5],[9,5]],
                  'qutip photon matrix': [[1,5],[2,5],[3,5],[4,5],[5,5],[6,5],[7,5],[8,5],[9,5]]}
functions_main = [euler_library.one_species_piqs_matrix,
                 euler_library.one_species_qutip_matrix]
functions_sec = [euler_library.one_species_piqs_matrix,
                 euler_library.one_species_qutip_matrix]
fig, ax, r1, r2 = euler_library.plot_times(functions_main, parameters_main,
                                      functions_sec, parameters_sec, repeats=2,
                                      file_name='./matrix_plot/one_species_matrix_phot.jpg',
                                      x_label_main=r'$[N]$', x_label_sec=r'$[N,nphot]$')

pd.DataFrame(r1).to_csv('./matrix_plot/one_species_matrix_phot_no_phot.csv')
pd.DataFrame(r2).to_csv('./matrix_plot/one_species_matrix_phot.csv')


# Two species without bosonic coupling
parameters_main = {'2species: piqs_matrix':  [[1,1],[2,2],[3,3],[4,4],[5,5]],
                   '2species: qutip_matrix': [[1,1],[2,2],[3,3],[4,4],[5,5]]}
parameters_sec = {'1species: piqs_matrix':  [[2],[4],[6],[8],[10]],
                  '1species: qutip_matrix': [[2],[4],[6],[8],[10]]}

functions_main = [euler_library.two_species_piqs_matrix,
                  euler_library.two_species_qutip_matrix]
functions_sec = [euler_library.one_species_piqs_matrix,
                 euler_library.one_species_qutip_matrix]

fig, ax, r1, r2 = euler_library.plot_times(functions_main, parameters_main,
                                     functions_sec, parameters_sec, repeats=2,
                                     file_name='./matrix_plot/two_species_matrix.jpg',
                                     x_label_main=r'$[N_1,N_2]$', x_label_sec=r'$[N]$')

pd.DataFrame(r1).to_csv('./matrix_plot/two_species_matrix_two_species.csv')
pd.DataFrame(r2).to_csv('./matrix_plot/two_species_matrix_one_species.csv')


# Two species with bosonic coupling
parameters_main = {'piqs matrix': [[1,1],[2,2],[3,3],[4,4],[5,5]],
                  'qutip matrix': [[1,1],[2,2],[3,3],[4,4],[5,5]]}
parameters_sec = {'piqs photon matrix':  [[1,1,3],[2,2,3],[3,3,3],[4,4,3],[5,5,3]],
                  'qutip photon matrix': [[1,1,3],[2,2,3],[3,3,3],[4,4,3],[5,5,3]]}
functions_main = [euler_library.two_species_piqs_matrix,
                 euler_library.two_species_qutip_matrix]
functions_sec = [euler_library.two_species_piqs_matrix,
                 euler_library.two_species_qutip_matrix]
fig, ax, r1, r2 = euler_library.plot_times(functions_main, parameters_main,
                                      functions_sec, parameters_sec, repeats=2,
                                      file_name='./matrix_plot/two_species_matrix_phot.jpg',
                                      x_label_main=r'$[N_1,N_2]$',
                                      x_label_sec=r'$[N_1,N_2,nphot]$')

pd.DataFrame(r1).to_csv('./matrix_plot/two_species_matrix_phot_no_phot.csv')
pd.DataFrame(r2).to_csv('./matrix_plot/two_species_matrix_phot.csv')
