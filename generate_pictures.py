import os
import sys
import src_disc.plotting_utils as plotting_utils
import matplotlib.pyplot as plt

if not os.path.exists('pickle_files'):
	os.makedirs('pickle_files')


def get_driver_command(nwaves, pickle_loc,\
	n, p, pde_type,domain_type,\
	xlim=1.0,ylim=1.0,isperiodic=False):

	disc_str   = ' --n %d --disc hps --p %d' % (n,p)
	domain_str = ' --domain %s --box_xlim %5.2f --box_ylim %5.2f' % (domain_type,xlim,ylim)
	bc_str     = ' --bc ones' + (' --periodic_bc' if isperiodic else '')
	pde_str    = ' --pde %s --nwaves %5.12f' % (pde_type,nwaves)
	pickle_str = ' --solver slabLU --pickle pickle_files/%s --store_sol' % (pickle_loc)

	command_str = 'python argparse_driver.py' + disc_str + domain_str + bc_str + pde_str + pickle_str
	print(command_str)
	return command_str


n = 600; p = 42

###########################################################################################
### generate plots for crystal waveguide

args_crystal = n,p,'bfield_crystal_waveguide', 'square'

nwaves1 = 24.623521102434587
nwaves2 = 24.673521102434584
os.system(get_driver_command(nwaves1,'crystal_picture1',*args_crystal))
os.system(get_driver_command(nwaves2,'crystal_picture2',*args_crystal))

plotting_utils.set_plt_params(plt,SMALL_SIZE=40)
plt.rcParams["figure.figsize"] = (32,10)
with plt.rc_context():
    fig, ax = plt.subplots(1,3)
pickle_file_freq0 = 'pickle_files/crystal_picture1'
pickle_file_freq1 = 'pickle_files/crystal_picture2'

plotting_utils.plot_bfield_from_pickle(pickle_file_freq0,fig,ax[0],plot_pad=0,\
                                       title_fontsize=50,axes_labeled=False)
plotting_utils.plot_solution_from_pickle(pickle_file_freq0,fig,ax[1],plot_pad=0,axes_labeled=False,\
                                        title_fontsize=50)
plotting_utils.plot_solution_from_pickle(pickle_file_freq1,fig,ax[2],plot_pad=0,axes_labeled=False,\
                                        title_fontsize=50)
plt.savefig('figures/picture_crystal.png',bbox_inches='tight')

###########################################################################################
### generate plots for annulus

args_annulus = n,p,'bfield_constant', 'annulus', 3.0, 1.0

nwaves1 = 15.0
os.system(get_driver_command(nwaves1,'annulus_picture1',*args_annulus))

plotting_utils.set_plt_params(plt,SMALL_SIZE=50)

plt.rcParams["figure.figsize"] = (20,10)
with plt.rc_context():
    fig, ax = plt.subplots(1,1)
pickle_file_freq1 = 'pickle_files/annulus_picture1'

plotting_utils.plot_solution_from_pickle(pickle_file_freq1,fig,ax,title_fontsize=50)
plt.savefig('figures/picture_annulus.png',bbox_inches='tight')

###########################################################################################
### generate plots for curvy annulus

n = 800; p = 42

args_curvy = n,p,'bfield_constant', 'curvy_annulus', 6.0, 1.0, True

nwaves1 = 10.0; nwaves2 = 30.0
os.system(get_driver_command(nwaves1,'curvy_annulus_picture1',*args_curvy))
os.system(get_driver_command(nwaves2,'curvy_annulus_picture2',*args_curvy))

plotting_utils.set_plt_params(plt,SMALL_SIZE=50)

plt.rcParams["figure.figsize"] = (40,20)
with plt.rc_context():
    fig, ax = plt.subplots(1,2)
pickle_file_freq0 = 'pickle_files/curvy_annulus_picture1'
pickle_file_freq1 = 'pickle_files/curvy_annulus_picture2'

plotting_utils.plot_solution_from_pickle(pickle_file_freq0,fig,ax[0],poly_pad=0.02,title_fontsize=60)
plotting_utils.plot_solution_from_pickle(pickle_file_freq1,fig,ax[1],poly_pad=0.04,title_fontsize=60,\
                                        axes_labeled=False)
plt.savefig('figures/picture_curvy_annulus.png',bbox_inches='tight')