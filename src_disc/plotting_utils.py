import torch
torch.set_default_dtype(torch.double)
import numpy as np
from matplotlib.patches import Polygon
from src_disc import pdo,built_in_funcs
import pickle

from scipy import interpolate
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable

####################################################################################
# plotting utility function that returns a polygon around the domain exterior
def get_poly_param_map(box_geom,geom_str,param_map,poly_pad=0,hres=100):
    
    if (poly_pad > 0):
        box_geom[1,0] += poly_pad
        print("adjusted box_geom ylim",box_geom)
    
    # make polygon
    ix = torch.tensor(np.linspace(box_geom[0,0],box_geom[0,1],hres))
    iy = torch.tensor(np.linspace(box_geom[1,0],box_geom[1,1],hres))
    ext_points = torch.zeros(hres*4,2)

    ext_points[:hres,0] = box_geom[0,0]
    ext_points[:hres,1] = iy
    ext_points[hres:2*hres,1] = box_geom[1,1]
    ext_points[hres:2*hres,0] = ix
    ext_points[2*hres:3*hres,0] = box_geom[0,1]
    ext_points[2*hres:3*hres,1] = torch.flip(iy,[0])
    ext_points[3*hres:,0] = torch.flip(ix,[0])
    ext_points[3*hres:,1] = box_geom[1,0]

    poly = Polygon(np.array(param_map(ext_points)), facecolor='none',edgecolor='none')
    return poly


def plot_solution(XX,box_geom,geom,kh,uu_sol,fig,ax,\
                  title=None,title_fontsize=45,plot_pad=0.1,poly_pad=0,\
                 axes_labeled=True,colorbar=True,resolution=1000):
    
    op, param_map, inv_param_map = pdo.get_param_map_and_pdo(geom, \
                                                             built_in_funcs.bfield_constant,kh)
    
    solution = uu_sol.reshape(XX.shape[0],)
    max_sol = torch.max(solution)
    min_sol = torch.min(solution)
    
    YY = param_map(XX)
    
    min_x = torch.min(YY[:,0]).item(); max_x = torch.max(YY[:,0]).item()
    min_y = torch.min(YY[:,1]).item(); max_y = torch.max(YY[:,1]).item()
    
    grid_x, grid_y    = np.mgrid[min_x:max_x:resolution*1j, min_y:max_y:resolution*1j]
    grid_solution     = griddata(YY, solution, (grid_x, grid_y), method='cubic').T
    
    im = ax.imshow(grid_solution, extent=(min_x-plot_pad,max_x+plot_pad,\
                                       min_y-plot_pad,max_y+plot_pad),\
                   vmin=min_sol, vmax=max_sol,\
                   origin='lower',cmap='jet')

    if (colorbar):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right',size='5%', pad=0.05)
        cbar = fig.colorbar(im,cax=cax,orientation='vertical')

    poly = get_poly_param_map(box_geom,geom,param_map,poly_pad=poly_pad)

    ax.add_patch(poly)
    im.set_clip_path(poly)
    
    if (not axes_labeled):
        ax.set_axis_off()
    
    if (title is None):
        ax.set_title('Solution for $\\kappa$=%4.2f'%(kh),fontsize=title_fontsize)
    else:
        ax.set_title(title,fontsize=title_fontsize)
    
def plot_bfield(XX,box_geom,geom,bfield,kh,fig,ax,\
                title=None,title_fontsize=45,plot_pad=0.1,poly_pad=0,axes_labeled=True,resolution=1000,\
               colorbar=True):
    
    if (title is None):
        title = 'Scattering field $b_{\\rm crystal}$'
    
    if (geom == 'square'):
        min_x = box_geom[0,0].item(); max_x = box_geom[0,1].item()
        min_y = box_geom[1,0].item(); max_y = box_geom[1,1].item()
        
        
        grid_x,grid_y = np.mgrid[min_x:max_x:resolution*1j, min_y:max_y:resolution*1j]
        positions = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
        bfield_val = bfield(torch.tensor(positions),kh) / (-kh**2)
        bfield_val = bfield_val.reshape(resolution,resolution)
        
        max_sol = torch.max(bfield_val)
        min_sol = torch.min(bfield_val)
        
        im = ax.imshow(bfield_val, extent=(min_x-plot_pad,max_x+plot_pad,\
                                           min_y-plot_pad,max_y+plot_pad),\
                       vmin=min_sol, vmax=max_sol,\
                       origin='lower',cmap='jet',interpolation='none')
        
        if (not axes_labeled):
            ax.set_axis_off()
        if (colorbar):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right',size='5%', pad=0.05)
            cbar = fig.colorbar(im,cax=cax,orientation='vertical')
    
        ax.set_title(title,fontsize=title_fontsize)
        
    else:
        op, param_map, inv_param_map = pdo.get_param_map_and_pdo(geom,bfield, kh)

        tmp_sol = bfield(param_map(XX),kh) / (-kh**2)
        plot_solution(XX,box_geom,geom,kh,tmp_sol,fig,ax,title=title,title_fontsize=title_fontsize,\
                      plot_pad=plot_pad,poly_pad=poly_pad,axes_labeled=axes_labeled,resolution=resolution)
    
    
def plot_bfield_from_pickle(pickle_file,fig,ax,title=None,\
                            title_fontsize=45,plot_pad=0.1,poly_pad=0,\
                           axes_labeled=True,resolution=1000):
    
    f = open(pickle_file,'rb')
    build_dict = pickle.load(f); solve_dict = pickle.load(f) 
    
    XX = solve_dict['xx']
    if ('box_geom' not in build_dict):
        box_geom = torch.tensor([[0,1.0],[0,1.0]])
    else:
        box_geom = build_dict['box_geom']
        
    kh = build_dict['kh']
    bfield_str = build_dict['pde']
    
    if ('domain' not in build_dict):
        geom = 'square'
    else:
        geom = build_dict['domain']
    
    if (bfield_str == 'bfield_constant'):
        bfield = built_in_funcs.bfield_constant
    elif (bfield_str == 'bfield_bumpy'):
        bfield = built_in_funcs.bfield_bumpy
    elif (bfield_str == 'bfield_gaussian_bumps'):
        bfield = built_in_funcs.bfield_gaussian_bumps
    elif (bfield_str == 'bfield_cavity'):
        bfield = built_in_funcs.bfield_cavity_scattering
    elif (bfield_str == 'bfield_crystal'):
        bfield = built_in_funcs.bfield_crystal
    elif (bfield_str == 'bfield_crystal_waveguide'):
        bfield = built_in_funcs.bfield_crystal_waveguide
    elif (bfield_str == 'bfield_crystal_rhombus'):
        bfield = built_in_funcs.bfield_crystal_rhombus
    else:
        raise ValueError
        
    plot_bfield(XX,box_geom,geom,bfield,kh,fig,ax,title=title,title_fontsize=title_fontsize,\
                plot_pad=plot_pad,poly_pad=poly_pad,axes_labeled=axes_labeled,resolution=resolution)
    
    
def plot_solution_from_pickle(pickle_file,fig,ax,title=None,title_fontsize=45,\
                              plot_pad=0.1,poly_pad=0,axes_labeled=True,colorbar=True):
        
    f = open(pickle_file,'rb')
    build_dict = pickle.load(f); solve_dict = pickle.load(f) 
    
    XX = solve_dict['xx']
    if ('box_geom' not in build_dict):
        box_geom = torch.tensor([[0,1.0],[0,1.0]])
    else:
        box_geom = build_dict['box_geom']
        
    kh = build_dict['kh']
    bfield_str = built_in_funcs.bfield_constant
    uu_sol = solve_dict['sol']
    
    if ('domain' not in build_dict):
        geom = 'square'
    else:
        geom = build_dict['domain']
        
    plot_solution(XX,box_geom,geom,kh,uu_sol,fig,ax,title=title,title_fontsize=title_fontsize,\
                  plot_pad=plot_pad,poly_pad=poly_pad,axes_labeled=axes_labeled,colorbar=colorbar)
    
    
def set_plt_params(plt, SMALL_SIZE = 30, MEDIUM_SIZE = 50, BIGGER_SIZE = 100):
    plt.rc('text',usetex=True)
    plt.rc('font',family='serif')
    plt.rc('text.latex',preamble=r'\usepackage{amsfonts,bm}')

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('lines', linewidth=5)