import torch
torch.set_default_dtype(torch.double)
import numpy as np
import warnings

class PDO_2d:
    def __init__(self, c11, c22, c12=None, \
                 c1= None, c2 = None, c = None):
        self.c11, self.c22 = c11, c22
        self.c12 = c12
        self.c1, self.c2    = c1, c2
        self.c = c

class PDO_3d:
    def __init__(self, c11, c22, c33, c12=None, c13 = None, c23 = None, \
                 c1= None, c2 = None, c3 = None, c = None):
        self.c11, self.c22, self.c33 = c11, c22, c33
        self.c12, self.c13, self.c23 = c12, c13, c23
        self.c1, self.c2, self.c3    = c1, c2, c3
        self.c = c
        
# Input: n x d matrix of locations of n points in d dimensions
# Output: n vector of function values
def ones(xxloc):
    tmp = torch.ones(xxloc.shape[0],device=xxloc.device)
    return tmp

# Input: n x d matrix of locations of n points in d dimensions
# Output: n vector of function values
def const(c=1):
    return lambda xxloc: c * torch.ones(xxloc.shape[0],device=xxloc.device)

################################################################################
#################### parameter map for curved domains ##########################

def c11_func(parameter_map,y1_d1, y1_d2):

    if ((y1_d1 is None) and (y1_d2 is None)):
        c11 = None
    else:

        def c11(xx):
            yy = parameter_map(xx)

            result = 0
            if (y1_d1 is not None):
                result += y1_d1(yy)**2
            if (y1_d2 is not None):
                result += y1_d2(yy)**2
            return result
    return c11


def c22_func(parameter_map,y2_d1, y2_d2):

    if ((y2_d1 is None) and (y2_d2 is None)):
        c22 = None
    else:

        def c22(xx):
            yy = parameter_map(xx)

            result = 0
            if (y2_d1 is not None):
                result += y2_d1(yy)**2
            if (y2_d2 is not None):
                result += y2_d2(yy)**2
            return result
    return c22

def c1_func(parameter_map,y1_d1d1, y1_d2d2):

    if ((y1_d1d1 is None) and (y1_d2d2 is None)):
        c1 = None
    else:

        def c1(xx):
            yy = parameter_map(xx)

            result = 0
            if (y1_d1d1 is not None):
                result -= y1_d1d1(yy)
            if (y1_d2d2 is not None):
                result -= y1_d2d2(yy)
            return result
    return c1


def c2_func(parameter_map,y2_d1d1, y2_d2d2):

    if ((y2_d1d1 is None) and (y2_d2d2 is None)):
        c2 = None
    else:

        def c2(xx):
            yy = parameter_map(xx)

            result = 0
            if (y2_d1d1 is not None):
                result -= y2_d1d1(yy)
            if (y2_d2d2 is not None):
                result -= y2_d2d2(yy)
            return result
    return c2

def c12_func(parameter_map,y1_d1, y1_d2, y2_d1, y2_d2):

    bool_expr1 = (y1_d1 is not None) and (y2_d1 is not None)
    bool_expr2 = (y1_d2 is not None) and (y2_d2 is not None) 
    if ((not bool_expr1) and (not bool_expr2)):
        c12 = None
    else:

        def c12(xx):
            yy = parameter_map(xx)

            result = 0
            if (bool_expr1):
                result += torch.mul(y1_d1(yy),y2_d1(yy))
            if (bool_expr2):
                result += torch.mul(y1_d2(yy),y2_d2(yy))
            return result
    return c12

#####################################################################################
# We solve variable-coefficient PDEs on curved domains Psi by solving
# on a reference rectangle Omega.
# parameter_map (z_func) : given point on Omega, maps to point on Psi

# y_func is maps points on Psi to points on Omega
# partial derivatives are taken with respect to y = (y_1,y_2)

def pdo_param(kh, bfield, z1, z2, y1, y2, y1_d1=None, y1_d2=None, y2_d1=None, y2_d2=None,\
       y1_d1d1=None, y1_d2d2=None, y2_d1d1=None, y2_d2d2=None):
    
    
    def parameter_map(xx):
        ZZ = xx.clone()
        ZZ[:,0] = z1(xx)
        ZZ[:,1] = z2(xx)
        return ZZ
    
    def inv_parameter_map(xx):
        YY = xx.clone()
        YY[:,0] = y1(xx)
        YY[:,1] = y2(xx)
        return YY

    
    c11 = c11_func(parameter_map,y1_d1,y1_d2)
    c22 = c22_func(parameter_map,y2_d1,y2_d2)
    c1  = c1_func (parameter_map,y1_d1d1,y1_d2d2)
    c2  = c2_func (parameter_map,y2_d1d1,y2_d2d2)
    c12 = c12_func(parameter_map,y1_d1,y1_d2,\
                   y2_d1,y2_d2)
    
    def c(xx):
        return bfield(parameter_map(xx),kh)
    
    pdo = PDO_2d(c11=c11,c22=c22,c1=c1,c2=c2,c12=c12,c=c)
    return pdo, parameter_map, inv_parameter_map


def get_param_helper(geom,bfield,kh):
    
    if ((geom == 'sinusoidal') or (geom== 'curved')):
        
        mag = 0.25
        psi    = lambda z: 1 - mag * torch.sin(6*z)
        dpsi   = lambda z:   - mag*6 * torch.cos(6*z)
        ddpsi  = lambda z:     mag*36  * torch.sin(6*z)
        
        y1_d1  = lambda xx: torch.ones(xx[:,0].shape,device=xx.device)
        y2_d1  = lambda xx: torch.mul(xx[:,1], dpsi(xx[:,0]))
        y2_d2  = lambda xx: psi(xx[:,0])

        y2_d1d1  = lambda xx: torch.mul(xx[:,1], ddpsi(xx[:,0]))
        
        z1   = lambda xx: xx[:,0]
        z2   = lambda xx: torch.div(xx[:,1],psi(xx[:,0]))
        
        y1   = lambda xx: xx[:,0]
        y2   = lambda xx: torch.mul(xx[:,1],psi(xx[:,0]))
        
        return pdo_param(kh, bfield, z1, z2, y1, y2,\
                       y1_d1=y1_d1, y2_d1=y2_d1, y2_d2=y2_d2, y2_d1d1=y2_d1d1)
    
    
    elif (geom == 'annulus'):
        
        const_theta = 1/(np.pi/3)
        r           = lambda zz: (zz[:,0]**2 + zz[:,1]**2)**0.5

        z1 = lambda zz: torch.mul( 1 + 1 * zz[:,1], torch.cos(zz[:,0]/const_theta) )
        z2 = lambda zz: torch.mul( 1 + 1 * zz[:,1], torch.sin(zz[:,0]/const_theta) )
        
        y1 = lambda zz: const_theta* torch.atan2(zz[:,1],zz[:,0])
        y2 = lambda zz: r(zz) - 1
        
        y1_d1    = lambda zz: -const_theta     * torch.div(zz[:,1], r(zz)**2)
        y1_d2    = lambda zz: +const_theta     * torch.div(zz[:,0], r(zz)**2)
        y1_d1d1  = lambda zz: +2*const_theta   * torch.div(torch.mul(zz[:,0],zz[:,1]), r(zz)**4)
        y1_d2d2  = lambda zz: -2*const_theta   * torch.div(torch.mul(zz[:,0],zz[:,1]), r(zz)**4)
        y1_d1d1 = None; y1_d2d2 = None


        y2_d1    = lambda zz: torch.div(zz[:,0], r(zz))
        y2_d2    = lambda zz: torch.div(zz[:,1], r(zz))
        y2_d1d1  = lambda zz: torch.div(zz[:,1]**2, r(zz)**3)
        y2_d2d2  = lambda zz: torch.div(zz[:,0]**2, r(zz)**3)
        
        return pdo_param(kh, bfield, z1,z2,y1,y2,\
                         y1_d1=y1_d1, y1_d2=y1_d2,\
                         y1_d1d1=y1_d1d1, y1_d2d2=y1_d2d2,\
                         y2_d1=y2_d1, y2_d2=y2_d2, y2_d1d1=y2_d1d1, y2_d2d2=y2_d2d2)
    
    elif (geom == 'curvy_annulus'):
        
        const_theta = 1/(np.pi/3)
        const_phase = 5
        const_amp   = 0.2
        
        r           = lambda zz: (zz[:,0]**2 + zz[:,1]**2)**0.5
        
        radius_zmap = lambda zz: 1 + const_amp*torch.cos(const_phase*zz[:,0]/const_theta) + zz[:,1]

        z1 = lambda zz: torch.mul( radius_zmap(zz), torch.cos(zz[:,0] / const_theta) )
        z2 = lambda zz: torch.mul( radius_zmap(zz), torch.sin(zz[:,0] / const_theta) )
        
        theta_ymap = lambda zz: torch.arctan2(zz[:,1], zz[:,0]) 
        
        y1 = lambda zz: const_theta * theta_ymap(zz)
        y2 = lambda zz: r(zz) - const_amp * torch.cos(const_phase*theta_ymap(zz)) - 1
        
        y1_d1    = lambda zz: -const_theta     * torch.div(zz[:,1], r(zz)**2)
        y1_d2    = lambda zz: +const_theta     * torch.div(zz[:,0], r(zz)**2)
        y1_d1d1  = lambda zz: +2*const_theta   * torch.div(torch.mul(zz[:,0],zz[:,1]), r(zz)**4)
        y1_d2d2  = lambda zz: -2*const_theta   * torch.div(torch.mul(zz[:,0],zz[:,1]), r(zz)**4)
        
        def y2_d1(zz):
            result = torch.div(zz[:,0], r(zz))
            
            tmp = torch.div( torch.sin(const_phase * theta_ymap(zz)), r(zz)**2)
            
            result -= const_amp*const_phase * torch.mul(zz[:,1], tmp)
            return result
        
        def y2_d2(zz):
            result = torch.div(zz[:,1], r(zz))
            tmp = torch.div( torch.sin(const_phase * theta_ymap(zz)), r(zz)**2)
            
            result += torch.mul(const_amp*const_phase* zz[:,0], tmp)
            return result
        
        def y2_d1d1(zz):
            result = torch.div(zz[:,1]**2, r(zz)**3)
            
            tmp1 = torch.mul(zz[:,0], torch.sin( const_phase * theta_ymap(zz)))
            tmp2 = torch.mul(zz[:,1], torch.cos( const_phase * theta_ymap(zz)))
                                           
            result += const_amp*const_phase * torch.div( torch.mul(zz[:,1],\
                                                                   2*tmp1+const_phase*tmp2), r(zz)**4) 
            return result
        
        def y2_d2d2(zz):
            result = torch.div(zz[:,0]**2, r(zz)**3)
            
            tmp1 = torch.mul(zz[:,0], torch.cos( const_phase * theta_ymap(zz)))
            tmp2 = torch.mul(zz[:,1], torch.sin( const_phase * theta_ymap(zz)))
                                           
            result += const_amp*const_phase * torch.div( torch.mul(zz[:,0],\
                                                                   const_phase*tmp1-2*tmp2), r(zz)**4)                           
                                           
            return result
        
        return pdo_param(kh, bfield, z1,z2,y1,y2,\
                         y1_d1=y1_d1, y1_d2=y1_d2,\
                         y1_d1d1=y1_d1d1, y1_d2d2=y1_d2d2,\
                         y2_d1=y2_d1, y2_d2=y2_d2, y2_d1d1=y2_d1d1, y2_d2d2=y2_d2d2)
    
    elif (geom == 'corner'):
        
        warnings.warn("Corner geometry does not work")
        
        z1 = lambda zz: zz[:,0]
        def z2(zz):
            bool_map = zz[:,0] >= 0.5
            zz[bool_map,1] *= 0.5
            return zz[:,1]
        
        y1 = lambda zz: zz[:,0]
        
        def y2(zz):
            bool_map = zz[:,0] >= 0.5
            zz[bool_map,1] *= 2
            return zz[:,1]
            
        y1_d1 = lambda zz: torch.ones(zz.shape[0],device=zz.device)
        
        def y2_d2(zz):
            bool_map = zz[:,0] >= 0.5
            result = torch.ones(zz.shape[0], device=zz.device)
            result[bool_map] *= 2
            return result
        
        return pdo_param(kh, bfield,z1,z2,y1,y2,\
                         y1_d1=y1_d1, y2_d2=y2_d2)
        
    else:
        raise ValueError("geom %s not available"%(geom))


#####################################################################################
# We solve variable-coefficient PDEs on curved domains Psi by solving
# on a reference rectangle Omega.
# parameter_map : given point on Omega, maps to point on Psi

# y_func is maps points on Psi to points on Omega
# partial derivatives are taken with respect to y = (y_1,y_2)

def get_param_map_and_pdo(geom, bfield, kh):
    
    if (geom == 'square'):
        # identity maps for parameter maps 
        param_map     = lambda xx: xx.clone()
        inv_param_map = lambda xx: xx.clone()
        
        def c_func(xx):
            return bfield(xx,kh)
        op = PDO_2d(const(1),const(1), c=c_func)
        
        return op,param_map,inv_param_map
    else:
        return get_param_helper(geom,bfield,kh)