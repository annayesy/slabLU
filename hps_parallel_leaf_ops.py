import torch
torch.set_default_dtype(torch.double)
import numpy as np

def get_nearest_div(n,div):
    while (np.mod(n,div) > 0):
        div -= 1
    return div

def get_Aloc_2d(p,xxloc,Ds,pdo,box_start,box_end,device):
    nboxes = box_end-box_start
    Aloc = torch.zeros(nboxes,p**2,p**2,device=device)
    xx_flat = xxloc[box_start:box_end].reshape(nboxes*p**2,2)

    args = p,2,nboxes,xx_flat,Aloc

    Aloc_acc(*args,pdo.c11,Ds[0],c=-1.)
    Aloc_acc(*args,pdo.c22,Ds[1],c=-1.)
    if (pdo.c12 is not None):
        Aloc_acc(*args,pdo.c12,Ds[2],c=-2.)
    if (pdo.c1 is not None):
        Aloc_acc(*args,pdo.c1,Ds[3])
    if (pdo.c2 is not None):
        Aloc_acc(*args,pdo.c2,Ds[4])
    if (pdo.c is not None):
        I = torch.eye(p**2,device=device)
        Aloc_acc(*args,pdo.c,I)
    return Aloc


def Aloc_acc(p,d,nboxes,xx_flat,Aloc,\
             func,D,c=1.):
    size_f    = p**d
    f_vals    = func(xx_flat).reshape(nboxes,size_f)
    f_vals    = f_vals[:,:,None]
    f_vals   *= c
    Aloc     += f_vals * D.unsqueeze(0)

    
def form_DtNs(p,d,xxloc,Nx,Jx,Jc,Jxreo,Ds,Intmap,pdo,
          box_start,box_end,device,mode,data,ff_body_func):
    if (d == 2):
        args = p,xxloc,Ds,pdo,box_start,box_end
        Aloc = get_Aloc_2d(*args,device)
    else:
        return ValueError
    Acc = Aloc[:,Jc,:][:,:,Jc]
    nrhs = data.shape[-1]

    if (mode == 'build'):
        
        if (pdo.c12 is None):
            S_tmp   = -torch.linalg.solve(Acc,Aloc[:,Jc][...,Jx])
            Irep    = torch.eye(Jx.shape[0],device=device).unsqueeze(0).repeat(box_end-box_start,1,1)
            S_full  = torch.concat((S_tmp,Irep),dim=1)
            Jtot    = torch.hstack((Jc,Jx))
            
            DtN     = Nx[...,Jtot].unsqueeze(0) @ S_full
        else:
            S_tmp   = -torch.linalg.solve(Acc,Aloc[:,Jc][...,Jxreo])
            Irep    = torch.eye(Jx.shape[0],device=device).unsqueeze(0).repeat(box_end-box_start,1,1)
            S_full  = torch.concat((S_tmp @ Intmap.unsqueeze(0),Irep),dim=1)
            
            Jtot    = torch.hstack((Jc,Jx))
            DtN     = Nx[...,Jtot].unsqueeze(0) @ S_full
            
            #S_tmp   = -torch.linalg.solve(Acc,Aloc[:,Jc][...,Jxreo])
            #Irep    = torch.eye(Jxreo.shape[0],device=device).unsqueeze(0).repeat(box_end-box_start,1,1)
            #S_full  = torch.concat((S_tmp,Irep),dim=1) @ Intmap.unsqueeze(0)
            #Jtot    = torch.hstack((Jc,Jxreo))
            #DtN     = Nx[...,Jtot].unsqueeze(0) @ S_full
        return DtN
    elif (mode == 'solve'):
        
        f_body = torch.zeros(box_end-box_start,Jc.shape[0],nrhs,device=device)
        if (ff_body_func is not None):
            xx_flat = xxloc[box_start:box_end].reshape((box_end-box_start)*p**2,2)
            tmp = ff_body_func(xx_flat)
            f_body = tmp.reshape(box_end-box_start,p**2,nrhs)[:,Jc]
        
       
        uu_sol = torch.zeros(box_end-box_start,p**2,2*nrhs,device=device)
        
        uu_sol[:,Jxreo,:nrhs] = Intmap.unsqueeze(0) @ data[box_start:box_end]
        if (pdo.c12 is None):
            uu_sol[:,Jc,:nrhs] = torch.linalg.solve(Acc, f_body - Aloc[:,Jc][...,Jx] @ data[box_start:box_end])
        else:
            uu_sol[:,Jc,:nrhs] = torch.linalg.solve(Acc, f_body - Aloc[:,Jc][...,Jxreo] @ uu_sol[:,Jxreo,:nrhs])
            
        # calculate residual
        uu_sol[:,Jc,nrhs:] = Aloc[:,Jc] @ uu_sol[...,:nrhs] - f_body
        return uu_sol
                                                      
    elif (mode == 'reduce_body'):
        # assume that the data is a function that you can apply to
        # xx locations
        xx_flat = xxloc[box_start:box_end].reshape((box_end-box_start)*p**2,2)
        f_body = ff_body_func(xx_flat)
        f_body = f_body.reshape(box_end-box_start,p**2,1)
        return - Nxc.unsqueeze(0) @ torch.linalg.solve(Acc,f_body[:,Jc])
    
def get_DtN_chunksize(p,device):
    if (device == torch.device('cuda')):
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a # in bytes
    else:
        f = 10e9 # 10 GB in bytes
    chunk_max = int(f / (p**4 * 8)) # 8 bytes in 64 bits memory
    return int(chunk_max/4)


def get_DtNs_helper(p,d,xxloc,Nx,Jx,Jc,Jxreo,Ds,Intmap,pdo,\
                    box_start,box_end,chunk_init,device,mode,data,ff_body_func):
    nboxes = box_end - box_start
    size_face = p-2
    if (mode == 'build'):
        DtNs = torch.zeros(nboxes,4*size_face,4*size_face,device=device)
    elif (mode == 'solve'):
        DtNs = torch.zeros(nboxes,p**2,2*data.shape[-1],device=device)
    elif (mode == 'reduce_body'):
        DtNs = torch.zeros(nboxes,4*size_face,1,device=device)
        
    chunk_size = chunk_init
    args = p,d,xxloc,Nx,Jx,Jc,Jxreo,Ds,Intmap,pdo
    chunk_list = torch.zeros(int(nboxes/chunk_init)+100,device=device).int(); 
    box_curr = 0; nchunks = 0
    while(box_curr < nboxes):

        b1 = box_curr + box_start
        b2 = np.min([box_end, b1 + chunk_size])
        
        tmp = form_DtNs(*args,b1,b2,device,mode,data,ff_body_func)
        
        DtNs[box_curr:box_curr + chunk_size] = tmp
        box_curr += chunk_size

        chunk_size = np.max([get_DtN_chunksize(p,device),chunk_init])
        chunk_list[nchunks] = b2-b1
        nchunks += 1
    return DtNs.cpu(),chunk_list[:nchunks]