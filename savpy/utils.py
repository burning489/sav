# coding:utf-8
"""
@author: Ding Zhao
@file: utils.py
@time: 2021/4
@file_desc:
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_args():
    """Get the command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Get default configuration for HiOSD model.')
    parser.add_argument('--Nx', default=128, type=int)
    parser.add_argument('--Ny', default=128, type=int)
    parser.add_argument('--Lx', default=1, type=float)
    parser.add_argument('--Ly', default=1, type=float)
    parser.add_argument('--T', default=5000, type=int)
    parser.add_argument('--dt', default=1e-5, type=float)
    parser.add_argument('--C', default=0, type=float)
    parser.add_argument('--beta', default=2, type=float)
    parser.add_argument('--eps', default=1e-2, type=float)
    return parser.parse_args()

def phi_init(xx, yy, **kwargs):
    """Initialize the order parameter for phase field."""
    return 5e-2*np.sin(2*np.pi*xx)*np.sin(2*np.pi*yy)

def r_init(phi, **kwargs):
    """Initialize auxiliary variable r = F(phi)/sqrt(E1(phi)+C0)"""
    Nx, Ny, Lx, Ly, C = kwargs["Nx"], kwargs["Ny"], kwargs["Lx"], kwargs["Ly"], kwargs["C"]
    hx, hy = Lx/Nx, Ly/Ny
    return np.sqrt(np.fft.fft2(F(phi, **kwargs))[0,0]*hx*hy+C)

def F(phi, **kwargs):
    """Free energy density function"""
    beta, eps = kwargs["beta"], kwargs["eps"]
    return (phi**2-beta-1)**2/(4*eps**2)

def F_der(phi, **kwargs):
    """Derivative of F"""
    beta, eps = kwargs["beta"], kwargs["eps"]
    return phi*(phi**2-beta-1)/(eps**2)

def prepare_fft2(**kwargs):
    """Prepare frequency domains for fft2"""
    Nx, Ny, Lx, Ly = kwargs["Nx"], kwargs["Ny"], kwargs["Lx"], kwargs["Ly"]
    x_positive_freq = np.arange(0, Nx//2+1)
    x_negative_freq = np.arange(-Nx//2+1, 0)
    x_freq = np.concatenate((x_positive_freq, x_negative_freq))
    y_positive_freq = np.arange(0, Ny//2+1)
    y_negative_freq = np.arange(-Ny//2+1, 0)
    y_freq = np.concatenate((y_positive_freq, y_negative_freq))
    k_x = 1j*x_freq*(2*np.pi/Lx)
    k_y = 1j*y_freq*(2*np.pi/Ly)
    k2x = k_x**2
    k2y = k_y**2
    kx, ky = np.meshgrid(k_x, k_y)
    kxx, kyy = np.meshgrid(k2x, k2y)
    k2 = kxx + kyy
    k4 = k2**2
    return kx, ky, kxx, kyy, k2, k4

def lap_diff(phi, **kwargs):
    """Lalpacian phi using fft2"""
    kx, ky, kxx, kyy, k2, k4 = prepare_fft2(**kwargs)
    lap=np.fft.ifft2((k2*np.fft.fft2(phi))).real
    return lap

def inv_A(phi, **kwargs):
    """A^{-1}*phi using fft2"""
    dt = kwargs["dt"]
    kx, ky, kxx, kyy, k2, k4 = prepare_fft2(**kwargs)
    return np.fft.ifft2(np.fft.fft2(phi)/(1-dt*k2)).real

def compute_r(phi, phi0, r0, b, **kwargs):
    """Compute auxiliary variable r"""
    Nx, Ny, Lx, Ly = kwargs["Nx"], kwargs["Ny"], kwargs["Lx"], kwargs["Ly"]
    hx, hy = Lx/Nx, Ly/Ny
    bphi0 = np.fft.fft2(b*phi0)[0,0]*hx*hy
    bphi1 = np.fft.fft2(b*phi)[0,0]*hx*hy
    return bphi1/2 -bphi0/2 + r0

def compute_b(phi, **kwargs):
    """Compute auxiliary variable b"""
    Nx, Ny, Lx, Ly, C = kwargs["Nx"], kwargs["Ny"], kwargs["Lx"], kwargs["Ly"], kwargs["C"]
    hx, hy = Lx/Nx, Ly/Ny
    return F_der(phi, **kwargs)/np.sqrt(np.fft.fft2(F(phi, **kwargs))[0,0]*hx*hy+C)

def compute_g(phi0, r0, b, **kwargs):
    """Compute righthand side g"""
    Nx, Ny, Lx, Ly, dt = kwargs["Nx"], kwargs["Ny"], kwargs["Lx"], kwargs["Ly"], kwargs["dt"]
    hx, hy = Lx/Nx, Ly/Ny
    bphi0 = np.fft.fft2(b*phi0)[0,0]*hx*hy
    return phi0 - dt*r0*b + dt/2*bphi0*b

def compute_energy(phi, r, **kwargs):
    """Compute Raw and Modified free energy"""
    Nx, Ny, Lx, Ly = kwargs["Nx"], kwargs["Ny"], kwargs["Lx"], kwargs["Ly"]
    hx, hy = Lx/Nx, Ly/Ny
    lphi = lap_diff(phi, **kwargs)
    philphi = np.sum(phi*lphi)
    r2 = np.sum(r*r)
    modified_energy = -1*philphi + r2
    f = np.sum(F(phi, **kwargs))
    raw_energy = -1*philphi + f
    modified_energy = modified_energy*hx*hy
    raw_energy = raw_energy*hx*hy
    return modified_energy, raw_energy

def animate(phi_list, **kwargs):
    Nx, Ny = kwargs["Nx"], kwargs["Ny"]
    T = phi_list.shape[0]
    for i in range(30):
        plt.matshow(phi_list[i].reshape(Nx,Ny))
        plt.savefig('./phase_{}.png'.format(i))
        plt.close()
    for i in range(30, T, (T-30)//30):
        plt.matshow(phi_list[i].reshape(Nx,Ny))
        plt.savefig('./phase_{}.png'.format(i))
        plt.close()

