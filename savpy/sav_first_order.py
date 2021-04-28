# coding:utf-8
"""
@author: Ding Zhao
@file: sav_first_order.py
@time: 2021/4
@file_desc:
"""

import numpy as np
from .utils import get_args, phi_init, r_init, compute_b, compute_g, inv_A, compute_r, animate

def sav_first_order(**kwargs):
	if len(kwargs) == 0:
		# kwargs = {"Nx":128, "Ny":128, "Lx":1, "Ly":1, "T":5000, "dt":1e-5, "beta":2, "C":0, "eps":1e-2}
		conf = get_args()
		Nx, Ny, Lx, Ly, T, dt, beta, C, eps = conf.Nx, conf.Ny, conf.Lx, conf.Ly, conf.T, conf.dt, conf.beta, conf.C, conf.eps
		kwargs = {"Nx":Nx, "Ny":Ny, "Lx":Lx, "Ly":Ly, "T":T, "dt":dt, "beta":beta, "C":C, "eps":eps}
	else:
		Nx = kwargs['Nx']; Ny = kwargs['Ny']; Lx = kwargs['Lx']; Ly = kwargs['Ly']; T = kwargs['T']
	dt = kwargs['dt']; beta = kwargs['beta']; C = kwargs['C']; eps = kwargs['eps']
	hx, hy = Lx/Nx, Ly/Ny

	x = hx * np.arange(Nx)
	y = hy * np.arange(Ny)
	xx, yy = np.meshgrid(x, y)
	phi0 = phi_init(xx, yy, **kwargs)
	r0 = r_init(phi0, **kwargs)
	r0 = r0.real
	phi_list = np.zeros((T,Nx*Ny))
	r_list = np.zeros((T, Nx*Ny))
	phi_list[0] = phi0.ravel()
	r_list[0] = r0.ravel()
	time = 0

	for t in range(1, T):
		time+=dt
		phi_star = phi0

		b = compute_b(phi_star, **kwargs)
		g = compute_g(phi0, r0, b, **kwargs)
		psi1 = inv_A(b, **kwargs)
		psi2 = inv_A(g, **kwargs)
		gamma = np.fft.fft2(b*psi1)[0,0]*hx*hy

		bphi = np.fft.fft2(b*psi2)[0,0]*hx*hy/(1+dt*gamma/2)

		phi = psi2 - dt/2*bphi*psi1

		r0 = compute_r(phi, phi0, r0, b, **kwargs)
		r0 = r0.real
		phi = phi.real
		phi0 = phi
		phi_list[t] = phi.ravel()
		r_list[t] = r0.ravel()
	return phi_list, r_list

if __name__ == '__main__':
	kwargs = {"Nx":128, "Ny":128, "Lx":1, "Ly":1, "T":5000, "dt":1e-5, "beta":2, "C":0, "eps":1e-2}
	phi_list, r_list = sav_first_order(**kwargs)
	animate(phi_list, **kwargs)