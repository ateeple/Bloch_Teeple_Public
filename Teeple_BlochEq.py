# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 12:16:40 2025

@author: Angela Teeple
"""

import numpy as np


def bloch_eq(t, M, delta_omega, AM, phase, T1, T2, M0, gamma):
    Mx, My, Mz = M
    B1x = (AM / gamma) * np.cos(phase)
    B1y = (AM / gamma) * np.sin(phase)
    
    dMx_dt = gamma * 2 * np.pi * (My * delta_omega / gamma - Mz * B1y)
    dMy_dt = gamma * 2 * np.pi * (Mz * B1x - Mx * delta_omega / gamma)
    dMz_dt = gamma * 2 * np.pi * (Mx * B1y - My * B1x)

    return np.array([dMx_dt, dMy_dt, dMz_dt])

def rk_step(h, t, M, f, *args):
    k1 = f(t, M, *args)
    k2 = f(t + 0.5 * h, M + 0.5 * h * k1, *args)
    k3 = f(t + 0.5 * h, M + 0.5 * h * k2, *args)
    k4 = f(t + h, M + h * k3, *args)
    return M + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def Bloch_sim(M0, T1, T2, AM_function, PM_function, tstep, off_Res, gamma, Tp):
    Mx_final = []
    My_final = []
    Mz_final = []
    Nt_Pts = len(AM_function)
    M_state = np.array([np.zeros(Nt_Pts), np.zeros(Nt_Pts), M0 * np.ones(Nt_Pts)])  # Initial magnetization

    for jj in range(Nt_Pts - 1):
        AM = AM_function[jj]
        phase = PM_function[jj]

        M_state[:, jj + 1] = rk_step(tstep, jj * tstep, M_state[:, jj], bloch_eq, off_Res, AM, phase, T1, T2, M0, gamma)

        Mx_final.append(M_state[0, jj + 1])
        My_final.append(M_state[1, jj + 1])
        Mz_final.append(M_state[2, jj + 1])

    return Mx_final, My_final, Mz_final