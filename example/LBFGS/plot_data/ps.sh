#!/bin/bash

rm proc000000_rho_kappa_mu_kernel.dat

./pre_plot

python plot_kernel_vs.py proc000000_rho_kappa_mu_kernel.dat 0
python plot_kernel_vp.py proc000000_rho_kappa_mu_kernel.dat 0
