# this should be run in my pymc environment

import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns

import ipdb

fname = "fit_effs_samples.cdf"
idata = az.from_netcdf(fname)

az.plot_ppc(idata, num_pp_samples=100);
plt.show()

