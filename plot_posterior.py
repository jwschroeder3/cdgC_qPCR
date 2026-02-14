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
trace = az.from_netcdf(fname)

az.plot_violin(data=trace, var_names=["primer_geno_effs"])
plt.show()
