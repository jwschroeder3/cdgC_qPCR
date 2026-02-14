# this should be run in my pymc environment
# in this program, I build a Bayesian model to fit the primer efficiencies for
# all of the data in this directory

# I will plot diagnostics on these estimates, but in practice I may want to
#  combine this inference with additional data to fit my real quantities of interest

import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns

import ipdb

n_chains = 6

infile_qpcr = "qPCR_results_noTemp_noD7_filtered.csv"

outprefix="fit_effs"

qpcr_tab = pd.read_csv( infile_qpcr )

print("Setting up data for fitting")
# set up some additional useful columns
qpcr_tab["log2dil"] = np.log2( qpcr_tab.relative_concentration )

# now pull out the dilution data that we will use for fitting
# in the process, we discretize the primer IDs since we need those later
dil_dat = qpcr_tab[qpcr_tab.type == "dill_ser"]
#print(dil_dat)
sns.scatterplot(dil_dat, x="log2dil", y="Cq", hue="primer_desc");
plt.savefig("cq_vs_log2dilution.png")
plt.close()

all_primers = list(set(dil_dat["primer_desc"]))
dil_dat["primer_inds"] = [all_primers.index(x) for x in dil_dat["primer_desc"] ]
dil_dat["dil_rand_name"] = dil_dat["date"] + "_" + dil_dat["relative_concentration"].astype("string") + "_" + dil_dat["sample_name"]
dil_rands = list(set(dil_dat["dil_rand_name"]))
dil_dat["dil_rand_ind"] = [dil_rands.index(x) for x in dil_dat["dil_rand_name"]]

# and also pull out the data for ct fitting
ct_dat = qpcr_tab[np.logical_or(qpcr_tab.type == "reference", qpcr_tab.type == "sample")].copy()
ct_dat["primer_inds"] = [all_primers.index(x) for x in ct_dat["primer_desc"] ]

## also figure out indexing for other factors that enter into the model
ct_dat["samp_rand_name"] = ct_dat["sample_name"] + ct_dat["date"]
ct_dat["rep_rand_name"] = ct_dat["sample_name"] + ct_dat["bio_Rep"].astype("str")
ct_dat["primer_rand_name"] = ct_dat["primer_desc"] + ct_dat["date"]
ct_dat["primer_geno"] = ct_dat["primer_desc"] + "_" + ct_dat["geno"]
ct_dat["samp_primer"] = ct_dat["sample_name"] + ct_dat["primer_desc"]

all_genos = list(set(ct_dat["geno"]))
primer_geno_ints = list(set(ct_dat["primer_geno"]))
all_samps = list(set(ct_dat["sample_name"]))
samp_rands = list(set(ct_dat["samp_rand_name"]))
rep_rands = list(set(ct_dat["rep_rand_name"]))
primer_rands = list(set(ct_dat["primer_rand_name"]))
samp_primers = list(set(ct_dat["samp_primer"]))

ct_dat["geno_ind"] = [all_genos.index(x) for x in ct_dat["geno"]]
ct_dat["primer_geno_ind"] = [primer_geno_ints.index(x) for x in ct_dat["primer_geno"]]
ct_dat["sample_ind"] = [all_samps.index(x) for x in ct_dat["sample_name"]]
ct_dat["samp_rand_ind"] = [samp_rands.index(x) for x in ct_dat["samp_rand_name"]]
ct_dat["rep_rand_ind"] = [rep_rands.index(x) for x in ct_dat["rep_rand_name"]]
ct_dat["primer_rand_ind"] = [primer_rands.index(x) for x in ct_dat["primer_rand_name"]]
ct_dat["samp_primer_ind"] = [samp_primers.index(x) for x in ct_dat["samp_primer"]]

#ipdb.set_trace()
#ct_dat["primer_geno_ind"] = [all_ints.index(x) for x in ct_dat["primer_geno"]]

distinct_geno_mask = ~ct_dat.duplicated("samp_primer_ind")
print(distinct_geno_mask)
distinct_geno_inds = np.where(distinct_geno_mask)[0]
print(ct_dat[["sample_name","primer_desc"]])
# I manually set the NA values from br.io to 999 to encode them as censored here.
# In the model, censored values are said to come from the same distribution, but
# truncated at a Cq of 40
is_uncensored = ct_dat.Cq != 999
is_censored = ct_dat.Cq == 999
uncensored = ct_dat[is_uncensored]
right_censored = ct_dat[is_censored]
n_right_censored = right_censored.shape[0]
        
# set up a model to fit the primer efficiency for each primer
this_model=pm.Model()

with this_model:

    # we define the model in two pieces: first, the fits for primer efficiency, and second, the fits to get the ddCt values that we want
    # the primer efficiency part begins here:

    ## here is what we want to fit -- the primer efficiency of each primer pair
    ## note that we also have an intercept for each primer, but we care less about this

    log2_inv_primer_effs = pm.Normal(
        'log2_inv_primer_effs',
        mu=1,
        sigma=0.5,
        shape=len(all_primers),
    )

    dil_primer_intercepts = pm.Gamma("dil_primer_intercepts", alpha=25, beta=1, shape=len(all_primers))

    dil_sigma = pm.HalfCauchy( 'dil_sigma', beta=5 )
    dil_nu = pm.Gamma('dil_nu', alpha=5, beta=0.5)
 
    ## now connect the observed Ct values with those parameters
    all_dilutions = -1 * dil_dat.log2dil.to_numpy()
    dil_cts = dil_dat.Cq.to_numpy()
    all_primer_inds = dil_dat.primer_inds.to_numpy()
    dil_rand_inds = dil_dat.dil_rand_ind.to_numpy()

    dil_mu = (
        # primer intercepts is just for the genotype for which the dilution series was performed
        dil_primer_intercepts[all_primer_inds]
        + log2_inv_primer_effs[all_primer_inds] * all_dilutions
    )

    dil_Ct_vals = pm.StudentT(
        'dil_Ct_vals',
        mu=dil_mu,
        observed=dil_cts,
        sigma=dil_sigma,
        nu=dil_nu,
    )

    # and now here is the part to fit the delta Cts

    # in general, we assume that the Ct for each well has the form:
    # Ct = intercept(primer) + intercept(sample) + pulldown(primer) + pulldown:condition(primer) + (1|sample_repid)
    # I may want to add in additional random effect terms?
    # note that the primer based terms hold over from the part of the model noted above

    # here are the additional parameters we need to fit
    #sample_intercepts = pm.Normal('sample_intercepts', mu=0, sigma=5, shape=len(all_samps))
    #geno_intercepts = pm.Normal('geno_intercepts', mu=0, sigma=5, shape=len(all_genos))

    #pulldown_effs = pm.Normal('pulldown_effs', mu=0, sigma=5, shape=len(all_primers))
    #interaction_effs = pm.Normal('interaction_effs', mu=0, sigma=5, shape=len(all_ints))

    ct_primer_inds = ct_dat.primer_inds.to_numpy()
    #ct_geno_inds = ct_dat.geno_ind.to_numpy()
    #ct_samp_primer_inds = ct_dat.samp_primer_ind.to_numpy()
    ct_primer_geno_inds = ct_dat.primer_geno_ind.to_numpy()
    ct_rep_geno_inds = ct_dat.rep_rand_ind.to_numpy()
    #is_not_dL_dR_Bstar = 1.0 * (ct_dat["geno"].to_numpy() != "dL_dR_Bstar")
    is_not_rpoC = 1.0 * (ct_dat["type"].to_numpy() == "sample")

    #rpoC_intercept = pm.Normal("rpoC_intercept", mu=0, sigma=1, shape=1)
    primer_intercepts = pm.Normal("primer_intercepts", mu=0, sigma=8, shape=len(all_primers))
    #samp_primer_effs = pm.Normal('samp_primer_effs', mu=0, sigma=5, shape=len(samp_primers))
    primer_geno_effs = pm.Normal('primer_geno_effs', mu=0, sigma=5, shape=len(primer_geno_ints))
    geno_rep_effs = pm.Gamma('geno_rep_effs', alpha=25, beta=1, shape = len(rep_rands))

    ct_sigma = pm.HalfCauchy( 'ct_sigma', beta=5 )
    ct_nu = pm.Gamma('ct_nu', alpha=5, beta=0.5)
 
    # censored vs uncensored is sliced below in the likelihoods
    samp_cts = ct_dat.Cq.to_numpy()

    # the Cq is equal to inverse log(primer efficiency) times log-quantity, where log-quantity is inferred
    ct_samp_mu = log2_inv_primer_effs[ct_primer_inds] * (
        # this term fits the amount of sample in a starting tube
        geno_rep_effs[ct_rep_geno_inds]
        # primer intercepts will be relative to rpoC
        + primer_intercepts[ct_primer_inds]
        # these are the terms of interest, and I'll have to back out the rpoC:genotype for each
        #   cdgC:genotype and SnrC:genotype to get the true measure of interest here.
        + primer_geno_effs[ct_primer_geno_inds]
    )

    samp_Ct_vals = pm.StudentT(
        'samp_Ct_vals',
        mu=ct_samp_mu[np.array(is_uncensored)],
        sigma=ct_sigma,
        nu=ct_nu,
        observed=samp_cts[np.array(is_uncensored)],
    )
    right_censored = pm.StudentT(
        "right_censored",
        mu=ct_samp_mu[np.array(is_censored)],
        sigma=ct_sigma,
        nu=ct_nu,
        transform=pm.distributions.transforms.Interval(37, None),
        shape=int(n_right_censored),
        initval=np.full(n_right_censored, 38),
    )

with this_model:
    #trace = pm.sample( 50, chains=1, tune=25, target_accept=0.95, backend="JAX" )
    trace = pm.sample( 5000, chains=n_chains, cores=n_chains, tune=2500, target_accept=0.99, max_treedepth=15, backend="JAX" )
    pm.sample_posterior_predictive(trace, extend_inferencedata=True)

with this_model:
    az.summary(trace).to_csv(outprefix + "_summary.txt")
    trace.to_netcdf(outprefix + "_samples.cdf")
    with open(outprefix + "parameter_names.txt", "w") as ostr:
        #for i,x in enumerate( all_samps):
        #    ostr.write("sample_intercepts[% 3i] = %s\n" % (i,x))

        for i,x in enumerate( all_genos):
            ostr.write("geno_rep_effs[% 3i] = %s\n" % (i,x))

        #for i,x in enumerate( samp_primers):
        #    ostr.write("samp_primer_effs[% 3i] = %s\n" % (i,x))

        for i,x in enumerate( primer_geno_ints):
            ostr.write("primer_geno_effs[% 3i] = %s\n" % (i,x))

    for param in [
            ["log2_inv_primer_effs", "dil_primer_intercepts"], #["dil_rand_effs"],
            #["sample_intercepts", "samp_primer_effs", 
            ["primer_intercepts", "primer_geno_effs", "geno_rep_effs"]
    ]:
        plt.figure()
        az.plot_trace(trace, var_names=param)
        param_str = "_".join(param)
        plt.savefig(outprefix + f'_{param_str}_trace.png')
     
