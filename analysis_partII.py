import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns
import re
import glob

import ipdb

def summarize(res):
    summary_geno = []
    summary_locus = []
    summary_mean = []
    summary_median = []
    summary_sd = []
    mean = np.mean(res, axis=2)
    median = np.median(res, axis=2)
    stdev = np.std(res, axis=2)
    for (i,geno) in enumerate(genotypes):
        for (j,locus) in enumerate(loci):
            summary_geno.append(geno)
            summary_locus.append(locus)
            summary_mean.append(mean[i,j])
            summary_median.append(median[i,j])
            summary_sd.append(stdev[i,j])

    out_summary = pd.DataFrame({
        "genotype": summary_geno,
        "locus": summary_locus,
        "mean": summary_mean,
        "median": summary_median,
        "stdev": summary_sd,
    })
    return out_summary

def arviz_summarize(res):
    
    summary_geno = []
    summary_locus = []
    summary_mean = []
    summary_lower = []
    summary_upper = []
    for (i,geno) in enumerate(genotypes):
        for (j,locus) in enumerate(loci):
            summary_geno.append(geno)
            summary_locus.append(locus)
            dat = az.convert_to_inference_data(res[i,j,:])
            dat_summary = az.summary(dat, hdi_prob=0.95)
            summary_mean.append(dat_summary["mean"][0])
            summary_lower.append(dat_summary["hdi_2.5%"][0])
            summary_upper.append(dat_summary["hdi_97.5%"][0])

    out_summary = pd.DataFrame({
        "genotype": summary_geno,
        "locus": summary_locus,
        "mean": summary_mean,
        "lower": summary_lower,
        "upper": summary_upper,
    })
    return out_summary



def subtract_rpoC(samps, key_df):
    genotypes = key_df.geno.unique()
    each_df = []
    for genotype in genotypes:
        geno_df = key_df[key_df.geno == genotype]
        rpoC_idx = geno_df.idx[geno_df.locus == "rpoC"]
        rpoC_samps = samps.primer_geno_effs[rpoC_idx,:].values
        for loc in ["cdgC", "SnrC"]:
            loc_idx = geno_df.idx[geno_df.locus == loc]
            loc_samps = samps.primer_geno_effs[loc_idx,:].values
            loc_effs = (loc_samps - rpoC_samps).flatten()
            tmp_df = pd.DataFrame(data={
                "value": loc_effs,
                "geno": [genotype for _ in range(len(loc_effs))],
                "locus": [loc for _ in range(len(loc_effs))],
            })
            each_df.append(tmp_df)
    results_df = pd.concat(each_df)
    return results_df

def get_evidence_ratios(samps_df, comparison_dict):
    K_df = []
    rope = 0.5
    for locus,geno_comps in comparison_dict.items():
        for comp_name,comp_key in geno_comps.items():

            #ipdb.set_trace()

            numer_samps = samps_df.value[
                (samps_df.locus == locus)
                & (samps_df.geno == comp_key["numerator"])
            ].values
            denom_samps = samps_df.value[
                (samps_df.locus == locus)
                & (samps_df.geno == comp_key["denominator"])
            ].values
            diffs = numer_samps - denom_samps
            n_gt = np.sum(diffs > 0.0)
            n_le = len(diffs) - n_gt
            K_gt = n_gt / n_le
            K_le = n_le / n_gt 

            n_in_rope = np.sum(np.logical_and(diffs >= -rope, diffs <= rope))
            n_not_in_rope = len(diffs) - n_in_rope
            K_in = n_in_rope / n_not_in_rope
            K_out = n_not_in_rope / n_in_rope

            tmp_df = pd.DataFrame(data={
                "locus": [locus],
                "comparison": [comp_name],
                "Kgt": K_gt,
                "Kle": K_le,
                "K_in_rope": K_in,
                "K_outside_rope": K_out,
            })
            K_df.append(tmp_df)

    final_df = pd.concat(K_df)
    return final_df


infile_qpcr = "qPCR_results_noTemp_noD7_filtered.csv"

param_summary_file = "fit_effs_summary.txt"
lookup_file = "fit_effsparameter_names.txt"
infile = "fit_effs_samples.cdf"

outprefix = "results"
out_raneff_summary_file = outprefix + "_raneffs_summary.csv"
out_fixedeff_summary_file = outprefix + "_fixeffs_summary.csv"

qpcr_tab = pd.read_csv( infile_qpcr )
ct_dat = qpcr_tab[np.logical_or(qpcr_tab.type == "reference", qpcr_tab.type == "sample")].copy()

trace = az.from_netcdf(infile)
trace.sel(draw=slice(2500,None), inplace=True)
#ipdb.set_trace()

dil_dat = qpcr_tab[
    qpcr_tab.type == "dill_ser"
]
all_primers = list(set(dil_dat["primer_desc"]))

ct_dat["primer_inds"] = [all_primers.index(x) for x in ct_dat["primer_desc"] ]

## also figure out indexing for other factors that enter into the model
ct_dat["samp_rand_name"] = ct_dat["sample_name"] + ct_dat["date"]
ct_dat["rep_rand_name"] = ct_dat["sample_name"] + ct_dat["bio_Rep"].astype("str")
ct_dat["primer_rand_name"] = ct_dat["primer_desc"] + ct_dat["date"]
ct_dat["primer_geno"] = ct_dat["primer_desc"] + "_" + ct_dat["geno"]
ct_dat["samp_primer"] = ct_dat["sample_name"] + ct_dat["primer_desc"]

primer_geno_ints = list(set(ct_dat["primer_geno"]))
all_samps = list(set(ct_dat["sample_name"]))
samp_rands = list(set(ct_dat["samp_rand_name"]))
primer_rands = list(set(ct_dat["primer_rand_name"]))
samp_primers = list(set(ct_dat["samp_primer"]))
rep_rands = list(set(ct_dat["rep_rand_name"]))

ct_dat["primer_geno_ind"] = [primer_geno_ints.index(x) for x in ct_dat["primer_geno"]]
ct_dat["sample_ind"] = [all_samps.index(x) for x in ct_dat["sample_name"]]
ct_dat["samp_rand_ind"] = [samp_rands.index(x) for x in ct_dat["samp_rand_name"]]
ct_dat["rep_rand_ind"] = [rep_rands.index(x) for x in ct_dat["rep_rand_name"]]
ct_dat["primer_rand_ind"] = [primer_rands.index(x) for x in ct_dat["primer_rand_name"]]
ct_dat["samp_primer_ind"] = [samp_primers.index(x) for x in ct_dat["samp_primer"]]

summary_df = pd.read_csv(param_summary_file)

lut = pd.read_csv(lookup_file, sep=" = ", names=["id","name"], engine="python")
lut[["param", "tmp"]] = lut["id"].str.split(r"\[", expand=True)
lut["idx"] = lut["tmp"].str.strip().str.rstrip(r"\]").astype("int")
geno_betas = lut[lut["param"] == "primer_geno_effs"].copy()
#pe_df = lut[lut["param"] == "log2_primer_effs"].copy()
#print(lut)

geno_names = list(geno_betas["name"])

geno_list = []
locus_list = []
for s in geno_names:
    e = s.split("_")
    geno = "_".join(e[1:len(e)])
    geno_list.append(geno)
    locus_list.append(e[0])

geno_betas["geno"] = geno_list
geno_betas["locus"] = locus_list

#geno_betas = geno_betas[geno_betas["geno"] != "py79"].copy()
geno_betas["search"] = geno_betas["id"].str.replace(" ", "")

#ipdb.set_trace()
samps = az.extract(trace)
results_df = subtract_rpoC(samps, geno_betas)
results_df["QrgB"] = results_df["geno"].apply(lambda x: "inactive" if "star" in x else "active")
genotypes_of_interest = results_df[
    (results_df.geno == "dL_dR_B")
    | (results_df.geno == "dL_dR_Bstar")
    | (results_df.geno == "dL_wt_B")
    | (results_df.geno == "dL_wt_Bstar")
]
genotypes_of_interest.loc[:,"genotype"] = genotypes_of_interest["geno"].str.split("_").apply(lambda x: '_'.join(x[0:-1]))

comparisons = {
    "SnrC": {
        "dR_B_vs_wt_B": {"numerator": "dL_dR_B", "denominator": "dL_wt_B"},
        "dR_Bstar_vs_wt_Bstar": {"numerator": "dL_dR_Bstar", "denominator": "dL_wt_Bstar"},
        "dR_B_vs_dR_Bstar": {"numerator": "dL_dR_B", "denominator": "dL_dR_Bstar"},
        "wt_B_vs_wt_Bstar": {"numerator": "dL_wt_B", "denominator": "dL_wt_Bstar"},
    },
    "cdgC": {
        "wt_B_vs_wt_Bstar": {"numerator": "dL_wt_B", "denominator": "dL_wt_Bstar"},
        "dR_B_vs_dR_Bstar": {"numerator": "dL_dR_B", "denominator": "dL_dR_Bstar"},
        "dR_B_vs_wt_B": {"numerator": "dL_dR_B", "denominator": "dL_wt_B"},
    },
}
K_df = get_evidence_ratios(results_df, comparisons)
K_df.to_csv("evidence_ratios.csv", index=False)

ax = sns.violinplot(
    data = results_df[results_df.geno.isin(
        [
            "dL_wt_B",
            "dL_wt_Bstar",
            "dL_dR_B",
            "dL_dR_Bstar",
            "dL_wt_PVC2224",
            "dL_dR_PVC2224",
            "CW2034_dJ_B",
        ]
    )],
    hue = "locus",
    x = "geno",
    y = "value",
    #split = True,
    order = [
        "dL_wt_B",
        "dL_wt_Bstar",
        "dL_dR_B",
        "dL_dR_Bstar",
        #"dL_wt_B_anticdgCJ",
        #"dL_wt_B_sensecdgCJ",
        "dL_wt_PVC2224",
        "dL_dR_PVC2224",
        "CW2034_dJ_B",
        #"CW2035_emptyVector",
    ],
)
ax.invert_yaxis()
plt.ylabel("change in Cq vs. rpoC vs. baseline")
plt.xlabel("Genotype")
plt.hlines(y=0.0, xmin=-0.5, xmax=6.5, colors='black', linestyles='dashed')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("20250724_rtqPCR.pdf")
plt.savefig("20250724_rtqPCR.png")


#primer_betas = lut[lut["param"] == "pulldown_effs"].copy()
#primer_betas["search"] = primer_betas["id"].str.replace(" ", "")

with_summary = geno_betas.merge(summary_df[["Unnamed: 0", "mean", "hdi_3%", "hdi_97%"]], how="left", left_on="search", right_on="Unnamed: 0")
with_summary[["name","geno","locus","mean", "hdi_3%", "hdi_97%"]].to_csv(
    out_fixedeff_summary_file, index=False
)

results_df.to_csv("results_df.csv", index=False)
## remove burn-in and stack the chains
#samps = trace.sel(draw=slice(2500,None))
#summary = az.summary(samps, stat_focus="median").reset_index()
#print(summary)
#print(np.any(summary["r_hat"] > 1.02))
#print(summary[summary["r_hat"] > 1.02])
#summary["var_name"] = summary["index"].str.split("[", expand=True)[0]
#summary["idx"] = summary["index"].str.split("[", expand=True)[1].str.strip("]")
#samples = samps.stack(sample=["chain","draw"])

