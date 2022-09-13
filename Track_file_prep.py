# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python [conda env:anaconda-karollus-tfhub]
#     language: python
#     name: conda-env-anaconda-karollus-tfhub-py
# ---

# %% [markdown]
# # Track File Construction

# %% [markdown]
# ## Setup

# %%
import pandas as pd


# %%
def dedupe_tracks(dataset_tracks):
    dataset_dupes = dataset_tracks["description"].duplicated(keep=False)
    assert dataset_dupes.sum() > 0
    dataset_dupe_descs = dataset_tracks[dataset_dupes]["description"].unique()
    for dupe_desc in dataset_dupe_descs:
        duped_idxs = dataset_tracks[dataset_tracks["description"] == dupe_desc].index
        i = 1
        for idx in duped_idxs:
            prev = dataset_tracks.at[idx, "description"]
            dataset_tracks.at[idx, "description"] = prev + "_" + str(i)
            i += 1
    assert dataset_tracks["description"].duplicated(keep=False).sum() == 0


# %% [markdown]
# ### Enformer and Basenji2 Data

# %% colab={"base_uri": "https://localhost:8080/", "height": 240} id="OlE6JAVfI08a" outputId="61f72dd8-e5f5-4764-d765-b6cbd47f6e90"
# Download targets from Basenji2 dataset 
# Cite: Kelley et al Cross-species regulatory sequence activity prediction. PLoS Comput. Biol. 16, e1008050 (2020).
targets_txt = 'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_human.txt'
target_df = pd.read_csv(targets_txt, sep='\t')
target_df

# %% [markdown]
# ### Basenji1 Data

# %%
b1_target_df = pd.read_csv("Models/Basenji/basenji1_targets.txt", sep="\t", 
                           header=None, names=["id", "path", "description"]).reset_index()
b1_target_df

# %% [markdown]
# ## Weiss Tracks

# %% [markdown]
# Tracks for `weiss_ingenome` and `weiss_constructs`.

# %%
weiss_tracks = target_df.loc[
    target_df.description.str.contains("osteoblast", case=False) 
    | target_df.description.str.contains("neuron", case=False)
][["description", "index"]]
weiss_tracks

# %%
weiss_dupes = weiss_tracks["description"].duplicated(keep=False)
weiss_tracks[weiss_dupes]

# %%
prev = weiss_tracks.at[1644, "description"]
weiss_tracks.at[1644, "description"] = prev + "_1"

# %%
prev = weiss_tracks.at[3358, "description"]
weiss_tracks.at[3358, "description"] = prev + "_2"

# %%
print(weiss_tracks.at[1644, "description"])

# %%
print(weiss_tracks.at[3358, "description"])

# %%
weiss_tracks["description"].duplicated(keep=False).sum()

# %% [markdown]
# <div class="alert alert-block alert-warning">
# <b>Note:</b> Only activate following cell if you want to overwrite the old tracks!
# </div>

# %% [raw]
# weiss_tracks.to_csv("Data/Weiss_human_specific_variants/weiss_tracks.yaml",
#                    index=False, header=False, sep="\t")

# %% [markdown]
# ## K562 Tracks

# %% [markdown]
# Tracks for `bergmann_exp` and `bergmann_promoteronly`

# %%
k562_tracks = target_df.loc[
    target_df.description.str.contains("k562", case=False)
][["description", "index"]]
k562_tracks

# %%
k562_tracks["description"].duplicated(keep=False).sum()

# %%
k562_dupes = k562_tracks["description"].duplicated(keep=False)
k562_dupe_descs = k562_tracks[k562_dupes]["description"].unique()

# %%
for dupe_desc in k562_dupe_descs:
    duped_idxs = k562_tracks[k562_tracks["description"] == dupe_desc].index
    i = 1
    for idx in duped_idxs:
        prev = k562_tracks.at[idx, "description"]
        k562_tracks.at[idx, "description"] = prev + "_" + str(i)
        i += 1

# %%
k562_tracks["description"].duplicated(keep=False).sum()

# %% [markdown]
# <div class="alert alert-block alert-warning">
# <b>Note:</b> Only activate following cell if you want to overwrite the old tracks!
# </div>

# %% [raw]
# k562_tracks.to_csv("Data/Tracks/k562_tracks.yaml",
#                    index=False, header=False, sep="\t")

# %% [markdown]
# ## Kircher Tracks

# %% [markdown]
# Tracks for `kircher_ingenome`

# %% [markdown]
# ### Enformer/Basenji2

# %%
kircher_tracks = target_df.loc[
    target_df.description.str.contains("cage", case=False)
    | target_df.description.str.contains("dnase", case=False)
][["description", "index"]]
kircher_tracks

# %%
kircher_tracks["description"].duplicated(keep=False).sum()

# %%
kircher_dupes = kircher_tracks["description"].duplicated(keep=False)
kircher_dupe_descs = kircher_tracks[kircher_dupes]["description"].unique()

# %%
for dupe_desc in kircher_dupe_descs:
    duped_idxs = kircher_tracks[kircher_tracks["description"] == dupe_desc].index
    i = 1
    for idx in duped_idxs:
        prev = kircher_tracks.at[idx, "description"]
        kircher_tracks.at[idx, "description"] = prev + "_" + str(i)
        i += 1

# %%
kircher_tracks["description"].duplicated(keep=False).sum()

# %% [markdown]
# <div class="alert alert-block alert-warning">
# <b>Note:</b> Only activate following cell if you want to overwrite the old tracks!
# </div>

# %%
kircher_tracks.to_csv("Data/Kircher_saturation_mutagenesis/kircher_tracks.yaml",
                   index=False, header=False, sep="\t")

# %% [markdown]
# ### Basenji1

# %%
b1_kircher_tracks = b1_target_df.loc[
    b1_target_df.description.str.contains("cage", case=False)
    | b1_target_df.description.str.contains("dnase", case=False)
][["description", "index"]]
b1_kircher_tracks

# %%
dedupe_tracks(b1_kircher_tracks)

# %% [markdown] tags=[]
# <div class="alert alert-block alert-warning">
# <b>Note:</b> Only activate following cell if you want to overwrite the old tracks!
# </div>

# %%
b1_kircher_tracks.to_csv("Data/Kircher_saturation_mutagenesis/basenji1_kircher_tracks.yaml",
                   index=False, header=False, sep="\t")

# %% [markdown]
# ## Arensbergen Tracks

# %% [markdown]
# Tracks for `arensbergen_ingenome`.

# %% [markdown]
# ### Enformer/Basenji2

# %%
arensbergen_tracks = target_df.loc[
    target_df.description.str.contains("k562", case=False) 
    | target_df.description.str.contains("hepg2", case=False)
][["description", "index"]]
arensbergen_tracks

# %%
arensbergen_dupes = arensbergen_tracks["description"].duplicated(keep=False)
arensbergen_dupes.sum()

# %%
arensbergen_dupe_descs = arensbergen_tracks[arensbergen_dupes]["description"].unique()

# %%
for dupe_desc in arensbergen_dupe_descs:
    duped_idxs = arensbergen_tracks[arensbergen_tracks["description"] == dupe_desc].index
    i = 1
    for idx in duped_idxs:
        prev = arensbergen_tracks.at[idx, "description"]
        arensbergen_tracks.at[idx, "description"] = prev + "_" + str(i)
        i += 1

# %%
arensbergen_tracks["description"].duplicated(keep=False).sum()

# %% [markdown]
# <div class="alert alert-block alert-warning">
# <b>Note:</b> Only activate following cell if you want to overwrite the old tracks!
# </div>

# %% [raw]
# arensbergen_tracks.to_csv("Data/Arensbergen_sure_mpra/arensbergen_tracks.yaml",
#                    index=False, header=False, sep="\t")

# %% [markdown]
# ### Basenji1

# %%
b1_arensbergen_tracks = b1_target_df.loc[
    b1_target_df.description.str.contains("k562", case=False) 
    | b1_target_df.description.str.contains("hepg2", case=False)
][["description", "index"]]
b1_arensbergen_tracks

# %%
dedupe_tracks(b1_arensbergen_tracks)

# %% [markdown] tags=[]
# <div class="alert alert-block alert-warning">
# <b>Note:</b> Only activate following cell if you want to overwrite the old tracks!
# </div>

# %%
b1_arensbergen_tracks.to_csv("Data/Arensbergen_sure_mpra/basenji1_arensbergen_tracks.yaml",
                             index=False, header=False, sep="\t")

# %% [markdown]
# ## Arensbergen Plasmid Tracks

# %% [markdown]
# Tracks for `arensbergen_plasmid`.

# %%
arensplasmid_tracks = target_df.loc[
    (target_df.description.str.contains("k562", case=False) 
      | target_df.description.str.contains("hepg2", case=False))
    & (target_df.description.str.contains("cage", case=False)
      | target_df.description.str.contains("dnase", case=False))
][["description", "index"]]
arensplasmid_tracks

# %%
dedupe_tracks(arensplasmid_tracks)
arensplasmid_tracks

# %% [markdown]
# <div class="alert alert-block alert-warning">
# <b>Note:</b> Only activate following cell if you want to overwrite the old tracks!
# </div>

# %%
arensplasmid_tracks.to_csv("Data/Arensbergen_sure_mpra/arensbergen_plasmid_tracks.yaml",
                   index=False, header=False, sep="\t")

# %% [markdown]
# ## Cohen Tracks

# %% [markdown]
# Tracks for `cohen_tripseq`.

# %% [markdown]
# ### Enformer/Basenji2

# %%
cohen_tracks = target_df.loc[
    target_df.description.str.contains("k562", case=False)
    & (target_df.description.str.contains("cage", case=False)
      | target_df.description.str.contains("dnase", case=False))
][["description", "index"]]
cohen_tracks

# %%
dedupe_tracks(cohen_tracks)
cohen_tracks

# %% [markdown]
# <div class="alert alert-block alert-warning">
# <b>Note:</b> Only activate following cell if you want to overwrite the old tracks!
# </div>

# %%
cohen_tracks.to_csv("Data/Cohen_genomic_environments/cohen_tracks.yaml",
                   index=False, header=False, sep="\t")

# %% [markdown]
# ### Basenji1

# %%
b1_cohen_tracks = b1_target_df.loc[
    b1_target_df.description.str.contains("k562", case=False)
    & (b1_target_df.description.str.contains("cage", case=False)
      | b1_target_df.description.str.contains("dnase", case=False))
][["description", "index"]]
b1_cohen_tracks

# %% tags=[]
dedupe_tracks(b1_cohen_tracks)

# %% [markdown] tags=[]
# <div class="alert alert-block alert-warning">
# <b>Note:</b> Only activate following cell if you want to overwrite the old tracks!
# </div>

# %%
b1_cohen_tracks.to_csv("Data/Cohen_genomic_environments/basenji1_cohen_tracks.yaml",
                             index=False, header=False, sep="\t")

# %% [markdown]
# ## Abramov Tracks

# %%
with open("Data/Abramov_ASB/tf_list.txt") as f:
    tf_list = f.readlines()
tf_list = list(map(str.rstrip, tf_list))

# %%
assert len(tf_list) == len(set(tf_list))

# %% [markdown]
# Construct a regex because `Series.str.contains` does not understand lists:

# %%
tf_regex = "|".join(tf_list)

# %%
abramov_tracks = target_df.loc[
    target_df.description.str.contains(tf_regex, case=True) 
    & target_df.description.str.contains("CHIP", case=False)
][["description", "index"]]
abramov_tracks

# %%
abramov_dupes = abramov_tracks["description"].duplicated(keep=False)
abramov_dupes.sum()

# %%
abramov_dupe_descs = abramov_tracks[abramov_dupes]["description"].unique()

# %%
for dupe_desc in abramov_dupe_descs:
    duped_idxs = abramov_tracks[abramov_tracks["description"] == dupe_desc].index
    i = 1
    for idx in duped_idxs:
        prev = abramov_tracks.at[idx, "description"]
        abramov_tracks.at[idx, "description"] = prev + "_" + str(i)
        i += 1

# %%
abramov_tracks["description"].duplicated(keep=False).sum()

# %% [markdown]
# <div class="alert alert-block alert-warning">
# <b>Note:</b> Only activate following cell if you want to overwrite the old tracks!
# </div>

# %% [raw]
# abramov_tracks.to_csv("Data/Abramov_ASB/abramov_tracks.yaml",
#                    index=False, header=False, sep="\t")

# %% [markdown]
# ## Fulco Preprocessing Tracks

# %%
fulcopp_tracks = target_df.loc[
    target_df.description.str.contains("K562", case=False)
    & (target_df.description.str.contains("CAGE", case=False)
    | target_df.description.str.contains("DNASE", case=False))
][["description", "index"]]
fulcopp_tracks

# %%
fulcopp_dupes = fulcopp_tracks["description"].duplicated(keep=False)
fulcopp_dupes.sum()

# %%
fulcopp_dupe_descs = fulcopp_tracks[fulcopp_dupes]["description"].unique()

# %%
for dupe_desc in fulcopp_dupe_descs:
    duped_idxs = fulcopp_tracks[fulcopp_tracks["description"] == dupe_desc].index
    i = 1
    for idx in duped_idxs:
        prev = fulcopp_tracks.at[idx, "description"]
        fulcopp_tracks.at[idx, "description"] = prev + "_" + str(i)
        i += 1

# %% [markdown]
# <div class="alert alert-block alert-warning">
# <b>Note:</b> Only activate following cell if you want to overwrite the old tracks!
# </div>

# %%
fulcopp_tracks.to_csv("Data/Fulco_CRISPRi/preprocessing_tracks.yaml",
                      index=False, header=False, sep="\t")

# %% [markdown]
# ## Sahu Tracks

# %%
sahu_tracks = target_df.loc[
    ((target_df.description.str.contains("CAGE", case=False) | target_df.description.str.contains("DNASE", case=False))
    & (target_df.description.str.contains("K562", case=False) | target_df.description.str.contains("colon", case=False)))
    | (target_df.description.str.contains("CHIP", case=False)) 
][["description", "index"]]
sahu_tracks

# %%
sahu_dupes = sahu_tracks["description"].duplicated(keep=False)
sahu_dupes.sum()

# %%
sahu_dupe_descs = sahu_tracks[sahu_dupes]["description"].unique()

# %%
for dupe_desc in sahu_dupe_descs:
    duped_idxs = sahu_tracks[sahu_tracks["description"] == dupe_desc].index
    i = 1
    for idx in duped_idxs:
        prev = sahu_tracks.at[idx, "description"]
        sahu_tracks.at[idx, "description"] = prev + "_" + str(i)
        i += 1

# %% [markdown]
# <div class="alert alert-block alert-warning">
# <b>Note:</b> Only activate following cell if you want to overwrite the old tracks!
# </div>

# %%
sahu_tracks.to_csv("Data/Sahu_enhancers/sahu_tracks.yaml",
                      index=False, header=False, sep="\t")

# %% [markdown]
# ## All Tracks (avsec_enhancercentered)

# %% [markdown]
# ### Enformer

# %%
alltracks = target_df[["description", "index"]]
alltracks

# %%
dedupe_tracks(alltracks)

# %% [markdown]
# <div class="alert alert-block alert-warning">
# <b>Note:</b> Only activate following cell if you want to overwrite the old tracks!
# </div>

# %%
alltracks.to_csv("Data/Tracks/all_tracks_deduped.yaml",
                      index=False, header=False, sep="\t")
