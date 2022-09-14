# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python [conda env:anaconda-sequencemodelbenchmark]
#     language: python
#     name: conda-env-anaconda-sequencemodelbenchmark-py
# ---

# %% [markdown]
# # Enformer experiments
#
# The following notebook contains all the analysis for our paper **Current sequence-based models of gene expression captures causal determinants of promoters but mostly ignore distal enhancers**. 

# %% [markdown]
# # Setup

# %% [markdown] id="MCDk7UQPG0Lr"
# ## Imports

# %% id="NRI9KisU11bM"
import itertools
import functools
import collections
import random
import re
import glob
import math
import os
import json
import pickle

import pyranges as pr

#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

#import joblib
import gzip
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import seaborn as sns
import statannot
import plotnine as p9
import sklearn
from sklearn import ensemble
from sklearn import pipeline
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# %% id="hTGOLrbZxNHK"
import tensorflow_hub as hub
import tensorflow as tf
# Make sure the GPU is enabled 
assert tf.config.list_physical_devices('GPU')

# %%
from Code.Utilities.seq_utils import one_hot_encode, rev_comp_sequence, rev_comp_one_hot, compute_offset_to_center_landmark
from Code.Utilities.enformer_utils import *

import Code.Utilities.basenji2_utils as basenji2_utils
import Code.Utilities.basenji1_utils as basenji1_utils
import kipoi

# %% id="g0F1A9AaCrkQ"
transform_path = 'gs://dm-enformer/models/enformer.finetuned.SAD.robustscaler-PCA500-robustscaler.transform.pkl'
model_path = 'https://tfhub.dev/deepmind/enformer/1'
fasta_file = 'Data/Genome/genome.fa'
fasta_file_hg19 = 'Data/Genome/genome_hg19.fa'
gtf_file = 'Data/Genome/genes.gtf'

# %% [markdown] id="Q8ZhswycGux3"
# ## Download files

# %% colab={"base_uri": "https://localhost:8080/", "height": 240} id="OlE6JAVfI08a" outputId="61f72dd8-e5f5-4764-d765-b6cbd47f6e90"
# Download targets from Basenji2 dataset 
# Cite: Kelley et al Cross-species regulatory sequence activity prediction. PLoS Comput. Biol. 16, e1008050 (2020).
targets_txt = 'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_human.txt'
df_targets = pd.read_csv(targets_txt, sep='\t')
df_targets.head(3)

# %%
df_targets.to_csv("Data/Targets/targets.tsv",sep="\t",index=None)

# %% [markdown] id="dowTJknFJOHu"
# Download and index the reference genome fasta file
#
# Credit to Genome Reference Consortium: https://www.ncbi.nlm.nih.gov/grc
#
# Schneider et al 2017 http://dx.doi.org/10.1101/gr.213611.116: Evaluation of GRCh38 and de novo haploid genome assemblies demonstrates the enduring quality of the reference assembly

# %% colab={"base_uri": "https://localhost:8080/"} id="flOUYxP7Fjvh" outputId="2f729a7e-1916-4998-ef4e-1794079037ea"
# !wget -O - http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz | gunzip -c > {fasta_file}
pyfaidx.Faidx(fasta_file)

# %% [markdown]
# Download and index hg19 reference genome fasta file

# %%
# !wget -O - http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz | gunzip -c > {fasta_file_hg19}
pyfaidx.Faidx(fasta_file_hg19)


# %% [markdown] id="A5xbQNZ6ljxm"
# Download the reference gtf

# %% colab={"base_uri": "https://localhost:8080/"} id="LjygWNOtlkN3" outputId="cd494e4e-4d3d-4fb3-d8da-3479dd291a7c"
# !wget -O - https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_39/gencode.v39.basic.annotation.gtf.gz | gunzip -c > {gtf_file}

# %% [markdown] id="Omj-KERcwSdB"
# ## Utility Code

# %%
def pearsonr_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = scipy.stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = scipy.stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

def pearson_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return scipy.stats.pearsonr(y, y_pred)[0]

class Kmerizer:
    
    def __init__(self, k, log=False, divide=False):
        self.k = k
        self.kmers = {"".join(x):i for i,x in zip(range(4**k), itertools.product("ACGT",repeat=k))}
        self.log = log
        self.divide = divide
        
    def kmerize(self, seq):
        counts = np.zeros(4**self.k)
        i = 0
        while i < len(seq) - self.k: 
            kmer = seq[i:i+self.k]
            counts[self.kmers[kmer]] += 1
            i += 1
        if self.divide:
            counts = counts/len(seq)
        if self.log:
            counts = np.log(counts + 1)
        return counts


# %%
def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
           [0.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.2, 0.0],
        ]),
        np.array([
           [1.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.8, 0.0],
        ]),
        np.array([
           [0.225, 0.45],
           [0.775, 0.45],
           [0.85, 0.3],
           [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(mpl.patches.Polygon((np.array([1,height])[None,:]*polygon_coords
                                                 + np.array([left_edge,base])[None,:]),
                                                facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(mpl.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(mpl.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(mpl.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))


def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(mpl.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(mpl.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(mpl.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(mpl.patches.Rectangle(xy=[left_edge+0.825, base+0.085*height], width=0.174, height=0.415*height,
                                            facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(mpl.patches.Rectangle(xy=[left_edge+0.625, base+0.35*height], width=0.374, height=0.15*height,
                                            facecolor=color, edgecolor=color, fill=True))


def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(mpl.patches.Rectangle(xy=[left_edge+0.4, base],
                  width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(mpl.patches.Rectangle(xy=[left_edge, base+0.8*height],
                  width=1.0, height=0.2*height, facecolor=color, edgecolor=color, fill=True))

default_colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}
default_plot_funcs = {0:plot_a, 1:plot_c, 2:plot_g, 3:plot_t}
def plot_weights_given_ax(ax, array,
                 figsize=(20,2),
                 height_padding_factor=0.2,
                 length_padding=1.0,
                 subticks_frequency=1.0,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 highlight={},
                 ylabel=""):
    if len(array.shape)==3:
        array = np.squeeze(array)
    assert len(array.shape)==2, array.shape
    if (array.shape[0]==4 and array.shape[1] != 4):
        array = array.transpose(1,0)
    assert array.shape[1]==4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        #sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i,:]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color=colors[letter[0]]
            if (letter[1] > 0):
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]                
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    #now highlight any desired positions; the key of
    #the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                mpl.patches.Rectangle(xy=[start_pos,min_depth],
                    width=end_pos-start_pos,
                    height=max_height-min_depth,
                    edgecolor=color, fill=False))
            
    ax.set_xlim(-length_padding, array.shape[0]+length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0]+1, subticks_frequency))
    height_padding = max(abs(min_neg_height)*(height_padding_factor),
                         abs(max_pos_height)*(height_padding_factor))
    ax.set_ylim(min_neg_height-height_padding, max_pos_height+height_padding)
    ax.set_ylabel(ylabel)
    ax.yaxis.label.set_fontsize(15)


def plot_weights(array,
                 figsize=(20,2),
                 despine=False,
                 **kwargs):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111) 
    plot_weights_given_ax(ax=ax, array=array,**kwargs)
    if despine:
        plt.axis('off')
    plt.show()
    return fig,ax


def plot_score_track_given_ax(arr, ax, threshold=None, **kwargs):
    ax.plot(np.arange(len(arr)), arr, **kwargs)
    if (threshold is not None):
        ax.plot([0, len(arr)-1], [threshold, threshold])
    ax.set_xlim(0,len(arr)-1)


def plot_score_track(arr, threshold=None, figsize=(20,2), **kwargs):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111) 
    plot_score_track_given_ax(arr, threshold=threshold, ax=ax, **kwargs) 
    plt.show()


# %% [markdown]
# ## Code to do predictions

# %%
def xpresso_batcherator(sample_generator,
                        xpresso_model,
                        batch_size=2,
                        verbose=True):
    results = []
    debug_output_dict = {}
    done = False
    batch_idx = 0
    while True:
        batch = []
        # fill batch
        for i in range(batch_size):
            try:
                sample = next(sample_generator)
                batch.append(sample)
            except StopIteration:
                done = True
                break
        if len(batch) == 0:
              break
        if len(batch) == 1:
              batch_seq = one_hot_encode(batch[0][0])[np.newaxis]
        else:
              batch_seq = np.stack([one_hot_encode(x[0]) for x in batch])
        # predict
        mean_half_life_features = np.zeros((batch_seq.shape[0],6), dtype='float32')
        predictions = xpresso_model.predict_on_batch([batch_seq,mean_half_life_features]).reshape(-1)
        # collect relevant data
        for idx, sample in enumerate(batch):
            _, metadata = sample
            result_dict = {k:v for k,v in metadata.items()}
            result_dict["sample_idx"] = batch_idx*batch_size + idx
            result_dict["Prediction"] = predictions[idx]
            results.append(result_dict)
        batch_idx += 1
        if verbose and (batch_idx % 100 == 0):
              print(batch_idx*batch_size)
        if done:
              break
    return results#, debug_output_dict


# %%
def batcherator(sample_generator,
                track_dict,
                batch_size=2,
                revcomp=True,
                verbose=True):
    results = []
    debug_output_dict = {}
    done = False
    batch_idx = 0
    while True:
        batch = []
        # fill batch
        for i in range(batch_size):
            try:
                sample = next(sample_generator)
                batch.append(sample)
            except StopIteration:
                done = True
                break
        if len(batch) == 0:
              break
        if len(batch) == 1:
              batch_seq = one_hot_encode(batch[0][0])[np.newaxis]
        else:
              batch_seq = np.stack([one_hot_encode(x[0]) for x in batch])
        # predict
        predictions = model.predict_on_batch(batch_seq)['human']
        if revcomp:
            predictions_rc = model.predict_on_batch(rev_comp_one_hot(batch_seq))['human']
        # collect relevant track data
        for idx, sample in enumerate(batch):
            _, metadata = sample
            result_dict = {k:v for k,v in metadata.items()}
            result_dict["sample_idx"] = batch_idx*batch_size + idx
            minbin = result_dict["minbin"]
            maxbin = result_dict["maxbin"]
            landmarkbin = result_dict["landmarkbin"]
            for track in track_dict:
                #result_dict[track] = predictions[idx,minbin:maxbin,track_dict[track]].copy()
                result_dict[track] = np.max(predictions[idx,minbin-1:maxbin+1,track_dict[track]])
                result_dict[track + "arg"] = np.argmax(predictions[idx,minbin-1:maxbin+1,track_dict[track]])
                result_dict[track + "_sum"] = np.sum(predictions[idx,minbin-1:maxbin+1,track_dict[track]])
                result_dict[track + "_localsum"] = np.sum(predictions[idx,minbin-1:maxbin+1,track_dict[track]][max(landmarkbin-1,0):landmarkbin+2])
                if revcomp:
                    result_dict[track + "_rc"] = np.max(predictions_rc[idx,minbin-1:maxbin+1,track_dict[track]])
                    result_dict[track + "arg_rc"] = np.argmax(predictions_rc[idx,minbin-1:maxbin+1,track_dict[track]])
                    result_dict[track + "_sum_rc"] = np.sum(predictions_rc[idx,minbin-1:maxbin+1,track_dict[track]])
                    result_dict[track + "_localsum_rc"] = np.sum(predictions_rc[idx,minbin-1:maxbin+1,track_dict[track]][max(landmarkbin-1,0):landmarkbin+2])
            results.append(result_dict)
        batch_idx += 1
        if verbose and (batch_idx % 5 == 0):
              print(batch_idx*batch_size)
        if done:
              break
    return results#, debug_output_dict


# %% [markdown] id="axJyXU13uuUC"
# # Prepare fasta, gtf and enformer

# %% id="5m1IEvOxux_t"
# import target df
target_df = pd.read_csv("Data/Targets/targets.tsv",sep="\t")
target_df_basenji1 = pd.read_csv("Models/Basenji/basenji1_targets.txt",sep="\t", names=["identifier","file","description"])
print("done")

# import gtf
gtf_df = pr.read_gtf(gtf_file)

# %%
# import fasta extractor
fasta_extractor = FastaStringExtractor(fasta_file)

# %% id="21USv_1VIXHz"
# import model
model = Enformer(model_path)

# %% [markdown] id="zEwfoz3cwOzt"
# ## Test that enformer is loaded correctly

# %% id="8u8Gt8WWyG53"
target_interval = kipoiseq.Interval('chr11', 35_082_742, 35_197_430)  # @param

sequence_one_hot = one_hot_encode(fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH)))
predictions = model.predict_on_batch(sequence_one_hot[np.newaxis])['human'][0]

# %%
tracks = {'DNASE:CD14-positive monocyte female': predictions[:, 41],
          'DNASE:keratinocyte female': predictions[:, 42],
          'CHIP:H3K27ac:keratinocyte female': predictions[:, 706],
          'CAGE:Keratinocyte - epidermal': np.log10(1 + predictions[:, 4799])}
plot_tracks(tracks, target_interval)

# %% [markdown] id="qomUc3saGZj-"
# # Observations about input/output
#
# This section contains some FYI about the model we considered useful to keep in mind during our analyses

# %% [markdown]
# ## Enformer

# %% [markdown]
# - Coordinates appear to be zero-based
# - Enformer takes as input 393216bp
# - Actually processed are 196608bp
# - It predicts for the central 114688, i.e. 896 128bp bins
# - The Plot tracks function only works for this size, it does not clip to display shorter intervals!
# - The center is at the boundary of bin 447 and bin 448
# - CAGE tracks start at channel 4675

# %% [markdown]
# ## Basenji2
#
# It is not super clear (to us) what the Basenji2 receptive field actually is. Below is an attempt to compute it.

# %%
basenji2 = tf.keras.models.load_model(
    "Data/Bassenji2/model_human.h5",
    custom_objects={
        "StochasticShift": basenji2_utils.StochasticShift,
        "GELU": basenji2_utils.GELU})

# %% [markdown]
# General formula:
#
# $r_l = s_l \times r_{l-1} + (k_l - s_l)$
#
# where:
# - $s_l$: stride
# - $k_l$: kernel size
# - $r_{l-1}$: receptive field size  (after layer l)
#
# For dilation, adapt kernel to:
#
# - $k_l = \alpha (K_l - 1) + 1$, where $K_l$ is the undilated kernel size and $\alpha$ is the dilation rate

# %%
layers = []
layers += [(15,1),(2,2)]
layers += [(5,1),(2,2)]*6
dilation_rates = [basenji2.get_layer("conv1d_{}".format(i)).dilation_rate[0] for i in range(7,7 + 2*11, 2)]
layers += [(d*(3 - 1) + 1, 1) for d in dilation_rates]

r = 0
for layer in layers[::-1]:
    r = layer[1]*r + (layer[0] - layer[1])
print(r)

# %% [markdown]
# # Run Xpresso
#
# For analyses where it makes sense, we also compare Xpresso.
#
# Since Xpresso runs very quickly, we simply do it in the notebook.
#
# Xpresso takes sequence as:
#
# - 7000 bp upstream of TSS
# - 3500 bp downstream of TSS

# %% [markdown]
# ## Xpresso - Segal

# %%
base_path_data = "Data/Segal_promoter/"
base_path_results = "Results/Segal_promoter_Xpresso/"

# %%
# Imports
Xpresso = kipoi.get_model("Models/Xpresso/human_median",source="dir")


# %%
Xpresso.model.summary()

# %%
with open(base_path_data + "PZDONOR EF1alpha Cherry ACTB GFP synthetic core promoter library (Ns).gbk") as file:
    plasmid_sequence = ""
    started_reading = False
    for line in file.readlines():
        if started_reading:
            line = line.strip().replace(" ","").replace("/","")
            line = re.sub(r'[0-9]+', '', line)
            plasmid_sequence += line
        if line.startswith("ORIGIN"):
            started_reading = True
    plasmid_sequence = plasmid_sequence.upper()

# %%
# extract different components
egfp_seq = rev_comp_sequence(plasmid_sequence[4189:4909])

# %%
# variable region: 4658 - 4822 (0-based)
full_insert = plasmid_sequence[658:6742]

# %% id="fZeyVTNflr5G"
# AAVS1: (PPP1R12C-201, intron 1)
#aavs1 = kipoiseq.Interval(chrom="chr19",start=55116972,end=55117385)
aavs1 = kipoiseq.Interval(chrom="chr19",start=55_115_750,end=55_115_780)

# %%
segal_df = pd.read_csv(base_path_data + "GSM3323461_oligos_measurements_processed_data.tab",sep="\t")

# %% [markdown]
# ### Test insertion

# %%
aavs1

# %%
idx = 15749 #15136#15749
prom = segal_df.loc[idx]["Oligo_sequence"] #15136
insert = full_insert[:4658] + prom + full_insert[4822:]
landmark = len(full_insert[:4658] + prom)

offset = compute_offset_to_center_landmark(landmark,insert)

modified_sequence, minbin, maxbin, landmarkbin = insert_sequence_at_landing_pad(insert, aavs1, fasta_extractor,landmark=landmark, shift_five_end=offset)

modified_sequence = modified_sequence[SEQUENCE_LENGTH//2-7000:SEQUENCE_LENGTH//2+3500]
sequence_one_hot = one_hot_encode(modified_sequence)
sequence_one_hot = sequence_one_hot[np.newaxis,...]

mean_half_life_features = np.zeros((sequence_one_hot.shape[0],6), dtype='float32')
pred = Xpresso.model.predict_on_batch([sequence_one_hot, mean_half_life_features])

# %%
re.search(prom,modified_sequence)

# %% [markdown]
# ### Run it 

# %%
15753*7*2

# %%
len(segal_df) * 7*2


# %%
def segal_sample_generator_factory(fasta_extractor):
    for crs_row in segal_df.iterrows():
        crs_row = crs_row[1]
        crs = crs_row["Oligo_sequence"]
        for insert_type in ["full", "minimal"]:
            if insert_type in ["full"]:
                insert = full_insert[:4658] + crs + full_insert[4822:]
                landmark = len(full_insert[:4658]) + len(crs)
            elif insert_type in ["minimal"]:
                insert = crs + egfp_seq
                landmark = len(crs)
            ideal_offset = compute_offset_to_center_landmark(landmark, insert)
            for offset in [-64,-32,-3,0,3,32,64]:
                    modified_sequence, minbin, maxbin, landmarkbin = \
                        insert_sequence_at_landing_pad(insert,aavs1,
                                                       fasta_extractor,
                                                       shift_five_end=ideal_offset + offset,
                                                       landmark=landmark)
                    modified_sequence = modified_sequence[SEQUENCE_LENGTH//2-7000:SEQUENCE_LENGTH//2+3500]
                    yield modified_sequence, {"Oligo_index":crs_row["Oligo_index"],
                                                "offset":offset,
                                                "insert_type":insert_type}

# prepare generator
sample_generator = \
    segal_sample_generator_factory(fasta_extractor=fasta_extractor)


# %%
# write jobs
xpresso_predictions = xpresso_batcherator(sample_generator, Xpresso.model, batch_size=512)

# %%
xpresso_predictions = pd.DataFrame(xpresso_predictions)

# %%
xpresso_predictions.to_csv(base_path_results + "xpresso_predictions.tsv", sep="\t", index=None)

# %% [markdown]
# ### Analysis

# %%
merged_df = pd.read_csv(base_path_results + "xpresso_predictions.tsv", sep="\t")

# %%
pred_col = "Prediction"

# %%
native_df = pd.read_csv(base_path_data + "native_core_promoters.tsv",sep="\t").rename(columns={"GFP_RFP_ratio(mean exp)":"mean_exp"})
native_df["Set"] = "Native"
pic_df = pd.read_csv(base_path_data + "pic_binding_sites.tsv",sep="\t").rename(columns={"GFP_RFP_ratio(mean exp)":"mean_exp"})
elements_df = pd.read_csv(base_path_data + "synthetic_configurations_of_core_promoters.tsv",sep="\t").rename(columns={"GFP_RFP_ratio(mean exp)":"mean_exp"})
tata_shift_df = pd.read_csv(base_path_data + "tata_inr_shift.tsv",sep="\t").rename(columns={"GFP_RFP_ratio(mean exp)":"mean_exp"})
tf_activity_df = pd.read_csv(base_path_data + "tf_activity_screen.tsv", sep="\t").rename(columns={"GFP_RFP_ratio(mean exp)":"mean_exp"})
tf_multiplicity_df = pd.read_csv(base_path_data + "tf_multiplicity_screen.tsv", sep="\t").rename(columns={"GFP_RFP_ratio(mean exp)":"mean_exp"})

# %% [markdown]
# ### Test impact of offset

# %%
native_tested = merged_df.merge(native_df,on="Oligo_index").dropna(subset=["mean_exp"])
pic_tested = merged_df.merge(pic_df,on="Oligo_index").dropna(subset=["mean_exp"])
columns = [pred_col, "Oligo_index", "mean_exp", "CV(std/mean)", "insert_type", "offset"]
endogenous = pd.concat([native_tested[columns],pic_tested[columns]])

for insert_type in ["full","minimal"]:
    print(insert_type)
    for offset in [-64,-32,-3,0,3,32,64]:
        print(offset)
        subset = endogenous.query('offset == @offset & insert_type == @insert_type')
        print(scipy.stats.pearsonr(subset[pred_col],
                        np.log2(subset["mean_exp"])))
        print(scipy.stats.spearmanr(subset[pred_col],subset["mean_exp"]))

# %% [markdown]
# ## Xpresso TSS

# %%
base_path_results = "Results/TSS_Xpresso/"
base_path_data = "Data/TSS_sim/"
base_path_model = "Models/Xpresso/"
base_path_data_gtex = "Data/GTEX/"

# %% [markdown]
# ### Run it

# %%
name_list = [
    "ts_id", "ts_ver", "ts_type", "chr", "strand", "ts_start", "ts_end", "tss",
    "ensembl_canonical", "mane", "tsl", "gene_id", "gene_name"
]
# conversion functions for some tsv columns
conv_funs = {
    "strand": lambda x: "+" if x == "1" else "-",
    "ensembl_canonical": lambda x: True if x == "1" else False,
    "mane": lambda x: x if not x == "" else pd.NA,
    "tsl": lambda x: x.split(" ")[0] if not x.split(" ")[0] == "" else pd.NA
}
# strand, ensembl_canonical, mane and tsl are handled by their converters
dtype_dict = {
    "ts_id": str,
    "ts_ver": int,
    "ts_type": str,
    "chr": str,
    "ts_start": int,
    "ts_end": int,
    "tss": int,
    "gene_id": str,
    "gene_name": str
}
# create the dataframe
tss_loc_df = pd.read_csv(os.path.join(base_path_data,
                                  "bigly_tss_from_biomart.txt"),
                     sep="\t",
                     dtype=dtype_dict,
                     names=name_list,
                     converters=conv_funs,
                     header=0)
# get protein_coding ensembl_canonical transcripts on the standard chromosomes
tss_loc_df = tss_loc_df[(tss_loc_df["ts_type"] == "protein_coding")
                & (tss_loc_df["ensembl_canonical"])
                & (tss_loc_df["chr"].isin(
                    list(map(lambda x: str(x), range(1, 23))) +
                    ["MT", "X", "Y"]))]

# needed to translate chromosome symbols from biomart output to fasta
chromosome_dict = {
    "1": "chr1",
    "2": "chr2",
    "3": "chr3",
    "4": "chr4",
    "5": "chr5",
    "6": "chr6",
    "7": "chr7",
    "8": "chr8",
    "9": "chr9",
    "10": "chr10",
    "11": "chr11",
    "12": "chr12",
    "13": "chr13",
    "14": "chr14",
    "15": "chr15",
    "16": "chr16",
    "17": "chr17",
    "18": "chr18",
    "19": "chr19",
    "20": "chr20",
    "21": "chr21",
    "22": "chr22",
    "MT": "chrM",
    "X": "chrX",
    "Y": "chrY"
}

# %%
stranded_fasta_extractor = kipoiseq.extractors.FastaStringExtractor(fasta_file, use_strand=True)

Xpresso = kipoi.get_model("Models/Xpresso/human_median",source="dir")


# %%
def tss_xpresso_sample_generator_factory():
    for crs_row in tss_loc_df.iterrows():
        crs_row = crs_row[1]
        tss = crs_row["tss"]
        chrom = chromosome_dict[crs_row["chr"]]
        strand = crs_row["strand"]
        if tss < 7000:
            continue
        if strand == "+":
            tss_interval =  kipoiseq.Interval(chrom=chrom,start=tss-1-7000,end=tss-1+3500, strand=strand)
        else:
            tss_interval =  kipoiseq.Interval(chrom=chrom,start=tss-1-3500,end=tss-1+7000, strand=strand)
        seq = stranded_fasta_extractor.extract(tss_interval).upper()
        if len(seq) != 7000+3500:
            seq = seq + "N"*(7000+3500-len(seq))
        yield seq, {"gene_id":crs_row["gene_id"]}


# %%
# run
xpresso_predictions = xpresso_batcherator(tss_xpresso_sample_generator_factory(), Xpresso.model, batch_size=512)

# %%
xpresso_pred_df = pd.DataFrame(xpresso_predictions)

# %%
xpresso_pred_df.to_csv(base_path_results + "xpresso_preds.tsv", sep="\t", index=None)

# %% [markdown] id="ypHneLKBYNAX"
# # Weingarten-Gabbay et al. (Segal lab) Promoter MPRA
#
# This section contains our analysis for the Weingarten-Gabbay et al. promoter MPRA.

# %% [markdown]
# Some background info, to keep in mind
#
# Experimental design:
#
# - Tested ~15k different promoters at the AAVS1 site
#
# - Cell type: K562
#
# Makeup of the insert (in genbank file, 1-based)
#
# - 659-1462: left homology arm
# - 1481-2815: EF1alpha promoter
# - 2832-3456: Kozak + mCherry
# - 3594-3818: BGH Poly(A)
# - 3931-4152: (rc) sv40 polyA
# - 4190-4909: (rc) eGFP
# - 4995-5127: (rc) chimeric intron
# - 5317-5480: (rc) crs region
# - 5905-6742: right homology arm

# %% [markdown] id="xgclGqIoouDn"
# The relevant columns are:
#
# \["Oligo_index", "Set", "GFP_RFP_ratio(mean exp)", "CV(std/mean)", "Oligo_Sequence"\]
#
# NB: Native core promoters have no column "Set"

# %%
base_path_data = "Data/Segal_promoter/"
base_path_results = "Results/Segal_promoter/"
base_path_results_xpresso = "Results/Segal_promoter_Xpresso/"

# %%
with open(base_path_data + "PZDONOR EF1alpha Cherry ACTB GFP synthetic core promoter library (Ns).gbk") as file:
    plasmid_sequence = ""
    started_reading = False
    for line in file.readlines():
        if started_reading:
            line = line.strip().replace(" ","").replace("/","")
            line = re.sub(r'[0-9]+', '', line)
            plasmid_sequence += line
        if line.startswith("ORIGIN"):
            started_reading = True
    plasmid_sequence = plasmid_sequence.upper()

# %%
# extract different components
egfp_seq = rev_comp_sequence(plasmid_sequence[4189:4909])

# %%
# variable region: 4658 - 4822 (0-based)
full_insert = plasmid_sequence[658:6742]

# %% id="fZeyVTNflr5G"
# AAVS1: (PPP1R12C-201, intron 1)
#aavs1 = kipoiseq.Interval(chrom="chr19",start=55116972,end=55117385)
aavs1 = kipoiseq.Interval(chrom="chr19",start=55_115_750,end=55_115_780)

# %%
segal_df = pd.read_csv(base_path_data + "GSM3323461_oligos_measurements_processed_data.tab",sep="\t")

# %%
len(segal_df)

# %%
segal_df["Oligo_sequence"].str.len().describe()

# %% [markdown]
# ## Test insertion
#
# We try out a particular promoter, to see if Enformer understands what we are trying to do at all

# %%
aavs1


# %%
def test_segal_insertion(insert, landmark, extra_offset=0, verbose=True):
    offset = compute_offset_to_center_landmark(landmark,insert)

    modified_sequence, minbin, maxbin, landmarkbin = insert_sequence_at_landing_pad(insert, aavs1, fasta_extractor,landmark=landmark, shift_five_end=offset + extra_offset)
    sequence_one_hot = one_hot_encode(modified_sequence)
    predictions = model.predict_on_batch(sequence_one_hot[np.newaxis])['human'][0]
    
    if verbose:
        tracks = {'DNASE:K562': predictions[:, 121],
                  'CAGE:chronic myelogenous leukemia cell line:K562': np.log10(1 + predictions[:,  4828]),
                 }
        plot_tracks(tracks, target_interval)

        print(predictions[minbin-1:maxbin+1,  4828])
        print(predictions[landmarkbin,  4828])
        print(sum(predictions[minbin-1:maxbin+1,  4828]))
    else:
        return (predictions[landmarkbin,  4828], sum(predictions[landmarkbin-1:landmarkbin+2,  4828]))

        

# %%
idx = 15136#15749
prom = segal_df.loc[idx]["Oligo_sequence"] #15136
insert = full_insert[:4658] + prom + full_insert[4822:]
landmark = len(full_insert[:4658]) + len(prom)//2


target_interval = kipoiseq.Interval(chrom="chr19",
                                    start=(aavs1.start+aavs1.end)//2 - 114688//2,
                                    end=(aavs1.start+aavs1.end)//2 + 114688//2,
                                   )


# %%
#basal
sequence_one_hot = one_hot_encode(fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH)))
predictions = model.predict_on_batch(sequence_one_hot[np.newaxis])['human'][0]

tracks = {'DNASE:K562': predictions[:, 121],
          'CAGE:chronic myelogenous leukemia cell line:K562': np.log10(1 + predictions[:,  4828]),
         }
plot_tracks(tracks, target_interval)

print(predictions[446:449,  4828])

# %%
test_segal_insertion(insert, landmark)

# %%
test_segal_insertion(insert, landmark, extra_offset = -64)

# %%
# revcomp
insert_rc = rev_comp_sequence(full_insert[:4658] + prom + full_insert[4822:])
landmark_rc = len(full_insert[4822:]) + len(prom)//2


test_segal_insertion(insert_rc, landmark_rc, extra_offset = 0)

# %%
test_segal_insertion(insert_rc, landmark_rc, extra_offset = -64)

# %% [markdown]
# ### Remove everything except the core promoter and the egfp
#
# This produces a decent amount of expression...

# %%
# with no offset
insert_min = prom + egfp_seq

landmark_min = len(prom)//2

test_segal_insertion(insert_min, landmark_min, extra_offset = 0)

# %%
# with -64 offset
test_segal_insertion(insert_min, landmark_min, extra_offset = -64)

# %% [markdown]
# ### Only insert a coding sequence
#
# Reassuringly, this does not create expression

# %%
insert_egfp = egfp_seq

landmark =  len(egfp_seq)//2

test_segal_insertion(insert_egfp, landmark, extra_offset = -64)

# %% [markdown]
# ## Analysis

# %%
# load enformer
pred_col = 'CAGE:chronic myelogenous leukemia cell line:K562  ENCODE, biol__landmark_sum'
pred_col_dnase = 'DNASE:K562_landmark_sum'

merged_df = pd.read_csv(base_path_results + "segal_promoters-enformer-latest_results.tsv", sep='\t')
merged_df = (merged_df
              .groupby(["Oligo_index","insert_type"])[[col for col in merged_df.keys() if col.startswith("CAGE") or col.startswith("DNASE")]]
              .mean()
              .reset_index())

merged_df_full = merged_df.query('insert_type == "full"')
merged_df_fullrv = merged_df.query('insert_type == "full_rv"')
merged_df_min = merged_df.query('insert_type == "minimal"')

# %%
# load basenji2
merged_df_basenji2 = pd.read_csv(base_path_results + "segal_promoters-basenji2-latest_results.tsv", sep='\t')
merged_df_basenji2 = (merged_df_basenji2
              .groupby(["Oligo_index","insert_type"])[[col for col in merged_df_basenji2.keys() if col.startswith("CAGE") or col.startswith("DNASE")]]
              .mean()
              .reset_index())

merged_df_full_basenji2 = merged_df_basenji2.query('insert_type == "full"')
merged_df_fullrv_basenji2 = merged_df_basenji2.query('insert_type == "full_rv"')
merged_df_min_basenji2 = merged_df_basenji2.query('insert_type == "minimal"')

# %%
# load basenji1
merged_df_basenji1 = pd.read_csv(base_path_results + "segal_promoters-basenji1-latest_results.tsv", sep='\t')
merged_df_basenji1 = (merged_df_basenji1
              .groupby(["Oligo_index","insert_type"])[[col for col in merged_df_basenji1.keys() if col.startswith("CAGE") or col.startswith("DNASE")]]
              .mean()
              .reset_index())

# combine replicates
merged_df_basenji1[pred_col] = (merged_df_basenji1['CAGE:chronic myelogenous leukemia cell line:K562 ENCODE, biol_rep1_landmark_sum']
                               + merged_df_basenji1['CAGE:chronic myelogenous leukemia cell line:K562 ENCODE, biol_rep2_landmark_sum']
                               + merged_df_basenji1['CAGE:chronic myelogenous leukemia cell line:K562 ENCODE, biol_rep3_landmark_sum']
                               )/3

merged_df_full_basenji1 = merged_df_basenji1.query('insert_type == "full"')
merged_df_fullrv_basenji1 = merged_df_basenji1.query('insert_type == "full_rv"')
merged_df_min_basenji1 = merged_df_basenji1.query('insert_type == "minimal"')

# %%
native_df = pd.read_csv(base_path_data + "native_core_promoters.tsv",sep="\t").rename(columns={"GFP_RFP_ratio(mean exp)":"mean_exp"})
native_df["Set"] = "Native"
pic_df = pd.read_csv(base_path_data + "pic_binding_sites.tsv",sep="\t").rename(columns={"GFP_RFP_ratio(mean exp)":"mean_exp"})
elements_df = pd.read_csv(base_path_data + "synthetic_configurations_of_core_promoters.tsv",sep="\t").rename(columns={"GFP_RFP_ratio(mean exp)":"mean_exp"})
tata_shift_df = pd.read_csv(base_path_data + "tata_inr_shift.tsv",sep="\t").rename(columns={"GFP_RFP_ratio(mean exp)":"mean_exp"})
tf_activity_df = pd.read_csv(base_path_data + "tf_activity_screen.tsv", sep="\t").rename(columns={"GFP_RFP_ratio(mean exp)":"mean_exp"})
tf_multiplicity_df = pd.read_csv(base_path_data + "tf_multiplicity_screen.tsv", sep="\t").rename(columns={"GFP_RFP_ratio(mean exp)":"mean_exp"})

# %%
tf_activity_df = tf_activity_df.drop_duplicates(subset=["Oligo_index","mean_exp"])

# %% [markdown]
# ### Native Promoters
#
# We see a good correlation with measured expression of endogenous promoters, regardless whether we use the full insert or the limited one.

# %% [markdown]
# #### Enformer

# %%
native_tested_full = merged_df_full.merge(native_df,on="Oligo_index").dropna(subset=["mean_exp"])
native_tested_fullrv = merged_df_fullrv.merge(native_df,on="Oligo_index").dropna(subset=["mean_exp"])
native_tested_min = merged_df_min.merge(native_df,on="Oligo_index").dropna(subset=["mean_exp"])

# %%
# full insert
subset = native_tested_fullrv
print(scipy.stats.pearsonr(np.log2(subset[pred_col]+1),
                           np.log2(subset["mean_exp"])))
print(scipy.stats.spearmanr(subset[pred_col],
                           subset["mean_exp"]))

subset = native_tested_full
print(scipy.stats.pearsonr(np.log2(subset[pred_col]+1),
                           np.log2(subset["mean_exp"])))
print(scipy.stats.spearmanr(subset[pred_col],
                           subset["mean_exp"]))

# min insert
subset = native_tested_min
print(scipy.stats.pearsonr(np.log2(subset[pred_col]+1),
                           np.log2(subset["mean_exp"])))
#print(pearsonr_ci(np.log2(subset[pred_col]+1),np.log2(subset["mean_exp"]),alpha=0.05))
print(scipy.stats.spearmanr(subset[pred_col],
                           subset["mean_exp"]))

# is enformer better than the invivo expression?
#print("\nK562 expression of associated gene vs MPRA")
#print(scipy.stats.pearsonr(np.log2(subset['K562_RPM_Mean']+1),
#                           np.log2(subset["mean_exp"])))
#print(pearsonr_ci(np.log2(subset['K562_RPM_Mean']+1),np.log2(subset["mean_exp"]),alpha=0.05))
#print(scipy.stats.spearmanr(subset['K562_RPM_Mean'],
#                           subset["mean_exp"]))
#print("\nK562 expression of associated gene vs Enformer prediction")
#print(scipy.stats.pearsonr(np.log2(subset['K562_RPM_Mean']+1),
#                           np.log2(subset[pred_col]+1)))
#print(scipy.stats.spearmanr(subset['K562_RPM_Mean'],
#                           subset[pred_col]))

# %%
scale = 1.3
p9.options.figure_size = (6.4*scale, 4.8*scale)
(p9.ggplot(data=native_tested_min,mapping=p9.aes(x=pred_col,y='mean_exp'))
 + p9.geom_point(alpha=0.2)
 #+ p9.geom_bin2d(binwidth = (0.05, 0.05))
 + p9.scale_x_log10()
 + p9.scale_y_log10()
 + p9.geom_smooth(method="lm",color="blue")
 + p9.labs(x="CAGE:K562 prediction (minimal insert)", y="Measured expression of Reporter Gene")
)

# %% [markdown]
# #### Basenji2

# %%
native_tested_full_basenji2 = merged_df_full_basenji2.merge(native_df,on="Oligo_index").dropna(subset=["mean_exp"])
native_tested_fullrv_basenji2 = merged_df_fullrv_basenji2.merge(native_df,on="Oligo_index").dropna(subset=["mean_exp"])
native_tested_min_basenji2 = merged_df_min_basenji2.merge(native_df,on="Oligo_index").dropna(subset=["mean_exp"])

# %%
# full insert
subset = native_tested_fullrv_basenji2
print(scipy.stats.pearsonr(np.log2(subset[pred_col]+1),
                           np.log2(subset["mean_exp"])))
print(scipy.stats.spearmanr(subset[pred_col],
                           subset["mean_exp"]))

subset = native_tested_full_basenji2
print(scipy.stats.pearsonr(np.log2(subset[pred_col]+1),
                           np.log2(subset["mean_exp"])))
print(scipy.stats.spearmanr(subset[pred_col],
                           subset["mean_exp"]))

# min insert
subset = native_tested_min_basenji2
print(scipy.stats.pearsonr(np.log2(subset[pred_col]+1),
                           np.log2(subset["mean_exp"])))
#print(pearsonr_ci(np.log2(subset[pred_col]+1),np.log2(subset["mean_exp"]),alpha=0.05))
print(scipy.stats.spearmanr(subset[pred_col],
                           subset["mean_exp"]))

# %% [markdown] tags=[]
# #### Basenji1

# %%
native_tested_full_basenji1 = merged_df_full_basenji1.merge(native_df,on="Oligo_index").dropna(subset=["mean_exp"])
native_tested_fullrv_basenji1 = merged_df_fullrv_basenji1.merge(native_df,on="Oligo_index").dropna(subset=["mean_exp"])
native_tested_min_basenji1 = merged_df_min_basenji1.merge(native_df,on="Oligo_index").dropna(subset=["mean_exp"])

# %%
# full insert
subset = native_tested_fullrv_basenji1
print(scipy.stats.pearsonr(np.log2(subset[pred_col]+1),
                           np.log2(subset["mean_exp"])))
print(scipy.stats.spearmanr(subset[pred_col],
                           subset["mean_exp"]))

subset = native_tested_full_basenji1
print(scipy.stats.pearsonr(np.log2(subset[pred_col]+1),
                           np.log2(subset["mean_exp"])))
print(scipy.stats.spearmanr(subset[pred_col],
                           subset["mean_exp"]))

# min insert
subset = native_tested_min_basenji1
print(scipy.stats.pearsonr(np.log2(subset[pred_col]+1),
                           np.log2(subset["mean_exp"])))
#print(pearsonr_ci(np.log2(subset[pred_col]+1),np.log2(subset["mean_exp"]),alpha=0.05))
print(scipy.stats.spearmanr(subset[pred_col],
                           subset["mean_exp"]))

# %% [markdown]
# ### Promoter directionality
#
#
# Weingarten-Gabbay et al. find that most promoters they tested are directional, i.e. they drive expression much stronger of the 3' reporter in one orientation than the other.
# As a result, the correlation in measured reporter gene expression between oligos where promoters are placed in forward or in reverse orientation, is very low.
#
# Enformer does not reproduce this result. In the full insert, there is a pretty strong correlation between forward and reverse expression. In the minimal insert, it is somewhat reduced but still present. The issue is that we cannot ask enformer in which direction it is predicting transcription to occur - the model only predicts CAGE.

# %%
pic_tested_fullrv = merged_df_fullrv.merge(pic_df,on="Oligo_index").dropna(subset=["mean_exp"])
pic_tested_min = merged_df_min.merge(pic_df,on="Oligo_index").dropna(subset=["mean_exp"])

# %%
print(scipy.stats.pearsonr(np.log2(pic_tested_fullrv[pred_col]+1),
                           np.log2(pic_tested_fullrv["mean_exp"])))
print(scipy.stats.spearmanr(np.log2(pic_tested_fullrv[pred_col]+1),
                           pic_tested_fullrv["mean_exp"]))

# %%
print(scipy.stats.pearsonr(np.log2(pic_tested_min[pred_col]+1),
                           np.log2(pic_tested_min["mean_exp"])))
print(scipy.stats.spearmanr(np.log2(pic_tested_min[pred_col]+1),
                           pic_tested_min["mean_exp"]))

# %%
# Compute correlation between forward and reverse for each
pic_fw = pic_tested_fullrv.query('CP_Orientation == "Fw"')
pic_rv = pic_tested_fullrv.query('CP_Orientation == "Rv"')
cols = ['Oligo_index',pred_col,'mean_exp','TFIIB_ID']
pic_orient = pic_fw[cols].merge(pic_rv[cols],on=['TFIIB_ID'],suffixes=("_fw","_rv"))

print("Measured")
print(scipy.stats.pearsonr(np.log2(pic_orient['mean_exp_fw']),
                           np.log2(pic_orient["mean_exp_rv"])))
print(scipy.stats.spearmanr((pic_orient['mean_exp_fw']),
                           pic_orient["mean_exp_rv"]))
print("Predicted")
print(scipy.stats.pearsonr(np.log2(pic_orient[pred_col + '_fw']),
                           np.log2(pic_orient[pred_col + "_rv"])))
print(scipy.stats.spearmanr((pic_orient[pred_col + '_fw']),
                           pic_orient[pred_col + "_rv"]))

# %%
scale = 1.0
p9.options.figure_size = (6.4*scale, 4.8*scale)
(p9.ggplot(data=pic_orient,mapping=p9.aes(x='mean_exp_fw',y='mean_exp_rv'))
 + p9.geom_point(alpha=1)
 + p9.scale_x_log10()
 + p9.scale_y_log10()
 + p9.labs(x="Measured Expression in forward orientation", y="Measured Expression in reverse orientation")
 + p9.coord_fixed(xlim=(0,1.6),ylim=(0,1.6))
)

# %%
scale = 1.0
p9.options.figure_size = (6.4*scale, 4.8*scale)
(p9.ggplot(data=pic_orient,mapping=p9.aes(x=pred_col+'_fw',y=pred_col+'_rv'))
 + p9.geom_point(alpha=1)
 + p9.scale_x_log10()
 + p9.scale_y_log10()
 + p9.labs(x="Predicted Expression in forward orientation", y="Predicted Expression in reverse orientation")
 #+ p9.coord_fixed(xlim=(0,2),ylim=(0,2))
)

# %% [markdown]
# ### All endogenous constructs with Promoter/PIC activity
#
# We combine the native promoter set with the set of tested constructs that were associated with in-vivo PIC activity, to get a bigger testset.
#
# This is our set of endogenous promoters.
#
# All models do reasonably well, but Enformer clearly performs the best. The minimal insert also (slightly) outperforms the full one.

# %% [markdown]
# #### Enformer

# %%
columns = [pred_col, "Oligo_index", "mean_exp", "CV(std/mean)", "Oligo_Sequence"]

native_tested_min = merged_df_min.merge(native_df,on="Oligo_index").dropna(subset=["mean_exp"])
pic_tested_min = merged_df_min.merge(pic_df.rename(columns={"Oligo_sequence":"Oligo_Sequence"}),on="Oligo_index").dropna(subset=["mean_exp"])
endogenous_min = pd.concat([native_tested_min[columns],pic_tested_min[columns]])
endogenous_min = endogenous_min.drop_duplicates()

native_tested_fullrv = merged_df_fullrv.merge(native_df,on="Oligo_index").dropna(subset=["mean_exp"])
pic_tested_fullrv = merged_df_fullrv.merge(pic_df.rename(columns={"Oligo_sequence":"Oligo_Sequence"}),on="Oligo_index").dropna(subset=["mean_exp"])
endogenous_fullrv = pd.concat([native_tested_fullrv[columns],pic_tested_fullrv[columns]])

# %%
print(scipy.stats.pearsonr(np.log2(endogenous_min[pred_col]+1),
                           np.log2(endogenous_min["mean_exp"])))
print(scipy.stats.spearmanr(endogenous_min[pred_col],
                           endogenous_min["mean_exp"]))

# %%
len(endogenous_min)

# %%
print(scipy.stats.pearsonr(np.log2(endogenous_fullrv[pred_col]+1),
                           np.log2(endogenous_fullrv["mean_exp"])))
print(scipy.stats.spearmanr(endogenous_fullrv[pred_col],
                           endogenous_fullrv["mean_exp"]))

# %%
scale = 1.3
p9.options.figure_size = (6.4*scale, 4.8*scale)
(p9.ggplot(data=endogenous_min,mapping=p9.aes(x=pred_col,y='mean_exp'))
 + p9.geom_point(alpha=0.15)
 #+ p9.geom_bin2d(binwidth = (0.05, 0.05))
 + p9.scale_x_log10()
 + p9.scale_y_log10()
 + p9.geom_smooth(method="lm",color="blue")
 + p9.labs(x="CAGE:K562 prediction (minimal insert)", y="Measured expression of Reporter Gene")
)

# %% [markdown]
# #### Basenji2

# %%
columns = [pred_col, "Oligo_index", "mean_exp", "CV(std/mean)", "Oligo_Sequence"]

native_tested_min_basenji2 = merged_df_min_basenji2.merge(native_df,on="Oligo_index").dropna(subset=["mean_exp"])
pic_tested_min_basenji2 = merged_df_min_basenji2.merge(pic_df.rename(columns={"Oligo_sequence":"Oligo_Sequence"}),on="Oligo_index").dropna(subset=["mean_exp"])
endogenous_min_basenji2 = pd.concat([native_tested_min_basenji2[columns],pic_tested_min_basenji2[columns]])
endogenous_min_basenji2 = endogenous_min_basenji2.drop_duplicates()

native_tested_fullrv_basenji2 = merged_df_fullrv_basenji2.merge(native_df,on="Oligo_index").dropna(subset=["mean_exp"])
pic_tested_fullrv_basenji2 = merged_df_fullrv_basenji2.merge(pic_df.rename(columns={"Oligo_sequence":"Oligo_Sequence"}),on="Oligo_index").dropna(subset=["mean_exp"])
endogenous_fullrv_basenji2 = pd.concat([native_tested_fullrv_basenji2[columns],pic_tested_fullrv_basenji2[columns]])

# %%
print(scipy.stats.pearsonr(np.log2(endogenous_min_basenji2[pred_col]+1),
                           np.log2(endogenous_min_basenji2["mean_exp"])))
print(scipy.stats.spearmanr(endogenous_min_basenji2[pred_col],
                           endogenous_min_basenji2["mean_exp"]))

print(scipy.stats.pearsonr(np.log2(endogenous_fullrv_basenji2[pred_col]+1),
                           np.log2(endogenous_fullrv_basenji2["mean_exp"])))
print(scipy.stats.spearmanr(endogenous_fullrv_basenji2[pred_col],
                           endogenous_fullrv_basenji2["mean_exp"]))

# %% [markdown]
# #### Basenji1

# %%
columns = [pred_col, "Oligo_index", "mean_exp", "CV(std/mean)", "Oligo_Sequence"]

native_tested_min_basenji1 = merged_df_min_basenji1.merge(native_df,on="Oligo_index").dropna(subset=["mean_exp"])
pic_tested_min_basenji1 = merged_df_min_basenji1.merge(pic_df.rename(columns={"Oligo_sequence":"Oligo_Sequence"}),on="Oligo_index").dropna(subset=["mean_exp"])
endogenous_min_basenji1 = pd.concat([native_tested_min_basenji1[columns],pic_tested_min_basenji1[columns]])
endogenous_min_basenji1 = endogenous_min_basenji1.drop_duplicates()

native_tested_fullrv_basenji1 = merged_df_fullrv_basenji1.merge(native_df,on="Oligo_index").dropna(subset=["mean_exp"])
pic_tested_fullrv_basenji1 = merged_df_fullrv_basenji1.merge(pic_df.rename(columns={"Oligo_sequence":"Oligo_Sequence"}),on="Oligo_index").dropna(subset=["mean_exp"])
endogenous_fullrv_basenji1 = pd.concat([native_tested_fullrv_basenji1[columns],pic_tested_fullrv_basenji1[columns]])

# %%
print(scipy.stats.pearsonr(np.log2(endogenous_min_basenji1[pred_col]+1),
                           np.log2(endogenous_min_basenji1["mean_exp"])))
print(scipy.stats.spearmanr(endogenous_min_basenji1[pred_col],
                           endogenous_min_basenji1["mean_exp"]))

print(scipy.stats.pearsonr(np.log2(endogenous_fullrv_basenji1[pred_col]+1),
                           np.log2(endogenous_fullrv_basenji1["mean_exp"])))
print(scipy.stats.spearmanr(endogenous_fullrv_basenji1[pred_col],
                           endogenous_fullrv_basenji1["mean_exp"]))

# %% [markdown]
# #### Compare the models

# %%
xpresso_df = pd.read_csv(base_path_results_xpresso + "xpresso_predictions.tsv", sep="\t")
xpresso_df = xpresso_df.query('offset in [-3,0] & insert_type == "full"')[["Oligo_index","insert_type","Prediction"]]
xpresso_df = (xpresso_df
              .groupby(["Oligo_index","insert_type"])["Prediction"]
              .mean()
              .reset_index())
xpresso_df = xpresso_df[["Oligo_index","Prediction"]].rename(columns={"Prediction":"xpresso_pred"})

endogenous_min = xpresso_df.merge(endogenous_min,on=["Oligo_index"])

endogenous_min["GC_content"] = (endogenous_min["Oligo_Sequence"].apply(lambda x: np.sum([nt in ["G","C"] for nt in x])))/164

endogenous_min = endogenous_min.merge(endogenous_min_basenji1[["Oligo_index",pred_col]].rename(columns={pred_col:"basenji1_pred"}),on=["Oligo_index"])

endogenous_min = endogenous_min.merge(endogenous_min_basenji2[["Oligo_index",pred_col]].rename(columns={pred_col:"basenji2_pred"}),on=["Oligo_index"])

# %%
len(endogenous_min)

# %%
task = "Endogenous\nPromoters"

row_list = [
    {"model":"GC content",
     "Dataset":task,
     "r":scipy.stats.pearsonr(endogenous_min["GC_content"],np.log2(endogenous_min["mean_exp"]))[0]},
    {"model":"Xpresso",
     "Dataset":task,
     "r":scipy.stats.pearsonr(endogenous_min["xpresso_pred"],np.log2(endogenous_min["mean_exp"]))[0]},
    {"model":"Basenji1",
     "Dataset":task,
     "r":scipy.stats.pearsonr(np.log2(endogenous_min["basenji1_pred"]+1),np.log2(endogenous_min["mean_exp"]))[0]},
    {"model":"Basenji2",
     "Dataset":task,
     "r":scipy.stats.pearsonr(np.log2(endogenous_min["basenji2_pred"]+1),np.log2(endogenous_min["mean_exp"]))[0]},
    {"model":"Enformer",
     "Dataset":task,
     "r":scipy.stats.pearsonr(np.log2(endogenous_min[pred_col]+1),np.log2(endogenous_min["mean_exp"]))[0]}
]
endogenous_corrs = pd.DataFrame(row_list)

# %%
endogenous_corrs

# %% [markdown]
# ### TATA and other core promoter elements
#
# Next, adding/removing core promoter motifs is tested in five backgrounds. In 3/5 backgrounds, all models fail to do well. But in 2/5 backgrounds Enformer does very well and better than the other models

# %%
elements_tested_fullrv = merged_df_fullrv.merge(elements_df,on="Oligo_index").dropna(subset=["mean_exp"])
elements_tested_min = merged_df_min.merge(elements_df,on="Oligo_index").dropna(subset=["mean_exp"])


# %%
def plot_motif_effect(motif, plot_df):
    dims = (4, 4)
    df = plot_df.query('Motif == "{}"'.format(motif))
    x = "x_lab"
    y = "log2_fc"
    #hue = "Motif_count"
    box_pairs=[
        (("-{}\nobserved".format(motif)), ("+{}\nobserved".format(motif))),
        (("-{}\npredicted".format(motif)), ("+{}\npredicted".format(motif)))
        ]
    fig, ax = plt.subplots(figsize=dims, dpi=100)
    ax = sns.boxplot(data=df, x=x, y=y, #hue=hue
                    )
    #plt.legend(bbox_to_anchor=(1, 0.5), title="Motif_count")
    statannot.add_stat_annotation(ax, data=df, x=x, y=y, #hue=hue, 
                                  box_pairs=box_pairs, comparisons_correction=None,
                        test='Mann-Whitney', loc='inside',  verbose=2)

    ax.set_xlabel("",fontsize=10, color="black")
    ax.set_ylabel("Expression change due to Motif\n(log2, relative to background)",fontsize=10, color="black")
    #plt.legend(bbox_to_anchor=(1, 0.5), title="Motif_count")
    #plt.setp(ax.get_legend().get_title(), fontsize='10')
    #plt.setp(ax.get_legend().get_texts(), fontsize='10')
    ax.tick_params(labelsize=10)
    plt.tight_layout()


# %%
def custom_boxplot(**kwargs):
    #df = plot_df.query('Motif == "{}"'.format(motif))
    #hue = "Motif_count"
    data = kwargs["data"]
    x = kwargs["x"]
    y = kwargs["y"]
    hue = kwargs.get("hue")
 
    motif = "Motif"

    box_pairs=[
        (("-{}\nobserved".format(motif)), ("+{}\nobserved".format(motif))),
        (("-{}\npredicted".format(motif)), ("+{}\npredicted".format(motif)))
        ]
    #fig, ax = plt.subplots(figsize=dims, dpi=100)
    ax = sns.boxplot(data=kwargs["data"],x=x, y=y)#, hue=kwargs["hue"])
    #ax.set_xlabel("",fontsize=10, color="black")
    #ax.set_ylabel("Expression change due to Motif\n(log2, relative to background)",fontsize=10, color="black")
    statannot.add_stat_annotation(ax, data=kwargs["data"], x=x, y=y, #hue=hue, 
                                  box_pairs=box_pairs, test='Mann-Whitney', 
                                  loc='inside',  verbose=0)

def plot_motif_effect_combined(df):
    df = df.copy()
    df["Log2 Fold Change vs Background"] = df["log2_fc"]
    df[" "] = pd.Categorical(df["x_lab"].apply(lambda x: x.split("\n")[0][0] + "Motif" + "\n" + x.split("\n")[1]),
                                categories=["-Motif\nobserved","+Motif\nobserved",
                                           "-Motif\npredicted","+Motif\npredicted"])
    
    # create plot
    g = sns.FacetGrid(df, col='Motif', height=5, col_wrap=3)
    g.map_dataframe(custom_boxplot, x = " ", y = "Log2 Fold Change vs Background")#, hue = "pred_obs")
    g.set_titles(col_template="{col_name}", y=1.00)

    plt.show()
    
#plot_motif_effect_combined(element_melt)


# %% [markdown]
# #### Enformer Full insert

# %%
print(scipy.stats.pearsonr(np.log2(elements_tested_fullrv[pred_col]+1),
                           np.log2(elements_tested_fullrv["mean_exp"])))
print(scipy.stats.spearmanr(np.log2(elements_tested_fullrv[pred_col]+1),
                           elements_tested_fullrv["mean_exp"]))

print(scipy.stats.pearsonr(np.log2(elements_tested_fullrv[pred_col_dnase]+1),
                           np.log2(elements_tested_fullrv["mean_exp"])))
print(scipy.stats.spearmanr(np.log2(elements_tested_fullrv[pred_col_dnase]+1),
                           elements_tested_fullrv["mean_exp"]))

# %%
scale = 1.3
p9.options.figure_size = (6.4*scale, 4.8*scale)
(p9.ggplot(data=elements_tested_fullrv,mapping=p9.aes(x=pred_col,y='mean_exp'))
 + p9.geom_point(alpha=1)
 + p9.scale_x_log10()
 + p9.scale_y_log10()
 + p9.geom_smooth(method="lm")
 + p9.labs(x="Predicted CAGE expression in K562", y="Measured Expression in MPRA")
)

# %%
for bg in set(elements_tested_fullrv["Background"]):
    print(bg)
    subset = elements_tested_fullrv.query('Background == @bg')
    print(scipy.stats.pearsonr(np.log2(subset[pred_col]),
                        np.log2(subset["mean_exp"])))

# %%
scale = 1.3
p9.options.figure_size = (6.4*scale, 4.8*scale)
(p9.ggplot(data=elements_tested_fullrv,mapping=p9.aes(x=pred_col,y='mean_exp'))
 + p9.geom_point(alpha=1)
 + p9.scale_x_log10()
 + p9.scale_y_log10()
 + p9.geom_smooth(method="lm", color="blue")
 + p9.labs(x="Predicted CAGE expression in K562", y="Measured Expression in MPRA")
 + p9.theme(subplots_adjust={'wspace': 0.2,'hspace': 0.2})
 + p9.facet_wrap('~Background', scales="free")
)

# %%
# compute log2 fc caused by adding/deleting elements with respect to the background sequences
elements = ['BREu_position', 'TATA_position', 'BREd_position', 'Int_position', 'MTE_position', 'DPE_position']
background = elements_tested_fullrv.query('Configuration_summary == 0')[['Background', 'mean_exp', pred_col]]
altered = elements_tested_fullrv.query('Configuration_summary != 0')[['Background', 'mean_exp', pred_col] + elements]
element_expr = background.merge(altered,on='Background',suffixes=('_background','_altered'))
element_expr['log2_fc_obs'] = np.log2(element_expr['mean_exp_altered']) - np.log2(element_expr['mean_exp_background'])
element_expr['log2_fc_pred'] = np.log2(element_expr[pred_col + '_altered'] + 1) - np.log2(element_expr[pred_col + '_background'] + 1)
# melt to have one row for predicted and observed for each motif
prefixes = []
for element in elements:
    prefix = element.split("_")[0]
    prefixes.append(prefix)
    element_expr[prefix] = element_expr[element].apply(lambda x: "-" + prefix if x == "-" else "+" + prefix)
element_melt = element_expr[['Background','log2_fc_obs','log2_fc_pred'] + prefixes].melt(id_vars=["Background"]+prefixes,var_name="pred_obs",value_name="log2_fc")
element_melt = element_melt.melt(id_vars=["Background","pred_obs","log2_fc"],var_name="Motif",value_name="Motif_presence")
element_melt["x_lab"] = element_melt["Motif_presence"] + "\n" + element_melt["pred_obs"].apply(lambda x: "observed" if x.split("_")[-1] == "obs" else "predicted")

# %%
plot_motif_effect_combined(element_melt)

# %%
tata_shift = merged_df_fullrv.merge(tata_shift_df, on="Oligo_index")[['Background',"mean_exp",pred_col,"TATA_position"]]
#tata_shift = background.merge(tata_shift,on='Background',suffixes=('_background','_altered'))
tata_shift['obs'] = np.log2(tata_shift['mean_exp'])
tata_shift['pred'] = np.log2(tata_shift[pred_col] + 1) 
tata_shift_melt = tata_shift[['Background','obs','pred',"TATA_position"]].melt(id_vars=["Background","TATA_position"],var_name="pred_obs",value_name="Log Expression")

# %%
for bg in set(tata_shift_melt["Background"]):
    print(bg)
    subset = tata_shift_melt.query('Background == @bg')
    print(scipy.stats.pearsonr(subset.query('pred_obs == "obs"')["Log Expression"],
                     subset.query('pred_obs == "pred"')["Log Expression"]))

# %%
(p9.ggplot(data=tata_shift_melt, mapping=p9.aes(x="TATA_position",y="Log Expression", color="pred_obs")) 
 + p9.geom_line()
 + p9.geom_point()
 + p9.facet_wrap("~Background")
)

# %% [markdown]
# #### Enformer Minimal insert

# %%
print(scipy.stats.pearsonr(np.log2(elements_tested_min[pred_col]+1),
                           np.log2(elements_tested_min["mean_exp"])))
print(scipy.stats.spearmanr(np.log2(elements_tested_min[pred_col]+1),
                           elements_tested_min["mean_exp"]))

# %%
print(scipy.stats.pearsonr(np.log2(elements_tested_min[pred_col_dnase]+1),
                           np.log2(elements_tested_min["mean_exp"])))
print(scipy.stats.spearmanr(np.log2(elements_tested_min[pred_col_dnase]+1),
                           elements_tested_min["mean_exp"]))

# %%
scale = 1.3
p9.options.figure_size = (6.4*scale, 4.8*scale)
(p9.ggplot(data=elements_tested_min,mapping=p9.aes(x=pred_col,y='mean_exp'))
 + p9.geom_point(alpha=1)
 + p9.scale_x_log10()
 + p9.scale_y_log10()
 + p9.geom_smooth(method="lm")
 + p9.labs(x="Predicted CAGE expression in K562", y="Measured Expression in MPRA")
)

# %%
for bg in set(elements_tested_min["Background"]):
    print(bg)
    subset = elements_tested_min.query('Background == @bg')
    print(scipy.stats.pearsonr(np.log2(subset[pred_col]),
                        np.log2(subset["mean_exp"])))

# %%
scale = 1.3
p9.options.figure_size = (6.4*scale, 4.8*scale)
p = (p9.ggplot(data=elements_tested_min,mapping=p9.aes(x=pred_col,y='mean_exp'))
 + p9.geom_point(alpha=1)
 + p9.scale_x_log10()
 + p9.scale_y_log10()
 + p9.geom_smooth(method="lm", color="blue")
 + p9.labs(x="Predicted CAGE expression in K562", y="Measured Expression in MPRA")
 + p9.theme(subplots_adjust={'wspace': 0.2,'hspace': 0.2})
 + p9.facet_wrap('~Background', scales="free")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9), legend_position=(0.7,0.25), legend_background=p9.element_blank(),
           axis_title=p9.element_text(size=10))
)
p

# %%
p.save("Graphics/" + "xsup_promoter_backgrounds" + ".svg", width=6.4, height=5.6, dpi=300)

# %%
# compute log2 fc caused by adding/deleting elements with respect to the background sequences
elements = ['BREu_position', 'TATA_position', 'BREd_position', 'Int_position', 'MTE_position', 'DPE_position']
background = elements_tested_min.query('Configuration_summary == 0')[['Background', 'mean_exp', pred_col]]
altered = elements_tested_min.query('Configuration_summary != 0')[['Background', 'mean_exp', pred_col] + elements]
element_expr = background.merge(altered,on='Background',suffixes=('_background','_altered'))
element_expr['log2_fc_obs'] = np.log2(element_expr['mean_exp_altered']) - np.log2(element_expr['mean_exp_background'])
element_expr['log2_fc_pred'] = np.log2(element_expr[pred_col + '_altered'] + 1) - np.log2(element_expr[pred_col + '_background'] + 1)
# melt to have one row for predicted and observed for each motif
prefixes = []
for element in elements:
    prefix = element.split("_")[0]
    prefixes.append(prefix)
    element_expr[prefix] = element_expr[element].apply(lambda x: "-" + prefix if x == "-" else "+" + prefix)
element_melt = element_expr[['Background','log2_fc_obs','log2_fc_pred'] + prefixes].melt(id_vars=["Background"]+prefixes,var_name="pred_obs",value_name="log2_fc")
element_melt = element_melt.melt(id_vars=["Background","pred_obs","log2_fc"],var_name="Motif",value_name="Motif_presence")
element_melt["x_lab"] = element_melt["Motif_presence"] + "\n" + element_melt["pred_obs"].apply(lambda x: "observed" if x.split("_")[-1] == "obs" else "predicted")

# %%
scale = 1.0

print(scipy.stats.pearsonr(element_expr['log2_fc_obs'],
                           element_expr['log2_fc_pred']))
print(scipy.stats.pearsonr(element_expr.query('Background in ["HIV1_CE_bg"]')['log2_fc_obs'],
                           element_expr.query('Background in ["HIV1_CE_bg"]')['log2_fc_pred']))
print(scipy.stats.pearsonr(element_expr.query('Background in ["C14orf166_CE_bg"]')['log2_fc_obs'],
                           element_expr.query('Background in ["C14orf166_CE_bg"]')['log2_fc_pred']))
print(scipy.stats.pearsonr(element_expr.query('Background in ["C14orf166_CE_bg","HIV1_CE_bg"]')['log2_fc_obs'],
                           element_expr.query('Background in ["C14orf166_CE_bg","HIV1_CE_bg"]')['log2_fc_pred']))


p9.options.figure_size = (6.4*scale, 4.8*scale)
p = (p9.ggplot(data=element_expr,#.query('Background in ["C14orf166_CE_bg","HIV1_CE_bg"]'),
               mapping=p9.aes(x='log2_fc_obs',y='log2_fc_pred'))
 + p9.geom_point(alpha=1)
 #+ p9.scale_x_log10()
 #+ p9.scale_y_log10()
 + p9.geom_smooth(method="lm", color="blue")
 + p9.labs(x="Predicted CAGE expression in K562", y="Measured Expression in MPRA")
 + p9.theme(subplots_adjust={'wspace': 0.2,'hspace': 0.2})
 + p9.facet_wrap('~Background')
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9), legend_position=(0.7,0.25), legend_background=p9.element_blank(),
           axis_title=p9.element_text(size=10))
)
p

# %%
plot_motif_effect_combined(element_melt)

# %%
plot_motif_effect("TATA", element_melt)

# %%
plot_motif_effect("Int", element_melt)

# %%
tata_shift = merged_df_min.merge(tata_shift_df, on="Oligo_index")[['Background',"mean_exp",pred_col,"TATA_position"]]
#tata_shift = background.merge(tata_shift,on='Background',suffixes=('_background','_altered'))
tata_shift['Measured'] = np.log2(tata_shift['mean_exp'])
tata_shift['Predicted'] = np.log2(tata_shift[pred_col] + 1) 
tata_shift_melt = tata_shift[['Background','Measured','Predicted',"TATA_position"]].melt(id_vars=["Background","TATA_position"],var_name="pred_obs",value_name="Log Expression")

# %%
for bg in set(tata_shift_melt["Background"]):
    print(bg)
    subset = tata_shift_melt.query('Background == @bg')
    print(scipy.stats.pearsonr(subset.query('pred_obs == "Measured"')["Log Expression"],
                     subset.query('pred_obs == "Predicted"')["Log Expression"]))
    print(scipy.stats.spearmanr(subset.query('pred_obs == "Measured"')["Log Expression"],
                     subset.query('pred_obs == "Predicted"')["Log Expression"]))

# %%
p=(p9.ggplot(data=tata_shift_melt, mapping=p9.aes(x="TATA_position",y="Log Expression", color="pred_obs")) 
 + p9.geom_line()
 + p9.geom_point()
 + p9.facet_wrap("~Background")
 + p9.labs(y='log2 Expression', x="TATA position\n relative to TSS", color="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9), legend_background=p9.element_blank(),
           axis_title=p9.element_text(size=10))
)
p

# %%
p.save("Graphics/" + "xsup_promoter_tata" + ".svg", width=6.4, height=6.4, dpi=300)

# %%

p = (p9.ggplot(data=tata_shift_melt.query('Background == "RPLP0_CE_bg"'), mapping=p9.aes(x="TATA_position",y="Log Expression", color="pred_obs")) 
 + p9.geom_line()
 + p9.geom_point()
 + p9.labs(y='$\mathregular{log_2}$ Expression', x="TATA position\n relative to TSS", color="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9), legend_position=(0.7,0.25), legend_background=p9.element_blank(),
           axis_title=p9.element_text(size=10))
)
p

# %%
p.save("Graphics/" + "promoter_tata" + ".svg", width=2.8, height=1.8, dpi=300)

# %% [markdown]
# #### Basenji2

# %%
elements_tested_fullrv_basenji2 = merged_df_fullrv_basenji2.merge(elements_df,on="Oligo_index").dropna(subset=["mean_exp"])
elements_tested_min_basenji2 = merged_df_min_basenji2.merge(elements_df,on="Oligo_index").dropna(subset=["mean_exp"])

# %%
print(scipy.stats.pearsonr(np.log2(elements_tested_fullrv_basenji2[pred_col]+1),
                           np.log2(elements_tested_fullrv_basenji2["mean_exp"])))
print(scipy.stats.spearmanr(np.log2(elements_tested_fullrv_basenji2[pred_col]+1),
                           elements_tested_fullrv_basenji2["mean_exp"]))

print(scipy.stats.pearsonr(np.log2(elements_tested_min_basenji2[pred_col]+1),
                           np.log2(elements_tested_min_basenji2["mean_exp"])))
print(scipy.stats.spearmanr(np.log2(elements_tested_min_basenji2[pred_col]+1),
                           elements_tested_min_basenji2["mean_exp"]))

# %%
print("full")
for bg in set(elements_tested_fullrv_basenji2["Background"]):
    print(bg)
    subset = elements_tested_fullrv_basenji2.query('Background == @bg')
    print(scipy.stats.pearsonr(np.log2(subset[pred_col]+1),
                        np.log2(subset["mean_exp"])))
print("min")    
for bg in set(elements_tested_min_basenji2["Background"]):
    print(bg)
    subset = elements_tested_min_basenji2.query('Background == @bg')
    print(scipy.stats.pearsonr(np.log2(subset[pred_col]+1),
                        np.log2(subset["mean_exp"])))

# %% [markdown]
# #### Basenji1

# %%
elements_tested_fullrv_basenji1 = merged_df_fullrv_basenji1.merge(elements_df,on="Oligo_index").dropna(subset=["mean_exp"])
elements_tested_min_basenji1 = merged_df_min_basenji1.merge(elements_df,on="Oligo_index").dropna(subset=["mean_exp"])

# %%
print(scipy.stats.pearsonr(np.log2(elements_tested_fullrv_basenji1[pred_col]+1),
                           np.log2(elements_tested_fullrv_basenji1["mean_exp"])))
print(scipy.stats.spearmanr(np.log2(elements_tested_fullrv_basenji1[pred_col]+1),
                           elements_tested_fullrv_basenji1["mean_exp"]))

print(scipy.stats.pearsonr(np.log2(elements_tested_min_basenji1[pred_col]+1),
                           np.log2(elements_tested_min_basenji1["mean_exp"])))
print(scipy.stats.spearmanr(np.log2(elements_tested_min_basenji1[pred_col]+1),
                           elements_tested_min_basenji1["mean_exp"]))

# %%
print("full")
for bg in set(elements_tested_fullrv_basenji1["Background"]):
    print(bg)
    subset = elements_tested_fullrv_basenji1.query('Background == @bg')
    print(scipy.stats.pearsonr(np.log2(subset[pred_col]+1),
                        np.log2(subset["mean_exp"])))
print("min")    
for bg in set(elements_tested_min_basenji1["Background"]):
    print(bg)
    subset = elements_tested_min_basenji1.query('Background == @bg')
    print(scipy.stats.pearsonr(np.log2(subset[pred_col]+1),
                        np.log2(subset["mean_exp"])))

# %% [markdown]
# #### Compare the models

# %%
xpresso_df = pd.read_csv(base_path_results_xpresso + "xpresso_predictions.tsv", sep="\t")
xpresso_df = xpresso_df.query('offset in [-3,0] & insert_type == "full"')[["Oligo_index","insert_type","Prediction"]]
xpresso_df = (xpresso_df
              .groupby(["Oligo_index","insert_type"])["Prediction"]
              .mean()
              .reset_index())
xpresso_df = xpresso_df[["Oligo_index","Prediction"]].rename(columns={"Prediction":"xpresso_pred"})

elements_tested_min = xpresso_df.merge(elements_tested_min,on=["Oligo_index"])

elements_tested_min["GC_content"] = (elements_tested_min["Oligo_sequence"].apply(lambda x: np.sum([nt in ["G","C"] for nt in x])))/164

elements_tested_min = elements_tested_min.merge(elements_tested_min_basenji1[["Oligo_index",pred_col]].rename(columns={pred_col:"basenji1_pred"}),on=["Oligo_index"])

elements_tested_min = elements_tested_min.merge(elements_tested_min_basenji2[["Oligo_index",pred_col]].rename(columns={pred_col:"basenji2_pred"}),on=["Oligo_index"])

# %%
task = "Core Promoter\nMotifs"

row_list = []
for bg in set(elements_tested_min["Background"]):
    subset =  elements_tested_min.query('Background == @bg')
    bg = bg.split("_")[0]
    row_list += [
        {"model":"GC content",
         "Background":bg,
         "Dataset":task,
         "r":scipy.stats.pearsonr(subset["GC_content"],np.log2(subset["mean_exp"]))[0],
         "p":scipy.stats.pearsonr(subset["GC_content"],np.log2(subset["mean_exp"]))[1]},
        {"model":"Xpresso",
         "Background":bg,
         "Dataset":task,
         "r":scipy.stats.pearsonr(subset["xpresso_pred"],np.log2(subset["mean_exp"]))[0],
         "p":scipy.stats.pearsonr(subset["xpresso_pred"],np.log2(subset["mean_exp"]))[1]},
        {"model":"Basenji1",
         "Background":bg,
         "Dataset":task,
         "r":scipy.stats.pearsonr(np.log2(subset["basenji1_pred"]+1),np.log2(subset["mean_exp"]))[0],
         "p":scipy.stats.pearsonr(np.log2(subset["basenji1_pred"]+1),np.log2(subset["mean_exp"]))[1]},
        {"model":"Basenji2",
         "Background":bg,
         "Dataset":task,
         "r":scipy.stats.pearsonr(np.log2(subset["basenji2_pred"]+1),np.log2(subset["mean_exp"]))[0],
         "p":scipy.stats.pearsonr(np.log2(subset["basenji2_pred"]+1),np.log2(subset["mean_exp"]))[1]},
        {"model":"Enformer",
         "Background":bg,
         "Dataset":task,
         "r":scipy.stats.pearsonr(np.log2(subset[pred_col]+1),np.log2(subset["mean_exp"]))[0],
         "p":scipy.stats.pearsonr(np.log2(subset[pred_col]+1),np.log2(subset["mean_exp"]))[1]}
    ]
prom_motif_corrs = pd.DataFrame(row_list)

# %%
prom_motif_corrs["padj"] = statsmodels.stats.multitest.fdrcorrection(prom_motif_corrs["p"])[1]
prom_motif_corrs["sig"] = statsmodels.stats.multitest.fdrcorrection(prom_motif_corrs["p"])[0]

# %%
prom_motif_corrs

# %%
prom_motif_corrs_sig = prom_motif_corrs.query('Background in ["HIV1","C14orf166"]')

# %% [markdown]
# ### TF binding site activity
#
# Generally, Enformer can predict with high accuracy which binding sites lead to high expression and which do not.

# %% [markdown]
# #### Enformer

# %%
tf_activity_fullrv = merged_df_fullrv.merge(tf_activity_df,on="Oligo_index").dropna(subset=["mean_exp"])
tf_activity_min = merged_df_min.merge(tf_activity_df,on="Oligo_index").dropna(subset=["mean_exp"])

tf_activity_fullrv["Background"] = tf_activity_fullrv["Background"].apply(lambda x: x.split("_")[0])
tf_activity_min["Background"] = tf_activity_min["Background"].apply(lambda x: x.split("_")[0])

# %%
print(scipy.stats.pearsonr(np.log2(tf_activity_fullrv[pred_col]+1),
                           np.log2(tf_activity_fullrv["mean_exp"])))
print(scipy.stats.spearmanr(tf_activity_fullrv[pred_col],
                           tf_activity_fullrv["mean_exp"]))

print(scipy.stats.pearsonr(np.log2(tf_activity_min[pred_col]+1),
                           np.log2(tf_activity_min["mean_exp"])))
print(scipy.stats.spearmanr(tf_activity_min[pred_col],
                           tf_activity_min["mean_exp"]))

# %%
backgrounds = set(tf_activity_fullrv["Background"])
print("full")
for bg in backgrounds:
    print(bg)
    subset = tf_activity_fullrv.query('Background == @bg')
    print(scipy.stats.pearsonr(np.log2(subset[pred_col]+1),
                               np.log2(subset["mean_exp"])))
    print(scipy.stats.spearmanr(subset[pred_col],
                               subset["mean_exp"]))
    
print("min")
backgrounds = set(tf_activity_min["Background"])
for bg in backgrounds:
    print(bg)
    subset = tf_activity_min.query('Background == @bg')
    print(scipy.stats.pearsonr(np.log2(subset[pred_col]+1),
                               np.log2(subset["mean_exp"])))
    print(scipy.stats.spearmanr(subset[pred_col],
                               subset["mean_exp"]))

# %%
scale = 1.0
p9.options.figure_size = (8*scale, 4.8*scale)
(p9.ggplot(data=tf_activity_min,mapping=p9.aes(x=pred_col,y='mean_exp'))
 + p9.geom_point(alpha=1)
 + p9.scale_x_log10()
 + p9.scale_y_log10()
 + p9.geom_smooth(method="lm", color="blue")
 + p9.labs(x="Predicted CAGE expression in K562 (full insert)", y="Measured Expression in MPRA")
 #+ p9.facet_wrap('~Background')
)

# %%
scale = 1.0
p9.options.figure_size = (8*scale, 4.8*scale)
p=(p9.ggplot(data=tf_activity_min,mapping=p9.aes(x=pred_col,y='mean_exp'))
 + p9.geom_point(alpha=1)
 + p9.scale_x_log10()
 + p9.scale_y_log10()
 + p9.geom_smooth(method="lm", color="blue")
 + p9.labs(x="Predicted CAGE expression in K562 (minimal insert)", y="Measured Expression in MPRA")
 + p9.facet_wrap('~Background')
)
p

# %%
p.save("Graphics/" + "xsup_promoter_tf_sites" + ".svg", width=6.4, height=6.4, dpi=300)

# %% [markdown]
# #### Basenji2

# %%
tf_activity_fullrv_basenji2 = merged_df_fullrv_basenji2.merge(tf_activity_df,on="Oligo_index").dropna(subset=["mean_exp"])
tf_activity_min_basenji2 = merged_df_min_basenji2.merge(tf_activity_df,on="Oligo_index").dropna(subset=["mean_exp"])

tf_activity_fullrv_basenji2["Background"] = tf_activity_fullrv_basenji2["Background"].apply(lambda x: x.split("_")[0])
tf_activity_min_basenji2["Background"] = tf_activity_min_basenji2["Background"].apply(lambda x: x.split("_")[0])

# %%
print(scipy.stats.pearsonr(np.log2(tf_activity_fullrv_basenji2[pred_col]+1),
                           np.log2(tf_activity_fullrv_basenji2["mean_exp"])))
print(scipy.stats.spearmanr(tf_activity_fullrv_basenji2[pred_col],
                           tf_activity_fullrv_basenji2["mean_exp"]))

print(scipy.stats.pearsonr(np.log2(tf_activity_min_basenji2[pred_col]+1),
                           np.log2(tf_activity_min_basenji2["mean_exp"])))
print(scipy.stats.spearmanr(tf_activity_min_basenji2[pred_col],
                           tf_activity_min_basenji2["mean_exp"]))

# %% [markdown]
# #### Basenji1

# %%
tf_activity_fullrv_basenji1 = merged_df_fullrv_basenji1.merge(tf_activity_df,on="Oligo_index").dropna(subset=["mean_exp"])
tf_activity_min_basenji1 = merged_df_min_basenji1.merge(tf_activity_df,on="Oligo_index").dropna(subset=["mean_exp"])

tf_activity_fullrv_basenji1["Background"] = tf_activity_fullrv_basenji1["Background"].apply(lambda x: x.split("_")[0])
tf_activity_min_basenji1["Background"] = tf_activity_min_basenji1["Background"].apply(lambda x: x.split("_")[0])

# %%
print(scipy.stats.pearsonr(np.log2(tf_activity_fullrv_basenji1[pred_col]+1),
                           np.log2(tf_activity_fullrv_basenji1["mean_exp"])))
print(scipy.stats.spearmanr(tf_activity_fullrv_basenji1[pred_col],
                           tf_activity_fullrv_basenji1["mean_exp"]))

print(scipy.stats.pearsonr(np.log2(tf_activity_min_basenji1[pred_col]+1),
                           np.log2(tf_activity_min_basenji1["mean_exp"])))
print(scipy.stats.spearmanr(tf_activity_min_basenji1[pred_col],
                           tf_activity_min_basenji1["mean_exp"]))

# %% [markdown]
# #### Compare the models

# %%
xpresso_df = pd.read_csv(base_path_results_xpresso + "xpresso_predictions.tsv", sep="\t")
xpresso_df = xpresso_df.query('offset in [-3,0] & insert_type == "full"')[["Oligo_index","insert_type","Prediction"]]
xpresso_df = (xpresso_df
              .groupby(["Oligo_index","insert_type"])["Prediction"]
              .mean()
              .reset_index())
xpresso_df = xpresso_df[["Oligo_index","Prediction"]].rename(columns={"Prediction":"xpresso_pred"})

tf_activity_min = xpresso_df.merge(tf_activity_min,on=["Oligo_index"])

tf_activity_min["GC_content"] = (tf_activity_min["Oligo_sequence"].apply(lambda x: np.sum([nt in ["G","C"] for nt in x])))/164

tf_activity_min = tf_activity_min.merge(tf_activity_min_basenji1[["Oligo_index",pred_col]].rename(columns={pred_col:"basenji1_pred"}),on=["Oligo_index"])

tf_activity_min = tf_activity_min.merge(tf_activity_min_basenji2[["Oligo_index",pred_col]].rename(columns={pred_col:"basenji2_pred"}),on=["Oligo_index"])

# %%
task_tf = "Transcription\nFactor Motifs"

row_list = [
    {"model":"GC content",
     "Dataset":task_tf,
     "r":scipy.stats.pearsonr(tf_activity_min["GC_content"],np.log2(tf_activity_min["mean_exp"]))[0]},
    {"model":"Xpresso",
     "Dataset":task_tf,
     "r":scipy.stats.pearsonr(tf_activity_min["xpresso_pred"],np.log2(tf_activity_min["mean_exp"]))[0]},
    {"model":"Basenji1",
     "Dataset":task_tf,
     "r":scipy.stats.pearsonr(np.log2(tf_activity_min["basenji1_pred"]+1),np.log2(tf_activity_min["mean_exp"]))[0]},
    {"model":"Basenji2",
     "Dataset":task_tf,
     "r":scipy.stats.pearsonr(np.log2(tf_activity_min["basenji2_pred"]+1),np.log2(tf_activity_min["mean_exp"]))[0]},
    {"model":"Enformer",
     "Dataset":task_tf,
     "r":scipy.stats.pearsonr(np.log2(tf_activity_min[pred_col]+1),np.log2(tf_activity_min["mean_exp"]))[0]}
]
tf_activity_corrs = pd.DataFrame(row_list)

# %%
tf_activity_corrs

# %% [markdown]
# ### Compare all models
#
# Making a plot that combines all models

# %%
all_corrs = pd.concat([endogenous_corrs, tf_activity_corrs, prom_motif_corrs_sig])[["model","Dataset","r","Background"]]
all_corrs = all_corrs.fillna("")
all_corrs['model'] = pd.Categorical(all_corrs["model"], all_corrs.query('Dataset == @task_tf').sort_values('r')["model"])
all_corrs["Dataset"] = all_corrs.apply(lambda x: x["Dataset"] if x["Background"] == "" else "{} ({})".format(x["Dataset"],x["Background"]),axis=1)


# %%
all_corrs.to_csv(base_path_results + "all_corrs.tsv", sep="\t", index=None)

# %%
all_corrs = pd.read_csv(base_path_results + "all_corrs.tsv", sep="\t")

# %%
task_tf = "Transcription\nFactor Motifs"
all_corrs['model'] = pd.Categorical(all_corrs["model"], all_corrs.query('Dataset == @task_tf').sort_values('r')["model"])

all_corrs['Dataset'] = pd.Categorical(all_corrs["Dataset"], all_corrs["Dataset"].drop_duplicates())

# %%
scale = 1.0
p9.options.figure_size = (6.4*scale, 4.8*scale)
p = (p9.ggplot(data=all_corrs,mapping=p9.aes(x="Dataset",y="r", fill="model"))
 + p9.geom_bar(stat="identity", position="dodge")
 + p9.labs(x="", y="Correlation between predicted\nand measured log Expression", fill="",
          title="Weingarten-Gabbay")
 + p9.theme(legend_box_margin=0,legend_key_size = 8, legend_text = p9.element_text(size=8),
            title=p9.element_text(size=10),
           axis_title=p9.element_text(size=10))
     #axis_text_x=p9.element_text(rotation=45, hjust=1))    
)
p

# %%
p.save("Graphics/" + "promoter_segal" + ".svg", width=6.0, height=2.5, dpi=300)

# %% [markdown]
# ### DNAse/Expression
#
# What is the relationship between the DNAse and CAGE prediction in Enformer?

# %%
scipy.stats.pearsonr(merged_df_min[pred_col],merged_df_min[pred_col_dnase])

# %%
scale = 1.3
p9.options.figure_size = (6.4*scale, 4.8*scale)
(p9.ggplot(data=merged_df_min,mapping=p9.aes(x=pred_col,y=pred_col_dnase))
 + p9.geom_point(alpha=0.2)
 #+ p9.geom_bin2d(binwidth = (0.05, 0.05))
 + p9.scale_x_log10()
 + p9.scale_y_log10()
 + p9.geom_smooth(method="lm",color="blue")
 + p9.labs(x=pred_col, y=pred_col_dnase)
)

# %%
(np.log2(merged_df_min[pred_col]+1) - np.log2(merged_df_min[pred_col_dnase]+1)).describe()

# %% [markdown]
# ### ISM for the different backgrounds

# %%
cols = ['Oligo_index','Background','Oligo_sequence']
dnase_example = segal_df.query('Oligo_index == 12885').copy()
dnase_example["Background"] = "DNAse_example"
background_seq_df = pd.concat([elements_df.query('Configuration_summary == 0')[cols], 
                               tf_activity_df.query('Motif_info == "-"')[cols],
                               dnase_example[cols]
                              ]).reset_index(drop=True)
background_seq_df["pos"] = -1
background_seq_df["alt"] = -1
background_seq_df["idx"] = -1

def ism(background_df):
    options = ["A", "C", "G", "T"]
    rows = []
    for _,crs_row in background_df.iterrows():
        seq = crs_row["Oligo_sequence"]
        for pos in range(len(seq)):
            for i in range(3):
                ref = seq[pos]
                alt = options[(options.index(ref) + i) % 4]
                new_seq = ''.join([seq[:pos], alt, seq[pos+1:]])
                rows.append({"Oligo_index":crs_row["Oligo_index"],
                             "Oligo_sequence":new_seq,
                             "Background":crs_row["Background"],
                             "pos":pos,
                             "alt":alt,
                             "idx":options.index(alt)
                            })
    return pd.DataFrame(rows)

ism_df = pd.concat([ism(background_seq_df), background_seq_df]).reset_index(drop=True)

# %%
ism_df.sort_values(['Background','pos']).query

# %% [markdown]
# ### Analyzing the TATA shift
#
# We saw in the experimental results that in the RPLP0 promoter, shifting the TATA to a position of -50 leads to a strong decrease in expression. What is the reason for this? We use Enformer to find out.

# %% [markdown]
# #### Occlude the tata positions

# %%
shifted_pos = tata_shift_df.query('Background == "RPLP0_CE_bg"')
shifted_pos['tata_pos'] = shifted_pos["Oligo_sequence"].apply(lambda x: x.find("TATATAAG"))

# %%
#baseline_seq = elements_df.query('Configuration_summary == 0 & Background == "RPLP0_CE_bg"').iloc[0]['Oligo_sequence']
baseline_seq = tata_shift_df.query('Background == "RPLP0_CE_bg" & TATA_position == -11').iloc[0]['Oligo_sequence']

# %%
chip_targets = target_df.loc[target_df.description.str.contains('CHIP') & target_df.description.str.contains('K562')]['description']

# %%
rows = []
chip_dict = {}

rng_insert = "".join(random.choices(["A","C","G","T"],k=8))

smpls = [(-1,baseline_seq)]
for _,df_row in shifted_pos.iterrows():
    tata_pos = df_row["tata_pos"]
    rel_pos = df_row["TATA_position"]
    seq = baseline_seq[:tata_pos] + "N"*8 + baseline_seq[tata_pos+8:]
    #seq = baseline_seq[:tata_pos] + rng_insert + baseline_seq[tata_pos+8:]
    assert len(seq) == len(baseline_seq)
    smpls.append((rel_pos,seq))

for crs_row in smpls:
    occlude_pos = crs_row[0]
    crs = crs_row[1]
    insert = crs + egfp_seq
    landmark = len(crs)//2
    ideal_offset = seq_utils.compute_offset_to_center_landmark(landmark, insert)
    for rev_comp in [False, True]:
        for offset in [-43, 0, 43]:
            modified_sequence, minbin, maxbin, landmarkbin = \
                insert_sequence_at_landing_pad(insert,
                                               aavs1,
                                               fasta_extractor,
                                               shift_five_end=ideal_offset + offset,
                                               landmark=landmark,
                                               rev_comp=rev_comp)
            # predict
            sequence_one_hot = one_hot_encode(modified_sequence)
            predictions = model.predict_on_batch(sequence_one_hot[np.newaxis])['human'][0]
            pred = np.sum(predictions[landmarkbin-1:landmarkbin+2,  5111])
            #append
            rows.append({
                "name": "base" if occlude_pos==-1 else "occlude_"+str(occlude_pos),
                "occlude_pos":occlude_pos,
                "offset":offset,
                "orient":rev_comp,
                "pred":pred
            })
            chip_dict[str(occlude_pos)+"_"+str(offset)+"_"+str(rev_comp)] = predictions[landmarkbin-1:landmarkbin+2,  [x for x in chip_targets.index]].sum(axis=0)

# %%
occlude_pred_df = pd.DataFrame(rows).groupby(['name','occlude_pos'])["pred"].mean().reset_index()
occlude_pred_df["log_pred"] = np.log2(occlude_pred_df["pred"]+1)

# %%
occlude_plot_df = occlude_pred_df.query('name != "base" and occlude_pos < -16')[["occlude_pos","log_pred"]]
occlude_plot_df["Group"] = "Occlusion_shift"
orig_preds = (tata_shift_melt.query('Background == "RPLP0_CE_bg" & pred_obs == "Predicted" & TATA_position < -16')
              [['TATA_position','Log Expression']]
              .rename(columns={'TATA_position':"occlude_pos","Log Expression":"log_pred"})
             )
orig_preds["Group"] = "TATA_shift"
occlude_plot_df = pd.concat([occlude_plot_df,orig_preds])

p = (p9.ggplot(data=occlude_plot_df, mapping=p9.aes(x="occlude_pos",y="log_pred",color="Group")) 
 + p9.geom_line()
 + p9.geom_point()
 + p9.labs(y='log2 Expression', x="TATA position\n relative to TSS", color="Type:")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9), legend_position=(0.7,0.25), legend_background=p9.element_blank(),
           axis_title=p9.element_text(size=10))
)
p

# %%
p.save("Graphics/" + "xsup_promoter_tata_occlude" + ".svg", width=6.4, height=4.8, dpi=300)


# %% [markdown]
# #### ISM around the critical position

# %%
def ism_around_pos(seq, mut_pos=66, mut_range=20):
    options = ["A", "C", "G", "T"]
    rows = []
    for pos in range(max(mut_pos-mut_range,0),mut_pos+mut_range):
        ref = seq[pos]
        for i in range(1,4):
            alt = options[(options.index(ref) + i) % 4]
            new_seq = ''.join([seq[:pos], alt, seq[pos+1:]])
            rows.append({"Oligo_sequence":new_seq,
                         "ref":ref,
                         "pos":pos,
                         "alt":alt,
                         "idx":options.index(alt)
                        })
    return pd.DataFrame(rows)

ism_df = ism_around_pos(baseline_seq)

# %%
ism_rows = []

for _,ism_row in ism_df.iterrows():
    crs = ism_row['Oligo_sequence']
    insert = crs + egfp_seq
    landmark = len(crs)//2
    ideal_offset = seq_utils.compute_offset_to_center_landmark(landmark, insert)
    batch = []
    for rev_comp in [False, True]:
        for offset in [-43, 0, 43]:
            modified_sequence, minbin, maxbin, landmarkbin = \
                insert_sequence_at_landing_pad(insert,
                                               aavs1,
                                               fasta_extractor,
                                               shift_five_end=ideal_offset + offset,
                                               landmark=landmark,
                                               rev_comp=rev_comp)
            batch.append((modified_sequence,landmarkbin))
    # predict
    sequence_one_hot = batch_seq = np.stack([one_hot_encode(x[0]) for x in batch])
    predictions = model.predict_on_batch(sequence_one_hot)['human']
    #append
    for idx, sample in enumerate(batch):
        ism_rows.append({
            "ref":ism_row["ref"],
            "pos":ism_row["pos"],
            "alt":ism_row["alt"],
            "idx":ism_row["idx"],
            "pred":np.sum(predictions[idx, 447-1:447+2,  5111])
        })

# %%
ism_pred_df = pd.DataFrame(ism_rows).groupby(["ref","pos","alt","idx"])["pred"].mean().reset_index()
ism_pred_df["log_pred"] = np.log2(ism_pred_df["pred"]+1)
base_pred = occlude_pred_df.query('name == "base"')["log_pred"].iloc[0]
ism_pred_df["log2_fc"] = ism_pred_df["log_pred"] - base_pred

# %%
ism_pred_df.to_csv(base_path_results + "ism_tata_shift_preds.tsv",sep="\t",index=None)

# %%
ism_pred_df = pd.read_csv(base_path_results + "ism_tata_shift_preds.tsv",sep="\t")

# %%
ism_array = np.zeros((4,40))
total_effect_array = np.zeros((4,40))

min_pos = ism_pred_df['pos'].min()

for _,ism_result in ism_pred_df.sort_values('pos',ascending=True).iterrows():
    ism_array[ism_result['idx'],ism_result['pos']-min_pos] = ism_result['log2_fc']
    
for _,ism_result in ism_pred_df.groupby('pos')['log2_fc'].min().reset_index().sort_values('pos',ascending=True).iterrows():
    total_effect_array[:,int(ism_result['pos'])-min_pos] = np.abs(ism_result['log2_fc'])
    
base_seq_onehot = one_hot_encode(baseline_seq[66-20:66+20])
total_effect_array = base_seq_onehot.transpose()*total_effect_array

# %%
fig, ax = plt.subplots(figsize=(2.8, 3.0))
im = ax.imshow(ism_array[:,20-8:20+8])
cbar = fig.colorbar(im,shrink=0.7,location="bottom")
cbar.ax.tick_params(labelsize=9)
# fix axis labels
ax.set_xticks([1,6,11])
labels = ['-55','-50','-45']
ax.set_xticklabels(labels)
ax.set_yticks([0,1,2,3])
ax.set_yticklabels(['A','C','G','T'])
ax.tick_params(axis='both', which='major', labelsize=9)
fig.savefig("Graphics/" + "promoter_imshow" + ".svg")

# %%
fig,ax = plot_weights(total_effect_array[:,20-8:20+8],despine=True)
fig.savefig("Graphics/" + "promoter_imshow_seq" + ".svg")

# %%
ism_pred_df

# %%
p = (p9.ggplot(data=ism_pred_df,mapping=p9.aes(x="ref",y="log2_fc"))
 + p9.geom_boxplot()
 #+ p9.geom_bin2d(binwidth = (0.05, 0.05))
 + p9.labs(x="Predicted $\mathregular{log_2}$ Fold Change\nof Expression", y="Predicted $\mathregular{log_2}$ Fold Change\nof Expression")
 + p9.theme(axis_text=p9.element_text(size=10))
)
p

# %%
p.save("Graphics/" + "xsup_promoter_ism_nt" + ".svg", width=6.0, height=2.5, dpi=300)

# %% [markdown]
# # Hong et al. (Cohen lab) TRIP-Seq and Patch-MPRA
#
# This section contains our results for the Hong et al experiment(s), who tested some promoters integrated in many backgrounds (TRIP-Seq) and many promoters in a few backgrounds (Patch-MPRA)

# %% [markdown]
# Positions on the plasmid:
#
# - 1810: 5'-ITR
# - 4294-4426: promoter insert
# - 4433-5137: tdTomato
# - 6128: 3'-ITR

# %%
base_path_data = "Data/Cohen_genomic_environments/"
base_path_results = "Results/Cohen_genomic_environments/" 
base_path_data_tss_sim = "Data/TSS_sim/"


# %%
with open(base_path_data + "mScarlet.txt") as file:
    lines = file.readlines()[1:]
    mScarlet_seq = "".join(lines).replace('\n', '').upper()
    
# get the plasmid sequence
with open(base_path_data + "trip_plasmid.gb") as file:
    plasmid_sequence = ""
    started_reading = False
    for line in file.readlines():
        if started_reading:
            line = line.strip().replace(" ","").replace("/","")
            line = re.sub(r'[0-9]+', '', line)
            plasmid_sequence += line
        if line.startswith("ORIGIN"):
            started_reading = True
    plasmid_sequence = plasmid_sequence.upper()

tdTomato = plasmid_sequence[4432:5137]
plasmid_left = plasmid_sequence[1809:4294]
plasmid_right = plasmid_sequence[4426:6128]

# %%
# coordinates here already seem to be hg38 according to methods section
trip_expression = pd.read_csv(base_path_data + "trip_seq.tsv", sep="\t").dropna()
trip_sequences = pd.read_csv(base_path_data + "trip_seq_sequences.tsv",sep="\t").rename(columns={"Name":"promoter"})
trip_sequences["Orientation"] = trip_sequences["Oligo_id"].apply(lambda x: "RC" if x.split("_")[-1] == "-" else "SS")

# add insertion interval
trip_expression["LP_interval"] = trip_expression.apply(lambda x: kipoiseq.Interval(chrom=x["chr"],
                                                                                  start=x["location"],
                                                                                  end=x["location"]+1)
                                                       ,axis=1)

trip = trip_expression.merge(trip_sequences, on="promoter")
trip['LP_interval'] = trip['LP_interval'].astype('str')

# %% [markdown]
# ## Analysis

# %%
pred_col = 'CAGE:chronic myelogenous leukemia cell line:K562 ENCODE, biol__landmark_sum'

# %%
results = pd.read_csv(base_path_results + "cohen_tripseq-enformer-latest_results.tsv",sep="\t")
# average over different offsets
results = (results.groupby(["promoter","LP_interval", "insert_type", "window_size"])[[x for x in results.keys() if "CAGE" in x]].mean().reset_index())
merged_df = results.merge(trip, on=["promoter","LP_interval"])


# %%
scipy.stats.pearsonr(results.query('insert_type == "full" & window_size == -1')[pred_col],
                     results.query('insert_type == "minimal" & window_size == -1')[pred_col]
                    )

# %% [markdown]
# ### Aggregate if the same location was measured several times

# %%
# average counts if the same promoter was measured several times at a location
merged_df = (merged_df.groupby(["promoter","LP_interval", "Orientation", "insert_type", "window_size"])
             [[pred_col,"log2(exp)"]]
            .mean().reset_index())

# %%
merged_df = merged_df.pivot(["promoter","LP_interval", "Orientation", "log2(exp)", "insert_type"], 
                             ["window_size"],
                            [pred_col]).reset_index()
merged_df.columns = ([x for x,y in zip(merged_df.columns.get_level_values(0),merged_df.columns.get_level_values(1)) if y == ""]
                   + [x + "_" + (str(y) if y != -1 else "ingenome") for x,y in zip(merged_df.columns.get_level_values(0),merged_df.columns.get_level_values(1)) if y != ""])

# %%
merged_df_full = merged_df.query('insert_type == "full"')
merged_df_min = merged_df.query('insert_type == "minimal"')

# %% [markdown]
# ### Overall correlation

# %%
print(scipy.stats.pearsonr(np.log2(merged_df_min[pred_col+"_ingenome"]+1),
                           merged_df_min['log2(exp)']))
print(scipy.stats.spearmanr(merged_df_min[pred_col+"_ingenome"],
                           merged_df_min['log2(exp)']))

# %%
print(scipy.stats.pearsonr(np.log2(merged_df_full[pred_col+"_ingenome"]+1),
                           merged_df_full['log2(exp)']))
print(scipy.stats.spearmanr(merged_df_full[pred_col+"_ingenome"],
                           merged_df_full['log2(exp)']))

# %%
scale = 1.0
p9.options.figure_size = (6.4*scale, 4.8*scale)
(p9.ggplot(data=merged_df_min,mapping=p9.aes(x=pred_col+"_ingenome",y='log2(exp)'))
 + p9.geom_point(alpha=0.2)
 + p9.scale_x_log10()
 + p9.labs(x="CAGE:K562 prediction (forward strand)", y="log2(Expression) in TRIP-Seq")
)

# %%
promoter_reduced = merged_df_min.groupby(['promoter']).median()[[pred_col+"_ingenome","log2(exp)"]]

# %%
print(scipy.stats.pearsonr(np.log2(promoter_reduced[pred_col+"_ingenome"]+1),
                           promoter_reduced['log2(exp)']))

# %%
print(scipy.stats.spearmanr(promoter_reduced[pred_col+"_ingenome"],
                           promoter_reduced['log2(exp)']))

# %%
promoter_reduced.reset_index()

# %%
scale = 1.0
p9.options.figure_size = (6.4*scale, 4.8*scale)
p = (p9.ggplot(data=promoter_reduced.reset_index(),mapping=p9.aes(x=pred_col+"_ingenome",y='log2(exp)'))
 + p9.geom_point()
 + p9.geom_smooth(method="lm", color="blue")
 + p9.geom_label(p9.aes(label="promoter"))
 + p9.scale_x_log10()
 + p9.labs(x="Median Enformer CAGE Prediction", y="Median log2 Expression (TRIP-Seq)")
)
p

# %%
scale = 1
p.save("Graphics/" + "xsup_distal_cohen_prom" + ".svg", width=6.4*scale, height=4.8*scale, dpi=300)

# %% [markdown]
# ### Correlation for each promoter

# %%
scale = 1.0
p9.options.figure_size = (6.4*scale, 4.8*scale)
p = (p9.ggplot(data=merged_df_min,mapping=p9.aes(x=pred_col+"_ingenome",y='log2(exp)'))
 #+ p9.geom_point(alpha=0.1)
 + p9.geom_bin2d(binwidth = (0.1, 0.2), raster=True)
 + p9.scale_x_log10()
 + p9.labs(x="Enformer CAGE Prediction", y="log2 Expression (TRIP-Seq)")
 + p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.geom_smooth(method="lm", color="blue")
 + p9.theme(legend_text=p9.element_text(size=8),legend_title=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,
            subplots_adjust={'wspace': 0.25,'hspace': 0.3})
 + p9.facet_wrap('~promoter',scales="free"))
p

# %%
scale = 1
p.save("Graphics/" + "xsup_distal_cohen_expr" + ".svg", width=6.4*scale, height=4.8*scale, dpi=300)

# %%
r_list = []
windows = [k.split("_")[-1] for k in merged_df_min.keys() if pred_col in k]
for promoter in ["hk1","hk2", "hk3", "dev1", "dev2", "dev3"]:
    for wdw in windows:
        subset = merged_df_min.query('promoter == @promoter')
        r = scipy.stats.pearsonr(subset['log2(exp)'],
                               np.log2(subset[pred_col+"_"+wdw]+1))[0]
        rho = scipy.stats.spearmanr(subset['log2(exp)'],
                                subset[pred_col+"_"+wdw])[0]
        r_list.append({
            "promoter":promoter,
            "window_size":wdw,
            "r":r,
            "rho":rho
        })
min_corrs = pd.DataFrame(r_list)

min_corrs["window_size"] = min_corrs["window_size"].apply(lambda x: x if x != "ingenome" else "Full")
min_corrs["window_size"] = pd.Categorical(min_corrs["window_size"], [x for x in windows if x != "ingenome"] + ["Full"])

# %%
min_corrs.query('window_size == "Full"')

# %%
scale = 0.5
p9.options.figure_size = (6.4*scale, 4.8*scale)
p = (p9.ggplot(data=merged_df_min.query('promoter == "hk1"'),mapping=p9.aes(x=pred_col+"_ingenome",y='log2(exp)'))
 #+ p9.geom_point(alpha=0.1)
 + p9.geom_bin2d(binwidth = (0.015, 0.1), raster=True)
 + p9.scale_x_log10()
 + p9.labs(x="Enformer CAGE Prediction", y="Measured log2 Expression\n(TRIP-Seq, hk1 Promoter)")
 + p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.geom_smooth(method="lm", color="blue")
 + p9.theme(legend_text=p9.element_text(size=8),legend_title=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,
            legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10))
)
p

# %%
p.save("Graphics/" + "distal_cohen" + ".svg", width=2.7, height=3.2, dpi=300)

# %%
scale = 1.0
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=min_corrs,mapping=p9.aes(x="window_size",y="r"))#,color="promoter"))
 + p9.geom_boxplot()
 + p9.geom_point(p9.aes(color="promoter"))
 #+ p9.geom_jitter()
 + p9.labs(x="Size of sequence window (bp)",y="Correlation with measured expression")
)
p

# %%
#scale = 1
#p.save("Graphics/" + "xsup_distal_cohen_windows" + ".svg", width=6.4*scale, height=4.8*scale, dpi=300)

# %%
scale = 1.0
p9.options.figure_size = (7*scale, 4.8*scale)
(p9.ggplot(data=merged_df_full,mapping=p9.aes(x=pred_col+"_ingenome",y='log2(exp)'))
 #+ p9.geom_point(alpha=0.1)
 + p9.geom_bin2d(binwidth = (0.1, 0.2))
 + p9.scale_x_log10()
 + p9.labs(x="CAGE:K562 prediction", y="Mean expression in TRIP-Seq")
 + p9.theme(subplots_adjust={'wspace': 0.25,'hspace': 0.3})
 + p9.facet_wrap('~promoter',scales="free"))

# %%
r_list = []
windows = [k.split("_")[-1] for k in merged_df_min.keys() if pred_col in k]
for promoter in ["hk1","hk2", "hk3", "dev1", "dev2", "dev3"]:
    for wdw in windows:
        subset = merged_df_full.query('promoter == @promoter')
        r = scipy.stats.pearsonr(subset['log2(exp)'],
                               np.log2(subset[pred_col+"_"+wdw]+1))[0]
        rho = scipy.stats.spearmanr(subset['log2(exp)'],
                                subset[pred_col+"_"+wdw])[0]
        r_list.append({
            "promoter":promoter,
            "window_size":wdw,
            "r":r,
            "rho":rho
        })
full_corrs = pd.DataFrame(r_list)

full_corrs["window_size"] = full_corrs["window_size"].apply(lambda x: x if x != "ingenome" else "Full")
full_corrs["window_size"] = pd.Categorical(full_corrs["window_size"], [x for x in windows if x != "ingenome"] + ["Full"])

# %%
full_corrs.query('window_size == "Full"')

# %%
scale = 1.0
p9.options.figure_size = (6.4*scale, 4.8*scale)

(p9.ggplot(data=full_corrs,mapping=p9.aes(x="window_size",y="r", color="promoter"))
 #+ p9.geom_boxplot()
 + p9.geom_point()
 #+ p9.geom_jitter()
 + p9.labs(x="Size of sequence window",y="Correlation with measured values")
)

# %%
scale = 1.0
p9.options.figure_size = (6.4*scale, 4.8*scale)

(p9.ggplot(data=full_corrs,mapping=p9.aes(x="window_size",y="r"))
 + p9.geom_boxplot()
 + p9.labs(x="Size of sequence window",y="Correlation with measured values")
)

# %% [markdown]
# ### Basenji2

# %%
results_basenji2 = pd.read_csv(base_path_results + "cohen_tripseq-basenji2-latest_results.tsv",sep="\t")
# average over different offsets
results_basenji2 = (results_basenji2.groupby(["promoter","LP_interval", "insert_type", "window_size"])[[x for x in results.keys() if "CAGE" in x]].mean().reset_index())
merged_df_basenji2 = results_basenji2.merge(trip, on=["promoter","LP_interval"])

# %%
# average counts if the same promoter was measured several times at a location
merged_df_basenji2 = (merged_df_basenji2.groupby(["promoter","LP_interval", "Orientation", "insert_type", "window_size"])
             [[pred_col,"log2(exp)"]]
            .mean().reset_index())

# %%
merged_df_full_basenji2 = merged_df_basenji2.query('insert_type == "full"')
merged_df_min_basenji2 = merged_df_basenji2.query('insert_type == "minimal"')

# %%
print(scipy.stats.pearsonr(np.log2(merged_df_min_basenji2[pred_col]+1),
                           merged_df_min_basenji2['log2(exp)']))
print(scipy.stats.spearmanr(merged_df_min_basenji2[pred_col],
                           merged_df_min_basenji2['log2(exp)']))

# %%
print(scipy.stats.pearsonr(np.log2(merged_df_full_basenji2[pred_col]+1),
                           merged_df_full_basenji2['log2(exp)']))
print(scipy.stats.spearmanr(merged_df_full_basenji2[pred_col],
                           merged_df_full_basenji2['log2(exp)']))

# %%
promoter_reduced_basenji2 = merged_df_min_basenji2.groupby(['promoter']).median()[[pred_col,"log2(exp)"]]

# %%
print(scipy.stats.pearsonr(np.log2(promoter_reduced_basenji2[pred_col]+1),
                           promoter_reduced_basenji2['log2(exp)']))

# %%
print(scipy.stats.spearmanr(promoter_reduced_basenji2[pred_col],
                           promoter_reduced_basenji2['log2(exp)']))

# %%
r_list = []
for promoter in ["hk1","hk2", "hk3", "dev1", "dev2", "dev3"]:
    subset = merged_df_min_basenji2.query('promoter == @promoter')
    r = scipy.stats.pearsonr(subset['log2(exp)'],
                           np.log2(subset[pred_col]+1))[0]
    rho = scipy.stats.spearmanr(subset['log2(exp)'],
                            subset[pred_col])[0]
    r_list.append({
        "promoter":promoter,
        "window_size":"Basenji2",
        "r":r,
        "rho":rho
    })
min_corrs_basenji2 = pd.DataFrame(r_list)


# %%
min_corrs_compare = pd.concat([min_corrs,min_corrs_basenji2])

# %%
scale = 1.0

compare_df = min_corrs_compare.query('window_size in ["3001","12501","Basenji2","39321","65537","98305","Full"]')
compare_df["model"] = (compare_df["window_size"]
                             .apply(lambda x: "Enformer" if x != "Basenji2" else "Basenji2"))

rename_dict = {
    "Basenji2": "Basenji2 (~40kb)",
    "3001":"Enformer 3kb",
    "12501":"12.5kb",
    "39321":"39kb",
    "65537":"65kb",
    "98305":"98kb",
    "Full":"Full (196kb)"
}
compare_df["window_size"] = (compare_df["window_size"]
                             .apply(lambda x: x if x not in rename_dict else rename_dict[x]))

compare_df["window_size"] = pd.Categorical(compare_df["window_size"], 
                                           [x for x in rename_dict.values()])

p9.options.figure_size = (6.4*scale, 4.8*scale)
p = (p9.ggplot(data=compare_df,mapping=p9.aes(x="window_size",y="r"))
 + p9.geom_boxplot()
 + p9.geom_point(p9.aes(color="promoter"),size=1)
 + p9.labs(x="Model and window size used", y="Correlation between predicted\nand measured log Expression", color="Promoter:")
 + p9.theme(legend_box_margin=0,legend_key_size = 9, 
            legend_text = p9.element_text(size=9),
            legend_title = p9.element_text(size=9),
           axis_title=p9.element_text(size=10),
     axis_text_x=p9.element_text(rotation=30, hjust=1))    
)
p

# %%
p.save("Graphics/" + "distal_cohen_compare" + ".svg", width=2.3, height=2.9, dpi=300)

# %% [markdown]
# ## Analysis - Patch-MPRA

# %%
pred_col_patchmpra = 'CAGE:chronic myelogenous leukemia cell line:K562  ENCODE, biol__landmark_sum'

# %%
obs_patchmpra = pd.read_csv(base_path_data + "patchMPRA_expression.tsv", sep="\t").rename(columns={'LP':'LP_name'})

# %%
results_patchmpra = pd.read_csv(base_path_results + "cohen_patchmpra-enformer-latest_results.tsv",sep="\t")
# average over different offsets
results_patchmpra = (results_patchmpra.groupby(["oligo_id","LP_name"])[[x for x in results_patchmpra.keys() if "CAGE" in x]].mean().reset_index())


# %%
patchmpra_merged = results_patchmpra.merge(obs_patchmpra,on=["oligo_id","LP_name"])

# %%
patchmpra_merged["log_pred_col"] = np.log2(patchmpra_merged[pred_col_patchmpra] + 1)

# %%
scipy.stats.pearsonr(np.log10(patchmpra_merged[pred_col_patchmpra] + 1),patchmpra_merged["mean_exp"])

# %% [markdown]
# ### Promoter and location performance

# %%
patchmpra_prom = patchmpra_merged.groupby(['oligo_id']).mean()[[pred_col_patchmpra,"mean_exp"]]

# %%
scipy.stats.pearsonr(np.log10(patchmpra_prom[pred_col_patchmpra] + 1),patchmpra_prom["mean_exp"])

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)
p = (p9.ggplot(data=patchmpra_prom,mapping=p9.aes(x=pred_col_patchmpra,y='mean_exp'))
 + p9.geom_point()
 #+ p9.geom_bin2d(binwidth = (0.015, 0.1), raster=True)
 + p9.scale_x_log10()
 + p9.labs(x="Enformer CAGE Prediction (mean over locations)", y="Measured log2 Expression\n(Patch-MPRA, mean over locations)")
 + p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.geom_smooth(method="lm", color="blue")
 + p9.theme(legend_text=p9.element_text(size=8),legend_title=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,
            axis_title=p9.element_text(size=10))
)
p

# %%
scale = 1
p.save("Graphics/" + "xsup_distal_cohen_patch_prom" + ".svg", width=6.4*scale, height=4.8*scale, dpi=300)

# %%
patchmpra_loc = patchmpra_merged.groupby(['LP_name']).mean()[[pred_col_patchmpra,"mean_exp"]]

# %%
scipy.stats.pearsonr(np.log10(patchmpra_loc[pred_col_patchmpra] + 1),patchmpra_loc["mean_exp"])

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)
p = (p9.ggplot(data=patchmpra_loc,mapping=p9.aes(x=pred_col_patchmpra,y='mean_exp'))
 + p9.geom_point()
 #+ p9.geom_bin2d(binwidth = (0.015, 0.1), raster=True)
 + p9.scale_x_log10()
 + p9.labs(x="Enformer CAGE Prediction (mean over promoters)", y="Measured log2 Expression\n(Patch-MPRA, mean over promoters)")
 + p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.geom_smooth(method="lm", color="blue")
 + p9.theme(legend_text=p9.element_text(size=8),legend_title=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,
            axis_title=p9.element_text(size=10))
)
p

# %%
scale = 1
p.save("Graphics/" + "xsup_distal_cohen_patch_prom" + ".svg", width=6.4*scale, height=4.8*scale, dpi=300)

# %% [markdown]
# ### Multiplicative model

# %% [markdown]
# #### Cohen

# %%
ols_cohen_prom = (statsmodels.regression.linear_model.OLS
                  .from_formula('mean_exp ~ oligo_id',
                                data=patchmpra_merged))

res_cohen_prom = ols_cohen_prom.fit()

preds = res_cohen_prom.predict()

print(scipy.stats.pearsonr(patchmpra_merged["mean_exp"], preds))
print(sklearn.metrics.r2_score(patchmpra_merged["mean_exp"], preds))

# %%
ols_cohen_mult = (statsmodels.regression.linear_model.OLS
                  .from_formula('mean_exp ~ oligo_id + LP_name',
                                data=patchmpra_merged))

res_cohen_mult = ols_cohen_mult.fit()

preds = res_cohen_mult.predict()

params_cohen = res_cohen_mult.params.reset_index().rename(columns={"index":"Name",0:"Cohen"})

print(scipy.stats.pearsonr(patchmpra_merged["mean_exp"], preds))
print(sklearn.metrics.r2_score(patchmpra_merged["mean_exp"], preds))

# %% [markdown]
# #### Enformer

# %%
ols_enformer_prom = (statsmodels.regression.linear_model.OLS
                  .from_formula('log_pred_col ~ oligo_id',
                                data=patchmpra_merged))

res_enformer_prom = ols_enformer_prom.fit()

preds = res_enformer_prom.predict()

print(scipy.stats.pearsonr(patchmpra_merged["log_pred_col"], preds))
print(sklearn.metrics.r2_score(patchmpra_merged["log_pred_col"], preds))

# %%
ols_enformer_mult = (statsmodels.regression.linear_model.OLS
                  .from_formula('log_pred_col ~ oligo_id + LP_name',
                                data=patchmpra_merged))

res_enformer_mult = ols_enformer_mult.fit()

preds = res_enformer_mult.predict()

params_enformer = res_enformer_mult.params.reset_index().rename(columns={"index":"Name",0:"Enformer"})

print(scipy.stats.pearsonr(patchmpra_merged["log_pred_col"], preds))
print(sklearn.metrics.r2_score(patchmpra_merged["log_pred_col"], preds))

# %% [markdown]
# #### Compare params

# %%
params_merged = params_enformer.merge(params_cohen, on="Name")

# %%
scipy.stats.pearsonr(params_merged.loc[params_merged.Name.str.startswith('oligo')]["Enformer"],params_merged.loc[params_merged.Name.str.startswith('oligo')]["Cohen"])

# %%
scipy.stats.pearsonr(params_merged.loc[params_merged.Name.str.startswith('LP')]["Enformer"],params_merged.loc[params_merged.Name.str.startswith('LP')]["Cohen"])

# %% [markdown]
# #### Induced variation

# %%
patchmpra_std = patchmpra_merged.groupby('oligo_id')[['mean_exp','log_pred_col']].std().reset_index()
patchmpra_std["Name"] = patchmpra_std["oligo_id"].apply(lambda x: "{}[T.{}]".format("oligo_id",x))
patchmpra_std = patchmpra_std.merge(params_merged,on="Name")

# %%
p=(p9.ggplot(data=patchmpra_std, 
            mapping=p9.aes(x="Enformer",y="log_pred_col"))
 + p9.geom_point()
 + p9.geom_smooth(color="blue")
 + p9.labs(x="Predicted Promoter Strength (a.u.)",y="Standard deviation of predicted\nlog2 Expression(over locations)")
 #+ p9.geom_smooth(method="lm", color="blue")
)
p

# %%
p.save("Graphics/" + "xsup_distal_cohen_patch_std" + ".svg", width=6.4*scale, height=4.8*scale, dpi=300)

# %%
p=(p9.ggplot(data=patchmpra_std, 
            mapping=p9.aes(x="Cohen",y="mean_exp"))
 + p9.geom_point()
 + p9.geom_smooth(color="blue")
 + p9.coord_cartesian(ylim=(0,2))
 + p9.labs(x="Measured Promoter Strength (a.u.)",y="Standard deviation of measured\nlog2 Expression(over locations)")
 #+ p9.geom_smooth(method="lm", color="blue")
)
p

# %%
p.save("Graphics/" + "xsup_distal_cohen_patch_stdobs" + ".svg", width=6.4*scale, height=4.8*scale, dpi=300)

# %% [markdown]
# # Bergman et al. Enhancer-Promoter Compatibility MPRA
#
# The experiment by Bergman et al. tested many combinations of promoter and enhancer in a plasmid background
#
# Uses hSTARR-seq_SCP1_vector_ 6243
#
# Locations:
#
# - Promoter: 18-118 (KpnI-ApaI, replaces SCP1 promoter)
# - Enhancer: 372-1838 (AgeI-SgrDI)
#
# In between there is a truncated GFP and some other stuff.
#
# Architecture:
#
# plasmid_left + five_prime_prom + promoter + three_prime_prom + tGFP + bc_left + bc + bc_right + five_prime_enhancer + enhancer + three_prime_enhancer + plasmid_right
#
# All enhancers are hg19, in the + direction.
#
#
# Lots of duplicated rows apparently. Maybe:
#
# - Aggregate by taking median accross replicates and then median across barcodes?
# - Throw out rows which are only present in one replicate?

# %%
base_path_data = "Data/Bergman_compatibility_logic/"
base_path_data_fulco = "Data/Fulco_CRISPRi/"
base_path_results = "Results/Bergman_compatibility_logic/"
base_path_results_fulco = "Results/Fulco_CRISPRi/"

# %%
promoters = pd.read_csv(os.path.join(base_path_data,"promoters.txt"),sep="\t")[["fragment","seq"]]
enhancers = pd.read_csv(os.path.join(base_path_data,"enhancers.txt"),sep="\t")[["fragment","seq"]]
crs_df = pd.concat([promoters, enhancers])

# %%
len(promoters)

# %%
promoters["seq"].str.len().describe()

# %%
enhancers["seq"].str.len().describe()

# %%
# get the plasmid sequence
with open(base_path_data + "hstarr_seq.gbk") as file:
    plasmid_sequence = ""
    started_reading = False
    for line in file.readlines():
        if started_reading:
            line = line.strip().replace(" ","").replace("/","")
            line = re.sub(r'[0-9]+', '', line)
            plasmid_sequence += line
        if line.startswith("ORIGIN"):
            started_reading = True
    plasmid_sequence = plasmid_sequence.upper()

plasmid_left = plasmid_sequence[17:]
tGFP = plasmid_sequence[119:372]
plasmid_right = plasmid_sequence[1838:]
five_prime_prom = "TCATGTGGGACATCAAGC"
three_prime_prom = "GCATAGTGAGTCCACCTT"
five_prime_enhancer = "GCTAACTTCTACCCATGC"
three_prime_enhancer = "GCAAGTTAAGTAGGTCGT"
bc_left = "TAGATTGATCTAGAGCATGCA" 
bc_right = "GAGTACTGGTATGTTCA"

# %%
#all(pd.read_csv(base_path_data + "GSE184426_ExP_1Kx1K_counts(1).txt",sep="\t") == pd.read_csv(base_path_data + "GSE184426_ExP_1Kx1K_counts.txt",sep="\t"))

# %%
exp_df = pd.read_csv(base_path_data + "GSE184426_ExP_1Kx1K_counts.txt",sep="\t")

# %%
exp_df["RNA_sum"] = (exp_df["weighted_reads_rep1"] + exp_df["weighted_reads_rep2"] + exp_df["weighted_reads_rep3"] + exp_df["weighted_reads_rep4"])
# subset DNA > 25 and RNA >= 1
exp_df =  exp_df.query('DNA_input > 25 & RNA_sum > 0')
# subset if < 2 barcodes
bc_count = (exp_df[["promoter","enhancer"]]
                  .groupby(["promoter","enhancer"])
                  .size()
                  .reset_index()
                  .rename(columns={0:"bc_count"})
                 ).query('bc_count > 1')
exp_df = exp_df.merge(bc_count, on=["promoter","enhancer"])

# %%
# check replicability analysis
exp_df["RNA_sum_1"] = (exp_df["weighted_reads_rep1"] + exp_df["weighted_reads_rep2"]) 
exp_df["RNA_sum_2"] = (exp_df["weighted_reads_rep3"] + exp_df["weighted_reads_rep4"])
exp_df["log(RNA/DNA)_1"] = np.log2(exp_df["RNA_sum_1"]/exp_df["DNA_input"])
exp_df["log(RNA/DNA)_2"] = np.log2(exp_df["RNA_sum_2"]/exp_df["DNA_input"])
scipy.stats.pearsonr(exp_df.replace([np.inf, -np.inf], np.nan).dropna()["log(RNA/DNA)_1"],
                     exp_df.replace([np.inf, -np.inf], np.nan).dropna()["log(RNA/DNA)_2"])[0]**2

# %%
# aggregate over barcodes
exp_df = (exp_df[["promoter","enhancer","DNA_input","RNA_sum"]]
          .groupby(["promoter","enhancer"])
          .sum()
          .reset_index()
         )

# %%
exp_df["log(RNA/DNA)"] = np.log2(exp_df["RNA_sum"]/exp_df["DNA_input"])

# %%
# merge with sequences
promoters = pd.read_csv(base_path_data + "promoters.txt",sep="\t").rename(columns={"fragment":"promoter","seq":"promoter_seq"})
enhancers = pd.read_csv(base_path_data + "enhancers.txt",sep="\t").rename(columns={"fragment":"enhancer","seq":"enhancer_seq"})

exp_df = exp_df.merge(promoters[["promoter","promoter_seq"]], on="promoter")
exp_df = exp_df.merge(enhancers[["enhancer","enhancer_seq"]], on="enhancer")


# %%
#prom_df = pd.read_csv(base_path_data + "GSE184426_HS-STARR-seq-promoters_counts.txt",sep="\t")
#prom_df["promoter"] = prom_df["chr"] + ":" + prom_df["start"].astype("str") + "-" + prom_df["end"].astype("str") + "-promoter"
#enhance_df = pd.read_csv(base_path_data + "GSE184426_HS-STARR-seq-enhancers_counts.txt",sep="\t")
#enhance_df["enhancer"] = enhance_df["chr"] + ":" + enhance_df["start"].astype("str") + "-" + enhance_df["end"].astype("str") + "-enhancer"

# %% [markdown]
# ## Test plasmid

# %%
def test_bergmann_plasmid(insert, landmark, extra_offset=0, verbose=True):
    offset = compute_offset_to_center_landmark(landmark,insert)

    modified_sequence, minbin, maxbin, landmarkbin = pad_sequence(insert, landmark=landmark, shift_five_end=offset + extra_offset)
    sequence_one_hot = one_hot_encode(modified_sequence)
    predictions = model.predict_on_batch(sequence_one_hot[np.newaxis])['human'][0]
    
    if verbose:
        tracks = {'DNASE:K562': predictions[:, 121],
                  'CAGE:chronic myelogenous leukemia cell line:K562': np.log10(1 + predictions[:,  4828]),
                 }
        plot_tracks(tracks, target_interval)

        print(predictions[minbin-1:maxbin+1,  4828])
        print(predictions[landmarkbin,  4828])
        print(sum(predictions[minbin-1:maxbin+1,  4828]))
    else:
        return sum(predictions[landmarkbin-1:landmarkbin+2,  4828])


# %%
promoter = exp_df.sort_values('log(RNA/DNA)').iloc[-1]["promoter_seq"]
enhancer = exp_df.sort_values('log(RNA/DNA)').iloc[-1]["enhancer_seq"]
bc = "N"*16 #dummy barcode

plasmid = plasmid_left + five_prime_prom + promoter + three_prime_prom + tGFP + bc_left + bc + bc_right + five_prime_enhancer + enhancer + three_prime_enhancer + plasmid_right
plasmid_min = five_prime_prom + promoter + three_prime_prom + tGFP + bc_left + bc + bc_right + five_prime_enhancer + enhancer + three_prime_enhancer 

landmark_plasmid = len(plasmid_left + five_prime_prom) + len(promoter)//2
landmark_min = len(five_prime_prom) + len(promoter)//2

target_interval = kipoiseq.Interval("chr0",0,(SEQUENCE_LENGTH - PADDING*2))

# %%
test_bergmann_plasmid(plasmid, landmark_plasmid, extra_offset=0)

# %%
test_bergmann_plasmid(plasmid, landmark_plasmid, extra_offset=-64)

# %%
test_bergmann_plasmid(plasmid_min, landmark_min, extra_offset=0)

# %%
test_bergmann_plasmid(plasmid_min, landmark_min, extra_offset=-64)


# %% [markdown]
# ## Test aavs1 insertion

# %%
def test_bergmann_insertion(insert, landmark, fasta_extractor=fasta_extractor, extra_offset=0, verbose=True):
    offset = compute_offset_to_center_landmark(landmark,insert)

    aavs1 = kipoiseq.Interval(chrom="chr19",start=55_115_750,end=55_115_780)
    
    modified_sequence, minbin, maxbin, landmarkbin = insert_sequence_at_landing_pad(insert, aavs1, fasta_extractor,landmark=landmark, shift_five_end=offset + extra_offset)
    sequence_one_hot = one_hot_encode(modified_sequence)
    predictions = model.predict_on_batch(sequence_one_hot[np.newaxis])['human'][0]
    
    if verbose:
        tracks = {'DNASE:K562': predictions[:, 121],
                  'CAGE:chronic myelogenous leukemia cell line:K562': np.log10(1 + predictions[:,  4828]),
                 }
        plot_tracks(tracks, target_interval)

        print(predictions[minbin-1:maxbin+1,  4828])
        print(predictions[landmarkbin,  4828])
        print(sum(predictions[minbin-1:maxbin+1,  4828]))
    else:
        return (predictions[landmarkbin,  4828], sum(predictions[landmarkbin-1:landmarkbin+2,  4828]))


# %%
test_bergmann_insertion(plasmid_min, landmark_min, extra_offset=0)

# %%
test_bergmann_insertion(plasmid_min, landmark_min, extra_offset=-64)

# %% [markdown]
# ## Analysis

# %%
with open(base_path_results  + "bergmann_exp-enformer-latest_results.tsv", 'r') as tsv:
    header = tsv.readline().strip()
    cols = header.split("\t")
cols = [k for k in cols if "CAGE" in k or "DNASE" in k and k.endswith("landmark_sum")]

results = pd.read_csv(base_path_results + "bergmann_exp-enformer-latest_results.tsv",sep="\t", usecols=["promoter","enhancer","insert_type"]+cols)
results = (results.groupby(["promoter","enhancer","insert_type"])[[x for x in results.keys() if "CAGE" in x or "DNASE" in x]].mean().reset_index())

# %%
merged_df_all = (results.merge(exp_df, on=["promoter","enhancer"]))
promoter_class = pd.read_csv(os.path.join(base_path_data,"promoter_classes.txt"),sep="\t")[["fragment","promoter_exp_class","randomBG"]].rename(columns={"fragment":"promoter", "randomBG":"randomBG_promoter"})
enhancer_class = pd.read_csv(os.path.join(base_path_data,"enhancer_classes.txt"),sep="\t")[["fragment","enhancer_exp_class","randomBG"]].rename(columns={"fragment":"enhancer", "randomBG":"randomBG_enhancer"})
merged_df_all = merged_df_all.merge(promoter_class,on="promoter").merge(enhancer_class,on="enhancer")

# %% [markdown]
# ### Subset df

# %%
# will need to change to 'CAGE:chronic myelogenous leukemia cell line:K562 ENCODE, biol__landmark_sum' soon
pred_col = 'CAGE:chronic myelogenous leukemia cell line:K562 ENCODE, biol__landmark_sum'
pred_col_dnase = 'DNASE:K562_4_landmark_sum'

merged_df_min = merged_df_all.query('insert_type == "min_plasmid"')
merged_df_full = merged_df_all.query('insert_type == "full_plasmid"')
merged_df_ingenome = merged_df_all.query('insert_type == "aavs1"')

# %%
# standardize DNA_input (for use in regression later)
merged_df_min["DNA_input_scaled"] = (merged_df_min["DNA_input"] - merged_df_min["DNA_input"].mean())/merged_df_min["DNA_input"].std()

# Prepare some variables in log form
log_pred_col = "log_" + pred_col.replace(":","_").replace(" ","_").replace(",","_")
log_pred_col_dnase = "log_" + pred_col_dnase.replace(":","_").replace(" ","_").replace(",","_")

merged_df_min["log_DNA_input"] = np.log2(merged_df_min["DNA_input"] + 1)
merged_df_min["log_RNA_sum"] = np.log2(merged_df_min["RNA_sum"] + 1)
merged_df_min[log_pred_col] = np.log2(merged_df_min[pred_col] + 1)
merged_df_min[log_pred_col_dnase] = np.log2(merged_df_min[pred_col_dnase] + 1)

# %%
merged_df_ingenome[log_pred_col] = np.log2(merged_df_ingenome[pred_col] + 1)

# %% [markdown]
# ### Overall correlation
#
# Enformer shows good, but not great, overall correlation with predictions

# %%
print(scipy.stats.pearsonr(np.log2(merged_df_full[pred_col]+1), merged_df_full['log(RNA/DNA)']))
print(scipy.stats.spearmanr(merged_df_full[pred_col], merged_df_full['log(RNA/DNA)']))

# %%
print(scipy.stats.pearsonr(np.log2(merged_df_min[pred_col]+1), merged_df_min['log(RNA/DNA)']))
print(scipy.stats.spearmanr(merged_df_min[pred_col], merged_df_min['log(RNA/DNA)']))

# %%
print(scipy.stats.pearsonr(np.log2(merged_df_ingenome[pred_col]+1), merged_df_ingenome['log(RNA/DNA)']))
print(scipy.stats.spearmanr(merged_df_ingenome[pred_col], merged_df_ingenome['log(RNA/DNA)']))

# %%
# try DNAse, just for fun
print(scipy.stats.pearsonr(np.log2(merged_df_min['DNASE:K562_1_landmark_sum']+1), merged_df_min['log(RNA/DNA)']))
print(scipy.stats.pearsonr(np.log2(merged_df_min['DNASE:K562_2_landmark_sum']+1), merged_df_min['log(RNA/DNA)']))
print(scipy.stats.pearsonr(np.log2(merged_df_min['DNASE:K562_3_landmark_sum']+1), merged_df_min['log(RNA/DNA)']))
print(scipy.stats.pearsonr(np.log2(merged_df_min['DNASE:K562_4_landmark_sum']+1), merged_df_min['log(RNA/DNA)']))

# %%
scale = 1.3
p9.options.figure_size = (6.4*scale, 4.8*scale)
(p9.ggplot(data=merged_df_min,mapping=p9.aes(x=pred_col,y='log(RNA/DNA)'))
 #+ p9.geom_point(alpha=0.1)
 + p9.geom_bin2d(binwidth = (0.025, 0.1))
 + p9.scale_x_log10()
 + p9.geom_smooth(method="lm",color="blue")
 + p9.labs(x="CAGE:K562 prediction", y="Measured log2(RNA/DNA)")
)

# %%
ols_pred_fit = (statsmodels.regression.linear_model.OLS
                  .from_formula('log_RNA_sum ~ log_DNA_input + {}'.format(log_pred_col),
                                data=merged_df_min))

res_pred_fit = ols_pred_fit.fit(maxiter=100)

# %%
res_pred_fit.summary()

# %%
scipy.stats.pearsonr(res_pred_fit.predict(), merged_df_min["log_RNA_sum"])

# %%
ols_pred_fit = (statsmodels.regression.linear_model.OLS
                  .from_formula('log_RNA_sum ~ log_DNA_input + {}'.format(log_pred_col_dnase),
                                data=merged_df_min))

res_pred_fit = ols_pred_fit.fit(maxiter=100)

# %%
scipy.stats.pearsonr(res_pred_fit.predict(), merged_df_min["log_RNA_sum"])

# %% [markdown]
# ### Compare (simple) average promoter and enhancer effects
#
# Predicted promoter strengths correlate very well with the average promoter effect measured in the MPRA
#
# However, this is not the case with enhancers, where the correlation is basically nonexistent

# %% [markdown]
# #### Correlation of average effects

# %%
prom_df = merged_df_min.groupby("promoter")[["log(RNA/DNA)","RNA_sum",pred_col,'DNASE:K562_1_landmark_sum']].mean().reset_index()
enhance_df = merged_df_min.groupby("enhancer")[["log(RNA/DNA)","RNA_sum",pred_col,'DNASE:K562_1_landmark_sum']].mean().reset_index()

# %%
print(scipy.stats.pearsonr(np.log2(prom_df[pred_col]+1), prom_df['log(RNA/DNA)']))
print(scipy.stats.spearmanr(prom_df[pred_col], prom_df['log(RNA/DNA)']))

print(scipy.stats.pearsonr(np.log2(prom_df[pred_col]+1), np.log2(prom_df['RNA_sum'] + 1)))
print(scipy.stats.spearmanr(prom_df[pred_col], prom_df['RNA_sum']))

# %%
# try DNAse, just for fun
print(scipy.stats.pearsonr(np.log2(prom_df['DNASE:K562_1_landmark_sum']+1), prom_df['log(RNA/DNA)']))
print(scipy.stats.spearmanr(prom_df['DNASE:K562_1_landmark_sum'], prom_df['log(RNA/DNA)']))

# %%
print(scipy.stats.pearsonr(np.log2(enhance_df[pred_col]+1), enhance_df['log(RNA/DNA)']))
print(scipy.stats.spearmanr(enhance_df[pred_col], enhance_df['log(RNA/DNA)']))

print(scipy.stats.pearsonr(np.log2(enhance_df[pred_col]+1), np.log2(enhance_df['RNA_sum'] + 1)))
print(scipy.stats.spearmanr(enhance_df[pred_col], enhance_df['RNA_sum']))

# %%
# try DNAse, just for fun
print(scipy.stats.pearsonr(np.log2(enhance_df['DNASE:K562_1_landmark_sum']+1), enhance_df['log(RNA/DNA)']))
print(scipy.stats.spearmanr(enhance_df[pred_col], enhance_df['log(RNA/DNA)']))

# %%
scale = 1.3
p9.options.figure_size = (6.4*scale, 4.8*scale)
(p9.ggplot(data=prom_df,mapping=p9.aes(x=pred_col,y='log(RNA/DNA)'))
 + p9.geom_point(alpha=0.25)
 #+ p9.geom_bin2d(binwidth = (0.025, 0.1))
 + p9.scale_x_log10()
 + p9.geom_smooth(method="lm",color="blue")
 + p9.labs(x="Average promoter strength in Enformer", y="Average promoter strength in MPRA")
)

# %%
scale = 1.3
p9.options.figure_size = (6.4*scale, 4.8*scale)
(p9.ggplot(data=enhance_df,mapping=p9.aes(x=pred_col,y='log(RNA/DNA)'))
 + p9.geom_point(alpha=0.25)
 #+ p9.geom_bin2d(binwidth = (0.025, 0.1))
 + p9.scale_x_log10()
 + p9.geom_smooth(method="lm",color="blue")
 + p9.labs(x="Average enhancer strength in Enformer", y="Average enhancer strength in MPRA")
)

# %%
# full insert
prom_df_full = merged_df_full.groupby("promoter")[["log(RNA/DNA)","RNA_sum",pred_col]].mean().reset_index()
enhance_df_full = merged_df_full.groupby("enhancer")[["log(RNA/DNA)","RNA_sum",pred_col]].mean().reset_index()

print(scipy.stats.pearsonr(np.log2(prom_df_full[pred_col]+1), prom_df_full['log(RNA/DNA)']))
print(scipy.stats.spearmanr(prom_df_full[pred_col], prom_df_full['log(RNA/DNA)']))

print(scipy.stats.pearsonr(np.log2(prom_df_full[pred_col]+1), np.log2(prom_df_full['RNA_sum'] + 1)))
print(scipy.stats.spearmanr(prom_df_full[pred_col], prom_df_full['RNA_sum']))

print(scipy.stats.pearsonr(np.log2(enhance_df_full[pred_col]+1), enhance_df_full['log(RNA/DNA)']))
print(scipy.stats.spearmanr(enhance_df_full[pred_col], enhance_df_full['log(RNA/DNA)']))

print(scipy.stats.pearsonr(np.log2(enhance_df_full[pred_col]+1), np.log2(enhance_df_full['RNA_sum'] + 1)))
print(scipy.stats.spearmanr(enhance_df_full[pred_col], enhance_df_full['RNA_sum']))

# %%
# in-genome
prom_df_ingenome = merged_df_ingenome.groupby("promoter")[["log(RNA/DNA)","RNA_sum",pred_col]].mean().reset_index()
enhance_df_ingenome = merged_df_ingenome.groupby("enhancer")[["log(RNA/DNA)","RNA_sum",pred_col]].mean().reset_index()

print(scipy.stats.pearsonr(np.log2(prom_df_ingenome[pred_col]+1), prom_df_ingenome['log(RNA/DNA)']))
print(scipy.stats.spearmanr(prom_df_ingenome[pred_col], prom_df_ingenome['log(RNA/DNA)']))

print(scipy.stats.pearsonr(np.log2(prom_df_ingenome[pred_col]+1), np.log2(prom_df_ingenome['RNA_sum'] + 1)))
print(scipy.stats.spearmanr(prom_df_ingenome[pred_col], prom_df_ingenome['RNA_sum']))

print(scipy.stats.pearsonr(np.log2(enhance_df_ingenome[pred_col]+1), enhance_df_ingenome['log(RNA/DNA)']))
print(scipy.stats.spearmanr(enhance_df_ingenome[pred_col], enhance_df_ingenome['log(RNA/DNA)']))

print(scipy.stats.pearsonr(np.log2(enhance_df_ingenome[pred_col]+1), np.log2(enhance_df_ingenome['RNA_sum'] + 1)))
print(scipy.stats.spearmanr(enhance_df_ingenome[pred_col], enhance_df_ingenome['RNA_sum']))

# %% [markdown]
# #### Variation within

# %%
merged_df_min[["RNA_sum",pred_col]].std()

# %%
prom_df_std = (merged_df_min.groupby("promoter")[["RNA_sum",pred_col]]
               .std()
               .reset_index()
               .rename(columns={pred_col:"Predicted Expression", "RNA_sum":"Measured Expression"})
               .melt(id_vars="promoter")
               .rename(columns={"promoter":"element"})
              )
prom_df_std["element_type"] = "Promoter"

enhance_df_std = (merged_df_min.groupby("enhancer")[["RNA_sum",pred_col]]
                   .std()
                   .reset_index()
                   .rename(columns={pred_col:"Predicted Expression", "RNA_sum":"Measured Expression"})
                   .melt(id_vars="enhancer")
                  .rename(columns={"enhancer":"element"})
                  )
enhance_df_std["element_type"] = "Enhancer"

std_df = pd.concat([prom_df_std, enhance_df_std])

# %%
prom_df_std.query('variable == "Predicted Expression"').dropna().sort_values('value')

# %%
(p9.ggplot(data=std_df,mapping=p9.aes(x="variable",y="value", color="element_type"))
 + p9.geom_boxplot()
 + p9.scale_y_log10()
 + p9.labs(x="", y="Standard deviation")
)

# %% [markdown]
# ### Modelling average enhancer/promoter effects
#
# We model the expression using promoter/enhancer fixed effects.
#
# For the measured expression we use the model:
#
# $RNA \sim Poisson(\lambda = \exp(\beta\log(DNA) + P + E))$
#
# For the enformer prediction, we model with:
#
# $\log(\hat{RNA} + 1) \sim Normal(\mu = P + E, \sigma^2)$
#
# $\log(\hat{RNA} + 1) = P + E + \epsilon$
#
# or
#
# $RNA \sim Gamma(\exp(P + E))$
#
# (Why Gamma? Because Enformer learns a Poisson rate parameter)

# %%
bg_prom = merged_df_min.groupby(['promoter','randomBG_promoter', 'promoter_exp_class'])['log(RNA/DNA)'].agg(["mean","size"]).reset_index().sort_values('mean').query('randomBG_promoter')
bg_prom.iloc[len(bg_prom)//2]

# %%
bg_enh = merged_df_min.groupby(['enhancer','randomBG_enhancer', 'enhancer_exp_class'])['log(RNA/DNA)'].agg(["mean","size"]).reset_index().sort_values('mean').query('randomBG_enhancer')
bg_enh.iloc[len(bg_enh)//2]

# %%
# baseline values
base_prom = "chr14:78352885-78353149-promoter"
base_enh = "chr22:17329655-17329919-enhancer"

# %% [markdown] toc-hr-collapsed=true
# #### Bergmann - promoter only model

# %%
poisson_bergmann_prom = (statsmodels.discrete.discrete_model.Poisson
                  .from_formula('RNA_sum ~ log_DNA_input + C(promoter, Treatment("{}"))'.format(base_prom),
                                data=merged_df_min))

#res_bergmann = poisson_bergmann.fit(disp=True,maxiter=100)
res_bergmann_prom = poisson_bergmann_prom.fit(method="minimize",min_method='dogleg',maxiter=100)

# %%
res_bergmann_prom.params.to_csv(base_path_results + "bergmann_promoter_poisson_params_baselined.tsv",sep="\t")

# %%
bergmann_prom_params = pd.read_csv(base_path_results + "bergmann_promoter_poisson_params_baselined.tsv",sep="\t").set_index("Unnamed: 0").squeeze()

poisson_bergmann_prom = (statsmodels.discrete.discrete_model.Poisson
                  .from_formula('RNA_sum ~ log_DNA_input + C(promoter, Treatment("{}"))'.format(base_prom),
                                data=merged_df_min))

preds = poisson_bergmann_prom.predict(params=bergmann_prom_params)
pred_df = merged_df_min[["promoter","enhancer","RNA_sum"]]
pred_df["pred"] = preds

pred_df.to_csv(base_path_results + "bergmann_promoter_poisson_preds_baselined.tsv",sep="\t",index=None)

print(scipy.stats.pearsonr(pred_df["RNA_sum"], pred_df["pred"]))
print(sklearn.metrics.r2_score(pred_df["RNA_sum"], pred_df["pred"]))
print(scipy.stats.pearsonr(np.log2(pred_df["RNA_sum"]+1), np.log2(pred_df["pred"]+1)))
print(sklearn.metrics.r2_score(np.log2(pred_df["RNA_sum"]+1), np.log2(pred_df["pred"]+1)))

# %%
pred_df2 = pd.read_csv(base_path_results + "bergmann_promoter_poisson_preds_baselined.tsv",sep="\t")

print(scipy.stats.spearmanr(pred_df2["RNA_sum"], pred_df2["pred"]))
print(scipy.stats.pearsonr(pred_df2["RNA_sum"], pred_df2["pred"]))

# %%
p = (p9.ggplot(data=pred_df2,mapping=p9.aes(x="pred",y="RNA_sum"))
 #+ p9.geom_col(position="dodge")
 + p9.scale_x_log10()
 + p9.scale_y_log10()
 + p9.geom_bin2d(bins=100, raster=True)
 + p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.geom_smooth(method="lm",color="blue")
 + p9.theme(legend_text=p9.element_text(size=8),legend_title=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,axis_title=p9.element_text(size=10))#
 + p9.labs(x="Poisson model prediction",y="RNA")
 #+ p9.scale_y_log10()
 #+ p9.labs(x="",y="Density")
)
p

# %% [markdown]
# #### Bergmann - multiplicative model

# %%
old_param = pd.read_csv(base_path_results + "bergmann_multiplicative_poisson_params.tsv",sep="\t").rename(columns={"Unnamed: 0":"Name", "0": "old"})
new_param = res_bergmann_mult.params.reset_index().rename(columns={"index":"Name", 0: "new"})
new_param["Name"] = new_param["Name"].str.replace('C(promoter, Treatment("{}"))'.format(base_prom),"promoter",regex=False).str.replace('C(enhancer, Treatment("{}"))'.format(base_enh),"enhancer",regex=False)
compare = old_param.merge(new_param,on="Name")
compare["class"] = compare["Name"].apply(lambda x: "prom" if x.startswith("prom") else "enh" if x.startswith("enh") else "other")

# %%
poisson_bergmann_mult = (statsmodels.discrete.discrete_model.Poisson
                  .from_formula('RNA_sum ~ log_DNA_input + C(promoter, Treatment("{}")) + C(enhancer, Treatment("{}"))'.format(base_prom,base_enh),
                                data=merged_df_min))

#res_bergmann = poisson_bergmann.fit(disp=True,maxiter=100)
res_bergmann_mult = poisson_bergmann_mult.fit(method="minimize",min_method='dogleg',maxiter=100)

# %%
res_bergmann_mult.params.to_csv(base_path_results + "bergmann_multiplicative_poisson_params_baselined.tsv",sep="\t")

# %%
bergmann_mult_params = pd.read_csv(base_path_results + "bergmann_multiplicative_poisson_params_baselined.tsv",sep="\t").set_index("Unnamed: 0").squeeze()

poisson_bergmann_mult = (statsmodels.discrete.discrete_model.Poisson
                  .from_formula('RNA_sum ~ log_DNA_input + C(promoter, Treatment("{}")) + C(enhancer, Treatment("{}"))'.format(base_prom,base_enh),
                                data=merged_df_min))

preds = poisson_bergmann_mult.predict(params=bergmann_mult_params)
pred_df = merged_df_min[["promoter","enhancer","RNA_sum"]]
pred_df["pred"] = preds

pred_df.to_csv(base_path_results + "bergmann_multiplicative_poisson_preds_baselined.tsv",sep="\t",index=None)

print(scipy.stats.pearsonr(pred_df["RNA_sum"], pred_df["pred"]))
print(sklearn.metrics.r2_score(pred_df["RNA_sum"], pred_df["pred"]))
print(scipy.stats.pearsonr(np.log2(pred_df["RNA_sum"]+1), np.log2(pred_df["pred"]+1)))
print(sklearn.metrics.r2_score(np.log2(pred_df["RNA_sum"]+1), np.log2(pred_df["pred"]+1)))

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

pred_df3 = pd.read_csv(base_path_results + "bergmann_multiplicative_poisson_preds_baselined.tsv",sep="\t")

p = (p9.ggplot(data=pred_df3,mapping=p9.aes(x="pred",y="RNA_sum"))
 #+ p9.geom_col(position="dodge")
 #+ p9.scale_x_log10()
 #+ p9.scale_y_log10()
 + p9.geom_bin2d(bins=100, raster=True)
 + p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.geom_smooth(method="lm",color="blue")
 + p9.theme(legend_text=p9.element_text(size=8),legend_title=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,axis_title=p9.element_text(size=10))#
 + p9.labs(x="Poisson model prediction",y="RNA")
 #+ p9.scale_y_log10()
 #+ p9.labs(x="",y="Density")
)
p

# %%
p = (p9.ggplot(data=pred_df3,mapping=p9.aes(x="pred",y="RNA_sum"))
 #+ p9.geom_col(position="dodge")
 + p9.scale_x_log10()
 + p9.scale_y_log10()
 + p9.geom_bin2d(bins=100, raster=True)
 + p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.geom_smooth(method="lm",color="blue")
 + p9.theme(legend_text=p9.element_text(size=8),legend_title=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,axis_title=p9.element_text(size=10))#
 + p9.labs(x="Poisson model prediction",y="RNA")
 #+ p9.scale_y_log10()
 #+ p9.labs(x="",y="Density")
)
p

# %%
90.1 - 53.5

# %% [markdown]
# #### Bergmann multiplicative log-linear model

# %%
ols_bergmann_mult = (statsmodels.regression.linear_model.OLS
                  .from_formula('log_RNA_sum ~ log_DNA_input + C(promoter, Treatment("{}")) + C(enhancer, Treatment("{}"))'.format(base_prom,base_enh),
                                data=merged_df_min))

#res_bergmann = poisson_bergmann.fit(disp=True,maxiter=100)
res_bergmann_mult = ols_bergmann_mult.fit()

# %%
res_bergmann_mult.params.to_csv(base_path_results + "bergmann_multiplicative_ols_params_baselined.tsv",sep="\t")

# %%
bergmann_mult_params = pd.read_csv(base_path_results + "bergmann_multiplicative_ols_params_baselined.tsv",sep="\t").set_index("Unnamed: 0").squeeze()

ols_bergmann_mult = (statsmodels.regression.linear_model.OLS
                  .from_formula('log_RNA_sum ~ log_DNA_input + C(promoter, Treatment("{}")) + C(enhancer, Treatment("{}"))'.format(base_prom,base_enh),
                                data=merged_df_min))

preds = ols_bergmann_mult.predict(params=bergmann_mult_params)
pred_df = merged_df_min[["promoter","enhancer","RNA_sum"]]
pred_df["pred"] = preds

pred_df.to_csv(base_path_results + "bergmann_multiplicative_ols_preds_baselined.tsv",sep="\t",index=None)

print(scipy.stats.pearsonr(np.log2(pred_df["RNA_sum"]+1), pred_df["pred"]))
print(sklearn.metrics.r2_score(np.log2(pred_df["RNA_sum"]+1), pred_df["pred"]))

# %% [markdown] tags=[]
# #### Bergmann - enhancer residual model

# %%
residual_df = pd.read_csv(base_path_results + "bergmann_promoter_poisson_preds_baselined.tsv",sep="\t")
residual_df["residual"] = np.log2(residual_df["RNA_sum"]+1) - np.log2(residual_df["pred"]+1)

# %%
ols_bergmann_residual = (statsmodels.regression.linear_model.OLS
                  .from_formula('residual ~ C(enhancer, Treatment("{}"))'.format(base_enh),
                                data=residual_df))

res_bergmann_residual = ols_bergmann_residual.fit()

# %%
res_bergmann_residual.params.to_csv(base_path_results + "bergmann_residual_ols_params_baselined.tsv",sep="\t")

# %%
res_bergmann_residual = pd.read_csv(base_path_results + "bergmann_residual_ols_params_baselined.tsv",sep="\t").set_index("Unnamed: 0").squeeze()

ols_bergmann_residual = (statsmodels.regression.linear_model.OLS
                  .from_formula('residual ~ C(enhancer, Treatment("{}"))'.format(base_enh),
                                data=residual_df))

preds = ols_bergmann_residual.predict(params=res_bergmann_residual)
residual_df["residual_pred"] = preds

residual_df.to_csv(base_path_results + "bergmann_residual_ols_preds_baselined.tsv",sep="\t",index=None)

print(scipy.stats.pearsonr(residual_df["residual"], residual_df["residual_pred"]))
print(sklearn.metrics.r2_score(residual_df["residual"], residual_df["residual_pred"]))

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=residual_df,mapping=p9.aes(x="residual_pred",y="residual"))
 #+ p9.geom_col(position="dodge")
 #+ p9.scale_x_log10()
 #+ p9.scale_y_log10()
 + p9.geom_bin2d(bins=100, raster=True)
 + p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.geom_smooth(method="lm",color="blue")
 + p9.theme(legend_text=p9.element_text(size=8),legend_title=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,axis_title=p9.element_text(size=10))#
 + p9.labs(x="Enhancer fixed effect model",y="Residual of Promoter-only model")
 #+ p9.scale_y_log10()
 #+ p9.labs(x="",y="Density")
)
p

# %% [markdown]
# #### Enformer - promoter only log-linear model

# %%
ols_enformer_prom = (statsmodels.regression.linear_model.OLS
                  .from_formula('{} ~ C(promoter, Treatment("{}"))'.format(log_pred_col,base_prom),
                                data=merged_df_min))

res_enformer_prom = ols_enformer_prom.fit()

# %%
res_enformer_prom.params.to_csv(base_path_results + "enformer_promoter_ols_params_baselined.tsv",sep="\t")

# %%
enformer_prom_params = pd.read_csv(base_path_results + "enformer_promoter_ols_params_baselined.tsv",sep="\t").set_index("Unnamed: 0").squeeze()

ols_enformer_prom = (statsmodels.regression.linear_model.OLS
                  .from_formula('{} ~ C(promoter, Treatment("{}"))'.format(log_pred_col,base_prom),
                                data=merged_df_min))

preds = ols_enformer_prom.predict(params=enformer_prom_params)
pred_df = merged_df_min[["promoter","enhancer",log_pred_col]]
pred_df["pred"] = preds

pred_df.to_csv(base_path_results + "enformer_promoter_ols_preds_baselined.tsv",sep="\t",index=None)

print(scipy.stats.pearsonr(pred_df[log_pred_col], pred_df["pred"]))
print(sklearn.metrics.r2_score(pred_df[log_pred_col], pred_df["pred"]))

# %%
0.957**2

# %% [markdown]
# #### Enformer - multiplicative log-linear model

# %%
ols_enformer_mult = (statsmodels.regression.linear_model.OLS
                  .from_formula('{} ~ C(promoter, Treatment("{}")) + C(enhancer, Treatment("{}"))'.format(log_pred_col,base_prom,base_enh),
                                data=merged_df_min))

res_enformer_mult = ols_enformer_mult.fit()

# %%
res_enformer_mult.params.to_csv(base_path_results + "enformer_multiplicative_ols_params_baselined.tsv",sep="\t")

# %%
enformer_mult_params = pd.read_csv(base_path_results + "enformer_multiplicative_ols_params_baselined.tsv",sep="\t").set_index("Unnamed: 0").squeeze()

ols_enformer_mult = (statsmodels.regression.linear_model.OLS
                  .from_formula('{} ~ C(promoter, Treatment("{}")) + C(enhancer, Treatment("{}"))'.format(log_pred_col,base_prom,base_enh),
                                data=merged_df_min))

preds = ols_enformer_mult.predict(params=enformer_mult_params)
pred_df = merged_df_min[["promoter","enhancer",log_pred_col]]
pred_df["pred"] = preds

pred_df.to_csv(base_path_results + "enformer_multiplicative_ols_preds_baselined.tsv",sep="\t",index=None)

print(scipy.stats.pearsonr(pred_df[log_pred_col], pred_df["pred"]))
print(sklearn.metrics.r2_score(pred_df[log_pred_col], pred_df["pred"]))

# %%
ols_enformer_mult = (statsmodels.regression.linear_model.OLS
                  .from_formula('{} ~ C(promoter, Treatment("{}")) + C(enhancer, Treatment("{}"))'.format(log_pred_col,base_prom,base_enh),
                                data=merged_df_ingenome))

res_enformer_mult = ols_enformer_mult.fit()

res_enformer_mult.params.to_csv(base_path_results + "enformer_multiplicative_ols_params_baselined_aavs1.tsv",sep="\t")

enformer_mult_params = pd.read_csv(base_path_results + "enformer_multiplicative_ols_params_baselined_aavs1.tsv",sep="\t").set_index("Unnamed: 0").squeeze()

ols_enformer_mult = (statsmodels.regression.linear_model.OLS
                  .from_formula('{} ~ C(promoter, Treatment("{}")) + C(enhancer, Treatment("{}"))'.format(log_pred_col,base_prom,base_enh),
                                data=merged_df_ingenome))

preds = ols_enformer_mult.predict(params=enformer_mult_params)
pred_df = merged_df_ingenome[["promoter","enhancer",log_pred_col]]
pred_df["pred"] = preds

pred_df.to_csv(base_path_results + "enformer_multiplicative_ols_preds_baselined_aavs1.tsv",sep="\t",index=None)

print(scipy.stats.pearsonr(pred_df[log_pred_col], pred_df["pred"]))
print(sklearn.metrics.r2_score(pred_df[log_pred_col], pred_df["pred"]))

# %% [markdown]
# #### Enformer - Gamma promoter-only model

# %%
gamma_enformer_prom = smf.glm(formula='{} ~ C(promoter, Treatment("{}"))'.format(pred_col.replace(":","_").replace(" ","_").replace(",","_"), base_prom), 
                       data=merged_df_min.rename(columns={pred_col:pred_col.replace(":","_").replace(" ","_").replace(",","_")}), 
                       family=sm.families.Gamma(link=sm.families.links.Log()))


res_enformer_prom = gamma_enformer_prom.fit(method="minimize",min_method='dogleg',maxiter=100, max_start_irls=0)
# IRLS seems to take insane amounts of memory here

# %%
res_enformer_prom.params.to_csv(base_path_results + "enformer_promoter_gamma_params_baselined.tsv",sep="\t")

# %%
enformer_prom_params = pd.read_csv(base_path_results + "enformer_promoter_gamma_params_baselined.tsv",sep="\t").set_index("Unnamed: 0").squeeze()


gamma_enformer_prom = smf.glm(formula='{} ~ C(promoter, Treatment("{}"))'.format(pred_col.replace(":","_").replace(" ","_").replace(",","_"), base_prom), 
                       data=merged_df_min.rename(columns={pred_col:pred_col.replace(":","_").replace(" ","_").replace(",","_")}), 
                       family=sm.families.Gamma(link=sm.families.links.Log()))

# %%
preds = gamma_enformer_prom.predict(params=enformer_prom_params)
pred_df = merged_df_min[["promoter","enhancer",pred_col,log_pred_col]]
pred_df["pred"] = preds

pred_df.to_csv(base_path_results + "enformer_promoter_gamma_preds_baselined.tsv",sep="\t",index=None)

print(scipy.stats.pearsonr(pred_df[pred_col], pred_df["pred"]))
print(sklearn.metrics.r2_score(pred_df[pred_col], pred_df["pred"]))
print(scipy.stats.pearsonr(np.log2(pred_df[pred_col]+1), np.log2(pred_df["pred"]+1)))
print(sklearn.metrics.r2_score(np.log2(pred_df[pred_col]+1), np.log2(pred_df["pred"]+1)))

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=pred_df,mapping=p9.aes(x="pred",y=pred_col,))
 #+ p9.geom_col(position="dodge")
 + p9.scale_x_log10()
 + p9.scale_y_log10()
 + p9.geom_bin2d(bins=100, raster=True)
 + p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.geom_smooth(method="lm",color="blue")
 + p9.theme(legend_text=p9.element_text(size=8),legend_title=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,axis_title=p9.element_text(size=10))#
 + p9.labs(x="Gamma model prediction",y="Enformer Prediction")
 #+ p9.scale_y_log10()
 #+ p9.labs(x="",y="Density")
)
p

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=pred_df,mapping=p9.aes(x="pred",y=pred_col))
 #+ p9.geom_col(position="dodge")
 #+ p9.scale_x_log10()
 #+ p9.scale_y_log10()
 + p9.geom_bin2d(bins=100, raster=True)
 + p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.geom_smooth(method="lm",color="blue")
 + p9.theme(legend_text=p9.element_text(size=8),legend_title=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,axis_title=p9.element_text(size=10))#
 + p9.labs(x="Gamma model prediction",y="Enformer Prediction")
 #+ p9.scale_y_log10()
 #+ p9.labs(x="",y="Density")
)
p

# %% [markdown]
# #### Enformer - Gamma multiplicative model

# %%
gamma_enformer_mult = smf.glm(formula='{} ~ C(promoter, Treatment("{}")) + C(enhancer, Treatment("{}"))'.format(pred_col.replace(":","_").replace(" ","_").replace(",","_"), base_prom, base_enh), 
                       data=merged_df_min.rename(columns={pred_col:pred_col.replace(":","_").replace(" ","_").replace(",","_")}), 
                       family=sm.families.Gamma(link=sm.families.links.Log()))


res_enformer_mult = gamma_enformer_mult.fit(method="minimize",min_method='dogleg',maxiter=100, max_start_irls=0)
# IRLS seems to take insane amounts of memory here

# %%
res_enformer_mult.params.to_csv(base_path_results + "enformer_multiplicative_gamma_params_baselined.tsv",sep="\t")

# %%
enformer_mult_params = pd.read_csv(base_path_results + "enformer_multiplicative_gamma_params_baselined.tsv",sep="\t").set_index("Unnamed: 0").squeeze()


gamma_enformer_mult = smf.glm(formula='{} ~ C(promoter, Treatment("{}")) + C(enhancer, Treatment("{}"))'.format(pred_col.replace(":","_").replace(" ","_").replace(",","_"), base_prom, base_enh), 
                       data=merged_df_min.rename(columns={pred_col:pred_col.replace(":","_").replace(" ","_").replace(",","_")}), 
                       family=sm.families.Gamma(link=sm.families.links.Log()))

# %%
preds = gamma_enformer_mult.predict(params=enformer_mult_params)
pred_df = merged_df_min[["promoter","enhancer",pred_col,log_pred_col]]
pred_df["pred"] = preds

pred_df.to_csv(base_path_results + "enformer_multiplicative_gamma_preds_baselined.tsv",sep="\t",index=None)

print(scipy.stats.pearsonr(pred_df[pred_col], pred_df["pred"]))
print(sklearn.metrics.r2_score(pred_df[pred_col], pred_df["pred"]))
print(scipy.stats.pearsonr(np.log2(pred_df[pred_col]+1), np.log2(pred_df["pred"]+1)))
print(sklearn.metrics.r2_score(np.log2(pred_df[pred_col]+1), np.log2(pred_df["pred"]+1)))

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=pred_df,mapping=p9.aes(x="pred",y=pred_col,))
 #+ p9.geom_col(position="dodge")
 + p9.scale_x_log10()
 + p9.scale_y_log10()
 + p9.geom_bin2d(bins=100, raster=True)
 + p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.geom_smooth(method="lm",color="blue")
 + p9.theme(legend_text=p9.element_text(size=8),legend_title=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,axis_title=p9.element_text(size=10))#
 + p9.labs(x="Gamma model prediction",y="Enformer Prediction")
 #+ p9.scale_y_log10()
 #+ p9.labs(x="",y="Density")
)
p

# %% [markdown]
# #### Enformer - enhancer residual model

# %%
residual_df = pd.read_csv(base_path_results + "enformer_promoter_ols_preds_baselined.tsv",sep="\t")
residual_df["residual"] = residual_df[log_pred_col] - residual_df["pred"]

# %%
ols_enformer_residual = (statsmodels.regression.linear_model.OLS
                  .from_formula('residual ~ C(enhancer, Treatment("{}"))'.format(base_enh),
                                data=residual_df))

res_enformer_residual = ols_enformer_residual.fit()

# %%
res_enformer_residual.params.to_csv(base_path_results + "enformer_residual_ols_params_baselined.tsv",sep="\t")

# %%
res_enformer_residual = pd.read_csv(base_path_results + "enformer_residual_ols_params_baselined.tsv",sep="\t").set_index("Unnamed: 0").squeeze()

ols_enformer_residual = (statsmodels.regression.linear_model.OLS
                  .from_formula('residual ~ C(enhancer, Treatment("{}"))'.format(base_enh),
                                data=residual_df))

preds = ols_enformer_residual.predict(params=res_enformer_residual)
residual_df["residual_pred"] = preds

residual_df.to_csv(base_path_results + "enformer_residual_ols_preds_baselined.tsv",sep="\t",index=None)

print(scipy.stats.pearsonr(residual_df["residual"], residual_df["residual_pred"]))
print(sklearn.metrics.r2_score(residual_df["residual"], residual_df["residual_pred"]))

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=residual_df,mapping=p9.aes(x="residual_pred",y="residual"))
 #+ p9.geom_col(position="dodge")
 #+ p9.scale_x_log10()
 #+ p9.scale_y_log10()
 + p9.geom_bin2d(bins=100, raster=True)
 + p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.geom_smooth(method="lm",color="blue")
 + p9.theme(legend_text=p9.element_text(size=8),legend_title=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,axis_title=p9.element_text(size=10))#
 + p9.labs(x="Enhancer fixed effect model",y="Residual of Promoter-only model")
 #+ p9.scale_y_log10()
 #+ p9.labs(x="",y="Density")
)
p

# %% [markdown]
# ### Compare variance explained

# %%
bergmann_prom_df = pd.read_csv(base_path_results + "bergmann_promoter_poisson_preds_baselined.tsv",sep="\t").rename(columns={"RNA_sum":"obs"})
bergmann_prom_df["pred"] = np.log2(bergmann_prom_df["pred"] + 1)
bergmann_prom_df["obs"] = np.log2(bergmann_prom_df["obs"] + 1)
bergmann_mult_df = pd.read_csv(base_path_results + "bergmann_multiplicative_poisson_preds_baselined.tsv",sep="\t").rename(columns={"RNA_sum":"obs"})
bergmann_mult_df["pred"] = np.log2(bergmann_mult_df["pred"] + 1)
bergmann_mult_df["obs"] = np.log2(bergmann_mult_df["obs"] + 1)
enformer_prom_df = pd.read_csv(base_path_results + "enformer_promoter_gamma_preds_baselined.tsv",sep="\t").rename(columns={pred_col:"obs"})
enformer_prom_df["pred"] = np.log2(enformer_prom_df["pred"] + 1)
enformer_prom_df["obs"] = np.log2(enformer_prom_df["obs"] + 1)
enformer_mult_df = pd.read_csv(base_path_results + "enformer_multiplicative_gamma_preds_baselined.tsv",sep="\t").rename(columns={pred_col:"obs"})
enformer_mult_df["pred"] = np.log2(enformer_mult_df["pred"] + 1)
enformer_mult_df["obs"] = np.log2(enformer_mult_df["obs"] + 1)

var_explain_list = [("Observed","Promoter",bergmann_prom_df),
                  ("Observed","Promoter\n+ Enhancer",bergmann_mult_df),
                  ("Predicted","Promoter",enformer_prom_df),
                  ("Predicted","Promoter\n+ Enhancer",enformer_mult_df)]

# %%
rows = []
for pred_type, reg_type, pred_df in var_explain_list:
    r2 = sklearn.metrics.r2_score(pred_df["obs"], pred_df["pred"])
    rows.append({
        "pred_type":pred_type,
        "reg_type":reg_type,
        "r2":r2
    })
var_explain_df = pd.DataFrame(rows)

# %%
var_explain_df

# %%
scale = 1.0
p9.options.figure_size = (6.4*scale, 4.8*scale)
p = (p9.ggplot(data=var_explain_df,mapping=p9.aes(x="reg_type",y="r2",fill="pred_type"))
 + p9.geom_bar(stat="identity", position="dodge")
 + p9.labs(x="Regulatory Element", y="Variance Explained", fill="", title="Bergman")
 + p9.theme(legend_box_margin=0,legend_key_size =9, legend_text = p9.element_text(size=9),
            title=p9.element_text(size=10),
            legend_background=p9.element_blank(),
            #legend_position=(0.25,0.8),
           axis_title=p9.element_text(size=10))
     #axis_text_x=p9.element_text(rotation=45, hjust=1))    
)
p

# %%
p.save("Graphics/" + "enhancer_varexpl" + ".svg", width=3.0, height=2.6, dpi=300)

# %% [markdown]
# ### Compare Model coefficients

# %% [markdown]
# #### Promoter only models

# %%
# Promoter only models

bergmann_prom_params = (pd.read_csv(base_path_results + "bergmann_promoter_poisson_params_baselined.tsv",sep="\t")
                        .rename(columns={"Unnamed: 0":"Name","0":"Bergmann"}))
enformer_prom_params = (pd.read_csv(base_path_results + "enformer_promoter_gamma_params_baselined.tsv",sep="\t")
                        .rename(columns={"Unnamed: 0":"Name","0":"Enformer"}))

parameters_prom = enformer_prom_params.merge(bergmann_prom_params, on="Name")
parameters_prom["Name"] = (parameters_prom["Name"]
                          .str.replace('C(promoter, Treatment("{}"))'.format(base_prom),"promoter",regex=False))

print(scipy.stats.pearsonr(parameters_prom.loc[parameters_prom["Name"].str.startswith("promoter")]["Bergmann"],
                           parameters_prom.loc[parameters_prom["Name"].str.startswith("promoter")]["Enformer"]))
print(scipy.stats.spearmanr(parameters_prom.loc[parameters_prom["Name"].str.startswith("promoter")]["Bergmann"],
                           parameters_prom.loc[parameters_prom["Name"].str.startswith("promoter")]["Enformer"]))

# %%
scale = 0.5
p9.options.figure_size = (6.4*scale, 4.8*scale)
p = (p9.ggplot(data=parameters_prom.loc[parameters_prom["Name"].str.startswith("promoter")],mapping=p9.aes(x='Enformer',y='Bergmann'))
 + p9.geom_point(alpha=0.25, size=0.5)
 #+ p9.geom_bin2d(binwidth = (0.025, 0.1))
 + p9.geom_smooth(method="lm",color="blue")
 + p9.labs(x="Predicted Promoter\nStrength (a.u.)", y="Measured Promoter\nStrength (a.u.)")
 + p9.theme(axis_title=p9.element_text(size=10))
)
p

# %% [markdown]
# #### Log-linear OLS models

# %%
# compare ols params
bergmann_ols_params = (pd.read_csv(base_path_results + "bergmann_multiplicative_ols_params_baselined.tsv",sep="\t")
                        .rename(columns={"Unnamed: 0":"Name","0":"Bergmann"}))
enformer_ols_params = (pd.read_csv(base_path_results + "enformer_multiplicative_ols_params_baselined.tsv",sep="\t")
                        .rename(columns={"Unnamed: 0":"Name","0":"Enformer"}))

parameters_ols = enformer_ols_params.merge(bergmann_ols_params, on="Name")
parameters_ols["Name"] = (parameters_ols["Name"]
                          .str.replace('C(promoter, Treatment("{}"))'.format(base_prom),"promoter",regex=False)
                          .str.replace('C(enhancer, Treatment("{}"))'.format(base_enh),"enhancer",regex=False))

print(scipy.stats.pearsonr(parameters_ols.loc[parameters_ols["Name"].str.startswith("promoter")]["Bergmann"],
                     parameters_ols.loc[parameters_ols["Name"].str.startswith("promoter")]["Enformer"]))
print(scipy.stats.spearmanr(parameters_ols.loc[parameters_ols["Name"].str.startswith("promoter")]["Bergmann"],
                     parameters_ols.loc[parameters_ols["Name"].str.startswith("promoter")]["Enformer"]))

print(scipy.stats.pearsonr(parameters_ols.loc[parameters_ols["Name"].str.startswith("enhancer")]["Bergmann"],
                     parameters_ols.loc[parameters_ols["Name"].str.startswith("enhancer")]["Enformer"]))
print(scipy.stats.spearmanr(parameters_ols.loc[parameters_ols["Name"].str.startswith("enhancer")]["Bergmann"],
                     parameters_ols.loc[parameters_ols["Name"].str.startswith("enhancer")]["Enformer"]))

# %%
scale = 0.5
p9.options.figure_size = (6.4*scale, 4.8*scale)
p = (p9.ggplot(data=parameters_ols.loc[parameters_ols["Name"].str.startswith("promoter")],mapping=p9.aes(x='Enformer',y='Bergmann'))
 + p9.geom_point(alpha=0.25, size=0.5)
 #+ p9.geom_bin2d(binwidth = (0.025, 0.1))
 + p9.geom_smooth(method="lm",color="blue")
 + p9.labs(x="Predicted Promoter\nStrength (a.u.)", y="Measured Promoter\nStrength (a.u.)")
 + p9.theme(axis_title=p9.element_text(size=10))
)
p

# %% [markdown]
# #### Poisson/Gamma models

# %%
bergmann_mult_params = (pd.read_csv(base_path_results + "bergmann_multiplicative_poisson_params_baselined.tsv",sep="\t")
                        .rename(columns={"Unnamed: 0":"Name","0":"Bergmann"}))
#bergmann_mult_params["Name"] = bergmann_mult_params["Name"].str.replace('C(promoter, Treatment("{}"))'.format(base_prom),"promoter",regex=False).str.replace('C(enhancer, Treatment("{}"))'.format(base_enh),"enhancer",regex=False)
enformer_mult_params = (pd.read_csv(base_path_results + "enformer_multiplicative_gamma_params_baselined.tsv",sep="\t")
                        .rename(columns={"Unnamed: 0":"Name","0":"Enformer"}))

parameters = enformer_mult_params.merge(bergmann_mult_params, on="Name")
parameters["Name"] = parameters["Name"].str.replace('C(promoter, Treatment("{}"))'.format(base_prom),"promoter",regex=False).str.replace('C(enhancer, Treatment("{}"))'.format(base_enh),"enhancer",regex=False)

# %%
print(scipy.stats.pearsonr(parameters.loc[parameters["Name"].str.startswith("promoter")]["Bergmann"],
                     parameters.loc[parameters["Name"].str.startswith("promoter")]["Enformer"]))
print(scipy.stats.spearmanr(parameters.loc[parameters["Name"].str.startswith("promoter")]["Bergmann"],
                     parameters.loc[parameters["Name"].str.startswith("promoter")]["Enformer"]))

# %%
scale = 0.5
p9.options.figure_size = (6.4*scale, 4.8*scale)
p = (p9.ggplot(data=parameters.loc[parameters["Name"].str.startswith("promoter")],mapping=p9.aes(x='Enformer',y='Bergmann'))
 + p9.geom_point(alpha=0.25, size=0.5, raster=True)
 #+ p9.geom_bin2d(binwidth = (0.025, 0.1))
 + p9.geom_smooth(method="lm",color="blue")
 + p9.labs(x="Predicted Promoter\nStrength (a.u.)", y="Measured Promoter\nStrength (a.u.)")
 + p9.theme(axis_title=p9.element_text(size=10))
)
p

# %%
p.save("Graphics/" + "xsup_bergmann_promoter" + ".svg", width=2.6, height=2.6, dpi=300)

# %%
print(scipy.stats.pearsonr(parameters.loc[parameters["Name"].str.startswith("enhancer")]["Bergmann"],
                     parameters.loc[parameters["Name"].str.startswith("enhancer")]["Enformer"]))
print(scipy.stats.spearmanr(parameters.loc[parameters["Name"].str.startswith("enhancer")]["Bergmann"],
                     parameters.loc[parameters["Name"].str.startswith("enhancer")]["Enformer"]))

# %%
scale = 1.0
p9.options.figure_size = (6.4*scale, 4.8*scale)
p = (p9.ggplot(data=parameters.loc[parameters["Name"].str.startswith("enhancer")],mapping=p9.aes(x='Enformer',y='Bergmann'))
 + p9.geom_point(alpha=0.25, size=0.5, raster=True)
 #+ p9.geom_bin2d(binwidth = (0.025, 0.1))
 + p9.geom_smooth(method="lm",color="blue")
 + p9.labs(x="Predicted Enhancer\nStrength (a.u.)", y="Measured Enhancer\nStrength (a.u.)")
 + p9.theme(axis_title=p9.element_text(size=10))
)
p

# %%
p.save("Graphics/" + "xsup_bergmann_enhancer" + ".svg", width=2.6, height=2.6, dpi=300)

# %% [markdown]
# ### How much variation do enhancers induce per promoter?

# %%
promoter_param = parameters.loc[parameters.Name.str.startswith('prom')]
promoter_param["promoter"] = promoter_param["Name"].apply(lambda x: x.split("[")[1].split("]")[0].split('.')[1])
enhancer_param = parameters.loc[parameters.Name.str.startswith('enh')]
enhancer_param["enhancer"] = enhancer_param["Name"].apply(lambda x: x.split("[")[1].split("]")[0].split('.')[1])

# %%
rows = []
for promoter in set(merged_df_min["promoter"]):
    subset = merged_df_min.query('promoter == @promoter')
    subset["RNA/DNA"] = subset["RNA_sum"]/subset["DNA_input"]
    if len(subset) < 2:
        continue
    r = scipy.stats.pearsonr(np.log2(subset[pred_col]+1),np.log2(subset["RNA/DNA"]))
    rows.append({
        "promoter":promoter,
        "class":subset["promoter_exp_class"].iloc[0],
        "n":len(subset),
        "avg":subset[pred_col].mean(),
        "max":subset[pred_col].max(),
        "min":subset[pred_col].min(),
        "std":subset[pred_col].std(),
        "cv":subset[pred_col].std()/subset[pred_col].mean(),
        "log2_avg":np.log2(subset[pred_col]+1).mean(),
        "log2_fc":np.log2(subset[pred_col]+1).max() - np.log2(subset[pred_col]+1).min(),
        "std_log2":np.log2(subset[pred_col]+1).std(),
        "avg_obs":subset["RNA/DNA"].mean(),
        "max_obs":subset["RNA/DNA"].max(),
        "min_obs":subset["RNA/DNA"].min(),
        "std_obs":subset["RNA/DNA"].std(),
        "cv_obs":subset["RNA/DNA"].std()/subset["RNA/DNA"].mean(),
        "log2_avg_obs":np.log2(subset["RNA/DNA"]).mean(),
        "log2_fc_obs":np.log2(subset["RNA/DNA"]).max() - np.log2(subset[pred_col]).min(),
        "std_log2_obs":np.log2(subset["RNA/DNA"]).std(),
        "r":r[0],
        "p":r[1]
    }) 
prom_var = pd.DataFrame(rows)

# %%
prom_var_filtered = prom_var.query('n > 500').sort_values('log2_fc').merge(promoter_param,on="promoter")
prom_var_filtered["class"] = pd.Categorical(prom_var_filtered["class"],[0,1,2])

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

(p9.ggplot(data=prom_var_filtered, 
            mapping=p9.aes(x="Enformer",y="std_log2",color="class"))
 + p9.geom_point()
 + p9.geom_smooth(color="black")
 + p9.labs(x="Predicted Promoter Strength (a.u.)",y="Standard deviation of predicted log2 Expression\n(over enhancers)")
 #+ p9.geom_smooth(method="lm", color="blue")
)

# %%
(p9.ggplot(data=prom_var_filtered, 
            mapping=p9.aes(x="Enformer",y="std_log2_obs",color="class"))
 + p9.geom_point()
 + p9.coord_cartesian(ylim=(0,2.5))
 + p9.labs(x="Predicted Promoter Strength (a.u.)",y="Standard deviation of observed log2 Expression\n(over enhancers)")
 #+ p9.geom_smooth(method="lm", color="blue")
)

# %%
(p9.ggplot(data=prom_var_filtered, 
            mapping=p9.aes(x="Bergmann",y="std_log2_obs",color="class"))
 + p9.geom_point()
 + p9.coord_cartesian(ylim=(0,2.5))
 + p9.labs(x="Observed Promoter Strength (a.u.)",y="Standard deviation of observed log2 Expression\n(over enhancers)")
 #+ p9.geom_smooth(method="lm", color="blue")
)

# %%
#strengths = prom_var_filtered[["promoter","Enformer","Bergmann"]].melt(id_vars="promoter",var_name="pred_obs",value_name="prom_strength")
#strengths["pred_obs"] = strengths["pred_obs"].apply(lambda x: "Predicted" if x == "Enformer" else "Observed")
strengths = prom_var_filtered[["promoter","Enformer"]].rename(columns={"Enformer":"prom_strength"})
stdevs = prom_var_filtered[["promoter","std_log2_obs","std_log2"]].melt(id_vars="promoter",var_name="pred_obs",value_name="std")
stdevs["pred_obs"] = stdevs["pred_obs"].apply(lambda x: "Predicted" if x == "std_log2" else "Observed")
plot_enh_std = strengths.merge(stdevs,on=["promoter"])

# %%
p = (p9.ggplot(data=plot_enh_std, 
            mapping=p9.aes(x="prom_strength",y="std", color="pred_obs"))
 + p9.geom_point(size=0.5)
 #+ p9.coord_cartesian(ylim=(0,2.5))
 #+ p9.facet_wrap('~pred_obs')
 + p9.geom_smooth(method="lowess")
 + p9.labs(x="Predicted Promoter Strength (a.u.)",y="Standard deviation of $\mathregular{log_2}$\nExpression (over enhancers)")
 #+ p9.geom_smooth(method="lm", color="blue")
 + p9.theme(legend_position='none',
           axis_title=p9.element_text(size=10))
)
p

# %%
p.save("Graphics/" + "enhancer_std" + ".svg", width=2.5, height=2.6, dpi=300)

# %% [markdown]
# # Enhancer Knockdown (Fulco et al. & Gasperini et al.)
#
# In these CRISPRi studies, enhancers were knocked down using CRISPR interference, to measure the impact on expression of target genes. We replicate this by shuffling enhancer sequences.

# %%
base_path_data = "Data/Fulco_CRISPRi/"
base_path_results = "Results/Fulco_CRISPRi/"
base_path_data_gtex = "Data/GTEX/"

# %%
# liftover gasperini
gasperini_df = pd.read_csv(base_path_data + "gasperini.tsv",sep="\t")

#((gasperini_df["chr.candidate_enhancer"]+":"+gasperini_df["start.candidate_enhancer"].astype('str')+"-"+gasperini_df["stop.candidate_enhancer"].astype('str'))
#.to_csv(base_path_data + "gasperini_liftover_input.tsv",index=None))

#gasperini_lift
gasperini_df["liftover_coord"] = pd.read_csv(base_path_data + "hglft_genome_2c397_7c34e0.bed",header=None)

# %%
gasperini_df["ziga_key"] = "gasperini2019_" + gasperini_df["target_gene_short"] + "_" + gasperini_df["liftover_coord"].apply(lambda x: x.split(':')[1])

# %%
gasperini_df["Absolute Pct Change"] = np.abs((1-gasperini_df["Diff_expression_test_fold_change"])*100)  #Proportion remaining transcript of target gene (performed as described in “Method Details: Digital gene expression quantification - Differential expression tests”). 

# %%
fulco_df = pd.read_csv(base_path_data + "enhancer_knockdown_effects.tsv",sep="\t")

# %% [markdown]
# ## Analysis

# %% [markdown]
# ### Get data

# %% [markdown]
# #### Knockdown analysis

# %%
missing_genes = {"ADSS2":'ADSS',
                 "ADSS1":'ADSSL1',
                 "ZFTA":'C11orf95',
                 "RUSF1":'C16orf58',
                 "HROB":'C17orf53',
                 "BRME1":'C19orf57',
                 "MIR9-1HG":'C1orf61',
                 "DNAAF9":'C20orf194',
                 "BBLN":'C9orf16',
                 "CARS1":'CARS',
                 "YJU2B":'CCDC130',
                 "MIX23":'CCDC58',
                 "UVSSA":'CRIPAK', #same strand
                 "EOLA2":'CXorf40B',
                 "PABIR1":'FAM122A',
                 "NALF1":'FAM155A',
                 "SLX9":'FAM207A', #same strand
                 "CYRIA":'FAM49A',
                 "CYRIB":'FAM49B',
                 "CEP43":'FGFR1OP',
                 "CEP20":'FOPNL',
                 "H1-10":'H1FX',
                 "H2AZ2":'H2AFV',
                 "MACROH2A1":'H2AFY',
                 "H3-3A":'H3F3A',
                 "HARS1":'HARS',
                 "H1-5":'HIST1H1B',
                 "H1-2":'HIST1H1C',
                 "H1-3":'HIST1H1D',
                 "H1-4":'HIST1H1E',
                 "H2AC11":'HIST1H2AG',
                 "H2AC13":'HIST1H2AI',
                 "H2BC4":'HIST1H2BC',
                 "H2BC5":'HIST1H2BD',
                 "H2BC7":'HIST1H2BF',
                 "H2BC8":'HIST1H2BG',
                 "H2BC11":'HIST1H2BJ',
                 "H2BC12":'HIST1H2BK',
                 "H2BC13":'HIST1H2BL',
                 "H2BC15":'HIST1H2BN',
                 "H3C4":'HIST1H3D',
                 "H4C2":'HIST1H4B',
                 "H4C3":'HIST1H4C',
                 "H4C8":'HIST1H4H',
                 "H4C11":'HIST1H4J',
                 "H2AW":'HIST3H2A',
                 "IARS1":'IARS',
                 "GARRE1":'KIAA0355',
                 "ELAPOR1":'KIAA1324',
                 "ELAPOR2":'KIAA1324L',
                 "MARCHF6":'MARCH6',
                 "MARCHF9":'MARCH9',
                 "PPP5D1P":'PPP5D1',
                 "QARS1":'QARS',
                 "METTL25B":'RRNAD1',
                 "TARS3":'TARSL2',
                 "DYNLT2B":'TCTEX1D2',
                 "STING1":'TMEM173',
                 "PEDS1":'TMEM189',
                 "DYNC2I1":'WDR60',
                 "DNAAF10":'WDR92',
                 "POLR1H":'ZNRD1'}

# %%
results = pd.read_csv(base_path_results + "avsec_fulltable_fixed-enformer-latest_results.tsv",sep="\t")

# %%
set(results["enhancer_wide_type"])

# %%
results = results.groupby(['ziga_key','fulco_key','gene','enhancer_modification'])[[k for k in results.keys() if ("CAGE" in k) or ("DNASE" in k)]].mean().reset_index()

# %% tags=[]
set(results["enhancer_modification"])

# %%
ziga_df = pd.read_csv(base_path_data + "ziga_additional_columns.tsv",sep="\t")

ziga_df["signed_tss_distance"] = (ziga_df['fix_enhancer_wide_start'] + ziga_df['fix_enhancer_wide_end'])/2 - (ziga_df['main_tss_start'] + ziga_df['main_tss_end'])/2

gene_locs = gtf_df.df.query('Feature == "gene"')[["Chromosome","Start","End","Score","Strand","Frame","gene_id","gene_name"]]
gene_locs["gene"] = gene_locs["gene_name"].apply(lambda x: missing_genes[x] if x in missing_genes else x) 

ziga_df = (ziga_df.merge(gene_locs[["Strand","gene"]].drop_duplicates(),on="gene"))

# %%
assert(len(ziga_df) == 8053)

# %%
ziga_df["Strand"] = ziga_df["Strand"].apply(lambda x: 1 if x=="+" else -1) 
# enhancer is upstream if sign of distance is opposite of strand
ziga_df["sign_of_dist"] = np.sign(ziga_df["signed_tss_distance"])
ziga_df["upstream"] = (ziga_df["sign_of_dist"] != ziga_df["Strand"])

# %%
ziga_df.keys()

# %%
baseline = results.query('enhancer_modification == "none"').drop(columns=["enhancer_modification"])
ko = results.query('enhancer_modification == "shuffle"').drop(columns=["enhancer_modification"])
merged_df = baseline.merge(ko, on=['ziga_key','fulco_key','gene'], suffixes=("_ref","_alt"))
merged_df[[k for k in merged_df.keys() if ("CAGE" in k) or ("DNASE" in k)]] = np.log2(merged_df[[k for k in merged_df.keys() if ("CAGE" in k) or ("DNASE" in k)]]+1)
merged_df = merged_df.merge(ziga_df, on=["ziga_key",'fulco_key','gene'])

pred_col = "CAGE:chronic myelogenous leukemia cell line:K562 ENCODE, biol__landmark_sum"
merged_df['log2_fc_pred'] = merged_df[pred_col+"_alt"] - merged_df[pred_col+"_ref"]
merged_df['abs_delta'] = np.abs(2**(merged_df[pred_col+"_alt"]) - 2**(merged_df[pred_col+"_ref"]) - 2)

# %%
ko_n = results.query('enhancer_modification == "replace_with_n"').drop(columns=["enhancer_modification"])
merged_df_n = baseline.merge(ko_n, on=['ziga_key','fulco_key','gene'], suffixes=("_ref","_alt"))
merged_df_n[[k for k in merged_df_n.keys() if ("CAGE" in k) or ("DNASE" in k)]] = np.log2(merged_df_n[[k for k in merged_df_n.keys() if ("CAGE" in k) or ("DNASE" in k)]]+1)
merged_df_n = merged_df_n.merge(ziga_df, on=["ziga_key",'fulco_key','gene'])

merged_df_n['log2_fc_pred'] = merged_df_n[pred_col+"_alt"] - merged_df_n[pred_col+"_ref"]

# %%
scipy.stats.pearsonr(merged_df['log2_fc_pred'],merged_df_n['log2_fc_pred'])

# %% [markdown]
# #### Local results

# %%
local_results = pd.read_csv(base_path_results + "ful_gas_localeffects-enformer-latest_results.tsv", sep="\t")
local_results = local_results.groupby(["id","chr","wide_start","wide_end","start","end","type","seqtype","insert_seq"])[[x for x in local_results.keys() if x.endswith("landmark_sum")]].mean().reset_index()
local_results = local_results.query('seqtype == "aavs1"')

# %%
local_results_prom = local_results.query('type == "promoter"')
local_results_enh = local_results.query('type == "enhancer"')
local_merged = local_results_prom.merge(local_results_enh,on="chr",suffixes=("_prom","_enh"))
local_merged["id"] = local_merged["id_enh"] + "_" + local_merged["id_prom"]

# %% [markdown]
# ### Overlaps

# %%
enhancer_pr = pr.PyRanges(merged_df.rename(columns={"fix_enhancer_wide_start":"Start",
                          "fix_enhancer_wide_end":"End",
                         "chromosome":"Chromosome"})
                         [["Chromosome","Start","End","ziga_key","actual_tss_distance","log2_fc_pred", "main_tss_start", "validated"]])

# %%
overlapped = enhancer_pr.join(enhancer_pr).df.query('ziga_key != ziga_key_b & main_tss_start == main_tss_start_b & validated < validated_b').sort_values('log2_fc_pred_b')

# %%
overlapped_regions = set(overlapped["ziga_key"])

# %% [markdown]
# ### Noise analysis

# %%
noise_df = merged_df.copy()
noise_df['log_tss_distance'] =  np.log10(noise_df['actual_tss_distance'])
noise_df['abs_pct_change'] = np.abs((2**noise_df['log2_fc_pred'] - 1)*100)
noise_df["mod"] = "shuffle"

noise_df.groupby(['validated'])['abs_pct_change'].median()

# %%
noise_df_n = merged_df_n.copy()
noise_df_n['log_tss_distance'] =  np.log10(noise_df_n['actual_tss_distance'])
noise_df_n['abs_pct_change'] = np.abs((2**noise_df_n['log2_fc_pred'] - 1)*100)
noise_df_n["mod"] = "n"

noise_df_n.groupby(['validated'])['abs_pct_change'].median()

# %%
combined_noise = pd.concat([noise_df[["actual_tss_distance",'abs_pct_change','mod','validated']],
                           noise_df_n[["actual_tss_distance",'abs_pct_change','mod','validated']]])

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=combined_noise.query('validated & actual_tss_distance < 100_000 & actual_tss_distance > 980')
           ,mapping=p9.aes(x="actual_tss_distance",y="abs_pct_change",fill="mod", color="mod"))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset')
 + p9.geom_point(size=0.5)
 + p9.labs(x="Distance (kb) between TSS and Enhancer",y="Change in Expression (%)\ndue to Enhancer Knockdown", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10)
           )
)
p

# %% [markdown]
# ### Compute Average Precision

# %% [markdown]
# #### Fulco

# %%
bins = [0, 3000, 12500, 34500]  + [float('inf')]
#labels = list(range(1,max_bin+1))
labels = ["0-3","3-12.5","12.5-34.5",'34.5-131']

merged_df['bin'] = pd.cut(merged_df['actual_tss_distance'], bins=bins, right=True, labels=labels, retbins=False, precision=3, include_lowest=False, duplicates='raise', ordered=True)


for pred_bin in labels:
    print(pred_bin)
    subset = merged_df.query('bin == @pred_bin & dataset_name == "fulco2019"')
    print("Enformer: {}".format(sklearn.metrics.average_precision_score(subset['validated'], np.abs(subset["log2_fc_pred"]))))
    print("Enformer Absolute: {}".format(sklearn.metrics.average_precision_score(subset['validated'], np.abs(subset["abs_delta"]))))
    print("ABC: {}".format(sklearn.metrics.average_precision_score(subset['validated'], subset['H3K27ac/abs(distance)'])))
    print("Random: {}".format(np.sum(subset['validated'])/len(subset['validated'])))
    print(np.sum(subset['validated']))
    print(len(subset['validated']) - np.sum(subset['validated']))



# %%
bins = [0, 3000, 12500, 34500]  + [float('inf')]
#labels = list(range(1,max_bin+1))
labels = ["0-3","3-12.5","12.5-34.5",'34.5-131']

merged_df['bin_ziga'] = pd.cut(np.abs(merged_df['relative_tss_distance']), bins=bins, right=True, labels=labels, retbins=False, precision=3, include_lowest=False, duplicates='raise', ordered=True)


for pred_bin in labels:
    print(pred_bin)
    subset = merged_df.query('bin_ziga == @pred_bin & dataset_name == "fulco2019"')
    print("Enformer: {}".format(sklearn.metrics.average_precision_score(subset['validated'], np.abs(subset["log2_fc_pred"]))))
    print("Enformer Absolute: {}".format(sklearn.metrics.average_precision_score(subset['validated'], np.abs(subset["abs_delta"]))))
    print("ABC: {}".format(sklearn.metrics.average_precision_score(subset['validated'], subset['H3K27ac/abs(distance)'])))
    print("Random: {}".format(np.sum(subset['validated'])/len(subset['validated'])))
    print(np.sum(subset['validated'])/len(subset['validated']))
    print(np.sum(subset['validated']))
    print(len(subset['validated']) - np.sum(subset['validated']))

# %% [markdown]
# #### Gasperini

# %%
for pred_bin in labels:
    print(pred_bin)
    subset = merged_df.query('bin == @pred_bin & dataset_name == "gasperini2019"')
    print("Enformer: {}".format(sklearn.metrics.average_precision_score(subset['validated'], np.abs(subset["log2_fc_pred"]))))
    print("Enformer Absolute: {}".format(sklearn.metrics.average_precision_score(subset['validated'], np.abs(subset["abs_delta"]))))
    print("ABC: {}".format(sklearn.metrics.average_precision_score(subset['validated'], subset['H3K27ac/abs(distance)'])))
    print("Random: {}".format(np.sum(subset['validated'])/len(subset['validated'])))
    print(np.sum(subset['validated'])/len(subset['validated']))
    print(np.sum(subset['validated']))
    print(len(subset['validated']) - np.sum(subset['validated']))

# %%
for pred_bin in labels:
    print(pred_bin)
    subset = merged_df.query('bin_ziga == @pred_bin & dataset_name == "gasperini2019"')
    print("Enformer: {}".format(sklearn.metrics.average_precision_score(subset['validated'], np.abs(subset["log2_fc_pred"]))))
    print("ABC: {}".format(sklearn.metrics.average_precision_score(subset['validated'], subset['H3K27ac/abs(distance)'])))
    print("Random: {}".format(np.sum(subset['validated'])/len(subset['validated'])))
    print(np.sum(subset['validated'])/len(subset['validated']))
    print(np.sum(subset['validated']))
    print(len(subset['validated']) - np.sum(subset['validated']))

# %% [markdown]
# ### Correlation - Fulco

# %%
fulco_df["fulco_key"] = fulco_df["Gene"] + "_" + fulco_df['Element name']

# %%
fulco_merged = merged_df.merge(fulco_df[["fulco_key",'Fraction change in gene expr', 'Adjusted p-value', 'Significant',
       'RNA readout method', 'Perturbation method', 'nKO', 'nCtrl', 'semKO',
       'semCtrl', 'Power to detect 25% effects', 'Gene TSS',
       'Normalized HiC Contacts', 'DHS (RPM)', 'H3K27ac (RPM)', 'Activity',
       'ABC Score', 'Reference']], on='fulco_key')

# %%
# We do not have data for all supposedly validated enhancers
print(len(merged_df.query('validated and dataset_name == "fulco2019"')))
print(len(fulco_merged.query('validated')))

# %%
#set(merged_df.query('validated and dataset_name == "fulco2019"')['fulco_key']) - set(fulco_df["fulco_key"])

# %%
fulco_merged['log2_fc_obs'] = np.log2(1 + fulco_merged['Fraction change in gene expr'])

# %%
print(scipy.stats.pearsonr(fulco_merged['log2_fc_pred'], fulco_merged['log2_fc_obs']))
print(scipy.stats.spearmanr(fulco_merged['log2_fc_pred'], fulco_merged['log2_fc_obs']))

# %%
scale = 1.0
p9.options.figure_size = (6.4*scale, 4.8*scale)

(p9.ggplot(data=fulco_merged,mapping=p9.aes(x="log2_fc_pred",y="log2_fc_obs"))
 + p9.geom_point(alpha=0.3)
 + p9.geom_smooth(method="lm", color="blue")
 + p9.labs(x="Predicted Log2 FC due to Enhancer Knockout",y="Observed Log2 FC due to Enhancer Knockout")
)

# %%
max_bin = 4
bins = [0] + fulco_merged['actual_tss_distance'].quantile(np.array(range(1,max_bin))/max_bin).tolist() + [float('inf')]
#labels = list(range(1,max_bin+1))
labels = ["First quartile","Second quartile","Third quartile","Fourth quartile"]

fulco_merged['quart_bin'] = pd.cut(fulco_merged['actual_tss_distance'], bins=bins, right=True, labels=labels, retbins=False, precision=3, include_lowest=False, duplicates='raise', ordered=True)

# %%
for pred_bin in ["First quartile","Second quartile","Third quartile","Fourth quartile"]:
    print(pred_bin)
    subset = fulco_merged.query('quart_bin == @pred_bin')
    print(scipy.stats.pearsonr(subset['log2_fc_pred'], subset['log2_fc_obs']))

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

(p9.ggplot(data=fulco_merged,mapping=p9.aes(x="log2_fc_pred",y="log2_fc_obs"))
 + p9.geom_point(alpha=0.3)
 + p9.geom_smooth(method="lm", color="blue")
 + p9.facet_wrap("~quart_bin", scales="free")
 + p9.theme(subplots_adjust={'hspace': 0.25,'wspace': 0.25})
 + p9.labs(x="Predicted Log2 FC due to Enhancer Knockout",y="Observed Log2 FC due to Enhancer Knockout")
)

# %% [markdown]
# ### Decay with Distance - Fulco

# %%
cols = ["fulco_key", "ziga_key", "validated", 'actual_tss_distance', 'upstream']

plot_df_fulco = fulco_merged.copy()
plot_df_fulco["Observed"] = np.abs(plot_df_fulco["log2_fc_obs"])
plot_df_fulco["Predicted"] = np.abs(plot_df_fulco["log2_fc_pred"])
plot_df_fulco["Observed_Percent"] = np.abs(plot_df_fulco['Fraction change in gene expr']*100)
plot_df_fulco["Predicted_Percent"] = np.abs(2**plot_df_fulco["log2_fc_pred"]-1)*100
plot_df_percent_fulco = (plot_df_fulco[cols+["Observed_Percent", "Predicted_Percent"]]
                   .rename(columns={"Observed_Percent":"Observed", "Predicted_Percent":"Predicted"})
                   .melt(id_vars=cols, var_name="Type"))



# %%
print(plot_df_percent_fulco.groupby(['Type','validated'])["value"].median())
print(plot_df_percent_fulco.query('actual_tss_distance < 100000 & actual_tss_distance > 990').groupby(['Type','validated'])["value"].median())

# %%
#plot_df_percent['type_valid'] = plot_df_percent.apply(lambda x: x["Type"] if x["validated"] else ("Not-Significant" if x["Type"] == "Predicted" else ""),axis=1)

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=plot_df_percent_fulco.query('validated & actual_tss_distance < 100000 & actual_tss_distance > 990'),mapping=p9.aes(x="actual_tss_distance",y="value",fill="Type", color="Type"))
 + p9.geom_point()
 #+ p9.geom_point(data=plot_df_percent_fulco.query('not validated & Distance < 100000 & Type == "Predicted"'),size=1,alpha=0.1,fill='grey')
 #+ p9.geom_smooth(data=plot_df_percent_fulco.query('not validated & Distance < 100000 & Type == "Predicted"'),method="lm", color="black", fill="")
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 #+ p9.geom_hline(yintercept = 3)
 + p9.geom_smooth(method="lm")
 + p9.labs(x="Distance (kb) between\nTSS and Enhancer",y="Change in Expression (%)\ndue to Enhancer Knockdown", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.29,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10))
)
p

# %%
#p.save("Graphics/" + "distal_fulco" + ".svg", width=2.6, height=2.9, dpi=300)

# %%
scale = 1.5
p9.options.figure_size = (6.4*scale, 4.8*scale)

(p9.ggplot(data=plot_df_percent_fulco.query('actual_tss_distance < 100000 & actual_tss_distance > 990'),mapping=p9.aes(x="actual_tss_distance",y="value",fill="Type", color="Type"))
 + p9.geom_point()
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 #+ p9.geom_hline(yintercept=10)
 #+ p9.geom_hline(yintercept=2)
 + p9.coord_cartesian(ylim=(-3,2))
 + p9.geom_smooth(method="lm")
 + p9.facet_wrap("~validated")
 + p9.labs(x="Distance (kb) between Gene TSS and Enhancer",y="Absolute Percentage change in Expression\ndue to Enhancer Knockout")
)

# %% [markdown]
# ### Correlation - Gasperini

# %%
gasperini_merged = merged_df.merge(gasperini_df[['ENSG', 'target_gene_short',
       'Diff_expression_test_raw_pval', 'Diff_expression_test_fold_change',
       'Diff_expression_test_Empirical_pval',
       'Diff_expression_test_Empirical_adjusted_pval',
       'high_confidence_subset', 'chr.candidate_enhancer',
       'start.candidate_enhancer', 'stop.candidate_enhancer', 'liftover_coord',
       'ziga_key']], on='ziga_key')

# %%
# Here we have data for all supposedly validated enhancers
print(len(merged_df.query('validated and dataset_name == "gasperini2019"')))
print(len(gasperini_merged.query('validated')))

# %%
gasperini_merged['log2_fc_obs'] = np.log2(gasperini_merged['Diff_expression_test_fold_change']) # Proportion remaining transcript of target gene (performed as described in “Method Details: Digital gene expression quantification - Differential expression tests”). 

# %%
scipy.stats.pearsonr(gasperini_merged['log2_fc_pred'], gasperini_merged['log2_fc_obs'])

# %%
scale = 1.0
p9.options.figure_size = (6.4*scale, 4.8*scale)

(p9.ggplot(data=gasperini_merged,mapping=p9.aes(x="log2_fc_pred",y="log2_fc_obs"))
 + p9.geom_point(alpha=0.3)
 + p9.geom_smooth(method="lm", color="blue")
 + p9.labs(x="Predicted Log2 FC due to Enhancer Knockout",y="Observed Log2 FC due to Enhancer Knockout")
)

# %%
max_bin = 4
bins = [0] + gasperini_merged['actual_tss_distance'].quantile(np.array(range(1,max_bin))/max_bin).tolist() + [float('inf')]
#labels = list(range(1,max_bin+1))
labels = ["First quartile","Second quartile","Third quartile","Fourth quartile"]

gasperini_merged['quart_bin'] = pd.cut(gasperini_merged['actual_tss_distance'], bins=bins, right=True, labels=labels, retbins=False, precision=3, include_lowest=False, duplicates='raise', ordered=True)

# %%
for pred_bin in ["First quartile","Second quartile","Third quartile","Fourth quartile"]:
    print(pred_bin)
    subset = gasperini_merged.query('quart_bin == @pred_bin')
    print(scipy.stats.pearsonr(subset['log2_fc_pred'], subset['log2_fc_obs']))

# %%
subset = gasperini_merged.query('actual_tss_distance < 20_000 & actual_tss_distance > 1000')
print(scipy.stats.pearsonr(subset['log2_fc_pred'], subset['log2_fc_obs']))

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

(p9.ggplot(data=gasperini_merged,mapping=p9.aes(x="log2_fc_pred",y="log2_fc_obs"))
 + p9.geom_point(alpha=0.3)
 + p9.geom_smooth(method="lm", color="blue")
 + p9.facet_wrap("~quart_bin", scales="free")
 + p9.theme(subplots_adjust={'hspace': 0.25,'wspace': 0.25})
 + p9.labs(x="Predicted Log2 FC due to Enhancer Knockout",y="Observed Log2 FC due to Enhancer Knockout")
)

# %% [markdown]
# ### Decay with Distance - Gasperini 

# %%
cols = ["ziga_key", "validated", 'actual_tss_distance', 'upstream']

plot_df_gasperini = gasperini_merged.copy()
plot_df_gasperini["Observed"] = np.abs(plot_df_gasperini["log2_fc_obs"])
plot_df_gasperini["Predicted"] = np.abs(plot_df_gasperini["log2_fc_pred"])
plot_df_gasperini["Observed_Percent"] = np.abs(1 - plot_df_gasperini['Diff_expression_test_fold_change'])*100
plot_df_gasperini["Predicted_Percent"] = np.abs(2**plot_df_gasperini["log2_fc_pred"]-1)*100
plot_df_percent_gasperini = (plot_df_gasperini[cols+["Observed_Percent", "Predicted_Percent"]]
                   .rename(columns={"Observed_Percent":"Observed", "Predicted_Percent":"Predicted"})
                   .melt(id_vars=cols, var_name="Type"))

# %%
plot_df_percent_nonvalid_gasperini = noise_df.query('dataset_name == "gasperini2019"')[['ziga_key', 'validated', 'actual_tss_distance', 'abs_pct_change', 'upstream']].rename(columns={"abs_pct_change":"value"})
plot_df_percent_nonvalid_gasperini["Type"] = "Predicted"
plot_df_percent_gasperini = pd.concat([plot_df_percent_gasperini,plot_df_percent_nonvalid_gasperini.query('not validated')])

# %%
plot_df_percent_gasperini.groupby(['Type','validated'])["value"].median()

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=plot_df_percent_gasperini.query('validated & actual_tss_distance < 100000 & actual_tss_distance > 990'),mapping=p9.aes(x="actual_tss_distance",y="value",fill="Type", color="Type"))
 #+ p9.geom_point(data=plot_df_percent_gasperini.query('not validated & actual_tss_distance < 100000 & actual_tss_distance > 500 & Type == "Predicted"'),size=1,alpha=0.05,color="grey",fill='grey')
 + p9.geom_point(size=0.5)
 #+ p9.geom_smooth(data=plot_df_percent_gasperini.query('not validated & actual_tss_distance < 100000 & actual_tss_distance > 500 & Type == "Predicted"'),method="lm", color="black", fill="")
 #+ p9.coords.coord_cartesian(ylim=(-2,2.3))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 #+ p9.geom_hline(yintercept = 3)
 + p9.geom_smooth(method="lm")
 + p9.labs(x="Distance (kb) between\nTSS and Enhancer",y="Change in Expression (%)\ndue to Enhancer Knockdown", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10)
           )
)
p


# %%
scale = 1.5
p9.options.figure_size = (6.4*scale, 4.8*scale)

(p9.ggplot(data=plot_df_percent_gasperini.query('actual_tss_distance < 100000  & actual_tss_distance > 990'),mapping=p9.aes(x="actual_tss_distance",y="value",fill="Type", color="Type"))
 + p9.geom_point()
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_hline(yintercept=10)
 + p9.geom_hline(yintercept=2)
 + p9.geom_smooth(method="lm")
 + p9.facet_wrap("~validated")
 + p9.labs(x="Distance (kb) between Gene TSS and Enhancer",y="Absolute Percentage change in Expression\ndue to Enhancer Knockout")
)

# %%
#p.save("Graphics/" + "distal_gasperini" + ".svg", width=2.6, height=2.9, dpi=300)

# %%
#p.save("Graphics/" + "xsup_distal_gasperini" + ".svg", width=6.4, height=4.8, dpi=300)

# %% [markdown]
# ### Median effect

# %%
fulco_pct = plot_df_percent_fulco.drop(columns="fulco_key")
fulco_pct["dataset"] = "Fulco"
gasperini_pct = plot_df_percent_gasperini
gasperini_pct["dataset"] = "Gasperini"

plot_df_percent = pd.concat([fulco_pct,gasperini_pct])

# %%
# all
print(np.sum(plot_df_percent.query('actual_tss_distance < 100_000 & actual_tss_distance > 990 & Type == "Observed"')["validated"]))
print(plot_df_percent.query('actual_tss_distance < 100_000 & actual_tss_distance > 990').groupby(['validated','Type'])['value'].min())
print(plot_df_percent.query('actual_tss_distance < 100_000 & actual_tss_distance > 990').groupby(['validated','Type'])['value'].median())
print(plot_df_percent.query('actual_tss_distance < 100_000 & actual_tss_distance > 990').groupby(['validated','Type'])['value'].max())
# only distal
print("> 10kb")
print(plot_df_percent.query('actual_tss_distance < 100_000 & actual_tss_distance > 9999').groupby(['validated','Type'])['value'].median())

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=plot_df_percent.query('validated & actual_tss_distance < 98_000 & actual_tss_distance > 999')
           ,mapping=p9.aes(x="actual_tss_distance",y="value",fill="Type", color="Type"))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_smooth(method="lm")
 #+ p9.geom_hline(yintercept=5)
 + p9.facet_wrap('~dataset',scales="free_y")
 + p9.geom_point(size=0.5)
 + p9.labs(x="Distance (bp) between TSS and Enhancer",y="Change in Expression (%)\ndue to Enhancer Knockdown", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.2}
           )
)
p

# %%
plot_df_percent_sat = plot_df_percent.copy()
plot_df_percent_sat["value_sat"] = plot_df_percent_sat["value"].apply(lambda x: x if (x > 0.1 and x < 100) else (0.1 if x <= 0.1 else 100))
plot_df_percent_sat["idx_sat"] = plot_df_percent_sat["value"].apply(lambda x: "Truncated" if (x <= 0.1 or x >= 100) else "")

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=plot_df_percent_sat.query('validated & actual_tss_distance < 98_000 & actual_tss_distance > 999')
           ,mapping=p9.aes(x="actual_tss_distance",y="value_sat"))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_smooth(method="lm",mapping=p9.aes(fill="Type", color="Type"))
 #+ p9.geom_hline(yintercept=5)
 + p9.facet_wrap('~dataset')#,scales="free_y")
 + p9.geom_point(size=0.5, mapping=p9.aes(fill="Type", color="Type",shape="idx_sat"))
 + p9.labs(x="Distance (bp) between TSS and Enhancer",y="Change in Expression (%)\ndue to Enhancer Knockdown", color="", fill="",shape="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.1}
           )
)
p

# %%
p.save("Graphics/" + "enhancer_crispr" + ".svg", width=6.5, height=2.9, dpi=300)

# %%
p = (p9.ggplot(data=plot_df_percent.query('validated & actual_tss_distance < 100_000 & actual_tss_distance > 999')
           ,mapping=p9.aes(x="actual_tss_distance",y="value",fill="Type", color="Type"))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset')
 + p9.geom_hline(yintercept=1)
 + p9.geom_point(size=0.02)
 + p9.labs(x="Distance (bp) between TSS and Enhancer",y="Change in Expression (%)\ndue to Enhancer Knockdown", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.3,0.22),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %%
#p.save("Graphics/" + "distal_enhancer" + ".svg", width=2.6, height=2.9, dpi=300)

# %%
p = (p9.ggplot(data=plot_df_percent.query('Type == "Predicted" & actual_tss_distance < 100_000 & actual_tss_distance > 999')
           ,mapping=p9.aes(x="actual_tss_distance",y="value",fill="validated", color="validated"))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_smooth(method="lm")
 + p9.facet_wrap('~dataset')
 #+ p9.geom_hline(yintercept=1)
 + p9.geom_point(size=0.1, alpha=0.3, raster=True)
 + p9.labs(x="Distance (bp) between TSS and Enhancer",y="Change in Expression (%)\ndue to Enhancer Knockdown", color="Validated:", fill="Validated:")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_title=p9.element_text(size=9),
            legend_position=(0.3,0.22),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %%
p.save("Graphics/" + "supfig_enhancer_nonvalid" + ".svg", width=6.4, height=4.8, dpi=300)

# %%
p = (p9.ggplot(data=plot_df_percent.query('Type == "Predicted" & actual_tss_distance < 100_000 & actual_tss_distance > 999')
           ,mapping=p9.aes(x="actual_tss_distance",y="value",fill="validated", color="validated"))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 #+ p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset')
 + p9.coord_cartesian(ylim=(1,2.2))
 #+ p9.geom_hline(yintercept=1)
 + p9.geom_point(size=1, alpha=1, raster=True)
 + p9.labs(x="Distance (bp) between TSS and Enhancer",y="Change in Expression (%)\ndue to Enhancer Knockdown", color="Validated:", fill="Validated:")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_title=p9.element_text(size=9),
            #legend_position=(0.3,0.22),
            legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %%
a = 100*100#10**4.4608
k = 1
#a = 10**3.5296
#k = 0.7222
exp_line = pd.DataFrame({"x":np.linspace(100,100_000,1000),
                         "y":a*(1/(np.linspace(100,100_000,1000)))})

scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=(plot_df_percent
                     .query('validated & actual_tss_distance < 100_000')
                    )
           ,mapping=p9.aes(x="actual_tss_distance",y="value"))#,fill="Type", color="Type"))
 #+ p9.scale_y_log10()
 #+ p9.scale_x_log10()
 #+ p9.geom_smooth()
 #+ p9.facet_wrap('~dataset')
 + p9.geom_point(size=0.5, alpha=0.5)
 + p9.geom_line(data=exp_line, mapping=p9.aes(x="x",y="y"),color="blue")
 + p9.facet_wrap('~Type')
 + p9.coord_cartesian(ylim=(0,100))
 + p9.labs(x="Distance (bp) between TSS and Enhancer",y="Change in Expression (%)\ndue to Enhancer Knockdown", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10)
           )
)
p

# %% [markdown]
# ### Impact of basal expression

# %%
promoter_effect = merged_df[["ziga_key","chromosome","main_tss_start","main_tss_end","log2_fc_pred","actual_tss_distance", "validated", pred_col+"_ref",pred_col+"_alt", "dataset_name"]]
promoter_effect["id"] = (promoter_effect["chromosome"] + "_" + ((promoter_effect["main_tss_start"] + promoter_effect["main_tss_end"])/2).astype('int').astype('str') + "_" + "promoter")
promoter_effect = promoter_effect.merge(local_results,on="id")
promoter_effect["abs_log2_fc_pred"] = np.abs(promoter_effect["log2_fc_pred"])
promoter_effect["abs_pct_change"] = np.abs(2**promoter_effect["log2_fc_pred"] - 1)*100
promoter_effect["basal"] = 2**promoter_effect[pred_col+"_ref"] - 1
promoter_effect["alt"] = 2**promoter_effect[pred_col+"_alt"] - 1
promoter_effect['delta'] = np.abs(promoter_effect["alt"] - promoter_effect["basal"])

# %%
p = (p9.ggplot(data=promoter_effect.query('actual_tss_distance < 100000 & actual_tss_distance > 990'),
               mapping=p9.aes(x=pred_col,y="abs_pct_change"))
 #+ p9.geom_point(data=plot_df_percent_gasperini.query('not validated & actual_tss_distance < 100000 & actual_tss_distance > 500 & Type == "Predicted"'),size=1,alpha=0.05,color="grey",fill='grey')
 + p9.geom_point(size=0.5)
 #+ p9.geom_smooth(data=plot_df_percent_gasperini.query('not validated & actual_tss_distance < 100000 & actual_tss_distance > 500 & Type == "Predicted"'),method="lm", color="black", fill="")
 #+ p9.coords.coord_cartesian(ylim=(-2,2.3))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 #+ p9.geom_hline(yintercept = 3)
 + p9.geom_smooth(method="lm",color="blue")
 + p9.labs(x="Predicted Local Expression",y="Change in Expression (%)\ndue to Enhancer Knockdown", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10)
           )
)
p

# %%
p = (p9.ggplot(data=promoter_effect.query('actual_tss_distance < 100_000 and actual_tss_distance > 990'),
               mapping=p9.aes(x="alt",y="abs_pct_change"))
 #+ p9.geom_point(data=plot_df_percent_gasperini.query('not validated & actual_tss_distance < 100000 & actual_tss_distance > 500 & Type == "Predicted"'),size=1,alpha=0.05,color="grey",fill='grey')
 + p9.geom_point(size=0.5)
 #+ p9.geom_smooth(data=plot_df_percent_gasperini.query('not validated & actual_tss_distance < 100000 & actual_tss_distance > 500 & Type == "Predicted"'),method="lm", color="black", fill="")
 #+ p9.coords.coord_cartesian(ylim=(-2,2.3))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_hline(yintercept = 10)
 + p9.geom_smooth(color="blue",method="lowess")#,method="lm")
 + p9.labs(x="Predicted Expression (w/o Enhancer)",y="Predicted Change in Expression (%)\ndue to Enhancer Knockdown", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10)
           )
)
p

# %%
p.save("Graphics/" + "supfig_enhancer_promrelative" + ".svg", width=6.4, height=4.8, dpi=300)

# %%
p = (p9.ggplot(data=promoter_effect.query('actual_tss_distance < 100_000 and actual_tss_distance > 990'),
               mapping=p9.aes(x="alt",y="abs_log2_fc_pred"))
 #+ p9.geom_point(data=plot_df_percent_gasperini.query('not validated & actual_tss_distance < 100000 & actual_tss_distance > 500 & Type == "Predicted"'),size=1,alpha=0.05,color="grey",fill='grey')
 + p9.geom_point(size=0.5)
 #+ p9.geom_smooth(data=plot_df_percent_gasperini.query('not validated & actual_tss_distance < 100000 & actual_tss_distance > 500 & Type == "Predicted"'),method="lm", color="black", fill="")
 #+ p9.coords.coord_cartesian(ylim=(-2,2.3))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 #+ p9.geom_hline(yintercept = 10)
 + p9.geom_smooth(color="blue",method="lowess")#,method="lm")
 + p9.labs(x="Predicted Expression (w/o Enhancer)",y="Predicted Change in Expression\n(unsigned log2 FC) due to Enhancer Knockdown", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10)
           )
)
p

# %%
p = (p9.ggplot(data=promoter_effect.query('actual_tss_distance < 100_000 and actual_tss_distance > 990'),
               mapping=p9.aes(x="alt",y="delta"))
 #+ p9.geom_point(data=plot_df_percent_gasperini.query('not validated & actual_tss_distance < 100000 & actual_tss_distance > 500 & Type == "Predicted"'),size=1,alpha=0.05,color="grey",fill='grey')
 + p9.geom_point(size=0.5)
 #+ p9.geom_smooth(data=plot_df_percent_gasperini.query('not validated & actual_tss_distance < 100000 & actual_tss_distance > 500 & Type == "Predicted"'),method="lm", color="black", fill="")
 #+ p9.coords.coord_cartesian(ylim=(-2,2.3))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_hline(yintercept = 10)
 + p9.geom_smooth(color="blue",method="lowess")#,method="lm")
 + p9.labs(x="Predicted Expression (w/o Enhancer)",y="Predicted Change in Expression (absolute)\ndue to Enhancer Knockdown", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10)
           )
)
p

# %%
p.save("Graphics/" + "supfig_enhancer_promabsolute" + ".svg", width=6.4, height=4.8, dpi=300)

# %%

dims = (6.4, 4.8)
statplot_df = promoter_effect.query('actual_tss_distance < 100_000 and actual_tss_distance > 990').copy()
statplot_df["validated"] = statplot_df["validated"].apply(lambda x: "Validated" if x else "Not Validated")
x = "validated"
y = "alt"
#hue = "Motif_count"
box_pairs=[
    ("Validated", "Not Validated")
    ]
fig, ax = plt.subplots(figsize=dims, dpi=100)
ax = sns.boxplot(data=statplot_df, x=x, y=y, #hue=hue
                )
#plt.legend(bbox_to_anchor=(1, 0.5), title="Motif_count")
statannot.add_stat_annotation(ax, data=statplot_df, x=x, y=y, #hue=hue, 
                              box_pairs=box_pairs, comparisons_correction=None,
                    test='Mann-Whitney', loc='inside',  verbose=2)

ax.set_xlabel("",fontsize=10, color="black")
ax.set_ylabel("Predicted Expression (after Enhancer-KO)",fontsize=10, color="black")
#plt.legend(bbox_to_anchor=(1, 0.5), title="Motif_count")
#plt.setp(ax.get_legend().get_title(), fontsize='10')
#plt.setp(ax.get_legend().get_texts(), fontsize='10')
ax.tick_params(labelsize=10)
plt.tight_layout()
fig.savefig("Graphics/" + "supfig_enhancer_basalvalid" + ".svg")

# %%
p = (p9.ggplot(data=promoter_effect.query('actual_tss_distance < 100_000 and actual_tss_distance > 990 & dataset_name == "fulco2019"'),
               mapping=p9.aes(x="alt",y="abs_pct_change"))
 #+ p9.geom_point(data=plot_df_percent_gasperini.query('not validated & actual_tss_distance < 100000 & actual_tss_distance > 500 & Type == "Predicted"'),size=1,alpha=0.05,color="grey",fill='grey')
 + p9.geom_point(size=0.5)
 #+ p9.geom_smooth(data=plot_df_percent_gasperini.query('not validated & actual_tss_distance < 100000 & actual_tss_distance > 500 & Type == "Predicted"'),method="lm", color="black", fill="")
 #+ p9.coords.coord_cartesian(ylim=(-2,2.3))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_hline(yintercept = 10)
 + p9.geom_smooth(color="blue",method="lm")
 + p9.labs(x="Predicted Expression (w/o Enhancer)",y="Change in Expression (%)\ndue to Enhancer Knockdown", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10)
           )
)
p

# %%
plot_df_promoter = plot_df_percent.merge(promoter_effect[["ziga_key",pred_col,"alt"]], on="ziga_key")

p = (p9.ggplot(data=plot_df_promoter.query('actual_tss_distance < 100000 & actual_tss_distance > 999 & Type == "Observed" & dataset == "Fulco"'),
               mapping=p9.aes(x="alt",y="value"))
 #+ p9.geom_point(data=plot_df_percent_gasperini.query('not validated & actual_tss_distance < 100000 & actual_tss_distance > 500 & Type == "Predicted"'),size=1,alpha=0.05,color="grey",fill='grey')
 + p9.geom_point(size=0.5)
 #+ p9.geom_smooth(data=plot_df_percent_gasperini.query('not validated & actual_tss_distance < 100000 & actual_tss_distance > 500 & Type == "Predicted"'),method="lm", color="black", fill="")
 #+ p9.coords.coord_cartesian(ylim=(-2,2.3))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 #+ p9.geom_hline(yintercept = 3)
 + p9.geom_smooth(method="lm",color="blue")
 + p9.labs(x="Predicted Expression (w/o Enhancer)",y="Change in Expression (%)\ndue to Enhancer Knockdown", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10)
           )
)
p

# %%
p = (p9.ggplot(data=promoter_effect.query('actual_tss_distance < 100_000 and actual_tss_distance > 990 & validated'),
               mapping=p9.aes(x="alt",y="abs_pct_change"))
 #+ p9.geom_point(data=plot_df_percent_gasperini.query('not validated & actual_tss_distance < 100000 & actual_tss_distance > 500 & Type == "Predicted"'),size=1,alpha=0.05,color="grey",fill='grey')
 + p9.geom_point(size=0.5)
 #+ p9.geom_smooth(data=plot_df_percent_gasperini.query('not validated & actual_tss_distance < 100000 & actual_tss_distance > 500 & Type == "Predicted"'),method="lm", color="black", fill="")
 #+ p9.coords.coord_cartesian(ylim=(-2,2.3))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_hline(yintercept = 10)
 + p9.geom_smooth(color="blue",method="lowess")#,method="lm")
 + p9.labs(x="Predicted Expression (w/o Enhancer)",y="Change in Expression (%)\ndue to Enhancer Knockdown", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10)
           )
)
p

# %%
plot_df_promoter = plot_df_percent.merge(promoter_effect[["ziga_key",pred_col,"alt"]], on="ziga_key")

p = (p9.ggplot(data=plot_df_promoter.query('actual_tss_distance < 100000 & actual_tss_distance > 999 & Type == "Observed" & validated'),
               mapping=p9.aes(x="alt",y="value"))
 #+ p9.geom_point(data=plot_df_percent_gasperini.query('not validated & actual_tss_distance < 100000 & actual_tss_distance > 500 & Type == "Predicted"'),size=1,alpha=0.05,color="grey",fill='grey')
 + p9.geom_point(size=0.5)
 #+ p9.geom_smooth(data=plot_df_percent_gasperini.query('not validated & actual_tss_distance < 100000 & actual_tss_distance > 500 & Type == "Predicted"'),method="lm", color="black", fill="")
 #+ p9.coords.coord_cartesian(ylim=(-2,2.3))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 #+ p9.geom_hline(yintercept = 3)
 + p9.geom_smooth(method="lowess",color="blue")
 + p9.labs(x="Predicted Expression (w/o Enhancer)",y="Change in Expression (%)\ndue to Enhancer Knockdown", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10)
           )
)
p

# %%
promoter_effect["log_dist"] = np.log10(promoter_effect["actual_tss_distance"])
promoter_effect["log_prom"] = np.log10(promoter_effect[pred_col]+1)
promoter_effect["log_pct"] = np.log10(promoter_effect["abs_pct_change"] + 0.001)

# %%
ols_dist_promoter = (statsmodels.regression.linear_model.OLS
                  .from_formula('log_pct ~ log_dist + log_prom + validated',
                                data=(promoter_effect
                                      .query('actual_tss_distance < 100_000 & actual_tss_distance > 999')))
                  )
res_ols_dist_promoter = ols_dist_promoter.fit()

res_ols_dist_promoter.summary()

# %% [markdown]
# ### Recall cutoffs

# %%
prec_for_cutoff_rows = []
for obs_cut in [0,10,20,25,50]:
    if obs_cut == 0:
        asym_df =  plot_df_percent.query('actual_tss_distance > 1000 and Type == "Predicted"')
    else:
        high_obs = set(plot_df_percent.query('actual_tss_distance > 1000 and Type == "Observed" and value > @obs_cut')["ziga_key"])
        asym_df = plot_df_percent.query('actual_tss_distance > 1000 and Type == "Predicted" and ziga_key in @high_obs')
    for cutoff in [0, 1, 3, 5,8,10,15,20,25, 50]:
        subset = asym_df.query('value > @cutoff')
        prec = np.sum(subset['validated'])/len(subset)
        recall = np.sum(subset['validated'])/np.sum(asym_df['validated'])
        prec_for_cutoff_rows.append({"obs_cut":obs_cut,
                                     "cutoff":cutoff,
                                     "tp":np.sum(subset['validated']),
                                     "total":len(subset),
                                     "% of candidates":len(subset)/len(asym_df),
                                     "Recall":recall,
                                     "inv_recall":1-recall,
                                     "Precision":prec,
                                     "inv_prec":1-prec,
                                    })
    
prec_for_cutoff_df = pd.DataFrame(prec_for_cutoff_rows)
#prec_for_cutoff_df

# %%
prec_for_cutoff_plt = prec_for_cutoff_df[["obs_cut","cutoff","Recall","Precision"]].melt(id_vars=["obs_cut","cutoff"]).query('obs_cut in [0] and cutoff in [1,3,5,8,10,20,50]')
prec_for_cutoff_plt["cutoff"] = pd.Categorical(prec_for_cutoff_plt["cutoff"], [1,3,5,8,10,20,50])


p = (p9.ggplot(data=prec_for_cutoff_plt
           ,mapping=p9.aes(x="variable",y="value", fill="cutoff"))
 + p9.geom_bar(stat="identity",position="dodge")
 + p9.labs(x="",y="Enhancer Classification Performance\n(using Enformer enhancer-KO predictions)",
           fill="Cutoff (%):")
 #+ p9.coord_cartesian(ylim=(0,0.32))
 + p9.theme(#legend_position=(0.35,0.745), 
            legend_box_margin=0,
            legend_background=p9.element_rect(color="none", size=2, fill='none'),
            legend_text=p9.element_text(size=8),legend_key_size=8,
            legend_title=p9.element_text(size=8),
           axis_text_x=p9.element_text(rotation=30, hjust=1),
           axis_title=p9.element_text(size=10))
 #+ p9.facet_wrap('~obs_cut')
)
p

# %%
p.save("Graphics/" + "supfig_enhancer_recall" + ".svg", width=6.4, height=4.8, dpi=300)

# %% [markdown]
# ### Can we recalibrate?

# %%
plot_df_percent["log_dist"] = np.log10(plot_df_percent["actual_tss_distance"])
plot_df_percent["log_pct"] = np.log10(plot_df_percent["value"])

ols_1_over_dist = (statsmodels.regression.linear_model.OLS
                  .from_formula('log_pct ~ log_dist',
                                data=(plot_df_percent
                                      .query('validated & actual_tss_distance < 100_000 & actual_tss_distance > 990 & Type == "Predicted"')))
                  )
res_ols_1_over_dist = ols_1_over_dist.fit()

res_ols_1_over_dist.summary()

# %%
plot_df_percent["pct_change_recalibrated"] = 10**(plot_df_percent["log_pct"] + 0.7222*plot_df_percent['log_dist'] - 3.5296)
plot_df_percent["pct_change_recalibrated"] = plot_df_percent.apply(lambda x: x["pct_change_recalibrated"] if x["Type"] == "Predicted" else x["value"] ,axis=1)

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=plot_df_percent.query('validated & actual_tss_distance < 100_000 & actual_tss_distance > 999')
           ,mapping=p9.aes(x="actual_tss_distance",y="pct_change_recalibrated",fill="Type", color="Type"))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset')
 + p9.geom_point(size=0.5)
 + p9.labs(x="Distance (bp) between TSS and Enhancer",y="Change in Expression (%)\ndue to Enhancer Knockdown", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=plot_df_percent.query('actual_tss_distance < 100_000 & actual_tss_distance > 999 & Type == "Predicted"')
           ,mapping=p9.aes(x="actual_tss_distance",y="pct_change_recalibrated",fill="validated", color="validated"))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset')
 + p9.geom_point(size=0.5)
 + p9.labs(x="Distance (bp) between TSS and Enhancer",y="Change in Expression (%)\ndue to Enhancer Knockdown", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %%
subset = plot_df_percent.query('Type == "Predicted" & actual_tss_distance > 1000 & ziga_key not in @ overlapped_regions')
print(sklearn.metrics.average_precision_score(subset['validated'], subset['value']))
print(sklearn.metrics.average_precision_score(subset['validated'], subset['pct_change_recalibrated']))

# %% [markdown]
# ### Only upstream

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=plot_df_percent.query('validated & actual_tss_distance < 100_000 & actual_tss_distance > 999 & upstream')
           ,mapping=p9.aes(x="actual_tss_distance",y="value",fill="Type", color="Type"))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_smooth(method="lm")
 + p9.facet_wrap('~dataset')
 + p9.geom_point(size=0.5)
 + p9.labs(x="Distance (bp) between TSS and Enhancer",y="Change in Expression (%)\ndue to Enhancer Knockdown", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %% [markdown]
# ### Strongest effect per gene

# %%
plot_df_percent["gene"] = plot_df_percent["ziga_key"].apply(lambda x: x.split("_")[1])

# %%
valid_genes = set(plot_df_percent.query('validated')["gene"])

# %%
print(len(valid_genes))
print(len(plot_df_percent.query('gene in @valid_genes & actual_tss_distance > 999 & Type == "Predicted"')))
print(len(plot_df_percent.query('gene in @valid_genes & actual_tss_distance > 999 & Type == "Predicted"'))/len(valid_genes))

# %%
plot_df_percent.query('gene in @valid_genes & actual_tss_distance > 999 & Type == "Predicted"').groupby('gene').size().describe()

# %%
strongest_effects = (plot_df_percent
                     .query('gene in @valid_genes & actual_tss_distance > 999')
                     .groupby(['gene','Type'])['value']
                     .agg({"median","max"})
                     .reset_index())

scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=strongest_effects#.query('Type == "Predicted"')
           ,mapping=p9.aes(x="max",color="Type"))
 #+ p9.scale_y_log10()
 + p9.scale_x_log10()
 #+ p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset')
 #+ p9.geom_vline(xintercept=6.5)
 #+ p9.geom_vline(xintercept=25)
 + p9.geom_vline(xintercept=10)
 + p9.stat_ecdf()
 + p9.labs(x="Maximal Change in Expression (%)\ndue to Enhancer Knockdown",y="Cumulative Fraction", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.25,0.75),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %%
p.save("Graphics/" + "supfig_enhancer_maxeffect" + ".svg", width=6.4, height=4.8, dpi=300)

# %%
strongest_effects_valid = (plot_df_percent
                     .query('gene in @valid_genes & actual_tss_distance > 999 & validated')
                     .groupby(['gene','Type'])['value']
                     .agg({"median","max"})
                     .reset_index())

scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=strongest_effects_valid#.query('Type == "Predicted"')
           ,mapping=p9.aes(x="max",color="Type"))
 #+ p9.scale_y_log10()
 + p9.scale_x_log10()
 #+ p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset')
 + p9.geom_vline(xintercept=10)
 #+ p9.geom_vline(xintercept=25)
 #+ p9.geom_vline(xintercept=10)
 + p9.stat_ecdf()
 + p9.labs(x="Maximal Change in Expression (%)\ndue to Enhancer Knockdown",y="Cumulative Fraction", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %%
genes_with_distal = set(plot_df_percent.query('actual_tss_distance > 10_000 & validated')["gene"])

# %%
strongest_effects_distal = (plot_df_percent
                     .query('gene in @genes_with_distal & actual_tss_distance > 10_000')
                     .groupby(['gene','Type'])['value']
                     .agg(["median","max"])
                     .reset_index())

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=strongest_effects_distal#.query('Type == "Predicted"')
           ,mapping=p9.aes(x="max",color="Type"))
 #+ p9.scale_y_log10()
 #+ p9.scale_x_log10()
 #+ p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset')
 + p9.geom_vline(xintercept=10)
 + p9.stat_ecdf()
 + p9.labs(x="Maximal Change in Expression (%)\ndue to Enhancer Knockdown",y="Cumulative Fraction", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %% [markdown]
# ### Which enhancer-gene pairs are seen in training?

# %%
valid_enhancers = ziga_df.query('validated')
valid_enhancers["tss_mid"] = (valid_enhancers["main_tss_end"] + valid_enhancers["main_tss_start"])/2
valid_enhancers["tss_id"] = valid_enhancers["chromosome"] + valid_enhancers["tss_mid"].astype('str')
valid_enhancers["enhancer_mid"] = (valid_enhancers['fix_enhancer_wide_start'] + valid_enhancers['fix_enhancer_wide_end'])/2

# %%
train_regions = (pd.read_csv(base_path_data_gtex + "human_regions.bed", names=["Chromosome","Start","End","set"], sep="\t")
                            .query('set == "train"')
                            .drop(columns="set"))
train_regions["id"] = train_regions["Chromosome"] + train_regions["Start"].astype('str') + train_regions["End"].astype('str')

train_regions_wide = train_regions.copy()
train_regions_wide["Start"] = train_regions_wide["Start"] - (SEEN_SEQUENCE_LENGTH - 131072)/2
train_regions_wide["End"] = train_regions_wide["End"] + (SEEN_SEQUENCE_LENGTH - 131072)/2
train_regions_wide = pr.PyRanges(train_regions_wide)

train_regions_predict = train_regions.copy()
train_regions_predict["Start"] = train_regions_predict["Start"] + (131072 - 896*128)/2
train_regions_predict["End"] = train_regions_predict["End"] - (131072 - 896*128)/2
train_regions_predict = pr.PyRanges(train_regions_predict)

# %%
tss_sites = pr.PyRanges(valid_enhancers[['chromosome', 'main_tss_start', 'main_tss_end','tss_id']]
                         .rename(columns={'chromosome':'Chromosome',
                                         "main_tss_start":"Start",
                                         "main_tss_end":"End"})
                        .drop_duplicates()
                       )

# %%
enhancer_sites = pr.PyRanges(valid_enhancers[['chromosome', "fix_enhancer_wide_start","fix_enhancer_wide_end",'tss_id','ziga_key']]
                         .rename(columns={'chromosome':'Chromosome',
                                         "fix_enhancer_wide_start":"Start",
                                         "fix_enhancer_wide_end":"End"})
                        .drop_duplicates()
                       )

# %%
tss_in_train = train_regions_predict.join(tss_sites, suffix="_tss",report_overlap=True).df
enhancer_in_train = train_regions_wide.join(enhancer_sites, suffix="_enhancer",report_overlap=True).df.query('Overlap == 2000')

# %%
both_in_train = tss_in_train.drop(columns="Overlap").merge(enhancer_in_train.drop(columns="Chromosome"),suffixes=("_predicted_region","_seen_region"), on=["id", "tss_id"])

# %%
#seen_pairs = both_in_train[["tss_mid","enhancer_mid","id"]].drop_duplicates().merge(valid_enhancers, on=["tss_mid","enhancer_mid"])
valid_enhancers_seen = (tss_in_train.groupby("tss_id")
                      .size().reset_index()
                      .rename(columns={0:"tss_seen_count"})
                      .merge(valid_enhancers, on=["tss_id"]))
valid_enhancers_seen = (both_in_train.groupby("ziga_key")
                      .size().reset_index()
                      .rename(columns={0:"enhancer_seen_count"})
                      .merge(valid_enhancers_seen, on=["ziga_key"],how="right"))
valid_enhancers_seen["enhancer_seen_count"] = valid_enhancers_seen["enhancer_seen_count"].fillna(0)
valid_enhancers_seen = valid_enhancers_seen.sort_values('enhancer_seen_count')

valid_enhancers_seen = (valid_enhancers_seen.groupby('tss_id')
                        .size().reset_index().rename(columns={0:"total_enhancers"})
                        .merge(valid_enhancers_seen, on=["tss_id"])
                       )

valid_enhancers_seen['fully_seen'] = (valid_enhancers_seen['enhancer_seen_count'] == valid_enhancers_seen['tss_seen_count'])

valid_enhancers_seen = (valid_enhancers_seen.groupby('tss_id')['fully_seen']
                        .sum().reset_index().rename(columns={'fully_seen':"fully_seen_enhancers"})
                        .merge(valid_enhancers_seen, on=["tss_id"])
                       )

# %%
# for tss, which are seen
# check for difference between enhancers which are seen fully and those which are not
seen_twice = valid_enhancers_seen.query('tss_seen_count > 0')
seen_twice = seen_twice.merge(merged_df[["ziga_key","log2_fc_pred"]],on="ziga_key")
seen_twice['abs_pct_change'] = np.abs((2**seen_twice['log2_fc_pred'] - 1)*100)
seen_twice["abs_log2_fc_pred"] = np.abs(seen_twice["log2_fc_pred"])
seen_twice["log_tss_distance"] = np.log10(seen_twice["actual_tss_distance"] + 1000)
seen_twice["log_abs_log2_fc_pred"] = np.log(np.abs(seen_twice["log2_fc_pred"]))
seen_twice['log_abs_pct_change'] = np.log(np.abs(seen_twice["abs_pct_change"]))

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=seen_twice.query('actual_tss_distance > 999 & actual_tss_distance < 100_000')
           ,mapping=p9.aes(x="actual_tss_distance",y="abs_pct_change",color="fully_seen"))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_hline(yintercept=10)
 + p9.geom_smooth(method="lm",color="black")
 + p9.geom_point(size=1)
 + p9.labs(x="Distance (kb) between TSS and Enhancer",y="Change in Expression (%)\ndue to Enhancer Knockdown", color="Seen with the TSS\nduring training")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_title=p9.element_text(size=9),
            legend_position=(0.3,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10)
           )
)
p

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=seen_twice.query('actual_tss_distance > 999  & actual_tss_distance < 100_000')
           ,mapping=p9.aes(x="actual_tss_distance",y="abs_pct_change",color="fully_seen"))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 #+ p9.geom_hline(yintercept=10)
 + p9.geom_smooth(method="lm")
 + p9.geom_point(size=0.25)
 + p9.labs(x="Distance (bb) between\nTSS and Enhancer",y="Predicted Change in Expression\ndue to Enhancer Knockdown (%)", color="Seen with the TSS\nduring training")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=8),
            legend_title=p9.element_text(size=8),
            legend_position=(0.4,0.24),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10)
           )
)
p

# %%
p.save("Graphics/" + "classimba_train" + ".svg", width=2.4, height=3.0, dpi=300)

# %%
p = (p9.ggplot(data=seen_twice.query('actual_tss_distance > 999')
           ,mapping=p9.aes(x="actual_tss_distance",y="abs_pct_change",color="fully_seen"))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 #+ p9.geom_hline(yintercept=10)
 + p9.geom_smooth(method="lm")
 + p9.geom_point(size=1)
 + p9.labs(x="Distance (bb) between TSS and Enhancer",y="Predicted Change in Expression (%)\ndue to Enhancer Knockdown", color="Seen with the TSS\nduring training")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_title=p9.element_text(size=9),
            legend_position=(0.3,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10)
           )
)
p

# %% [markdown]
# ### The Enformer ABC score

# %%
with open(base_path_results+"avsec_enhancercentered-enformer-latest_results.tsv", 'r') as tsv:
    header = tsv.readline()
    cols = header.split("\t")
cols = [k for k in cols if k.endswith("landmark_sum_wide") and ("k562" in k.lower() or "H3K27ac" in k)]

enhancer_results = pd.read_csv(base_path_results+"avsec_enhancercentered-enformer-latest_results.tsv",sep="\t",
                     usecols = ["chr","old_index","enh_start", "enh_end","enh_mid","orient","offset"] + cols)

enhancer_results = enhancer_results.groupby(["chr","old_index","enh_start", "enh_end","enh_mid"])[cols].mean().reset_index()

enhancer_results = enhancer_results.rename(columns={x:x+"_enh" for x in cols})

enhancer_results["enhancer_id"] = enhancer_results["chr"] + "_" + enhancer_results["enh_start"].astype('str') + ":" + enhancer_results["enh_end"].astype('str')

# %%
ko_results = merged_df.copy()
ko_results["enhancer_id"] = ko_results["chromosome"] + "_" + ko_results["enhancer_start"].astype('str') + ":" + ko_results["enhancer_end"].astype('str')
ko_results = ko_results.query('actual_tss_distance > 1000')
ko_results = ko_results.merge(enhancer_results, on="enhancer_id")
ko_results["basal"] = 2**ko_results[pred_col+"_ref"] - 1
ko_results["delta"] = np.abs(2**(ko_results[pred_col+"_ref"]-1) - 2**(ko_results[pred_col+"_alt"]-1))
ko_results["abs_log2_fc_pred"] = np.abs(ko_results["log2_fc_pred"])
ko_results["pct_change"] = (2**ko_results["log2_fc_pred"] - 1)*100
ko_results["abs_pct_change"] = np.abs(2**ko_results["log2_fc_pred"] - 1)*100
ko_results["log_abs_pct_change"] = np.log10(ko_results["abs_pct_change"])
ko_results["log_dist"] = np.log10(ko_results["actual_tss_distance"]+1000)
ko_results["one_over_dist"] = 1/ko_results["actual_tss_distance"]
ko_results["valid_num"] = ko_results["validated"].astype('int')

# %%
track_types = ["H3K27ac", "DNASE"] #["H3K27ac","H3K27me", "H3K9me", "H3K9ac", "H3K79me", "DNASE"]
track_types_dist = [x+"_over_dist" for x in track_types] + ["combined_over_dist"]

for track_type in track_types:
    ko_results[track_type] = ko_results[[x for x in enhancer_results.keys() if x.endswith('landmark_sum_wide_enh') and track_type in x]].mean(axis=1)
    ko_results[track_type + "_over_dist"] = ko_results[track_type]/ko_results["actual_tss_distance"]
ko_results["combined"] = scipy.stats.mstats.gmean(ko_results[track_types],axis=1)
ko_results["combined_over_dist"] = ko_results["combined"]/ko_results["actual_tss_distance"]

# %%
print(scipy.stats.pearsonr(ko_results['H3K27ac/abs(distance)'],ko_results["combined_over_dist"]))
print(scipy.stats.spearmanr(ko_results['H3K27ac/abs(distance)'],ko_results["combined_over_dist"]))

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=ko_results
           ,mapping=p9.aes(x="combined_over_dist",y='H3K27ac/abs(distance)'))#,fill="validated", color="validated"))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset_name')
 + p9.geom_point(size=0.5,alpha=0.15)
 + p9.labs(x="Enformer ABC score",y="ABC Score (H3K27ac/distance)", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %%
p.save("Graphics/" + "supfig_enhancer_eabc" + ".svg", width=6.4, height=4.8, dpi=300)

# %%
auprc_rows = []

predictors = {
    "delta":"Predicted Absolute Change",
    "H3K27ac/abs(distance)":"ABC score (H3K27ac/distance)",
    "abs_log2_fc_pred":"Predicted (unsigned) Log-Fold Change",
    "abs_pct_change":"Predicted Percentage Change",
    "combined_over_dist":"Enformer-ABC",
    #pred_col+"_ref":"Predicted Basal Expression"
}

def bootstrap_auprc(subset, predictor, n=99, cutoff=0.05):
    samples = []
    for i in range(n):
        sample = subset.sample(frac=1.0,replace=True)
        samples.append(sklearn.metrics.average_precision_score(sample['validated'], sample[predictor]))
    samples = sorted(samples)
    return samples[math.floor((n+1)*cutoff/2)],samples[math.ceil((n+1)*(1-cutoff/2))]

#pred_bins = ["Full"] + [x for x in set(ko_results["bin"])]
datasets = {"Fulco":"fulco2019", "Gasperini":"gasperini2019", "Combined":"Combined"}

#for pred_bin in pred_bins:
for dataset in datasets:
    dataset_key = datasets[dataset]
    if dataset == "Combined":
        subset = ko_results.copy()
    else:
        #subset = ko_results.query('bin == @pred_bin')
        subset = ko_results.query('dataset_name == @dataset_key')
    for predictor in predictors.keys():
        auprc = sklearn.metrics.average_precision_score(subset['validated'], subset[predictor])
        min_auprc, max_auprc = bootstrap_auprc(subset, predictor)
        auprc_rows.append({
            "dataset":dataset,
            "predictor":predictors[predictor],
            "auprc":auprc,
            "min":min_auprc,
            "max":max_auprc
        })
    random = np.sum(subset['validated'])/len(subset['validated'])
    auprc_rows.append({
        "dataset":dataset,
        "predictor":"Random",
        "auprc":random,
        "min":random,
        "max":random
    })
    
auprc_df = pd.DataFrame(auprc_rows)

# %%
auprc_df["predictor"] = pd.Categorical(auprc_df["predictor"], categories = [x for x in predictors.values()] + ["Random"])


p = (p9.ggplot(data=auprc_df
           ,mapping=p9.aes(x="dataset",y="auprc", fill="predictor",ymin="min",ymax="max"))
 + p9.geom_bar(stat="identity",position="dodge")
 + p9.geom_errorbar(position=p9.position_dodge(0.9))
 + p9.labs(x="",y="AUPRC (Bars: 95% C.I.)",
           fill="Predictor:")
 + p9.theme(#legend_position=(0.35,0.745), 
            legend_box_margin=0,
            #legend_position=(0.75,0.75),
            legend_background=p9.element_rect(color="none", size=2, fill='none'),
            legend_text=p9.element_text(size=8),legend_key_size=8,
            legend_title=p9.element_text(size=8),
           axis_text_x=p9.element_text(rotation=30, hjust=1),
           axis_title=p9.element_text(size=10))
)
p

# %%
p.save("Graphics/" + "supfig_enhancer_auprc" + ".svg", width=6.4, height=4.8, dpi=300)

# %%
print(sklearn.metrics.average_precision_score(ko_results['validated'], preds))

# %% [markdown]
# ### Which pairs could be useful for in-silico analysis?
#
# We posit the following criteria for enhancer-promoter pairs:
# - high predicted effect
# - not too close (> 3kb)
# - some weak but close enhancers
# - some intermediate strength enhancers
#
# We posit the following criteria for locations:
# - the strongest effects
# - very far ones with strong effect
# - some close but weak ones

# %%
# strong effects
candidates = (plot_df_percent.query('Type == "Predicted"')
              .query('actual_tss_distance > 3000 and actual_tss_distance < 90_000 and (value > 30)')
              #.groupby('ziga_key').size().reset_index()
              #.rename(columns={0:"count"})
              #.query('count == 2')
             )
strong_candidates = set(candidates["ziga_key"])
candidate_genes = set(candidates["gene"])
# weak enhancers
weak_candidates = set((plot_df_percent.query('Type == "Predicted"')
              .query('actual_tss_distance > 3000 and actual_tss_distance < 8_000 and (value < 0.1)')
             )["ziga_key"])
weak_for_gene_candidates = set((plot_df_percent.query('Type == "Predicted" and gene in @candidate_genes')
              .query('actual_tss_distance > 3000 and actual_tss_distance < 20_000 and (value < 1)')
             )["ziga_key"])
intermediate_candidates = set((plot_df_percent.query('Type == "Predicted" and gene in @candidate_genes')
              .query('actual_tss_distance > 3_000 and actual_tss_distance < 20_000 and (value < 8 and value > 4)')
             )["ziga_key"])


candidate_set = strong_candidates | weak_candidates | weak_for_gene_candidates | intermediate_candidates

candidate_locs = (merged_df.query('ziga_key in @candidate_set')
                  [["ziga_key","gene","main_tss_start","main_tss_end","main_tss_bin",'enhancer_start', 'enhancer_end','actual_tss_distance','chromosome',"validated"]
                   + ['fix_enhancer_wide_start', 'fix_enhancer_wide_end', "sequence_start","sequence_end"]
                   + [pred_col + "_ref",pred_col + "_alt", 'log2_fc_pred']]
                  .sort_values('log2_fc_pred'))
candidate_locs["abs_log2_fc_pred"] = np.abs(candidate_locs["log2_fc_pred"])
candidate_locs["candidate_type"] = candidate_locs["ziga_key"].apply(lambda x: "strong" if x in strong_candidates else ("intermediate" if x in intermediate_candidates else ("weak")))

# %%
candidate_locs["enhancer_size"] = candidate_locs["enhancer_end"] - candidate_locs["enhancer_start"]

# %%
n_total = 2_000_000
n_weak = 6
n_far = 6

print(len(candidate_locs["gene"].drop_duplicates()))
print(len(candidate_locs[["chromosome","enhancer_start"]].drop_duplicates()))
n_pred = len(candidate_locs["gene"].drop_duplicates()) * len(candidate_locs[["chromosome","enhancer_start"]].drop_duplicates())
print(n_pred)
print(n_total/(n_pred*6))

# %%
10235*32

# %%
test_locs = set(candidate_locs.sort_values("abs_log2_fc_pred",ascending=False).head(math.floor(n_total/(n_pred*6)) - n_weak - n_far)["ziga_key"])
test_locs = test_locs | set(candidate_locs.query('candidate_type == "weak"').head(n_weak)["ziga_key"])
test_locs = set(candidate_locs.query('ziga_key not in @test_locs').sort_values("actual_tss_distance",ascending=False).head(n_far)["ziga_key"]) | test_locs
candidate_locs["test_location"] = candidate_locs["ziga_key"].apply(lambda x: x in test_locs)

# %%
print(candidate_locs["validated"].describe())
print(candidate_locs.query('test_location')["actual_tss_distance"].describe())
print(((2**candidate_locs["log2_fc_pred"] - 1)*100).describe())
print(np.abs(((2**candidate_locs["log2_fc_pred"] - 1)*100)).describe())
print((2**candidate_locs[pred_col+"_alt"] - 1).describe())
print(np.sum(candidate_locs["log2_fc_pred"] > 0)/len(candidate_locs))
print(np.sum(candidate_locs.query('test_location')["log2_fc_pred"] > 0)/len(candidate_locs.query('test_location')))

# %%
(2**candidate_locs[pred_col+"_ref"] - 1).describe()

# %%
candidate_locs.query('test_location').groupby('gene').size().reset_index().rename(columns={0:"count"}).query('count > 1')

# %%
candidate_locs.to_csv(base_path_data + "in_silico_candidate_locs.tsv",sep="\t", index=None)

# %% [markdown]
# # Is Enformer multiplicative?
#
# In this in-silico experiment, we analyze whether Enformer follows a multiplicative logic to determine expression (background and enhancer scale an innate promoter strength). For this purpose, we identify promoter, enhancer and backgrounds which are such that enformer links the enhancer with the promoter. We then test every combination of promoter and enhancer (in 32 backgrounds) and assess how well a multiplicative model explains the results

# %%
base_path_data = "Data/Fulco_CRISPRi/"
base_path_results = "Results/Fulco_CRISPRi/"
base_path_data_gtex = "Data/GTEX/"

# %%
candidate_locs = pd.read_csv(base_path_data + "in_silico_candidate_locs.tsv",sep="\t").rename(columns={"ziga_key":"location_key"})
candidate_locs["enhancer_id"] = candidate_locs["chromosome"] + ":" + candidate_locs["fix_enhancer_wide_start"].astype('str') + "-" +  candidate_locs["fix_enhancer_wide_end"].astype('str')
#candidate_locs["main_tss_mid"] = (candidate_locs["main_tss_start"] + candidate_locs["main_tss_end"])//2
candidate_locs["promoter_id"] = candidate_locs["chromosome"] + ":" + candidate_locs["main_tss_start"].astype('str') + "-" +  candidate_locs["main_tss_end"].astype('str')

# %% [markdown]
# ## Analysis

# %%
pred_col = "CAGE:chronic myelogenous leukemia cell line:K562 ENCODE, biol__landmark_sum"

# %%
results = pd.read_csv(base_path_results + "fulco_in_fulco-enformer-latest_results.tsv",sep="\t")
results = results.groupby(["location_key","promoter_id","enhancer_id"])[[k for k in results.keys() if ("CAGE" in k) or ("DNASE" in k)]].mean().reset_index()

# %%
results["log_pred_col"] = np.log2(results[pred_col] + 1)


# %% [markdown]
# ### Promoter and multiplicative model for each location

# %%
def fit_and_predict(df,dep_var,indep_vars,model_name,id_name,id_col="location_key"):
    df = df[[dep_var] + indep_vars]
    res_enformer = (statsmodels.regression.linear_model.OLS
                    .from_formula("{} ~ {}".format(dep_var, " + ".join(indep_vars)),
                                  data=df)).fit()
    df["model"] = model_name
    df[id_col] = id_name 
    df["preds"] = res_enformer.predict()
    params = res_enformer.params.reset_index().rename(columns={"index":"Name",0:"param"})
    params["model"] = model_name
    params[id_col] = id_name
    r2 = sklearn.metrics.r2_score(df[dep_var], df["preds"])
    return df, params, r2


# %%
locations = set(results["location_key"])

rows = []
param_df_list_prom = []
param_df_list_mult = []
pred_df_list_prom = []
pred_df_list_mult = []
for loc in locations:
    subset = results.query('location_key == @loc')
    # promoter model
    prom_preds, prom_params, prom_r2 = fit_and_predict(subset,"log_pred_col",["promoter_id"],"promoter",loc)
    # promoter + enhancer model
    mult_preds, mult_params, mult_r2 = fit_and_predict(subset,"log_pred_col",["promoter_id","enhancer_id"],"multiplicative",loc)
    # metrics
    rows.append({"location_key":loc,
                 "r2_prom":prom_r2,
                 "r2_mult":mult_r2, 
                 "avg_expr":subset["log_pred_col"].mean(),
                 "total_std":subset["log_pred_col"].std(),
                 "median_std":subset.groupby('promoter_id')["log_pred_col"].std().median(),
                 "max_std":subset.groupby('promoter_id')["log_pred_col"].std().max()})
    # params
    param_df_list_prom.append(prom_params)
    param_df_list_mult.append(mult_params)
    # preds
    pred_df_list_prom.append(prom_preds)
    pred_df_list_mult.append(mult_preds)
    
var_explained = pd.DataFrame(rows).merge(candidate_locs[["location_key","log2_fc_pred","abs_log2_fc_pred","candidate_type","actual_tss_distance"]],on="location_key")
mult_params = pd.concat(param_df_list_mult)
mult_preds = pd.concat(pred_df_list_mult)

# %%
candidate_effects = candidate_locs[["enhancer_id","log2_fc_pred","actual_tss_distance"]].rename(columns={"enhancer_id":"Name"})
candidate_effects["Name"] = candidate_effects["Name"].apply(lambda x: "enhancer_id[T.{}]".format(x))
corr_with_ko = (mult_params.merge(candidate_effects,on="Name")
                .groupby('location_key')[['param','log2_fc_pred']].corr()
                .reset_index().query('level_1 == "param"')[["location_key","log2_fc_pred"]]
                .sort_values('log2_fc_pred').rename(columns={"log2_fc_pred":"corr_with_ko"}))

# %%
var_explained = var_explained.merge(corr_with_ko,on='location_key').sort_values('r2_prom')

# %% [markdown]
# #### Variance explained

# %%
var_explained_molten = var_explained[["location_key","r2_prom","r2_mult"]].melt(id_vars="location_key")
var_explained_molten["variable"] = var_explained_molten["variable"].apply(lambda x: "Promoter" if x =="r2_prom" else "Promoter +\nEnhancer")
var_explained_molten["location_key"] = pd.Categorical(var_explained_molten["location_key"],categories=[x for x in var_explained.sort_values("r2_prom")["location_key"]])

# %%
scale = 1.0
p9.options.figure_size = (6.4*scale, 4.8*scale)
p = (p9.ggplot(data=var_explained_molten,mapping=p9.aes(x="location_key",y="value",fill="variable", color="variable"))
 + p9.geom_point(size = 1.5)
 + p9.labs(x="Background", y="Variance Explained", fill="",color="")
 + p9.theme(legend_box_margin=0,legend_key_size =9, 
            legend_text = p9.element_text(size=9),
            legend_background=p9.element_blank(),
            legend_position=(0.7,0.3),
           axis_title=p9.element_text(size=10),
            axis_text_x=p9.element_blank(),
            #axis_ticks_major_x=p9.element_blank(),
            #axis_text_x=p9.element_text(rotation=90, hjust=1)
           )
    )
p

# %%
p.save("Graphics/" + "supfig_insilico_location" + ".svg", width=6.4/2, height=3.0, dpi=300)

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=var_explained
           ,mapping=p9.aes(x="avg_expr",y='r2_prom'))
 #+ p9.scale_y_log10()
 #+ p9.scale_x_log10()
 #+ p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset_name')
 + p9.geom_point()
 + p9.labs(x="Average predicted expression",y="Variance Explained by Promoter", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=var_explained
           ,mapping=p9.aes(x="max_std",y='r2_prom',color="candidate_type"))
 #+ p9.scale_y_log10()
 #+ p9.scale_x_log10()
 #+ p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset_name')
 + p9.geom_point()
 + p9.labs(x="Maximal st. dev. induced by enhancer",y="Variance Explained by Promoter", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            #legend_position=(0.21,0.2),
            legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=var_explained
           ,mapping=p9.aes(x="actual_tss_distance",y='r2_prom',color="candidate_type"))
 #+ p9.scale_y_log10()
 #+ p9.scale_x_log10()
 #+ p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset_name')
 + p9.geom_point()
 + p9.labs(x="Distance to TSS",y="Variance Explained by Promoter", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            #legend_position=(0.21,0.2),
            legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=var_explained
           ,mapping=p9.aes(x="actual_tss_distance",y='median_std',color="candidate_type"))
 #+ p9.scale_y_log10()
 #+ p9.scale_x_log10()
 #+ p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset_name')
 + p9.geom_point()
 + p9.labs(x="Distance to TSS",y="Median standard deviation per promoter", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            #legend_position=(0.21,0.2),
            legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=var_explained
           ,mapping=p9.aes(x="abs_log2_fc_pred",y='r2_prom',color="candidate_type"))
 #+ p9.scale_y_log10()
 #+ p9.scale_x_log10()
 #+ p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset_name')
 + p9.geom_point()
 + p9.labs(x="KO effect",y="Variance Explained by Promoter", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %% [markdown]
# #### What happens to the real promoter at each location?

# %%
real_prom = results.merge(candidate_locs[["location_key","promoter_id","log2_fc_pred","abs_log2_fc_pred",
                                          "candidate_type","actual_tss_distance",pred_col + "_ref",pred_col + "_alt"]],on=["location_key","promoter_id"])
real_prom["dev_from_ref"] = np.abs(real_prom[pred_col + "_ref"] - real_prom["log_pred_col"])

# %%
# sanity check
real_pair = real_prom.merge(candidate_locs[["location_key","enhancer_id","main_tss_bin"]],on=["location_key","enhancer_id"])
real_pair[["location_key","promoter_id","enhancer_id","dev_from_ref","actual_tss_distance",pred_col+"_ref","log_pred_col","main_tss_bin"]].sort_values("dev_from_ref").tail(5)

# %%
real_prom_mean = real_prom.groupby(['location_key','promoter_id'])[["log_pred_col"]].mean().reset_index()
real_prom_std = real_prom.groupby(['location_key','promoter_id'])[["log_pred_col"]].std().reset_index().rename(columns={"log_pred_col":"std"})
real_prom_merged = (real_prom[['location_key','promoter_id',"log2_fc_pred","abs_log2_fc_pred","candidate_type","actual_tss_distance",
                               pred_col + "_ref",pred_col + "_alt"]]
                    .drop_duplicates()
                    .merge(real_prom_mean.merge(real_prom_std,on=["location_key","promoter_id"]),on=["location_key","promoter_id"]))
real_prom_merged["dev_from_ref"] = np.abs(real_prom_merged[pred_col + "_ref"] - real_prom_merged["log_pred_col"])
real_prom_merged["dev_from_alt"] = np.abs(real_prom_merged[pred_col + "_alt"] - real_prom_merged["log_pred_col"])
real_prom_merged = real_prom_merged.merge(var_explained[["location_key","r2_prom"]], on="location_key")


# %%
# the deviation of the mean expression from the reference (the one with the real enhancer)
# if low: every enhancer works like the "real" one
# if high: no enhancer works like the "real" one
real_prom_merged.sort_values("dev_from_ref")

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=real_prom_merged
           ,mapping=p9.aes(x="actual_tss_distance",y='std',color="candidate_type"))
 #+ p9.scale_y_log10()
 #+ p9.scale_x_log10()
 #+ p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset_name')
 + p9.geom_point()
 + p9.labs(x="Distance to TSS",y="Standard deviation induced by enhancers", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            #legend_position=(0.21,0.2),
            legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=real_prom_merged#.query('actual_tss_distance > 50_000')
           ,mapping=p9.aes(x="log2_fc_pred",y='std',color="candidate_type"))
 #+ p9.scale_y_log10()
 #+ p9.scale_x_log10()
 #+ p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset_name')
 + p9.geom_point()
 + p9.labs(x="KO effect",y="Standard deviation induced by enhancers", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            #legend_position=(0.21,0.2),
            legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %% [markdown]
# #### Correlation of per-region enhancer effect with ko-effect

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=var_explained
           ,mapping=p9.aes(x="r2_prom",y='corr_with_ko'))
 #+ p9.scale_y_log10()
 #+ p9.scale_x_log10()
 #+ p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset_name')
 + p9.geom_point()
 + p9.labs(x="Variance Explained by Promoter",y="Correlation of enhancer effect with KO effect", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %% [markdown]
# #### Correlate promoter effects

# %%
# correlate parameters
params_pivot = mult_params[["Name","param","location_key"]].pivot('Name','location_key')
params_pivot.columns = params_pivot.columns.get_level_values(1)
params_pivot = params_pivot.reset_index()
prom_corr = params_pivot.loc[params_pivot.Name.str.startswith('prom')].corr()
enh_corr = params_pivot.loc[params_pivot.Name.str.startswith('enh')].corr()

# %%
sns_plot = sns.clustermap(prom_corr)

# %%
cluster_df = prom_corr.copy().reset_index()
cluster_df["location_key"] = pd.Categorical(cluster_df["location_key"], 
                                            categories=[x for x in var_explained.sort_values('avg_expr')["location_key"]])
cluster_df = cluster_df.sort_values("location_key")
cluster_df = cluster_df.set_index('location_key')
cluster_df = cluster_df[[x for x in var_explained.sort_values('avg_expr')["location_key"]]]

sns_plot = sns.clustermap(cluster_df,col_cluster=False,row_cluster=False)

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=params_pivot.loc[params_pivot.Name.str.startswith('prom')]
           ,mapping=p9.aes(x="fulco2019_NFE2_54286541-54287101",y='gasperini2019_ANXA9_150976845-150977758'))
 #+ p9.scale_y_log10()
 #+ p9.scale_x_log10()
 #+ p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset_name')
 + p9.geom_point()
 #+ p9.labs(x="",y="", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %%
x_loc = "fulco2019_NFE2_54286541-54287101"
y_loc = "gasperini2019_ANXA9_150976845-150977758"
loc_list = [x_loc,y_loc]

comparison = (results
              .query('location_key in @loc_list')
              [["location_key","promoter_id","enhancer_id","log_pred_col"]]
              .pivot(['enhancer_id','promoter_id'],'location_key')
             )
comparison.columns = comparison.columns.get_level_values(1)
comparison = comparison.reset_index()

scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=comparison
           ,mapping=p9.aes(x=x_loc,y=y_loc))
 #+ p9.scale_y_log10()
 #+ p9.scale_x_log10()
 #+ p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset_name')
 + p9.geom_point(size=0.5,alpha=0.2)
 #+ p9.labs(x="",y="", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %% [markdown]
# #### Correlate enhancer effects

# %%
cluster_df = enh_corr.copy().reset_index()
cluster_df["location_key"] = pd.Categorical(cluster_df["location_key"], 
                                            categories=[x for x in var_explained.sort_values('r2_prom')["location_key"]])
cluster_df = cluster_df.sort_values("location_key")
cluster_df = cluster_df.set_index('location_key')
cluster_df = cluster_df[[x for x in var_explained.sort_values('r2_prom')["location_key"]]]

sns_plot = sns.clustermap(cluster_df,col_cluster=False,row_cluster=False)

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=params_pivot.loc[params_pivot.Name.str.startswith('enh')]
           ,mapping=p9.aes(x="gasperini2019_RHAG_49646128-49646757",y='gasperini2019_CD69_9764660-9765179'))
 #+ p9.scale_y_log10()
 #+ p9.scale_x_log10()
 + p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset_name')
 + p9.geom_point(size=0.5)
 + p9.labs(x="",y="", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=params_pivot.loc[params_pivot.Name.str.startswith('enh')]
           ,mapping=p9.aes(x="gasperini2019_ANXA9_150976845-150977758",y='gasperini2019_RHAG_49646128-49646757'))
 #+ p9.scale_y_log10()
 #+ p9.scale_x_log10()
 + p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset_name')
 + p9.geom_point(size=0.5)
 + p9.labs(x="",y="", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=params_pivot.loc[params_pivot.Name.str.startswith('enh')]
           ,mapping=p9.aes(x="gasperini2019_ALAS2_55116629-55116926",y='gasperini2019_RHAG_49646128-49646757'))
 #+ p9.scale_y_log10()
 #+ p9.scale_x_log10()
 + p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset_name')
 + p9.geom_point(size=0.5)
 + p9.labs(x="",y="", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=params_pivot.loc[params_pivot.Name.str.startswith('enh')]
           ,mapping=p9.aes(x="gasperini2019_C3_6652567-6653052",y='gasperini2019_RHAG_49646128-49646757'))
 #+ p9.scale_y_log10()
 #+ p9.scale_x_log10()
 + p9.geom_smooth(method="lm")
 #+ p9.facet_wrap('~dataset_name')
 + p9.geom_point(size=0.5)
 + p9.labs(x="",y="", color="", fill="")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_position=(0.21,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10), subplots_adjust={'wspace': 0.05}
           )
)
p

# %% [markdown]
# #### Heatmap for "best" location

# %%

# %% [markdown]
# ### Location and Multplicative model for each Promoter

# %%
promoters = set(results["promoter_id"])

rows = []
param_df_list_loc = []
param_df_list_mult = []
pred_df_list_loc = []
pred_df_list_mult = []
for prom in promoters:
    subset = results.query('promoter_id == @prom')
    # promoter model
    loc_preds, loc_params, loc_r2 = fit_and_predict(subset,"log_pred_col",["location_key"],"location",prom,id_col="promoter_id")
    # promoter + enhancer model
    mult_preds, mult_params, mult_r2 = fit_and_predict(subset,"log_pred_col",["location_key","enhancer_id"],"multiplicative",prom,id_col="promoter_id")
    # metrics
    rows.append({"promoter_id":prom,
                 "r2_loc":loc_r2,
                 "r2_mult":mult_r2, 
                 "avg_expr":subset["log_pred_col"].mean(),
                 "total_std":subset["log_pred_col"].std(),
                 "median_std":subset.groupby('location_key')["log_pred_col"].std().median(),
                 "max_std":subset.groupby('location_key')["log_pred_col"].std().max()})
    # params
    param_df_list_loc.append(loc_params)
    param_df_list_mult.append(mult_params)
    # preds
    pred_df_list_loc.append(loc_preds)
    pred_df_list_mult.append(mult_preds)
    
var_explained_prom = pd.DataFrame(rows)
mult_params_prom = pd.concat(param_df_list_mult)
mult_preds_prom = pd.concat(pred_df_list_mult)

# %% [markdown]
# #### Variance explained

# %%
var_explained_prom_molten = var_explained_prom[["promoter_id","r2_loc","r2_mult"]].melt(id_vars="promoter_id")
var_explained_prom_molten["variable"] = var_explained_prom_molten["variable"].apply(lambda x: "Background" if x =="r2_loc" else "Background +\nEnhancer")
var_explained_prom_molten["promoter_id"] = pd.Categorical(var_explained_prom_molten["promoter_id"],categories=[x for x in var_explained_prom.sort_values("r2_loc")["promoter_id"]])

# %%
scale = 1.0
p9.options.figure_size = (6.4*scale, 4.8*scale)
p = (p9.ggplot(data=var_explained_prom_molten,mapping=p9.aes(x="promoter_id",y="value",fill="variable",color="variable"))
 #+ p9.geom_bar(stat="identity", position="dodge")
 + p9.geom_point(size = 1.5)
 + p9.labs(x="Promoter", y="Variance Explained", fill="",color="")
 + p9.theme(legend_box_margin=0,legend_key_size =9, 
            legend_text = p9.element_text(size=9),
            legend_background=p9.element_blank(),
            legend_position=(0.7,0.3),
           axis_title=p9.element_text(size=10),
            axis_text_x=p9.element_blank(),
            #axis_ticks_direction_x=p9.element_blank(),
            #axis_text_x=p9.element_text(rotation=90, hjust=1)
           )    
)
p

# %%
p.save("Graphics/" + "supfig_insilico_promoter" + ".svg", width=6.7/2, height=3.0, dpi=300)

# %% [markdown]
# #### Correlate enhancer effect

# %%
params_pivot_prom = mult_params_prom[["Name","param","promoter_id"]].pivot('Name','promoter_id')
params_pivot_prom.columns = params_pivot_prom.columns.get_level_values(1)
params_pivot_prom = params_pivot_prom.reset_index()
enh_corr_prom = params_pivot_prom.loc[params_pivot_prom.Name.str.startswith('enh')].corr()

# %%
cluster_df = enh_corr_prom.copy().reset_index()
cluster_df["promoter_id"] = pd.Categorical(cluster_df["promoter_id"], 
                                            categories=[x for x in var_explained_prom.sort_values('r2_loc')["promoter_id"]])
cluster_df = cluster_df.sort_values("promoter_id")
cluster_df = cluster_df.set_index('promoter_id')
cluster_df = cluster_df[[x for x in var_explained_prom.sort_values('r2_loc')["promoter_id"]]]

sns_plot = sns.clustermap(cluster_df,col_cluster=False,row_cluster=False)

# %% [markdown]
# ### Overall model: Promoter + Enhancer + Location

# %%
dep_var = "log_pred_col"
indep_vars = ["location_key"]

res_enformer = (statsmodels.regression.linear_model.OLS
                .from_formula("{} ~ {}".format(dep_var, " + ".join(indep_vars)),
                              data=results)).fit()
sklearn.metrics.r2_score(results[dep_var], res_enformer.predict())

# %%
dep_var = "log_pred_col"
indep_vars = ["promoter_id"]

res_enformer = (statsmodels.regression.linear_model.OLS
                .from_formula("{} ~ {}".format(dep_var, " + ".join(indep_vars)),
                              data=results)).fit()
sklearn.metrics.r2_score(results[dep_var], res_enformer.predict())

# %%
dep_var = "log_pred_col"
indep_vars = ["promoter_id","location_key"]

res_enformer = (statsmodels.regression.linear_model.OLS
                .from_formula("{} ~ {}".format(dep_var, " + ".join(indep_vars)),
                              data=results)).fit()
print(sklearn.metrics.r2_score(results[dep_var], res_enformer.predict()))

results["loc_prom_residuals"] = results["log_pred_col"] - res_enformer.predict()

# %%
dep_var = "log_pred_col"
indep_vars = ["promoter_id","enhancer_id","location_key"]

res_enformer = (statsmodels.regression.linear_model.OLS
                .from_formula("{} ~ {}".format(dep_var, " + ".join(indep_vars)),
                              data=results)).fit()
sklearn.metrics.r2_score(results[dep_var], res_enformer.predict())

# %%
res_enformer = (statsmodels.regression.linear_model.OLS
                .from_formula("log_pred_col ~ promoter_id:location_key",
                              data=results)).fit()
sklearn.metrics.r2_score(results[dep_var], res_enformer.predict())

# %% [markdown]
# ### Compute consistency of enhancer effect per promoter
#
# Here we remove the average effect per promoter and location (this is like the residual of a model with promoter:location interaction term)
#
# So all remaining variation should be caused by the enhancer
#
# We then check whether, holding the promoter fixed, does the same enhancer have similar effects across locations?

# %%
prom_pivot = results[["location_key","enhancer_id","promoter_id","log_pred_col"]]
prom_pivot = (prom_pivot.groupby(['location_key','promoter_id'])['log_pred_col']
                              .mean().reset_index().rename(columns={"log_pred_col":"loc_prom_mean"})
                              .merge(prom_pivot,on=['location_key','promoter_id'])
                             )
print(sklearn.metrics.r2_score(prom_pivot["log_pred_col"],prom_pivot["loc_prom_mean"]))
prom_pivot["loc_prom_residual"] = prom_pivot["log_pred_col"] - prom_pivot["loc_prom_mean"]
#prom_pivot = prom_pivot[["location_key","enhancer_id","promoter_id","loc_prom_residual"]].pivot(["location_key","enhancer_id"],"promoter_id")
#prom_pivot.columns = prom_pivot.columns.get_level_values(1)

# %%
res_enformer = (statsmodels.regression.linear_model.OLS
                .from_formula("loc_prom_residual ~ enhancer_id",
                              data=prom_pivot)).fit()
sklearn.metrics.r2_score(prom_pivot["loc_prom_residual"], res_enformer.predict())

# %%
res_enformer.summary()

# %%
prom_pivot = prom_pivot[["location_key","enhancer_id","promoter_id","loc_prom_residual"]].pivot(["location_key","enhancer_id"],"promoter_id")
prom_pivot.columns = prom_pivot.columns.get_level_values(1)

# %%
sns_plot = sns.clustermap(prom_pivot.corr())

# %% [markdown] tags=[]
# # What fraction of the sequence influences predictions? (GTEx + (eQTL) + Cardoso-Moreira et al. (Kaessmann lab))
#
# Here we:
#
# - Analyze the performance of Xpresso, Basenji2 and Enformer on the GTEx data
# - Analyze to what extent proximal and distal sequence contribute
# - Test Enformer on GTEx eQTL

# %%
base_path_data = "Data/GTEX/"
base_path_data_kaessmann = "Data/Kaessmann/"
base_path_data_gtex_fc = "Data/gtex_aFC/"
base_path_data_tss_sim = "Data/TSS_sim/"
base_path_data_fulco = "Data/Fulco_CRISPRi/"
base_path_results = "Results/TSS_sim/"
base_path_results_xpresso = "Results/TSS_Xpresso/"
base_path_results_gtex_fc = "Results/gtex_aFC/"

# %%
# load gtex data
gtex_df = (pd.read_csv(base_path_data + "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct", sep="\t")
           .rename(columns={'Name':'gene_id','Description':'gene_name'})
          )
tissues = gtex_df.keys()[2:]
gtex_df = gtex_df.loc[~gtex_df.gene_id.str.endswith("PAR_Y")]
gtex_molten = gtex_df.melt(id_vars=['gene_id','gene_name'], var_name="tissue", value_name='observed_tpm')
gtex_molten['gene_id'] = gtex_molten['gene_id'].apply(lambda x: x.split('.')[0])

# remove genes which are never expressed (otherwise removing colMean will create phantom deviation for these)
nonexpressed_genes = set(gtex_molten.groupby('gene_id')["observed_tpm"].max().reset_index().query('observed_tpm == 0')["gene_id"])
gtex_molten = gtex_molten.query('gene_id not in @nonexpressed_genes')

# %%
# load kaessmann data
kaessmann_df = (pd.read_csv(base_path_data_kaessmann + "Human.RPKM.txt", sep=" ")
                .reset_index()
           .rename(columns={'index':'gene_id'})
          )
dev_tissues = kaessmann_df.keys()[1:]
kaessmann_molten = kaessmann_df.melt(id_vars=['gene_id'], var_name="sample", value_name='observed_tpm')
kaessmann_molten['gene_id'] = kaessmann_molten['gene_id'].apply(lambda x: x.split('.')[0])

# remove genes which are never expressed (otherwise removing colMean will create phantom deviation for these)
nonexpressed_genes = (kaessmann_molten.groupby('gene_id')["observed_tpm"].max().reset_index().query('observed_tpm == 0')["gene_id"])
kaessmann_molten = kaessmann_molten.query('gene_id not in @nonexpressed_genes')

# %%
test_set = pd.read_csv(base_path_data + "human_regions.bed", names=["Chromosome","Start","End","set"], sep="\t").query('set == "test"')

# %%
length_df = gtf_df.df.query('Feature == "exon"')
length_df["len"] = length_df["End"] - length_df["Start"]
length_df = length_df.groupby(['gene_id','transcript_id'])['len'].sum().reset_index()
length_df = length_df.groupby(['gene_id'])['len'].median().reset_index()

# %%
# compute length for each gene
gene_locs = gtf_df.df.query('Feature == "gene"')[["Chromosome","Start","End","Score","Strand","Frame","gene_id"]]
gene_locs = gene_locs.merge(length_df, on="gene_id")
gene_locs = gene_locs.loc[~gene_locs.gene_id.str.endswith("PAR_Y")]
gene_locs["gene_id"] = gene_locs["gene_id"].apply(lambda x: x.split('.')[0])

# %%
# remove y and MT chromosome from gtex and kaessmann
y_mt_chrom_genes = set(gene_locs.query('Chromosome == "chrY" or Chromosome == "chrMT"')["gene_id"])
gtex_molten = gtex_molten.query('gene_id not in @y_mt_chrom_genes')
kaessmann_molten = kaessmann_molten.query('gene_id not in @y_mt_chrom_genes')


# %%
# test genes are those which are fully contained in a test region
test_genes = set(test_set.merge(gene_locs, on="Chromosome",suffixes=("","_gene"))
                 .query('Start < Start_gene & End > End_gene')["gene_id"])

# %%
# train genes are all those which do not at all intersect a test region
intersect_genes = set(test_set.merge(gene_locs, on="Chromosome",suffixes=("","_gene"))
                      .query('(Start < Start_gene & End > Start_gene) or (Start < End_gene & End > End_gene)')["gene_id"])


# %%
# sanity check: ensure that no test gene touches a train region
#non_test_set = pd.read_csv(base_path_data + "human_regions.bed", names=["Chromosome","Start","End","set"], sep="\t").query('set != "test"')
#train_genes = set(non_test_set.merge(gene_locs, on="Chromosome",suffixes=("","_gene"))
#                  .query('(Start < Start_gene & End > Start_gene) | (Start < End_gene & End > End_gene)')["gene_id"])
#assert len(train_genes & test_genes) == 0

# another way to do it
#assert len(set(non_test_set.merge(gene_locs.query('gene_id in @test_genes'), on="Chromosome",suffixes=("","_gene"))
#                  .query('(Start < Start_gene & End > Start_gene) | (Start < End_gene & End > End_gene)')["gene_id"])) == 0

# %%
def fit_and_pred(lm, X_train, X_test, y_train, y_test):
    lm.fit(X_train, y_train)
    y_pred_train = lm.predict(X_train).reshape(-1)
    y_pred_test = lm.predict(X_test).reshape(-1)
    return lm, y_pred_train, y_pred_test

def null_model_pred(y_train, y_test, subset_train, subset_test, covariates):
    if len(covariates) > 0:
        X_train = subset_train[covariates]
        X_test = subset_test[covariates]
        lm_null = sklearn.linear_model.LinearRegression()
    else:
        X_train = np.ones(len(y_train))[:,np.newaxis]
        X_test = np.ones(len(y_test))[:,np.newaxis]
        lm_null = sklearn.linear_model.LinearRegression(fit_intercept=False)
    lm_null, y_pred_train_null, y_pred_test_null = fit_and_pred(lm_null, X_train, X_test, y_train, y_test)
    return np.concatenate([y_pred_train_null,y_pred_test_null])


def linearmodel_match_windowed(gene_df, obs_df, samples, sample_col, locs_df, windows, 
                               model_name, dataset_name,
                               lm = sklearn.linear_model.Lasso(alpha=0.0004), print_alpha=False,
                               remove_mean=True, covariates=[]):
    import warnings
    warnings.filterwarnings("ignore")

    windows = [str(x) if x != -1 else "ingenome" for x in windows]
    cage_tracks = [k for k in gene_df.keys() if "CAGE" in k]
    
    
    rows = []
    predictions = []
    model_dict = collections.defaultdict(dict)
    for sample in set(samples):
        print(sample)
        # prepare data
        subset = (obs_df.query('{} == @sample'.format(sample_col))
                  .merge(locs_df[['gene_id','len']],on="gene_id")
                  .merge(gene_df.drop(columns="gene_name"), on=["gene_id"]))
        subset = subset.query('(gene_id in @ test_genes) or (gene_id not in @intersect_genes)')
        subset['log_obs'] = np.log10(subset['observed_tpm']+1)
        subset["expressed"] = subset["log_obs"] > 0
        subset['log_len'] = np.log10(subset['len'])
        subset[[k for k in subset.keys() if "CAGE" in k]] = np.log10(subset[[k for k in subset.keys() if "CAGE" in k]]+1)
        subset_train = subset.query('gene_id not in @intersect_genes')
        subset_test = subset.query('gene_id in @test_genes')
        if remove_mean:
            mean = subset_train['log_obs'].mean()
            subset_train['log_obs'] = subset_train['log_obs'] - mean
            subset_test['log_obs'] = subset_test['log_obs'] - mean
        subset_pred = pd.concat([subset_train[["gene_id",sample_col,"expressed","log_obs"]],subset_test[["gene_id",sample_col,"expressed","log_obs"]]])
        y_train = np.array(subset_train['log_obs'])
        y_test = np.array(subset_test['log_obs'])
        print(np.mean(y_train))
        print(np.mean(y_test))
        subset_pred['log_pred_null'] = null_model_pred(y_train, y_test, subset_train, subset_test, covariates)
        for wdw in windows:
            wdw_cage_tracks = [x for x in cage_tracks if wdw in x] + covariates
            X_train = np.array(subset_train[wdw_cage_tracks])
            X_test = np.array(subset_test[wdw_cage_tracks])
            lm_pipe = pipeline.Pipeline([('scaler', sklearn.preprocessing.StandardScaler()),
                                       ('model', lm)])
            lm_pipe, y_pred_train, y_pred_test = fit_and_pred(lm_pipe, X_train, X_test, y_train, y_test)
            model_dict[sample][wdw] = lm_pipe
            tpl = (y_test, y_pred_test)
            r = scipy.stats.pearsonr(*tpl)[0]
            print(r)
            r2 = sklearn.metrics.r2_score(*tpl)
            print(r2)
            if print_alpha:
                print(lm_pipe['model'].alpha_)
            rho = scipy.stats.spearmanr(*tpl)[0]
            rows.append({sample_col:sample,
                         "window_size":wdw,
                         "r":r,
                         "R2":r2,
                         "RMSE":sklearn.metrics.mean_squared_error(*tpl,squared=False),
                         "rho":rho})
            subset_pred['log_pred_'+wdw] = np.concatenate([y_pred_train,y_pred_test])
        predictions.append(subset_pred)
            

    corrs_lm = pd.DataFrame(rows)
    corrs_lm["window_size"] = corrs_lm["window_size"].apply(lambda x: x if x != "ingenome" else "Full")
    #corrs_lm["window_size"] = pd.Categorical(corrs_lm["window_size"], [x for x in windows if x != "ingenome"] + ["Full"])
    corrs_lm.to_csv(base_path_results + dataset_name + "_corrs_modelmatched_{}.tsv".format(model_name), index=None, sep="\t")

    with open(base_path_results + dataset_name + "_{}_lm_models_pseudocount1.pkl".format(model_name), 'wb') as handle:
        pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    pred_df = pd.concat(predictions)
    pred_df.to_csv(base_path_results + dataset_name + "_pred_modelmatched_{}.tsv".format(model_name), index=None, sep="\t")

# %% [markdown]
# ## Prepare eQTL data

# %% [markdown]
# ### Prepare SuSiE credible sets
#
# Filter criteria:
#
# - The credible set is small (at most 5k)
# - Whole credible set can be scored by Enformer (furthest variant is within ~100k of canonical TSS)
# - Whole credible set is 5' of the whole gene (no splicing, no NMD)
# - No other protein coding transcript is 5' of the canonical one

# %%
susie_to_gtex = {'LCL':'Cells - EBV-transformed lymphocytes',
                 'adipose_subcutaneous':'Adipose - Subcutaneous',
                 'adipose_visceral':'Adipose - Visceral (Omentum)',
                 'adrenal_gland':'Adrenal Gland',
                 'artery_aorta':'Artery - Aorta',
                 'artery_coronary':'Artery - Coronary',
                 'artery_tibial':'Artery - Tibial',
                 'blood':'Whole Blood',
                 'brain_amygdala':'Brain - Amygdala',
                 'brain_anterior_cingulate_cortex':'Brain - Anterior cingulate cortex (BA24)',
                 'brain_caudate':'Brain - Caudate (basal ganglia)',
                 'brain_cerebellar_hemisphere':'Brain - Cerebellar Hemisphere',
                 'brain_cerebellum':'Brain - Cerebellum',
                 'brain_cortex':'Brain - Cortex',
                 'brain_frontal_cortex':'Brain - Frontal Cortex (BA9)',
                 'brain_hippocampus':'Brain - Hippocampus',
                 'brain_hypothalamus':'Brain - Hypothalamus',
                 'brain_nucleus_accumbens':'Brain - Nucleus accumbens (basal ganglia)',
                 'brain_putamen':'Brain - Putamen (basal ganglia)',
                 'brain_spinal_cord':'Brain - Spinal cord (cervical c-1)',
                 'brain_substantia_nigra':'Brain - Substantia nigra',
                 'breast':'Breast - Mammary Tissue',
                 'colon_sigmoid':'Colon - Sigmoid',
                 'colon_transverse':'Colon - Transverse',
                 'esophagus_gej':'Esophagus - Gastroesophageal Junction',
                 'esophagus_mucosa':'Esophagus - Mucosa',
                 'esophagus_muscularis':'Esophagus - Muscularis',
                 'fibroblast':'Cells - Cultured fibroblasts',
                 'heart_atrial_appendage':'Heart - Atrial Appendage', 
                 'heart_left_ventricle':'Heart - Left Ventricle', 
                 'kidney_cortex':'Kidney - Cortex',
                 'liver':'Liver', 
                 'lung':'Lung', 
                 'minor_salivary_gland':'Minor Salivary Gland',
                 'muscle':'Muscle - Skeletal', 
                 'nerve_tibial':'Nerve - Tibial', 
                 'ovary':'Ovary', 
                 'pancreas':'Pancreas', 
                 'pituitary':'Pituitary',
                 'prostate':'Prostate', 
                 'skin_not_sun_exposed':'Skin - Not Sun Exposed (Suprapubic)',
                 'skin_sun_exposed':'Skin - Sun Exposed (Lower leg)', 
                 'small_intestine':'Small Intestine - Terminal Ileum',
                 'spleen':'Spleen', 
                 'stomach':'Stomach', 
                 'testis':'Testis', 
                 'thyroid':'Thyroid', 
                 'uterus':'Uterus', 
                 'vagina':'Vagina',}

# %%
canonical_tss = pd.read_csv(base_path_data_tss_sim + 'only_protein_coding_ensembl_canonical_tss.tsv',sep="\t").rename(columns={"ts_id":"transcript_id"})
canonical_tss['tss'] = canonical_tss['tss'] - 1

all_tss = pd.read_csv('Data/TSS_sim/bigly_tss_from_biomart.txt',sep="\t").rename(columns={"Transcript stable ID":"transcript_id",
                                                                                         "Gene stable ID":"gene_id",
                                                                                         "Transcription start site (TSS)":"tss_noncanonical",
                                                                                         "Chromosome/scaffold name":"chromosome",
                                                                                         "Transcript type":"transcript_type"})
all_tss["chromosome"] = "chr" + all_tss["chromosome"].astype('str')
all_tss["tss_noncanonical"] = all_tss["tss_noncanonical"] - 1
all_tss = all_tss.query('transcript_type == "protein_coding"')

canonical_tss = canonical_tss.merge(all_tss[["gene_id","transcript_id","Strand"]],on=["gene_id","transcript_id"])

# throw out genes which have another protein coding TSS 5' of the canonical TSS

noncanonical_tss = all_tss[["gene_id","transcript_id","chromosome","tss_noncanonical","Strand"]].merge(canonical_tss[["gene_id","tss"]],on=["gene_id"])
# distance > 0 means ==TSS===tss==, where small letters denotes noncanonical
noncanonical_tss["distance"] = noncanonical_tss["tss_noncanonical"] - noncanonical_tss["tss"]
noncanonical_tss["abs_distance"] = np.abs(noncanonical_tss["distance"])
noncanonical_tss = noncanonical_tss.query('distance != 0')
noncanonical_tss["sign_of_dist"] = np.sign(noncanonical_tss["distance"])
# genes which have a protein coding transcript (significantly: >1 bin) 5' of the canoncial transcript
# if distance > 0 and strand = -1, then we have ==tss===TSS==
unclear_genes = set(noncanonical_tss.query('sign_of_dist != Strand & abs_distance > 128')["gene_id"])

# %%
susie_paths = glob.glob(base_path_data_gtex_fc + "susie_credible_sets/GTEx_ge*.txt.gz")
susie_df_list = []
for path in susie_paths:
    susie_df = pd.read_csv(path, sep="\t").rename(columns={"molecular_trait_id":"gene_id","variant":"variant_id"})
    susie_df["tissue"] = path.split("/")[-1].split('.')[0][8:]
    susie_df_list.append(susie_df)

# %%
susie_df = pd.concat(susie_df_list)

susie_df = susie_df.merge(canonical_tss[["gene_id","tss","Strand"]],on="gene_id")
susie_df["tss_distance"] = (susie_df["position"]-1) - susie_df["tss"]
susie_df["abs_tss_distance"] = np.abs(susie_df["tss_distance"])
# if tss_distance > 0, then ==TSS===VAR==
susie_df["sign_of_distance"] = np.sign(susie_df["tss_distance"])
# end_distance is the distance between the last base of the reference allele and the TSS
susie_df['ref_len'] = susie_df['ref'].str.len()
susie_df['end_distance'] = (susie_df["position"]-1) + (susie_df['ref_len']-1) - susie_df["tss"]
# if this has a different sign than the distance to the variant start, the variant touches the tss
susie_df['crossing_variant'] = (np.sign(susie_df['end_distance']) != np.sign(susie_df['tss_distance'])).astype('int')

susie_gene_blocks = (susie_df
                     .groupby(['cs_id','gene_id','tissue'])[["position","tss_distance","abs_tss_distance","Strand","sign_of_distance","crossing_variant"]]
                     .agg({"position":["max","min"],
                           "tss_distance":["min","max","mean"],
                           "abs_tss_distance":["min","max","mean"],
                           "sign_of_distance":["min","max"],
                           "crossing_variant":["max"],
                           "Strand":["min","max"]},axis="columns")
                     .reset_index()
                    )
susie_gene_blocks.columns = ([x for x,y in zip(susie_gene_blocks.columns.get_level_values(0),susie_gene_blocks.columns.get_level_values(1)) if y == ""]
                   + [x + "_" + str(y) for x,y in zip(susie_gene_blocks.columns.get_level_values(0),susie_gene_blocks.columns.get_level_values(1)) if y != ""])
assert np.sum(susie_gene_blocks["Strand_min"] == susie_gene_blocks["Strand_max"]) == len(susie_gene_blocks)
susie_gene_blocks["block_size"] = susie_gene_blocks["position_max"] - susie_gene_blocks["position_min"]
 
# throw out blocks which are 3' of the canonical TSS
# if tss_distance > 0, then ==TSS===VAR==
# so if strand = 1, we want TSS distance to be negative: ==VAR==TSS[CDS]
# else, if strand = -1, we want TSS distance to be positive: [CDS]TSS==VAR==
# we also exclude blocks with inconsistent sign (i.e. blocks which span the TSS)
susie_gene_blocks = susie_gene_blocks.query('Strand_min != sign_of_distance_min & sign_of_distance_min == sign_of_distance_max')

# throw out blocks of genes which have another 5' protein coding transcript
susie_gene_blocks = susie_gene_blocks.query('gene_id not in @unclear_genes')

# limit to small blocks Enformer can score and throw out blocks that hit the TSS
susie_gene_blocks = susie_gene_blocks.query('abs_tss_distance_max < 95_000 & abs_tss_distance_min > 0 & crossing_variant_max < 1 & block_size < 5_000')

susie_df_small_blocks = susie_df.merge(susie_gene_blocks,on=["cs_id", 'gene_id', "tissue"])

susie_df_small_blocks["tissue"] = susie_df_small_blocks["tissue"].apply(lambda x: susie_to_gtex[x])

# %%
susie_gene_blocks['abs_tss_distance_min'].describe()

# %%
susie_df_small_blocks

# %%
susie_df_small_blocks.to_csv(base_path_data_gtex_fc + "susie_df_small_blocks.tsv",sep="\t",index=None)

# %% [markdown]
# ### Intersect with CRE

# %%
cre_pr = pr.read_bed(base_path_data_gtex_fc + "GRCh38-cCREs.bed")

# %%
susie_df_small_blocks["chr_chrom"] = "chr" + susie_df_small_blocks["chromosome"].astype('str')
susie_df_small_blocks["actual_start"] = susie_df_small_blocks["position"] - 1
susie_df_small_blocks["actual_end"] = susie_df_small_blocks["position"] - 1 + susie_df_small_blocks["ref_len"]

susie_pr = (susie_df_small_blocks[["chr_chrom","actual_start","actual_end","variant_id","gene_id","cs_id","tissue","tss_distance","abs_tss_distance","tss"]]
            .rename(columns={"chr_chrom":"Chromosome","actual_start":"Start", "actual_end":"End"}))
susie_pr = pr.PyRanges(susie_pr)

# %%
cre_merged_df = susie_pr.join(cre_pr,strandedness=False).df.rename(columns={"Strand":"cre_type"})

# %%
susie_df_small_blocks

# %% [markdown]
# ### Look at effect size distribution

# %%
gtex_eqtl_paths = glob.glob(base_path_data_gtex_fc + "GTEx_Analysis_v8_eQTL/*pairs.txt.gz")
gtex_eqtl_df_list = []
for path in gtex_eqtl_paths:
    gtex_eqtl_df = pd.read_csv(path, sep="\t")
    gtex_eqtl_df['gene_id'] = gtex_eqtl_df['gene_id'].apply(lambda x: x.split('.')[0])
    gtex_eqtl_df["tissue"] = path.split("/")[-1].split('.')[0]
    gtex_eqtl_df_list.append(gtex_eqtl_df)

# %%
gtex_eqtl_df = pd.concat(gtex_eqtl_df_list)
gtex_eqtl_df["variant_id"] = gtex_eqtl_df["variant_id"].apply(lambda x: x[:-4]) 

gtex_to_gtex = {}
for tissue in set(gtex_eqtl_df["tissue"]):
    tissue_split = tissue.split("_")
    for real_tissue in tissues:
        real_tissue_split = [y for y in real_tissue.replace("(","").replace(")","").split(" ") if y != "-"] 
        if all([x == y for x,y in zip(tissue_split,real_tissue_split)]):
            gtex_to_gtex[tissue] = real_tissue
    if tissue not in gtex_to_gtex:
        assert False
        
gtex_eqtl_df["tissue"] = gtex_eqtl_df["tissue"].apply(lambda x: gtex_to_gtex[x]) 

# %%
gtex_eqtl_df_merged = gtex_eqtl_df[['variant_id','gene_id','slope','slope_se','tissue']].merge(susie_df_small_blocks, on=["gene_id","variant_id","tissue"])
gtex_eqtl_df_merged["abs_slope"] = np.abs(gtex_eqtl_df_merged["slope"])
gtex_eqtl_df_merged = gtex_eqtl_df_merged.merge(gtex_eqtl_df_merged.groupby(['cs_id','tissue'])["abs_slope"]
                                                  .max()
                                                  .reset_index()
                                                  .rename(columns={"abs_slope":"max_slope"}),
                                                  on=["cs_id",'tissue'])

# %%
effect_by_dist = gtex_eqtl_df_merged[['cs_id','tissue','abs_tss_distance_min','max_slope']].drop_duplicates()

# %%
effect_by_dist.to_csv(base_path_data_gtex_fc + "effect_by_dist.tsv",sep="\t",index=None)

# %% [markdown]
# ## Prepare CRE data
#
# - Find all CRE within range (+/- (196/2)kb) of canonical TSS
# - Exclude MT and Y chromosome
# - Exclude CRE that intersect the TSS
# - Note their location

# %%
cre_pr = pr.read_bed(base_path_data_gtex_fc + "GRCh38-cCREs.bed")

# %%
canonical_tss = pd.read_csv(base_path_data_tss_sim + 'only_protein_coding_ensembl_canonical_tss.tsv',sep="\t").rename(columns={"ts_id":"transcript_id"})
canonical_tss['tss'] = canonical_tss['tss'] - 1
gene_chr = gtf_df.df[["Chromosome","gene_id"]].drop_duplicates()
gene_chr["gene_id"] = gene_chr["gene_id"].apply(lambda x: x.split('.')[0])
canonical_tss = canonical_tss.merge(gene_chr, on="gene_id")
canonical_tss = canonical_tss.query('Chromosome != "chrY" and Chromosome != "chrMT"')

canonical_tss["Start"] = canonical_tss["tss"] - (SEEN_SEQUENCE_LENGTH)/2
canonical_tss["End"] = canonical_tss["tss"] + (SEEN_SEQUENCE_LENGTH)/2
canonical_tss_pr = pr.PyRanges(canonical_tss)

# %%
cre_joined = canonical_tss_pr.join(cre_pr,strandedness=False,suffix="_cre",report_overlap=True).df

# %%
cre_joined["cre_len"] = cre_joined["End_cre"] - cre_joined["Start_cre"]

# %%
# remove cre that intersect the TSS
intersects = cre_joined.loc[(np.sign(cre_joined["Start_cre"] - cre_joined["tss"]) != np.sign(cre_joined["End_cre"] - cre_joined["tss"]))]
cre_joined = cre_joined.loc[(np.sign(cre_joined["Start_cre"] - cre_joined["tss"]) == np.sign(cre_joined["End_cre"] - cre_joined["tss"]))]

# %%
cre_joined["abs_tss_distance_mean"] = np.abs((cre_joined["End_cre"] + cre_joined["Start_cre"])/2 - cre_joined["tss"])

# %%
cre_joined.groupby('gene_id').size().describe()

# %%
cre_joined["cre_len"].describe()

# %%
379*350

# %%
267*112

# %%
cre_joined.rename(columns={"Strand":"cre_type"}).to_csv(base_path_data_tss_sim + "cre_positions.tsv",sep="\t")

# %% [markdown] tags=[]
# ## Analysis - GTEx

# %%
with open(base_path_results  + "tss_sim-enformer-latest_results.tsv", 'r') as tsv:
    header = tsv.readline()
    cols = header.split("\t")
cols = [k for k in cols if "CAGE" in k and k.endswith("landmark_sum")]

results = pd.read_csv(base_path_results + "tss_sim-enformer-latest_results.tsv",sep="\t",
                     usecols = ['gene_id', 'gene_name', "transcript_id","window_type","window_size"] + cols)

# %%
results = (results.groupby(["gene_id","gene_name","window_size"])[[x for x in results.keys() if "CAGE" in x]].mean().reset_index())

# %%
windows = sorted(list(set(results["window_size"])))

# %% [markdown] tags=[]
# ### K562 - Across genes

# %%
pred_col = 'CAGE:chronic myelogenous leukemia cell line:K562 ENCODE, biol__landmark_sum'

# %%
k562_df = results[["gene_id","gene_name","window_size",pred_col]]
k562_df = k562_df[["gene_id","gene_name","window_size",pred_col]].pivot(["gene_id","gene_name"],["window_size"],[pred_col]).reset_index()
k562_df.columns = ([x for x,y in zip(k562_df.columns.get_level_values(0),k562_df.columns.get_level_values(1)) if y == ""]
                   + [x + "_" + str(y) for x,y in zip(k562_df.columns.get_level_values(0),k562_df.columns.get_level_values(1)) if y != ""])
k562_df = k562_df.rename(columns={pred_col+"_-1":pred_col+"_ingenome"})

# %%
rows = []
for wdw in windows:
    if wdw == -1:
        continue
    tpl = (np.log2(k562_df[pred_col+"_ingenome"]+1),np.log2(k562_df[pred_col+"_"+str(wdw)]+1))
    r = scipy.stats.pearsonr(*tpl)[0]
    rows.append({
        "window_size": wdw,
        "R": r,
        "R-squared": sklearn.metrics.r2_score(*tpl),
        "R-squared (after Rescaling)": r ** 2,
        "Explained Variance": sklearn.metrics.explained_variance_score(*tpl)
    })
k562_corrs = pd.DataFrame(rows)
k562_corrs["window_size"] = pd.Categorical(k562_corrs["window_size"], k562_corrs["window_size"])

# %%
k562_corrs

# %%
(p9.ggplot(data=k562_corrs,mapping=p9.aes(x="window_size",y="Explained Variance"))
 + p9.geom_bar(stat="identity")
 + p9.labs(x="Size of sequence window around TSS (bp), max = 196608",y="Variance in Enformer log2(CAGE) predictions\nexplained by sequence window")
)

# %%
(p9.ggplot(data=k562_df,mapping=p9.aes(x=pred_col+"_1001",y=pred_col+"_ingenome"))
 + p9.geom_bin2d(binwidth = (0.05, 0.1))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_abline()
 + p9.geom_smooth(method="lm")
 + p9.labs(x="K562 CAGE predicition, with 1000 bp of sequence",y="K562 CAGE predicition")
)

# %%
(p9.ggplot(data=k562_df,mapping=p9.aes(x=pred_col+"_3001",y=pred_col+"_ingenome"))
 + p9.geom_bin2d(binwidth = (0.05, 0.1))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_smooth(method="lm")
 + p9.labs(x="K562 CAGE predicition, with 3000 bp of sequence",y="K562 CAGE predicition")
)

# %%
(p9.ggplot(data=k562_df,mapping=p9.aes(x=pred_col+"_39321",y=pred_col+"_ingenome"))
 + p9.geom_bin2d(binwidth = (0.05, 0.1))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_smooth(method="lm")
 + p9.labs(x="K562 CAGE predicition, with 1/5-th of the input sequence",y="K562 CAGE predicition")
)

# %%
(p9.ggplot(data=k562_df,mapping=p9.aes(x=pred_col+"_65537",y=pred_col+"_ingenome"))
 + p9.geom_bin2d(binwidth = (0.05, 0.1))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_smooth(method="lm")
 + p9.labs(x="K562 CAGE predicition, with 1/3-rd of the input sequence",y="K562 CAGE predicition")
)

# %%
(p9.ggplot(data=k562_df,mapping=p9.aes(x=pred_col+"_98305",y=pred_col+"_ingenome"))
 + p9.geom_bin2d(binwidth = (0.05, 0.1))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_smooth(method="lm")
 + p9.labs(x="K562 CAGE predicition, with 1/2 of the input sequence",y="K562 CAGE predicition")
)

# %% [markdown]
# ### Across Genes

# %%
universal_samples = ['Clontech Human Universal Reference Total RNA', 'SABiosciences XpressRef Human Universal Total RNA', 'CAGE:Universal RNA - Human Normal Tissues Biochain']
track_cols = [k for k in results.keys() if "CAGE" in k and k.endswith("landmark_sum") and not any(x in k for x in universal_samples)]

gene_df = results[["gene_id","gene_name","window_size"] + track_cols]
gene_df = gene_df[["gene_id","gene_name","window_size"]+track_cols].pivot(["gene_id","gene_name"],["window_size"],track_cols).reset_index()
gene_df.columns = ([x for x,y in zip(gene_df.columns.get_level_values(0),gene_df.columns.get_level_values(1)) if y == ""]
                   + [x + "_" + str(y) for x,y in zip(gene_df.columns.get_level_values(0),gene_df.columns.get_level_values(1)) if y != ""])
gene_df = gene_df.rename(columns={k+"_-1":k+"_ingenome" for k in track_cols})

# %%
rows = []
for wdw in windows:
    if wdw == -1:
        continue
    for col in track_cols:
        tpl = (np.log2(gene_df[col+"_ingenome"]+1),np.log2(gene_df[col+"_"+str(wdw)]+1))
        r = scipy.stats.pearsonr(*tpl)[0]
        rows.append({
            "track_name":col,
            "window_size": wdw,
            "R": r,
            "R-squared": sklearn.metrics.r2_score(*tpl),
            "R-squared (after Rescaling)": r ** 2,
            "Explained Variance": sklearn.metrics.explained_variance_score(*tpl)
        })
gene_corrs = pd.DataFrame(rows)
gene_corrs["window_size"] = pd.Categorical(gene_corrs["window_size"], [x for x in windows if x != -1])

# %%
(p9.ggplot(data=gene_corrs,mapping=p9.aes(x="window_size",y="Explained Variance"))
 + p9.geom_boxplot()
 + p9.labs(x="Size of sequence window with TSS at the center (bp, max = 196608)",y="Variance in Enformer log2(CAGE) predictions\nexplained by sequence window")
)

# %%
gene_corrs.groupby('window_size').median()

# %% [markdown]
# ### GTEX - Across genes, matching with linear model

# %% [markdown]
# #### Train linear model for every tissue and window size and gather predictions

# %% [markdown]
# ##### Enformer

# %%
#with open(base_path_results  + "tss_sim_groupby_20_05_2022.tsv", 'r') as tsv:
#    header = tsv.readline()
#    cols = header.split("\t")
#cols = [k for k in cols if "CAGE" in k and k.endswith("landmark_sum")]

#results = pd.read_csv(base_path_results + "tss_sim_groupby_20_05_2022.tsv",sep="\t",
#                     usecols = ['gene_id', 'gene_name', "transcript_id","window_type","window_size"] + cols)

with open(base_path_results  + "tss_sim-enformer-latest_results.tsv", 'r') as tsv:
    header = tsv.readline()
    cols = header.split("\t")
cols = [k for k in cols if "CAGE" in k and k.endswith("landmark_sum")]

results = pd.read_csv(base_path_results + "tss_sim-enformer-latest_results.tsv",sep="\t",
                     usecols = ['gene_id', 'gene_name', "transcript_id","window_type","window_size"] + cols)

results = (results.groupby(["gene_id","gene_name"])[[x for x in results.keys() if "CAGE" in x]].mean().reset_index())

universal_samples = ['Clontech Human Universal Reference Total RNA', 'SABiosciences XpressRef Human Universal Total RNA', 'CAGE:Universal RNA - Human Normal Tissues Biochain']
track_cols = [k for k in results.keys() if "CAGE" in k and k.endswith("landmark_sum") and not any(x in k for x in universal_samples)]

gene_df = results[["gene_id","gene_name","window_type","window_size"] + track_cols]
gene_df = gene_df[["gene_id","gene_name","window_size"]+track_cols].pivot(["gene_id","gene_name"],["window_size"],track_cols).reset_index()
gene_df.columns = ([x for x,y in zip(gene_df.columns.get_level_values(0),gene_df.columns.get_level_values(1)) if y == ""]
                   + [x + "_" + str(y) for x,y in zip(gene_df.columns.get_level_values(0),gene_df.columns.get_level_values(1)) if y != ""])
gene_df = gene_df.rename(columns={k+"_-1":k+"_ingenome" for k in track_cols})

# %%
windows = sorted(list(set(results["window_size"])))

# %%
linearmodel_match_windowed(gene_df, gtex_molten, tissues, "tissue", gene_locs, windows, 
                           model_name = "enformer", dataset_name="gtex",
                           lm=sklearn.linear_model.RidgeCV(), print_alpha=True)

# %% [markdown]
# ##### Basenji2

# %%
with open(base_path_results + "tss_sim-basenji2-latest_results.tsv", 'r') as tsv:
    header = tsv.readline()
    cols = header.split("\t")
cols = [k for k in cols if "CAGE" in k and k.endswith("landmark_sum")]

basenji2_results = pd.read_csv(base_path_results + "tss_sim-basenji2-latest_results.tsv",sep="\t",
                              usecols = ['gene_id', 'gene_name', 'orient', 'offset', 'window_size'] + cols)
basenji2_results = (basenji2_results.groupby(["gene_id","gene_name"])[[x for x in basenji2_results.keys() if "CAGE" in x]].mean().reset_index())

universal_samples = ['Clontech Human Universal Reference Total RNA', 'SABiosciences XpressRef Human Universal Total RNA', 'CAGE:Universal RNA - Human Normal Tissues Biochain']
track_cols = [k for k in basenji2_results.keys() if "CAGE" in k and k.endswith("landmark_sum") and not any(x in k for x in universal_samples)]

basenji2_gene_df = basenji2_results[["gene_id","gene_name"] + track_cols].rename(columns={t:t+"_basenji2" for t in track_cols})

# %%
linearmodel_match_windowed(basenji2_gene_df, gtex_molten, tissues, "tissue", gene_locs, windows=["basenji2"], 
                           model_name="basenji2", dataset_name="gtex",
                           lm=sklearn.linear_model.RidgeCV(), print_alpha=True)

# %% [markdown]
# #### Analyze correlations

# %%
tissue_corrs_lasso = pd.read_csv(base_path_results + "gtex_corrs_modelmatched_enformer.tsv", sep="\t")
tissue_corrs_lasso["window_size"] = (pd.Categorical(tissue_corrs_lasso["window_size"], 
                                                    [str(x) for x in sorted([int(x) for x in set(tissue_corrs_lasso["window_size"]) if x != "Full"])] + ["Full"]))

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=tissue_corrs_lasso,mapping=p9.aes(x="window_size",y="r"))
 + p9.geom_boxplot()
 + p9.labs(x="Size of sequence window (bp)",y="Correlation between predicted and\nmeasured log expression (GTEx)")
+ p9.theme(axis_text_x=p9.element_text(rotation=40, hjust=1), axis_title=p9.element_text(size=10))
)
p

# %%
p.save("Graphics/" + "distal_gtex" + ".svg", width=2.8, height=2.9, dpi=300)

# %% [markdown]
# #### Add exon-intron ratio
#

# %%
tissue_pred_df = pd.read_csv(base_path_results + "gtex_pred_modelmatched_enformer.tsv", sep="\t")
tissues = set(tissue_pred_df["tissue"])

# %%
gene_pos = gtf_df.df.query('Feature == "gene"')[["Chromosome","Start","End","gene_id"]]
gene_pos = gene_pos.loc[~gene_pos.gene_id.str.endswith("PAR_Y")]
gene_pos["gene_id"] = gene_pos["gene_id"].apply(lambda x: x.split('.')[0])
tissue_pred_df = gene_pos.merge(tissue_pred_df, on="gene_id")

# %%
exint = (pd.read_csv(base_path_data + "genomic_sequence_plus_features_hl_all_tissues.csv")
         .rename(columns={"Unnamed: 0":"transcript_id"}))
exint["transcript_id"] = exint["transcript_id"].apply(lambda x: x.split('.')[0])


# %%
transcript_to_gene = (pd.read_csv(base_path_data + "gene2transcript.txt", sep="\t")
                      .rename(columns={"Gene stable ID":"gene_id",
                                       "Transcript stable ID":"transcript_id"})
                     )

# %%
exint = transcript_to_gene.merge(exint, on="transcript_id")

# %%
# find matching to tissues
exint_tissue_match_dict = {k.replace(" - "," ").replace("(","").replace(")","").replace(" ","_").replace("-","_"):k for k in tissues}
exint_tissue_match_dict['Cells_Transformed_fibroblasts'] = 'Cells - Cultured fibroblasts'
matched = [k for k in exint.keys() if k in exint_tissue_match_dict]

exint_tissue_match_df = pd.DataFrame([{"key":k,"tissue":v} for k,v in exint_tissue_match_dict.items()])

# %%
# impute nan with median
exint = exint[["gene_id"]+matched]
exint_array = np.array(exint[matched])
row_medians = np.nanmedian(exint_array, axis=1)
idxs = np.where(np.isnan(exint_array))
exint_array[idxs] = np.take(row_medians, idxs[0])
exint[matched] = exint_array

# %%
# melt and match
exint_molten = (exint
                .melt(id_vars="gene_id",var_name="key",value_name="exon_intron_ratio")
                .merge(exint_tissue_match_df, on="key")
                .drop(columns=["key"])
                .groupby(['gene_id','tissue'])
                .mean()
                .reset_index()
               )

# %%
tissue_pred_df = exint_molten.merge(tissue_pred_df, on=["gene_id","tissue"], how="right")

# %%
# mean normalize exon-intron ratio for each tissue
exint_means = (tissue_pred_df
               .groupby('tissue')["exon_intron_ratio"]
               .mean()
               .reset_index()
               .rename(columns={"exon_intron_ratio":"mean_ratio"})
              )
tissue_pred_df = tissue_pred_df.merge(exint_means, on="tissue")
tissue_pred_df["mean_ratio"] = tissue_pred_df["mean_ratio"].fillna(0)
tissue_pred_df["exon_intron_ratio"] = tissue_pred_df["exon_intron_ratio"] - tissue_pred_df["mean_ratio"]
# replace nan with 0
tissue_pred_df["exon_intron_ratio"] = tissue_pred_df["exon_intron_ratio"].fillna(0)

# %%
row_list = []
for tissue in set(tissue_pred_df.tissue):
    subset_train = tissue_pred_df.query('tissue == @tissue & gene_id not in @intersect_genes')
    subset_test = tissue_pred_df.query('tissue == @tissue & gene_id in @test_genes')
    y_test = np.array(subset_test['log_obs'])
    y_train = np.array(subset_train['log_obs'])
    # train exon_intron model
    X_train = np.array(subset_train[["log_pred_ingenome","exon_intron_ratio"]])
    model = sklearn.linear_model.RidgeCV()
    model.fit(X_train, y_train)
    # test exon_intron model
    X_test = np.array(subset_test[["log_pred_ingenome","exon_intron_ratio"]])
    y_pred_exon_intron = model.predict(X_test).reshape(-1)
    row_list.append({
        "tissue":tissue,
        "Enformer":scipy.stats.pearsonr(subset_test["log_obs"],subset_test["log_pred_ingenome"])[0],
        "Enformer +\nExon-Intron Ratio":scipy.stats.pearsonr(subset_test["log_obs"],y_pred_exon_intron)[0],
        "r_exon_intron_obs":scipy.stats.pearsonr(subset_test["exon_intron_ratio"],subset_test["log_pred_ingenome"])[0],
        "r_exon_intron_pred":scipy.stats.pearsonr(subset_test["exon_intron_ratio"],subset_test["log_pred_ingenome"])[0]
    }
    )
gtex_corrs_full = pd.DataFrame(row_list)

# %% [markdown]
# #### Compare to other models
#
# - Xpresso
# - Basenji1
# - Basenji2

# %%
gtex_corrs_full = gtex_corrs_full[["tissue","Enformer","Enformer +\nExon-Intron Ratio"]].melt(id_vars="tissue", var_name="model",value_name="r")

# %%
xpresso_pred_df = (pd.read_csv(base_path_results_xpresso + "xpresso_preds.tsv", sep = "\t")
                   [["gene_id","Prediction"]]
                   .rename(columns={"Prediction":"xpresso_pred"})
                  )

tissue_pred_df = (tissue_pred_df
                  .merge(xpresso_pred_df, on="gene_id")
                  .merge(gene_locs[["gene_id","len"]], on="gene_id")
                 )

tissue_pred_df["log_len"] = np.log10(tissue_pred_df["len"])

row_list = []
for tissue in tissues:
    subset_train = tissue_pred_df.query('tissue == @tissue & gene_id not in @intersect_genes')
    subset_test = tissue_pred_df.query('tissue == @tissue & gene_id in @test_genes')
    # train ridge
    y_train = np.array(subset_train['log_obs'])
    X_train = np.array(subset_train[['xpresso_pred']])
    model = sklearn.linear_model.RidgeCV()
    model.fit(X_train, y_train)
    # test ridge
    y_test = np.array(subset_test['log_obs'])
    X_test = np.array(subset_test[['xpresso_pred']])
    y_pred = model.predict(X_test).reshape(-1)
    tpl = (y_test, y_pred)
    r = scipy.stats.pearsonr(*tpl)
    r2 = sklearn.metrics.r2_score(*tpl)
    row_list.append({
        "tissue":tissue,
        "model":"Xpresso",
        "r":r[0],
        
    })
xpresso_corrs = pd.DataFrame(row_list)

# %%
basenji2_corrs = pd.read_csv(base_path_results + "gtex_corrs_modelmatched_basenji2.tsv", sep="\t")[["tissue","r"]]
basenji2_corrs["model"] = "Basenji2"

# %%
combined_corrs = pd.concat([gtex_corrs_full, xpresso_corrs, basenji2_corrs])

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

combined_corrs["model_cat"] = pd.Categorical(combined_corrs["model"],
                                             categories=["Xpresso", "Basenji2", "Enformer", "Enformer +\nExon-Intron Ratio"])

p = (p9.ggplot(data=combined_corrs,mapping=p9.aes(x="model_cat",y="r"))
 + p9.geom_boxplot()
 + p9.labs(x="Model",y="Pearson correlation between predicted\nand measured log expression", title="GTEx")
 + p9.theme(axis_title=p9.element_text(size=10), title=p9.element_text(size=10),
           axis_text_x=p9.element_text(rotation=30, hjust=1))
)
p

# %%
combined_corrs.groupby('model').median()

# %%
p.save("Graphics/" + "gtex_gtex" + ".svg", width=2.6, height=3.2, dpi=300)

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

plot_df = combined_corrs.query('model in ["Basenji2","Enformer"]')
plot_df["tissue_cat"] = pd.Categorical(plot_df['tissue'],
                                       plot_df.query('model == "Enformer"').sort_values('r')['tissue'])

p = (p9.ggplot(data=plot_df,mapping=p9.aes(x="tissue_cat",y="r",color="model"))
 #+ p9.geom_col(position="dodge")
 + p9.geom_point(size=2.5)
 + p9.coords.coord_cartesian(ylim=(0.67,0.85))
 + p9.labs(x="",y="Correlation of predicted with\nmeasured log expression (GTEx)")
 + p9.theme(axis_title=p9.element_text(size=10),
           axis_text_x=p9.element_text(rotation=90, hjust=1),
           legend_box_margin=0, legend_key_size=9, 
            legend_text=p9.element_text(size=9),
            legend_background=p9.element_blank(),
            legend_title=p9.element_text(size=9), legend_position=(0.25,0.8))
)
p

# %%
p.save("Graphics/" + "xsup_gtex_tissues" + ".svg", width=7.2, height=4.8, dpi=300)

# %% [markdown]
# ### GTEX - Across tissues

# %% [markdown]
# #### Analyze correlations

# %%
tissue_pred_df = pd.read_csv(base_path_results + "gtex_pred_modelmatched_enformer.tsv", sep="\t")
tissue_var_df_molten = tissue_pred_df

# %%
rows = []

tissue_var_df_molten_indexed = tissue_var_df_molten.set_index("gene_id")
for gene in set(tissue_var_df_molten.query('gene_id in @test_genes')["gene_id"]):
    subset = tissue_var_df_molten_indexed.loc[gene]
    # skip genes which are constant across dev-stages (or never expressed)
    if not (subset["log_obs"].std() > 0):
        continue
    for col in [x for x in subset.keys() if x.startswith("log_pred")]:
        row_dict = {"gene_id":gene}
        row_dict["window_size"] = col.split("_")[-1]
        tpl = (subset["log_obs"],subset[col])
        corr_test =  scipy.stats.pearsonr(*tpl)
        r = corr_test[0]
        row_dict["R"] = r
        row_dict["p"] = corr_test[1]
        row_dict["rho"] = scipy.stats.spearmanr(*tpl)[0]
        row_dict["R-squared"]  = sklearn.metrics.r2_score(*tpl)
        row_dict["R-squared (after Rescaling)"] = r ** 2
        row_dict["Explained Variance"]  = sklearn.metrics.explained_variance_score(*tpl)
        row_dict["pred_mean_log_expr"] = subset["log_pred_ingenome"].mean()
        row_dict["pred_std_log_expr"] = subset["log_pred_ingenome"].std()
        row_dict["mean_log_expr"] = subset["log_obs"].mean()
        row_dict["std_log_expr"] = subset["log_obs"].std()
        rows.append(row_dict)
        
between_tissue_corrs = pd.DataFrame(rows)
between_tissue_corrs["window_size"] = between_tissue_corrs["window_size"].apply(lambda x: x if x != "ingenome" else "Full")
between_tissue_corrs["window_size"] = pd.Categorical(between_tissue_corrs["window_size"], 
                                                     ["null"] + [str(x) for x in sorted([int(x) for x in set(between_tissue_corrs["window_size"]) if x != "Full" and x!= "null"])] + ["Full"])

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=between_tissue_corrs,mapping=p9.aes(x="window_size",y="R"))
 + p9.geom_boxplot()
 + p9.labs(x="Size of sequence window (bp)",y="Correlation of predicted with\nmeasured log expression (GTEx)")
)
p

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=between_tissue_corrs.query('window_size != "null"'),mapping=p9.aes(x="window_size",y="R"))
 + p9.geom_boxplot()
 + p9.labs(x="Size of sequence window (bp)",y="Correlation of predicted with\nmeasured log expression (GTEx)")
)
p

# %%
p.save("Graphics/" + "xsup_distal_gtexbtwn" + ".svg", width=6.4, height=4.8, dpi=300)

# %%
between_tissue_corrs.groupby('window_size').median()

# %%
between_tissue_corrs.to_csv(base_path_results + "between_tissue_corrs_windows.tsv",index=None,sep="\t")

# %% [markdown]
# #### Compare models

# %%
tissue_pred_df = (pd.read_csv(base_path_results + "gtex_pred_modelmatched_enformer.tsv", sep="\t")
                  [["gene_id","tissue","log_obs","log_pred_ingenome","expressed"]]
                  .rename(columns={"log_pred_ingenome":"log_pred_enformer"})
                 )
tissue_pred_df_basenji2 = (pd.read_csv(base_path_results + "gtex_pred_modelmatched_basenji2.tsv", sep="\t")
                          [["gene_id","tissue","log_pred_basenji2"]]
                         )
tissue_pred_df = (tissue_pred_df
                  .merge(tissue_pred_df_basenji2, on=["gene_id", "tissue"])
                 )

# %%
tissue_var_df_molten = tissue_pred_df

# %%
expresed_in_all_tissues = set(tissue_pred_df.groupby('gene_id')['expressed'].min().reset_index().query('expressed')["gene_id"])

# %%
rows = []

tissue_var_df_molten_indexed = tissue_var_df_molten.set_index("gene_id")
for gene in set(tissue_var_df_molten.query('gene_id in @test_genes')["gene_id"]):
    subset = tissue_var_df_molten_indexed.loc[gene]
    for col in [x for x in subset.keys() if x.startswith("log_pred")]:
        row_dict = {"gene_id":gene}
        row_dict["model"] = col.split("_")[-1]
        tpl = (subset["log_obs"],subset[col])
        corr_test =  scipy.stats.pearsonr(*tpl)
        r = corr_test[0]
        row_dict["R"] = r
        row_dict["p"] = corr_test[1]
        row_dict["rho"] = scipy.stats.spearmanr(*tpl)[0]
        row_dict["R-squared"]  = sklearn.metrics.r2_score(*tpl)
        row_dict["pred_mean_log_expr_enformer"] = subset["log_pred_enformer"].mean()
        row_dict["pred_std_log_expr_enformer"] = subset["log_pred_enformer"].std()
        row_dict["pred_mean_log_expr_basenji2"] = subset["log_pred_basenji2"].mean()
        row_dict["pred_std_log_expr_basenji2"] = subset["log_pred_basenji2"].std()
        row_dict["mean_log_expr"] = subset["log_obs"].mean()
        row_dict["std_log_expr"] = subset["log_obs"].std()
        row_dict["mae_log_expr"] = np.abs(subset["log_obs"] - subset["log_obs"].mean()).mean()
        row_dict["dyn_range_log_expr"] = subset["log_obs"].quantile([0.05,0.95]).max() - subset["log_obs"].quantile([0.05,0.95]).min()
        rows.append(row_dict)
        
between_tissue_corrs = pd.DataFrame(rows)

# %%
between_tissue_corrs["dyn_range_log2_expr"] = between_tissue_corrs["dyn_range_log_expr"]/np.log10(2)
between_tissue_corrs["mae_log2_expr"] = between_tissue_corrs["mae_log_expr"]/np.log10(2)
between_tissue_corrs["sig"] = between_tissue_corrs["p"] < 0.05/(len(between_tissue_corrs)/2)
between_tissue_corrs["correct_direction"] = between_tissue_corrs["R"] > 0
between_tissue_corrs["trend_predicted"] = between_tissue_corrs["correct_direction"] & between_tissue_corrs["sig"] 
between_tissue_corrs["always_expressed"] = between_tissue_corrs['gene_id'].apply(lambda x: x in expresed_in_all_tissues)

# %%
between_tissue_corrs["mae_log2_expr"].describe()

# %%
between_tissue_corrs["dyn_range_log2_expr"].describe()

# %%
between_tissue_corrs.query('not always_expressed')["dyn_range_log2_expr"].describe()

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=between_tissue_corrs,mapping=p9.aes(x="always_expressed",y="R", fill="model"))
 + p9.geom_boxplot()
 + p9.labs(x="Is the gene always expressed?",y="Correlation of predicted with measured\n(log) expression (GTEx)")
)
p

# %%
between_tissue_corrs["Dynamic\nRange"] = between_tissue_corrs["dyn_range_log2_expr"].apply(lambda x: "< 2-fold" if x < 1 else ('2 to 8-fold' if x < 3 else '> 8-fold'))

between_tissue_corrs["Dynamic\nRange"] = pd.Categorical(between_tissue_corrs["Dynamic\nRange"], 
                                                        ["< 2-fold", '2 to 8-fold', '> 8-fold'])

scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=between_tissue_corrs,mapping=p9.aes(x="Dynamic\nRange",y="R", fill="model"))
 + p9.geom_boxplot()
 + p9.labs(x="Dynamic Range",y="Correlation of predicted with measured\n(log) expression (GTEx)")
)
p

# %%
between_tissue_corrs["Dynamic\nRange"] = between_tissue_corrs["dyn_range_log2_expr"].apply(lambda x: "< 2-fold" if x < 1 else ('2 to 8-fold' if x < 3 else '> 8-fold'))

between_tissue_corrs["Dynamic\nRange"] = pd.Categorical(between_tissue_corrs["Dynamic\nRange"], 
                                                        ["< 2-fold", '2 to 8-fold', '> 8-fold'])

scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

plot_df = (between_tissue_corrs
           .groupby(['Dynamic\nRange','model'])['trend_predicted']
          .agg(['sum','size'])
          .reset_index()
          )
plot_df["pct"] = 100*(plot_df["sum"]/plot_df["size"])

p = (p9.ggplot(data=plot_df,mapping=p9.aes(x="Dynamic\nRange", y="pct",fill="model"))
 + p9.geom_bar(stat="identity",position="dodge")
 + p9.labs(x="Dynamic Range",y="Percentage of Genes with\nsignificant prediction of Trend")
)
p

# %%
p.save("Graphics/" + "xsup_gtex_gtexbtwntissues" + ".svg", width=6.4, height=4.8, dpi=300)

# %% [markdown]
# #### RMSE analysis

# %%
gene_mean_df = (between_tissue_corrs
           .groupby('gene_id')
           [["pred_mean_log_expr_enformer","pred_std_log_expr_enformer","pred_mean_log_expr_basenji2","pred_std_log_expr_basenji2","mean_log_expr" ,"std_log_expr"]]
           .mean()
           .reset_index()
          )
# prediction of gene mean
print(scipy.stats.pearsonr(gene_mean_df["mean_log_expr"],gene_mean_df["pred_mean_log_expr_basenji2"]))
print(scipy.stats.pearsonr(gene_mean_df["mean_log_expr"],gene_mean_df["pred_mean_log_expr_enformer"]))
# prediction of gene standard deviation
print(scipy.stats.pearsonr(gene_mean_df["std_log_expr"],gene_mean_df["pred_std_log_expr_basenji2"]))
print(scipy.stats.pearsonr(gene_mean_df["std_log_expr"],gene_mean_df["pred_std_log_expr_enformer"]))


# %%
sklearn.metrics.r2_score(tissue_pred_df.query('gene_id in @test_genes')["log_obs"],
                         tissue_pred_df.query('gene_id in @test_genes')["log_pred_enformer"])

# %%
gene_dev_df = tissue_pred_df.merge(gene_mean_df, on="gene_id")

# %%
# compute deviations
# y^bar_g - y^bar: deviation of gene mean around global mean
# y^bar_gc - y^bar: deviation of genes around global mean
# y_gc - y^bar_g : deviation of genes around gene means
# y_gc - y^hat_gc : residual

gene_dev_df["total"] = gene_dev_df["log_obs"] - gene_mean_df['mean_log_expr'].mean()
gene_dev_df["condition"] = gene_dev_df["log_obs"] - gene_dev_df['mean_log_expr']
gene_dev_df["basenji2"] = gene_dev_df["log_obs"] - gene_dev_df['log_pred_basenji2']
gene_dev_df["enformer"] = gene_dev_df["log_obs"] - gene_dev_df['log_pred_enformer']

# %%
1 - (np.sum(gene_dev_df["condition"]**2)/np.sum(gene_dev_df["total"]**2))

# %%
1 - (np.sum(gene_dev_df["enformer"]**2)/np.sum(gene_dev_df["total"]**2))

# %%
1 - (gene_dev_df["enformer"].var()/gene_dev_df["total"].var())

# %%
# compute the dynamic ranges in log10
qtls = [0.025,0.975]
print("log10")
print(gene_dev_df["total"].quantile(qtls).max() - gene_dev_df["total"].quantile(qtls).min())
print(gene_dev_df["condition"].quantile(qtls).max() - gene_dev_df["condition"].quantile(qtls).min())
print(gene_dev_df["enformer"].quantile(qtls).max() - gene_dev_df["enformer"].quantile(qtls).min())
# log2
print("log2")
print((gene_dev_df["total"].quantile(qtls).max() - gene_dev_df["total"].quantile(qtls).min())/np.log10(2))
print((gene_dev_df["condition"].quantile(qtls).max() - gene_dev_df["condition"].quantile(qtls).min())/np.log10(2))
print((gene_dev_df["enformer"].quantile(qtls).max() - gene_dev_df["enformer"].quantile(qtls).min())/np.log10(2))

# %%
print(2**7.7832250862224805)
print(2**3.962575653356749)
print(2**5.690521605101862)

# %%
# compute the RMSE
print(gene_dev_df["total"].std()/np.log10(2))
print(gene_dev_df["condition"].std()/np.log10(2))
print(gene_dev_df["enformer"].std()/np.log10(2))
print(gene_dev_df["basenji2"].std()/np.log10(2))

# %%
print(2**2.261978863822522)
print(2**0.9668217018328089)
print(2**1.3861203036366847)

# %%
# compute the MAE
print(np.abs(gene_dev_df["total"]).mean()/np.log10(2))
print(np.abs(gene_dev_df["condition"]).mean()/np.log10(2))
print(np.abs(gene_dev_df["enformer"]).mean()/np.log10(2))
print(np.abs(gene_dev_df["basenji2"]).mean()/np.log10(2))

# %%
print(2**1.9089314796080616)
print(2**0.6050264184534644)
print(2**1.0210755008529042)

# %%
#gene_mean_dev = (gene_mean_df["mean_log_expr"] - gene_mean_df['mean_log_expr'].mean())
#print(np.sum(np.abs(gene_mean_dev.quantile([0.05,0.95]))))

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

plot_df = (gene_dev_df.sample(frac=0.2,random_state=10)
           [['gene_id','tissue',
             "total","condition",
             "basenji2","enformer"]]
           .melt(id_vars=['gene_id','tissue'],value_name='deviation',var_name='type')
           .query('type != "basenji2"')
          )
plot_df['deviation_log2'] = plot_df['deviation']/np.log10(2)

deviation_rename_dict = {"condition":"Gene Mean",
                        "enformer":"Enformer\nprediction",
                        "total":"Global Mean"}

plot_df["group"] = plot_df["type"].apply(lambda x: deviation_rename_dict[x])

plot_df["group"] = pd.Categorical(plot_df["group"],[x for x in deviation_rename_dict.values()])

p = (p9.ggplot(data=plot_df,mapping=p9.aes(x="deviation",fill="group"))
 #+ p9.geom_col(position="dodge")
 #+ p9.geom_point(size=2.5)
 #+ p9.coords.coord_cartesian(ylim=(0.67,0.85))
 #+ p9.geom_histogram(position="identity", alpha=0.5)
 + p9.geom_density(position="identity", alpha=0.5)
 #+ p9.scale_y_log10()
 + p9.labs(x="Deviation of\nmeasured log10 expression",y="Density",fill="Deviation from:")
 + p9.theme(axis_title=p9.element_text(size=10),
           axis_text_x=p9.element_text(rotation=90, hjust=1),
           legend_box_margin=0, legend_key_size=9, 
            legend_text=p9.element_text(size=9),
            legend_background=p9.element_blank(),
            legend_title=p9.element_text(size=9), legend_position=(0.68,0.70))
)
p

# %%
p.save("Graphics/" + "gtex_deviation" + ".svg", width=2.6, height=3.0, dpi=300)

# %% [markdown]
# #### Prediction of deviation

# %%
gene_dev_df['log2_fc'] = gene_dev_df['condition']/np.log10(2)
gene_dev_df['pred_log2_fc'] = (gene_dev_df['log_pred_enformer'] - gene_dev_df['pred_mean_log_expr_enformer'])/np.log10(2)
gene_dev_df['pred_log2_fc_basenji2'] = (gene_dev_df['log_pred_basenji2'] - gene_dev_df['pred_mean_log_expr_basenji2'])/np.log10(2)
gene_dev_df["abs_log2_fc"] = np.abs(gene_dev_df["log2_fc"])
gene_dev_df["abs_pred_log2_fc"] = np.abs(gene_dev_df["pred_log2_fc"])

# %%
print(scipy.stats.pearsonr(gene_dev_df['log2_fc'], gene_dev_df['pred_log2_fc']))
print(scipy.stats.pearsonr(gene_dev_df['log2_fc'], gene_dev_df['pred_log2_fc_basenji2']))

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=gene_dev_df,mapping=p9.aes(x="pred_log2_fc",y="log2_fc",))
 #+ p9.geom_col(position="dodge")
 + p9.geom_bin2d(bins=100, raster=True)
 + p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.geom_smooth(method="lm",color="blue")
 + p9.geom_vline(xintercept = 1)
 + p9.geom_vline(xintercept = -1)
 + p9.geom_hline(yintercept = 1)
 + p9.geom_hline(yintercept = -1)
 + p9.labs(x="Predicted $\mathregular{log_2}$ deviation\nfrom gene mean",y="Measured $\mathregular{log_2}$ deviation\nfrom gene mean")
 + p9.theme(legend_text=p9.element_text(size=9),legend_title=p9.element_text(size=9),
            legend_box_margin=0,legend_key_width=5,axis_title=p9.element_text(size=10))#
 #+ p9.scale_y_log10()
 #+ p9.labs(x="",y="Density")
)
p

# %%
p.save("Graphics/" + "gtex_btwn" + ".svg", width=2.6, height=3.0, dpi=300)

# %%
classify_df = gene_dev_df.copy()
classify_df["differential"] = classify_df['abs_log2_fc'] > 1
classify_df["pred_differential"] = classify_df['abs_pred_log2_fc'] > 1

classify_rows_gtex = []
classify_rows_gtex.append({"Dataset":"GTEx","Metric":"FPR","Score":np.sum(classify_df["pred_differential"])/np.sum(~classify_df["differential"])})
classify_rows_gtex.append({"Dataset":"GTEx","Metric":"Precision","Score":sklearn.metrics.precision_score(classify_df["differential"], classify_df["pred_differential"])})
classify_rows_gtex.append({"Dataset":"GTEx","Metric":"Recall","Score":sklearn.metrics.recall_score(classify_df["differential"], classify_df["pred_differential"])})
classify_rows_gtex.append({"Dataset":"GTEx","Metric":"MCC","Score":sklearn.metrics.matthews_corrcoef(classify_df["differential"], classify_df["pred_differential"])})
classify_rows_gtex.append({"Dataset":"GTEx","Metric":"Prevalence","Score":np.sum(classify_df["differential"])/len(classify_df["differential"])})

# %%
pd.DataFrame(classify_rows_gtex)

# %%
sklearn.metrics.accuracy_score(classify_df["differential"], classify_df["pred_differential"])

# %%
sklearn.metrics.confusion_matrix(classify_df["differential"],classify_df["pred_differential"], normalize="all")

# %%
0.05/(0.05+0.115)

# %%
cmatrix = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(classify_df["differential"],classify_df["pred_differential"], normalize="all")
ax,fig = cmatrix.ax_, cmatrix.figure_

# %%
classify_df = gene_dev_df.query('abs_log2_fc > 1 or abs_log2_fc < 0.3')
classify_df["differential"] = classify_df['abs_log2_fc'] > 1
classify_df["three_class_obs"] = classify_df['log2_fc'].apply(lambda x: "Up" if x > 1 else ("Down" if x < - 1 else "Neutral"))
classify_df["three_class_pred"] = classify_df['pred_log2_fc'].apply(lambda x: "Up" if x > 1 else ("Down" if x < - 1 else "Neutral"))

# %%
#print(sklearn.metrics.accuracy_score(classify_df["three_class_obs"], ["Neutral"]*len(classify_df)))
print(sklearn.metrics.balanced_accuracy_score(classify_df["three_class_obs"], classify_df["three_class_pred"]))
print(sklearn.metrics.precision_score(classify_df["three_class_obs"], classify_df["three_class_pred"],
                                      labels=["Up","Neutral","Down"],average=None))
print(sklearn.metrics.recall_score(classify_df["three_class_obs"],classify_df["three_class_pred"],
                                   labels=["Up","Neutral","Down"],average=None))

# %% [markdown] tags=[]
# ### Kaessmann - Across genes, linear model matching

# %% [markdown]
# #### Train linear models and generate predictions

# %% [markdown]
# ##### Enformer

# %%
linearmodel_match_windowed(gene_df, kaessmann_molten, dev_tissues, "sample", gene_locs, windows, 
                           model_name = "enformer", dataset_name="kaessmann",
                          lm=sklearn.linear_model.RidgeCV(), print_alpha=True)

# %% [markdown]
# ##### Basenji2

# %%
with open(base_path_results + "tss_sim-basenji2-latest_results.tsv", 'r') as tsv:
    header = tsv.readline()
    cols = header.split("\t")
cols = [k for k in cols if "CAGE" in k and k.endswith("landmark_sum")]

basenji2_results = pd.read_csv(base_path_results + "tss_sim-basenji2-latest_results.tsv",sep="\t",
                              usecols = ['gene_id', 'gene_name', 'orient', 'offset', 'window_size'] + cols)
basenji2_results = (basenji2_results.groupby(["gene_id","gene_name"])[[x for x in basenji2_results.keys() if "CAGE" in x]].mean().reset_index())

universal_samples = ['Clontech Human Universal Reference Total RNA', 'SABiosciences XpressRef Human Universal Total RNA', 'CAGE:Universal RNA - Human Normal Tissues Biochain']
track_cols = [k for k in basenji2_results.keys() if "CAGE" in k and k.endswith("landmark_sum") and not any(x in k for x in universal_samples)]

basenji2_gene_df = basenji2_results[["gene_id","gene_name"] + track_cols].rename(columns={t:t+"_basenji2" for t in track_cols})

# %%
linearmodel_match_windowed(basenji2_gene_df, kaessmann_molten, dev_tissues, "sample", gene_locs, 
                           windows=["basenji2"], 
                           model_name = "basenji2", dataset_name="kaessmann",
                           lm=sklearn.linear_model.RidgeCV(), print_alpha=True)

# %% [markdown]
# #### Analyze correlations

# %%
kaessmann_corrs_lasso = pd.read_csv(base_path_results + "kaessmann_corrs_modelmatched_enformer.tsv", sep="\t")
kaessmann_corrs_lasso["window_size"] = (pd.Categorical(kaessmann_corrs_lasso["window_size"], 
                                                    [str(x) for x in sorted([int(x) for x in set(kaessmann_corrs_lasso["window_size"]) if x != "Full"])] + ["Full"]))

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=kaessmann_corrs_lasso,mapping=p9.aes(x="window_size",y="r"))
 + p9.geom_boxplot()
 + p9.labs(x="Size of sequence window (bp)",y="Correlation of predicted with measured\nlog expression (Cardoso-Moreira)")
)
p

# %%
p.save("Graphics/" + "xsup_distal_kaess" + ".svg", width=6.4, height=4.8, dpi=300)

# %% [markdown]
# #### Compare to other models

# %%
kaessmann_corrs = pd.read_csv(base_path_results + "kaessmann_corrs_modelmatched_enformer.tsv", sep="\t").query('window_size == "Full"')
kaessmann_corrs["model"] = "Enformer"
kaessmann_corrs = kaessmann_corrs[["sample","model","r"]]

# %%
kaessmann_pred_df = pd.read_csv(base_path_results + "kaessmann_pred_modelmatched_enformer.tsv", sep="\t")

xpresso_pred_df = (pd.read_csv(base_path_results_xpresso + "xpresso_preds.tsv", sep = "\t")
                   [["gene_id","Prediction"]]
                   .rename(columns={"Prediction":"xpresso_pred"})
                  )

kaessmann_pred_df = (kaessmann_pred_df
                  .merge(xpresso_pred_df, on="gene_id")
                  .merge(gene_locs[["gene_id","len"]], on="gene_id")
                 )

kaessmann_pred_df["log_len"] = np.log10(kaessmann_pred_df["len"])

row_list = []
for sample in dev_tissues:
    subset_train = kaessmann_pred_df.query('sample == @sample & gene_id not in @intersect_genes')
    subset_test = kaessmann_pred_df.query('sample == @sample & gene_id in @test_genes')
    # train ridge
    y_train = np.array(subset_train['log_obs'])
    X_train = np.array(subset_train[['log_len','xpresso_pred']])
    model = sklearn.linear_model.RidgeCV()
    model.fit(X_train, y_train)
    # test ridge
    y_test = np.array(subset_test['log_obs'])
    X_test = np.array(subset_test[['log_len','xpresso_pred']])
    y_pred = model.predict(X_test).reshape(-1)
    tpl = (y_test, y_pred)
    r = scipy.stats.pearsonr(*tpl)
    row_list.append({
        "sample":sample,
        "model":"Xpresso",
        "r":r[0]
    })
xpresso_corrs = pd.DataFrame(row_list)

# %%
basenji2_corrs = pd.read_csv(base_path_results + "kaessmann_corrs_modelmatched_basenji2.tsv", sep="\t")[["sample","r"]]
basenji2_corrs["model"] = "Basenji2"

# %%
combined_corrs = pd.concat([kaessmann_corrs, xpresso_corrs,  basenji2_corrs])

# %%
combined_corrs.groupby('model').median()

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

combined_corrs["model_cat"] = pd.Categorical(combined_corrs["model"],
                                             categories=["Xpresso", "Basenji2", "Enformer"])

p = (p9.ggplot(data=combined_corrs
           ,mapping=p9.aes(x="model_cat",y="r"))
 + p9.geom_boxplot()
 + p9.labs(x="Model",y="Correlation of predicted with measured\nlog expression (Cardoso-Moreira)")
 + p9.theme(axis_title=p9.element_text(size=10))
)
p

# %%
p.save("Graphics/" + "xsup_gtex_kaessmann" + ".svg", width=6.4, height=4.8, dpi=300)

# %%
combined_corrs["tissue"] = combined_corrs["sample"].apply(lambda x: x.split('.')[0])
combined_corrs["stage"] = combined_corrs["sample"].apply(lambda x: x.split('.')[1].lower())
combined_corrs["stage_num"] = combined_corrs["sample"].apply(lambda x: int(x.split('.')[2]))

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

plot_df = combined_corrs.query('model in ["Enformer"]')

plot_df["tissue"] =  pd.Categorical(plot_df["tissue"],
                                     categories=(combined_corrs
                                                 .query('model in ["Enformer"]')
                                                 .groupby(['tissue','model'])['r'].mean().reset_index()
                                                 .sort_values('r')["tissue"]))

p = (p9.ggplot(data=plot_df
           ,mapping=p9.aes(x="tissue",y="r"))#, color="model_cat"))
 #+ p9.geom_col(position="dodge")
 + p9.geom_boxplot()
 #+ p9.coords.coord_cartesian(ylim=(0.7,0.85))
 + p9.labs(x="",y="Correlation of predictions with measured\ngene expression (Cardoso-Moreira)",color="Model")
 + p9.theme(axis_title=p9.element_text(size=10),legend_background=p9.element_blank(),
            legend_text=p9.element_text(size=8),legend_key_size=8,
            legend_title=p9.element_text(size=8))
)
p

# %%
p.save("Graphics/" + "xsup_gtex_kaessmann_tissues" + ".svg", width=6.4, height=4.8, dpi=300)

# %%
stage_cats = ['4wpc','5wpc','6wpc','7wpc','8wpc','9wpc','10wpc','11wpc','12wpc','13wpc','16wpc','18wpc','19wpc','20wpc','newborn','infant','toddler',
              'school','youngteenager','teenager','oldteenager','youngadult','youngmidage','oldermidage','senior']

plot_df = combined_corrs.query('model in ["Basenji2","Enformer"]').groupby(['stage','model','tissue'])['r'].mean().reset_index().sort_values('r')

scale = 1.5
p9.options.figure_size = (6.4*scale, 4.8*scale)

plot_df["model_cat"] = pd.Categorical(plot_df["model"],
                                      categories=["Basenji2", "Enformer"])

plot_df["stage"] =  pd.Categorical(plot_df["stage"],
                                     categories=stage_cats)

(p9.ggplot(data=plot_df
           ,mapping=p9.aes(x="stage",y="r",color='model_cat'))
 #+ p9.geom_col(position="dodge")
 + p9.geom_point(size=2.5)
 + p9.coords.coord_cartesian(ylim=(0.63,0.84))
 + p9.labs(x="",y="Average Pearson Correlation",color="Model")
 + p9.facet_wrap('~tissue')
 + p9.theme(axis_title=p9.element_text(size=10),legend_background=p9.element_blank(),
            legend_text=p9.element_text(size=8),legend_key_size=8,
            legend_title=p9.element_text(size=8), axis_text_x=p9.element_text(rotation=90, hjust=1))
)

# %%
stage_cats = ['4wpc','5wpc','6wpc','7wpc','8wpc','9wpc','10wpc','11wpc','12wpc','13wpc','16wpc','18wpc','19wpc','20wpc','newborn','infant','toddler',
              'school','youngteenager','teenager','oldteenager','youngadult','youngmidage','oldermidage','senior']

plot_df = combined_corrs.query('model in ["Enformer"]')



scale = 1.0
p9.options.figure_size = (6.4*scale, 4.8*scale)


plot_df = plot_df.query('stage not in ["youngteenager","oldteenager"]')
plot_df["stage"] = pd.Categorical(plot_df["stage"],
                                     categories=stage_cats)

p = (p9.ggplot(data=plot_df
           ,mapping=p9.aes(x="stage",y="r"))
 #+ p9.geom_col(position="dodge")
 + p9.geom_boxplot(outlier_size=0.5)
 + p9.geom_point(size=0.5)
 #+ p9.coords.coord_cartesian(ylim=(0.7,0.84))
 + p9.labs(x="Development Stage",y="", title="Cardoso-Moreira et al.")
 + p9.theme(axis_title=p9.element_text(size=10), title=p9.element_text(size=10),
            legend_background=p9.element_blank(),
            legend_text=p9.element_text(size=8),legend_key_size=8,
            legend_title=p9.element_text(size=8), axis_text_x=p9.element_text(rotation=90, hjust=1))
)
p

# %%
p.save("Graphics/" + "gtex_kaessmann_stages" + ".svg", width=3.3, height=3.3, dpi=300)

# %% [markdown]
# ### Kaessmann - Across development stages

# %% [markdown]
# #### Analyze correlations

# %%
kaessmann_pred_df = pd.read_csv(base_path_results + "kaessmann_pred_modelmatched_enformer.tsv", sep="\t")

dev_var_df = kaessmann_pred_df

dev_var_df["col"] = dev_var_df["sample"].apply(lambda x: "_".join(x.split("_")[1:]))
dev_var_df["tissue"] = dev_var_df["sample"].apply(lambda x: x.split(".")[0])
dev_var_df["dev_stage"] = dev_var_df["sample"].apply(lambda x: "_".join(x.split("_")[0].split(".")[1:]))
dev_var_df["dev_stage_number"] = dev_var_df["dev_stage"].apply(lambda x: int(x.split("_")[1]))
dev_var_df["dev_stage"] = dev_var_df["dev_stage"].apply(lambda x: x.split("_")[0])

dev_var_df_aggreg = dev_var_df.groupby(["gene_id","tissue","dev_stage"]).mean().reset_index()

# %%
rows = []

dev_var_df_indexed = dev_var_df_aggreg.set_index("gene_id")

for tissue in set(dev_var_df_aggreg["tissue"]):
    tissue_subset_df = dev_var_df_indexed.query('tissue == @tissue')
    for gene in set(tissue_subset_df.index[tissue_subset_df.index.isin(test_genes)]):
        subset = tissue_subset_df.loc[gene]
        # skip genes which are constant across dev-stages (or never expressed)
        if not (subset["log_obs"].std() > 0):
            continue
        for col in [x for x in subset.keys() if x.startswith("log_pred")]:
            row_dict = {"gene_id":gene, "tissue":tissue}
            row_dict["window_size"] = col.split("_")[-1]
            tpl = (subset["log_obs"],subset[col])
            corr_test =  scipy.stats.pearsonr(*tpl)
            r = corr_test[0]
            row_dict["R"] = r
            row_dict["p"] = corr_test[1]
            row_dict["rho"] = scipy.stats.spearmanr(*tpl)[0]
            row_dict["R-squared"]  = sklearn.metrics.r2_score(*tpl)
            row_dict["R-squared (after Rescaling)"] = r ** 2
            row_dict["Explained Variance"]  = sklearn.metrics.explained_variance_score(*tpl)
            row_dict["pred_mean_log_expr"] = subset["log_pred_ingenome"].mean()
            row_dict["pred_std_log_expr"] = subset["log_pred_ingenome"].std()
            row_dict["mean_log_expr"] = subset["log_obs"].mean()
            row_dict["std_log_expr"] = subset["log_obs"].std()
            rows.append(row_dict)
        
between_dev_corrs = pd.DataFrame(rows)
between_dev_corrs["window_size"] = between_dev_corrs["window_size"].apply(lambda x: x if x != "ingenome" else "Full")
between_dev_corrs["window_size"] = pd.Categorical(between_dev_corrs["window_size"], 
                                                  ['null'] + [str(x) for x in sorted([int(x) for x in set(between_dev_corrs["window_size"]) if x != "Full" and x != "null"])] + ["Full"])

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

(p9.ggplot(data=between_dev_corrs.groupby(["window_size","gene_id"])["R"].median().reset_index(),mapping=p9.aes(x="window_size",y="R"))
 + p9.geom_boxplot()
 + p9.labs(x="Size of sequence window",y="Correlation of Enformer predictions with\nmeasured gene expression (Kaessmann)")
)

# %% [markdown]
# #### Compare to other models

# %%
kaessmann_tissues = set([x.split(".")[0] for x in dev_tissues])

kaessmann_pred_df = (pd.read_csv(base_path_results + "kaessmann_pred_modelmatched_enformer.tsv", sep="\t")
                  [["gene_id","sample","log_obs","log_pred_ingenome"]]
                  .rename(columns={"log_pred_ingenome":"log_pred_enformer"})
                 )

kaessmann_pred_df_basenji2 = (pd.read_csv(base_path_results + "kaessmann_pred_modelmatched_basenji2.tsv", sep="\t")
                          [["gene_id","sample","log_pred_basenji2"]]
                         )

kaessmann_pred_df = (kaessmann_pred_df
                  .merge(kaessmann_pred_df_basenji2, on=["gene_id", "sample"])
                 )

# %%
dev_var_df = kaessmann_pred_df

dev_var_df["tissue"] = dev_var_df["sample"].apply(lambda x: x.split(".")[0])
dev_var_df["dev_stage"] = dev_var_df["sample"].apply(lambda x: "_".join(x.split("_")[0].split(".")[1:]))
dev_var_df["dev_stage_number"] = dev_var_df["dev_stage"].apply(lambda x: int(x.split("_")[1]))
dev_var_df["dev_stage"] = dev_var_df["dev_stage"].apply(lambda x: x.split("_")[0])

dev_var_df_aggreg = dev_var_df.groupby(["gene_id","tissue","dev_stage"]).mean().reset_index()

# %% tags=[]
rows = []

dev_var_df_indexed = dev_var_df_aggreg.set_index("gene_id")

for tissue in kaessmann_tissues:
    tissue_subset_df = dev_var_df_indexed.query('tissue == @tissue')
    for gene in set(tissue_subset_df.index[tissue_subset_df.index.isin(test_genes)]):
        subset = tissue_subset_df.loc[gene]
        # skip genes which are constant across dev-stages (or never expressed)
        if not (subset["log_obs"].std() > 0):
            continue
        for col in [x for x in subset.keys() if x.startswith("log_pred")]:
            row_dict = {"gene_id":gene}
            row_dict["model"] = col.split("_")[-1]
            row_dict["tissue"] = tissue
            tpl = (subset["log_obs"],subset[col])
            corr_test =  scipy.stats.pearsonr(*tpl)
            r = corr_test[0]
            row_dict["R"] = r
            row_dict["p"] = corr_test[1]
            row_dict["rho"] = scipy.stats.spearmanr(*tpl)[0]
            row_dict["R-squared"]  = sklearn.metrics.r2_score(*tpl)
            row_dict["pred_mean_log_expr_enformer"] = subset["log_pred_enformer"].mean()
            row_dict["pred_std_log_expr_enformer"] = subset["log_pred_enformer"].std()
            row_dict["pred_mean_log_expr_basenji2"] = subset["log_pred_basenji2"].mean()
            row_dict["pred_std_log_expr_basenji2"] = subset["log_pred_basenji2"].std()
            row_dict["mean_log_expr"] = subset["log_obs"].mean()
            row_dict["std_log_expr"] = subset["log_obs"].std()
            rows.append(row_dict)
        
between_dev_corrs = pd.DataFrame(rows)

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

(p9.ggplot(data=between_dev_corrs.groupby(["model","gene_id"])["R"].median().reset_index(),mapping=p9.aes(x="model",y="R"))
 + p9.geom_boxplot()
 + p9.labs(x="Model",y="Correlation of predictions with\nmeasured gene expression (Kaessmann)")
)

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

(p9.ggplot(data=between_dev_corrs,mapping=p9.aes(x="tissue",y="R", fill="model"))
 + p9.geom_boxplot()
 + p9.labs(x="Tissue",y="Correlation of predictions with\nmeasured gene expression (Kaessmann)")
)

# %% [markdown]
# #### Look at DDG genes particularly

# %%
ddg_df = pd.read_csv(base_path_data_kaessmann + "human_ddg.tsv",sep="\t").rename(columns={"Human_ID":"gene_id"})

# %%
pattern_cols = [x for x in ddg_df.keys() if "pattern" in x]
ddg_df = (ddg_df[["gene_id"] + pattern_cols]
          .fillna("Not DDG")
          .rename(columns={x:x.split("_")[0] for x in pattern_cols})
          .melt(id_vars="gene_id",var_name="tissue",value_name="ddg_status")
         )

ddg_df["is_ddg"] = ddg_df["ddg_status"] != "Not DDG"

# %%
ddg_df = between_dev_corrs.merge(ddg_df, on=["gene_id","tissue"])

# %%
scale = 2
p9.options.figure_size = (6.4*scale, 4.8*scale)

(p9.ggplot(data=ddg_df,mapping=p9.aes(x="tissue",y="R", fill="model"))
 + p9.geom_boxplot()
 + p9.labs(x="Tissue",y="Correlation of predictions with\nmeasured gene expression (Kaessmann)")
 + p9.facet_wrap('~ddg_status')
)

# %%
scale = 1.5
p9.options.figure_size = (6.4*scale, 3.2*scale)

(p9.ggplot(data=ddg_df,mapping=p9.aes(x="tissue",y="R", fill="model"))
 + p9.geom_boxplot()
 + p9.labs(x="Tissue",y="Correlation of predictions with\nmeasured gene expression (Kaessmann)")
 + p9.facet_wrap('~is_ddg')
)

# %%
ddg_df.query('model == "enformer"').groupby(['tissue','is_ddg'])['R'].median().reset_index()

# %% [markdown]
# #### RMSE analysis

# %%
gene_mean_df = (between_dev_corrs
           .groupby('gene_id')
           [["pred_mean_log_expr_enformer","pred_std_log_expr_enformer","pred_mean_log_expr_basenji2","pred_std_log_expr_basenji2","mean_log_expr" ,"std_log_expr"]]
           .mean()
           .reset_index()
          )
# prediction of gene mean
print(scipy.stats.pearsonr(gene_mean_df["mean_log_expr"],gene_mean_df["pred_mean_log_expr_basenji2"]))
print(scipy.stats.pearsonr(gene_mean_df["mean_log_expr"],gene_mean_df["pred_mean_log_expr_enformer"]))
# prediction of gene standard deviation
print(scipy.stats.pearsonr(gene_mean_df["std_log_expr"],gene_mean_df["pred_std_log_expr_basenji2"]))
print(scipy.stats.pearsonr(gene_mean_df["std_log_expr"],gene_mean_df["pred_std_log_expr_enformer"]))

# %%
sklearn.metrics.r2_score(dev_var_df_aggreg.query('gene_id in @test_genes')["log_obs"],
                         dev_var_df_aggreg.query('gene_id in @test_genes')["log_pred_enformer"])

# %%
gene_dev_df = dev_var_df_aggreg.merge(gene_mean_df, on="gene_id")

# %%
# compute deviations
# y^bar_g - y^bar: deviation of gene mean around global mean
# y^bar_gc - y^bar: deviation of genes around global mean
# y_gc - y^bar_g : deviation of genes around gene means
# y_gc - y^hat_gc : residual

gene_dev_df["total"] = gene_dev_df["log_obs"] - gene_mean_df['mean_log_expr'].mean()
gene_dev_df["condition"] = gene_dev_df["log_obs"] - gene_dev_df['mean_log_expr']
gene_dev_df["basenji2"] = gene_dev_df["log_obs"] - gene_dev_df['log_pred_basenji2']
gene_dev_df["enformer"] = gene_dev_df["log_obs"] - gene_dev_df['log_pred_enformer']

# %%
# Variance explained by gene means
1 - (np.sum(gene_dev_df["condition"]**2)/np.sum(gene_dev_df["total"]**2))

# %%
# Variance explained by enformer
1 - (np.sum(gene_dev_df["enformer"]**2)/np.sum(gene_dev_df["total"]**2))

# %%
# compute the dynamic ranges in log10
print("log10")
print(gene_dev_df["total"].quantile([0.05,0.95]).max() - gene_dev_df["total"].quantile([0.05,0.95]).min())
print(gene_dev_df["condition"].quantile([0.05,0.95]).max() - gene_dev_df["condition"].quantile([0.05,0.95]).min())
print(gene_dev_df["enformer"].quantile([0.05,0.95]).max() - gene_dev_df["enformer"].quantile([0.05,0.95]).min())

print("log2")
print((gene_dev_df["total"].quantile([0.05,0.95]).max() - gene_dev_df["total"].quantile([0.05,0.95]).min())/np.log10(2))
print((gene_dev_df["condition"].quantile([0.05,0.95]).max() - gene_dev_df["condition"].quantile([0.05,0.95]).min())/np.log10(2))
print((gene_dev_df["enformer"].quantile([0.05,0.95]).max() - gene_dev_df["enformer"].quantile([0.05,0.95]).min())/np.log10(2))

# %%
# compute the RMSE
print(gene_dev_df["total"].std()/np.log10(2))
print(gene_dev_df["condition"].std()/np.log10(2))
print(gene_dev_df["enformer"].std()/np.log10(2))
print(gene_dev_df["basenji2"].std()/np.log10(2))

# %%
# compute the MAE
print(np.abs(gene_dev_df["total"]).mean()/np.log10(2))
print(np.abs(gene_dev_df["condition"]).mean()/np.log10(2))
print(np.abs(gene_dev_df["enformer"]).mean()/np.log10(2))
print(np.abs(gene_dev_df["basenji2"]).mean()/np.log10(2))

# %%
#gene_mean_dev = (gene_mean_df["mean_log_expr"] - gene_mean_df['mean_log_expr'].mean())
#print(np.sum(np.abs(gene_mean_dev.quantile([0.05,0.95]))))

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

plot_df = (gene_dev_df.sample(frac=0.1,random_state=10)
           [['gene_id','tissue',
             "total","condition",
             "basenji2","enformer"]]
           .melt(id_vars=['gene_id','tissue'],value_name='deviation',var_name='type')
           .query('type != "basenji2"')
          )
plot_df['deviation_log2'] = plot_df['deviation']/np.log10(2)

rename_types = {
    "total":"Deviation from Global Mean",
    "condition":"Deviation from Gene Mean",
    "enformer":"Deviation from Enformer Prediction",
}

plot_df['type'] = plot_df['type'].apply(lambda x: rename_types[x])

p = (p9.ggplot(data=plot_df,mapping=p9.aes(x="deviation",fill="type"))
 #+ p9.geom_col(position="dodge")
 #+ p9.geom_point(size=2.5)
 #+ p9.coords.coord_cartesian(ylim=(0.67,0.85))
 #+ p9.geom_histogram(position="identity", alpha=0.5)
 + p9.geom_density(position="identity", alpha=0.5)
 #+ p9.scale_y_log10()
 + p9.labs(x="Log Residual",y="Density",legend_title="Residual Type")
 + p9.theme(axis_title=p9.element_text(size=10),
           axis_text_x=p9.element_text(rotation=90, hjust=1),
           legend_box_margin=0, legend_key_size=9, 
            legend_text=p9.element_text(size=9),
            legend_background=p9.element_blank(),
            legend_title=p9.element_text(size=9), legend_position=(0.6,0.8))
)
p

# %% [markdown]
# #### Prediction of deviation

# %%
gene_dev_df['log2_fc'] = gene_dev_df['condition']/np.log10(2)
gene_dev_df['pred_log2_fc'] = (gene_dev_df['log_pred_enformer'] - gene_dev_df['pred_mean_log_expr_enformer'])/np.log10(2)
gene_dev_df['pred_log2_fc_basenji2'] = (gene_dev_df['log_pred_basenji2'] - gene_dev_df['pred_mean_log_expr_basenji2'])/np.log10(2)
gene_dev_df["abs_log2_fc"] = np.abs(gene_dev_df["log2_fc"])
gene_dev_df["abs_pred_log2_fc"] = np.abs(gene_dev_df["pred_log2_fc"])

# %%
print(scipy.stats.pearsonr(gene_dev_df['log2_fc'], gene_dev_df['pred_log2_fc_basenji2']))
print(scipy.stats.pearsonr(gene_dev_df['log2_fc'], gene_dev_df['pred_log2_fc']))

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=gene_dev_df,mapping=p9.aes(x="pred_log2_fc",y="log2_fc",))
 #+ p9.geom_col(position="dodge")
 + p9.geom_bin2d(bins=100, raster=True)
 + p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.geom_smooth(method="lm",color="blue")
 + p9.geom_vline(xintercept = 1)
 + p9.geom_vline(xintercept = -1)
 + p9.geom_hline(yintercept = 1)
 + p9.geom_hline(yintercept = -1)
 + p9.theme(legend_text=p9.element_text(size=8),legend_title=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,axis_title=p9.element_text(size=10))#
 + p9.labs(x="Predicted log2 deviation from mean expression",y="Measured log2 deviation\nfrom mean expression")
 #+ p9.scale_y_log10()
 #+ p9.labs(x="",y="Density")
)
p

# %% [markdown]
# ### Performance improvements due to local vs long-range information

# %%
tissue_pred_df = (pd.read_csv(base_path_results + "gtex_pred_modelmatched_enformer.tsv", sep="\t")
                  .rename(columns={"log_pred_ingenome":"log_pred_enformer"})
                 )
tissue_pred_df_basenji2 = (pd.read_csv(base_path_results + "gtex_pred_modelmatched_basenji2.tsv", sep="\t")
                          [["gene_id","tissue","log_pred_basenji2"]]
                         )
tissue_pred_df = (tissue_pred_df
                  .merge(tissue_pred_df_basenji2, on=["gene_id", "tissue"])
                  .rename(columns={"tissue":"sample"})
                 )
tissue_pred_df["dataset"] = "GTEx"

# %%
kaessmann_pred_df = (pd.read_csv(base_path_results + "kaessmann_pred_modelmatched_enformer.tsv", sep="\t")
                  .rename(columns={"log_pred_ingenome":"log_pred_enformer"})
                 )
kaessmann_pred_df_basenji2 = (pd.read_csv(base_path_results + "kaessmann_pred_modelmatched_basenji2.tsv", sep="\t")
                          [["gene_id","sample","log_pred_basenji2"]]
                         )

kaessmann_pred_df = (kaessmann_pred_df
                  .merge(kaessmann_pred_df_basenji2, on=["gene_id", "sample"])
                 )
kaessmann_pred_df["dataset"] = "Cardoso-Moreira"

# %%
pred_df = pd.concat([tissue_pred_df,kaessmann_pred_df])
pred_df = pred_df.query('gene_id in @test_genes')

pred_df_means = (pred_df.groupby(['dataset','gene_id'])
                 .mean()
                 .reset_index()
                 .rename(columns={x:x+"_mean" for x in pred_df.keys() if "log" in x})
                )

pred_df_merged = pred_df.merge(pred_df_means,on=["dataset","gene_id"])


# %%
def compute_metrics(tpl, target, dataset, col):
    rmse = sklearn.metrics.mean_squared_error(*tpl,squared=False)
    mae = sklearn.metrics.mean_absolute_error(*tpl)
    r = scipy.stats.pearsonr(*tpl)
    r2 = sklearn.metrics.r2_score(*tpl)
    return ({
        "target":target,
        "dataset":dataset,
        "predictor":col.split("_")[-1],
        "rmse":rmse,
        "mae":mae,
        "r":r[0],
        "r2":r2
    })

rows = []
for dataset in ["GTEx","Cardoso-Moreira"]:
    subset = pred_df_merged.query('dataset == @dataset')
    for col in [k for k in pred_df.keys() if "pred" in k and "null" not in k]:
        tpl = (subset["log_obs"],subset[col]) # total variation
        rows.append(compute_metrics(tpl, "total", dataset, col))
        tpl = (subset["log_obs_mean"],subset[col+"_mean"]) # between gene
        rows.append(compute_metrics(tpl, "between-gene", dataset, col))
        tpl = (subset["log_obs"] - subset["log_obs_mean"],
               subset[col] - subset[col+"_mean"]) # between condition
        rows.append(compute_metrics(tpl, "between-condition", dataset, col))
        
perf_df = pd.DataFrame(rows) 

rows = []
for dataset in ["GTEx","Cardoso-Moreira"]:
    subset = pred_df_merged.query('dataset == @dataset')
    for col in [k for k in pred_df.keys() if "pred" in k and "null" not in k and "enformer" not in k]:
        tpl = (subset["log_pred_enformer"],subset[col]) # total variation
        rows.append(compute_metrics(tpl, "total", dataset, col))
        tpl = (subset["log_pred_enformer_mean"],subset[col+"_mean"]) # between gene
        rows.append(compute_metrics(tpl, "between-gene", dataset, col))
        tpl = (subset["log_pred_enformer"] - subset["log_pred_enformer_mean"],
               subset[col] - subset[col+"_mean"]) # between condition
        rows.append(compute_metrics(tpl, "between-condition", dataset, col))
        
self_corr_df = pd.DataFrame(rows)

# %%
perf_df.query('target == "total"')

# %%
predictor_cats = ['basenji2',#'1001','12501',
                  '39321','49153','65537','98305','enformer']
#pairs = [(x,y) for x, y in zip(predictor_cats[:-1],predictor_cats[1:])]
pairs = [('basenji2',x) for x in predictor_cats[1:]]

perf_df_indexed = perf_df.set_index(['dataset','target','predictor'])
    
rows = []
for dataset in ["GTEx","Cardoso-Moreira"]:
    for target in ["total","between-gene","between-condition"]:
        for pair in pairs:
            delta = (perf_df_indexed.loc[dataset,target,pair[1]] 
                     - perf_df_indexed.loc[dataset,target,pair[0]])
            metrics = {
                "dataset":dataset,
                "target":target,
                "predictor":pair[1]
            }
            for k in delta.keys():
                metrics[k] = delta[k]
            rows.append(metrics)
            
delta_over_basenji2 = pd.DataFrame(rows)

# %%
delta_over_basenji2

# %%
plot_df = delta_over_basenji2.query('target == "total"')

rename_dict = {
    "39321":"39kb (Basenji2 RF)",
    "49153":"49kb (1/4th RF)",
    "65537":"65kb (1/3rd RF)",
    "98305":"98kb (1/2 RF)",
    "enformer": "196kb (Full)"
}
cats = ["39kb (Basenji2 RF)","49kb (1/4th RF)","65kb (1/3rd RF)","98kb (1/2 RF)","196kb (Full)"]
plot_df["predictor"] = plot_df["predictor"].apply(lambda x: rename_dict[x])
plot_df["predictor"] = pd.Categorical(plot_df["predictor"], cats)

plot_df["mae"] = plot_df['mae']*-1
plot_df["rmse"] = plot_df['rmse']*-1

p = (p9.ggplot(data=plot_df
           ,mapping=p9.aes(x="dataset",y="rmse", fill="predictor"))
 + p9.geom_bar(stat="identity",position="dodge")
 + p9.labs(x="",y="Improvement over Basenji2\n(Reduction in RMSE)",fill="Sequence window size (kb)")
 + p9.theme(legend_position=(0.35,0.745), legend_background=p9.element_rect(color="none", size=2, fill='none'),
            legend_text=p9.element_text(size=8),legend_key_size=8,
            legend_title=p9.element_text(size=8),
           axis_text_x=p9.element_text(rotation=30, hjust=1),
           axis_title=p9.element_text(size=10))
)
p

# %%
plot_df = delta_over_basenji2.query('target != "total"')

rename_dict = {
    "1001":"1kb (~0.5%)",
    "12501":"12.5kb (~6%)",
    "39321":"39kb (1/5th)",
    "49153":"49kb (1/4th)",
    "65537":"65kb (1/3rd)",
    "98305":"98kb (1/2)",
    "enformer": "196kb (Full)"
}
cats = [x[1] for x in rename_dict.items()]
plot_df["predictor"] = plot_df["predictor"].apply(lambda x: rename_dict[x])
plot_df["predictor"] = pd.Categorical(plot_df["predictor"], cats)

plot_df["target"] = plot_df["target"].apply(lambda x: x.replace("-","\n"))
#plot_df["dataset"] = plot_df["dataset"].apply(lambda x: x.replace("-","\n"))

p = (p9.ggplot(data=plot_df
           ,mapping=p9.aes(x="target",y="r2", fill="predictor"))
 + p9.geom_bar(stat="identity",position="dodge")
 + p9.labs(x="",y="Improvement over Basenji2\n(Additional Variance Explained)",fill="Window size:")
 + p9.theme(#legend_position=(0.35,0.745),
            legend_box_margin=0,
            legend_background=p9.element_rect(color="none", size=2, fill='none'),
            legend_text=p9.element_text(size=8),legend_key_size=8,
            legend_title=p9.element_text(size=8),
           axis_text_x=p9.element_text(rotation=30, hjust=1),
           axis_title=p9.element_text(size=10))
 + p9.facet_wrap('~dataset')
)
p

# %%
p.save("Graphics/" + "windows_basenji" + ".svg", width=6.4, height=4.8, dpi=300)

# %%
0.45/0.66

# %%
plot_df = perf_df.query('target != "total"')

rename_dict = {
    "1001":"1kb (~0.5%)",
    "12501":"12.5kb (~6%)",
    "39321":"39kb (1/5th)",
    "49153":"49kb (1/4th)",
    "65537":"65kb (1/3rd)",
    "98305":"98kb (1/2)",
    "enformer": "196kb (Full)"
}
cats = [x[1] for x in rename_dict.items()]
plot_df = plot_df.query('predictor in @rename_dict.keys()')
plot_df["predictor"] = plot_df["predictor"].apply(lambda x: rename_dict[x])
plot_df["predictor"] = pd.Categorical(plot_df["predictor"], cats)

plot_df["target"] = plot_df["target"].apply(lambda x: x.replace("-","\n"))

p = (p9.ggplot(data=plot_df
           ,mapping=p9.aes(x="target",y="r2", fill="predictor"))
 + p9.geom_bar(stat="identity",position="dodge")
 + p9.labs(x="",y="Fraction of Variance Explained",
           fill="Window size:")
 + p9.theme(#legend_position=(0.35,0.745), 
            legend_box_margin=0,
            legend_background=p9.element_rect(color="none", size=2, fill='none'),
            legend_text=p9.element_text(size=8),legend_key_size=8,
            legend_title=p9.element_text(size=8),
           axis_text_x=p9.element_text(rotation=30, hjust=1),
           axis_title=p9.element_text(size=10))
 + p9.facet_wrap('~dataset')
)
p

# %%
p.save("Graphics/" + "windows_total" + ".svg", width=2.8, height=2.9, dpi=300)

# %%
plot_df = self_corr_df.query('target != "total" & dataset == "GTEx"')

rename_dict = {
    "1001":"1kb (~0.5% RF)",
    "12501":"12.5kb (~6% RF)",
    "39321":"39kb (1/5th RF)",
    "49153":"49kb (1/4th RF)",
    "65537":"65kb (1/3rd RF)",
    "98305":"98kb (1/2 RF)",
    "enformer": "196kb (Full)"
}
cats = [x[1] for x in rename_dict.items()]
plot_df = plot_df.query('predictor in @rename_dict.keys()')
plot_df["predictor"] = plot_df["predictor"].apply(lambda x: rename_dict[x])
plot_df["predictor"] = pd.Categorical(plot_df["predictor"], cats)

plot_df["target"] = plot_df["target"].apply(lambda x: x.replace("-","\n"))

p = (p9.ggplot(data=plot_df
           ,mapping=p9.aes(x="target",y="r2", fill="predictor"))
 + p9.geom_bar(stat="identity",position="dodge")
 + p9.labs(x="",y="Fraction of Variation in Predictions\nExplained by Sequence Window",
           fill="Sequence window size (kb)")
 + p9.theme(#legend_position=(0.35,0.745), 
            legend_background=p9.element_rect(color="none", size=2, fill='none'),
            legend_text=p9.element_text(size=8),legend_key_size=8,
            legend_title=p9.element_text(size=8),
           axis_text_x=p9.element_text(rotation=30, hjust=1),
           axis_title=p9.element_text(size=10))
 #+ p9.facet_wrap('~target')
)
p

# %% [markdown]
# ## Analysis - eQTL

# %%
# Check consistency of ordering

with open(base_path_results_gtex_fc  + "gtex_eqtl_at_tss-enformer-latest_results.tsv", 'r') as tsv:
    header = tsv.readline()
    cols = header.split("\t")
cols_1 = [k for k in cols if "CAGE" in k and k.endswith("landmark_sum")]

with open(base_path_results  + "tss_sim_groupby_20_05_2022.tsv", 'r') as tsv:
    header = tsv.readline()
    cols = header.split("\t")
cols_2 = [k for k in cols if "CAGE" in k and k.endswith("landmark_sum")]

assert cols_1 == cols_2

# %%
with open(base_path_results_gtex_fc  + "gtex_eqtl_at_tss-enformer-latest_results.tsv", 'r') as tsv:
    header = tsv.readline()
    cols = header.split("\t")
universal_samples = ['Clontech Human Universal Reference Total RNA', 'SABiosciences XpressRef Human Universal Total RNA', 'CAGE:Universal RNA - Human Normal Tissues Biochain']
cage_tracks = [k for k in cols if "CAGE" in k and k.endswith("landmark_sum") and not any(x in k for x in universal_samples)]

metadata_cols = ['chromosome','variant_pos','variant_id','cs_id','tissue','variant_seq','allele']

results = pd.read_csv(base_path_results_gtex_fc  + "gtex_eqtl_at_tss-enformer-latest_results.tsv",sep="\t",
                      #nrows=4000,
                     usecols = metadata_cols + ['offset','orient'] + cage_tracks)

# %%
results = (results.groupby(metadata_cols)[cage_tracks].mean().reset_index())

# %% [markdown]
# ### Prepare

# %% [markdown]
# #### Match to tissue

# %%
with open(base_path_results + "gtex_enformer_lm_models_pseudocount1.pkl", 'rb') as handle:
    model_dict = pickle.load(handle)

# %%
df_list = []

for sample in set(results["tissue"]):
    subset = results.query('tissue == @sample')
    lm_pipe = model_dict[sample]["ingenome"]
    X = np.array(np.log10(subset[cage_tracks] + 1))
    y_pred = lm_pipe.predict(X).reshape(-1)
    subset_pred = subset[metadata_cols].copy()
    subset_pred['log_pred'] = y_pred 
    df_list.append(subset_pred)

eqtl_pred_df = pd.concat(df_list)

# %%
eqtl_pred_df.to_csv(base_path_results_gtex_fc + 'modelmatched_preds.tsv',sep="\t",index=None)

# %% [markdown]
# #### Compute fold changes

# %%
eqtl_pred_df = pd.read_csv(base_path_results_gtex_fc + 'modelmatched_preds.tsv',sep="\t")

# %%
eqtl_pred_ref = eqtl_pred_df.query('allele == "ref"').drop(columns="allele")
eqtl_pred_alt = eqtl_pred_df.query('allele == "alt"').drop(columns="allele")

eqtl_pred_merged = eqtl_pred_ref.merge(eqtl_pred_alt,on=["chromosome","variant_pos","variant_id","cs_id","tissue"],suffixes=["_ref","_alt"])

# %%
eqtl_pred_merged["log2_fc"] = (eqtl_pred_merged["log_pred_alt"] - eqtl_pred_merged["log_pred_ref"])/np.log10(2)
eqtl_pred_merged["abs_log2_fc"] = np.abs(eqtl_pred_merged["log2_fc"])
eqtl_pred_merged["pct_change"] = (2**eqtl_pred_merged["log2_fc"] - 1)*100
eqtl_pred_merged["abs_pct_change"] = np.abs(eqtl_pred_merged["pct_change"])

# %% [markdown]
# #### Merge with metadata

# %%
susie_df_small_blocks = pd.read_csv(base_path_data_gtex_fc + "susie_df_small_blocks.tsv",sep="\t")
susie_df_small_blocks["chromosome"] = "chr" + susie_df_small_blocks["chromosome"] 
#susie_df_small_blocks = susie_df_small_blocks.melt([k for k in susie_df_small_blocks.keys() if k not in ["ref","alt"]],value_name="variant_seq",var_name="allele")

# %%
susie_df_merged = eqtl_pred_merged.merge(susie_df_small_blocks.drop(columns=['chromosome', 'position', 'ref', 'alt']),
                                         on=["variant_id","cs_id","tissue"])

# %% [markdown]
# #### Resolve blocks
#
# For each credible set, we determine the maximal effect

# %%
max_effects =(susie_df_merged.groupby(['gene_id','cs_id','tissue'])
              [["log2_fc","abs_log2_fc", "abs_pct_change", 'abs_tss_distance_min','abs_tss_distance_mean']]
              .max().reset_index().sort_values('abs_log2_fc'))

# %%
susie_df_max_merged = susie_df_merged.merge((max_effects[['gene_id','cs_id','tissue',"abs_pct_change"]]
                                          .rename(columns={"abs_pct_change":"max_abs_pct_change"})),
                                           on=['gene_id','cs_id','tissue'])

# %% [markdown]
# ### Maximal effect by block

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=susie_df_max_merged.query('abs_pct_change == max_abs_pct_change'),
               mapping=p9.aes(x="abs_tss_distance",y="abs_pct_change"))
 #+ p9.geom_point(size=0.5)
+ p9.geom_bin2d(bins=50, raster=True)
 + p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_hline(yintercept=1)
 + p9.geom_smooth(method="lm")
 + p9.labs(x="Distance (bp) between TSS and eQTL\n(strongest variant in credible set)",
           y="Predicted Change in Expression (%)\n(strongest variant in credible set)")
 + p9.theme(legend_key_size=8,
            legend_title=p9.element_text(size=8),
            legend_text=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,
            legend_position=(0.22,0.3),
            legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10))
)
p

# %%
p.save("Graphics/" + "distal_eQTL_pred" + ".svg", width=2.7, height=2.9, dpi=300)

# %%
max_effects["abs_pct_change"].median()

# %%
susie_df_max_merged.query('abs_pct_change == max_abs_pct_change').query('abs_tss_distance > 10_000')["abs_pct_change"].median()

# %%
one_over_dist_df = susie_df_max_merged.query('abs_pct_change == max_abs_pct_change')

one_over_dist_df["log_dist"] = np.log10(one_over_dist_df["abs_tss_distance"])
one_over_dist_df["log_pct"] = np.log10(one_over_dist_df["abs_pct_change"])

ols_1_over_dist = (statsmodels.regression.linear_model.OLS
                  .from_formula('log_pct ~ log_dist',
                                data=(one_over_dist_df))
                  )
res_ols_1_over_dist = ols_1_over_dist.fit()

res_ols_1_over_dist.summary()

# %% [markdown]
# ### Distribution over distance

# %%
#sample = scipy.stats.pareto.rvs(b=1.0, loc=-5000, scale=6000, size=100_000)
sample = scipy.stats.pareto.rvs(1.8423501778362956, -17829.909798770816, 17831.90979877046, size=100_000)

p = (p9.ggplot(data=max_effects[["abs_tss_distance_mean","gene_id"]].drop_duplicates()
               ,mapping=p9.aes(x="abs_tss_distance_mean"))
 #+ p9.geom_point(size=0.5)
 #+ p9.scale_x_log10()
 + p9.stat_ecdf()
 + p9.stat_ecdf(data=pd.DataFrame({"abs_tss_distance_mean":sample}).query("abs_tss_distance_mean < 100_000"),
                color="red")
 #+ p9.geom_vline(xintercept=10_000)
 #+ p9.geom_hline(yintercept=0.8)
 + p9.labs(x="Distance (bp) between TSS and eQTL credible set midpoint",
           y="Cumulative Fraction")
 + p9.theme(legend_key_size=8,
            legend_title=p9.element_text(size=8),
            legend_text=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,
            #legend_position=(0.29,0.2),
            legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10))
)
p

# %%
#a = 1658070#2_000_000
#exp_line = pd.DataFrame({"x":np.linspace(1000,100_000),
#                         "y":a*(1/(np.linspace(1000,100_000)))})

p = (p9.ggplot(data=max_effects[["abs_tss_distance_mean","gene_id"]].drop_duplicates()
               ,mapping=p9.aes(x="abs_tss_distance_mean"))
 #+ p9.geom_point(size=0.5)
 #+ p9.scale_x_log10()
 + p9.geom_histogram(bins=50)#binwidth=2000)
 #+ p9.scale_y_log10()
 #+ p9.geom_line(data=exp_line,
 #                mapping=p9.aes(x="x",y="y"),
 #            color="blue")
 + p9.labs(x="Distance (bp) between TSS and eQTL\n(credible set midpoint)",
           y="Count")
 + p9.theme(legend_key_size=8,
            legend_title=p9.element_text(size=8),
            legend_text=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,
            #legend_position=(0.29,0.2),
            legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10))
)
p

# %%
a = 10**(1.6316)#2_000_000
k = 0.6
exp_line = pd.DataFrame({"x":np.linspace(0.25,100_000,10000),
                         "y":a*(1/(np.linspace(0.25,100_000,10000)**k))})

p = (p9.ggplot(data=susie_df_max_merged.query('abs_pct_change == max_abs_pct_change'),
               mapping=p9.aes(x="abs_tss_distance", y="abs_pct_change"))
 + p9.geom_point(size=0.5, color="black")
 #+ p9.scale_x_log10()
 #+ p9.geom_histogram(bins=50)#binwidth=2000)
 #+ p9.scale_y_log10()
 #+ p9.geom_line(data=exp_line,
 #                mapping=p9.aes(x="x",y="y"),
 #            color="blue")
 + p9.labs(x="Distance (bp) between TSS and eQTL (strongest variant)",
           y="Predicted Change in Expression (%)\n(strongest variant in credible set)")
 #+ p9.geom_line(data=exp_line,
 #                mapping=p9.aes(x="x",y="y"),
 #            color="blue")
 + p9.theme(legend_key_size=8,
            legend_title=p9.element_text(size=8),
            legend_text=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,
            #legend_position=(0.29,0.2),
            legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10))
)
p

# %%
a = 1000000#2_000_000
exp_line = pd.DataFrame({"x":np.linspace(1000,100_000),
                         "y":a*(1/(np.linspace(1000,100_000)))})

p = (p9.ggplot(data=max_effects[["abs_tss_distance_mean","gene_id"]].drop_duplicates()
               ,mapping=p9.aes(x="abs_tss_distance_mean"))
 #+ p9.geom_point(size=0.5)
 + p9.scale_x_log10()
 + p9.geom_histogram(breaks=np.linspace(1,100_000))#binwidth=2000)
 + p9.scale_y_log10()
 + p9.coords.coord_cartesian(xlim=(3,5))
 + p9.geom_point(data=exp_line,
                 mapping=p9.aes(x="x",y="y"))
 + p9.labs(x="Distance (bp) between TSS and eQTL credible set midpoint",
           y="Count")
 + p9.theme(legend_key_size=8,
            legend_title=p9.element_text(size=8),
            legend_text=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,
            #legend_position=(0.29,0.2),
            legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10))
)
p

# %% [markdown]
# ### Normalized Effect size over distance

# %%
effect_by_dist = pd.read_csv(base_path_data_gtex_fc + "effect_by_dist.tsv",sep="\t")

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p=(p9.ggplot(data=effect_by_dist,mapping=p9.aes(x="abs_tss_distance_min",y="max_slope")) 
 #+ p9.geom_point(alpha=0.2)
 + p9.geom_bin2d(bins=50, raster=True)
 + p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.scale_x_log10()
 + p9.scale_y_log10()
 + p9.geom_smooth(method="lm")
 + p9.labs(x="Distance (bp) between TSS and eQTL\n(strongest variant in credible set)",y="Observed Effect Size (Slope)\n(strongest variant in credible set)")
 + p9.theme(legend_key_size=8,
            legend_title=p9.element_text(size=8),
            legend_text=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,
            #legend_position=(0.22,0.3),
            legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10))
)
p

# %%
p.save("Graphics/" + "distal_eQTL_obs" + ".svg", width=2.7, height=2.9, dpi=300)

# %% [markdown]
# ### Which eQTL are seen in training

# %%
valid_eqtl = susie_df_max_merged.query('abs_pct_change == max_abs_pct_change')
valid_eqtl["tss_id"] = valid_eqtl["chromosome"] + valid_eqtl["tss"].astype('str')
valid_eqtl["tss_start"] = valid_eqtl["tss"]
valid_eqtl["tss_end"] = valid_eqtl["tss"] + 1
valid_eqtl["var_start"] = valid_eqtl['variant_pos'] - 1

# %%
train_regions = (pd.read_csv(base_path_data + "human_regions.bed", names=["Chromosome","Start","End","set"], sep="\t")
                            .query('set == "train"')
                            .drop(columns="set"))
train_regions["id"] = train_regions["Chromosome"] + train_regions["Start"].astype('str') + train_regions["End"].astype('str')

train_regions_wide = train_regions.copy()
train_regions_wide["Start"] = train_regions_wide["Start"] - (SEEN_SEQUENCE_LENGTH - 131072)/2
train_regions_wide["End"] = train_regions_wide["End"] + (SEEN_SEQUENCE_LENGTH - 131072)/2
train_regions_wide = pr.PyRanges(train_regions_wide)

train_regions_predict = train_regions.copy()
train_regions_predict["Start"] = train_regions_predict["Start"] + (131072 - 896*128)/2
train_regions_predict["End"] = train_regions_predict["End"] - (131072 - 896*128)/2
train_regions_predict = pr.PyRanges(train_regions_predict)

# %%
tss_sites = pr.PyRanges(valid_eqtl[['chromosome', 'tss_start', 'tss_end','tss_id']]
                         .rename(columns={'chromosome':'Chromosome',
                                         "tss_start":"Start",
                                         "tss_end":"End"})
                        .drop_duplicates()
                       )

eqtl_sites = pr.PyRanges(valid_eqtl[['chromosome', 'variant_pos','var_start','variant_id','tss_id']]
                         .rename(columns={'chromosome':'Chromosome',
                                         "var_start":"Start",
                                         "variant_pos":"End"})
                        .drop_duplicates()
                       )

# %%
tss_in_train = train_regions_predict.join(tss_sites, suffix="_tss").df
eqtl_in_train = train_regions_wide.join(eqtl_sites, suffix="_eqtl").df

# %%
both_in_train = tss_in_train.merge(eqtl_in_train.drop(columns="Chromosome"),suffixes=("_predicted_region","_seen_region"), on=["tss_id","id"])

# %%
#seen_pairs = both_in_train[["tss_mid","enhancer_mid","id"]].drop_duplicates().merge(valid_enhancers, on=["tss_mid","enhancer_mid"])
valid_eqtl_seen = (tss_in_train.groupby("tss_id")
                      .size().reset_index()
                      .rename(columns={0:"tss_seen_count"})
                      .merge(valid_eqtl, on=["tss_id"]))
valid_eqtl_seen = (both_in_train.groupby(['variant_id','tss_id'])
                      .size().reset_index()
                      .rename(columns={0:"eqtl_seen_count"})
                      .merge(valid_eqtl_seen, on=['variant_id','tss_id'],how="right"))

valid_eqtl_seen = (valid_eqtl_seen.groupby('tss_id')
                        .size().reset_index().rename(columns={0:"total_eqtl"})
                        .merge(valid_eqtl_seen, on=["tss_id"])
                       )

valid_eqtl_seen['fully_seen'] = (valid_eqtl_seen['eqtl_seen_count'] == valid_eqtl_seen['tss_seen_count'])

valid_eqtl_seen = (valid_eqtl_seen.groupby('tss_id')['fully_seen']
                        .sum().reset_index().rename(columns={'fully_seen':"fully_seen_eqtl"})
                        .merge(valid_eqtl_seen, on=["tss_id"])
                       )

# %%
# for tss, which are seen
# check for difference between enhancers which are seen fully and those which are not
seen_twice = valid_eqtl_seen.query('tss_seen_count > 0')

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

p = (p9.ggplot(data=seen_twice
           ,mapping=p9.aes(x="abs_tss_distance",y="abs_pct_change",color="fully_seen"))
 + p9.scale_y_log10()
 + p9.scale_x_log10()
 + p9.geom_hline(yintercept=10)
 + p9.geom_smooth(method="lm")#,color="black")
 + p9.geom_point(size=1,alpha=0.2)
 + p9.labs(x="Distance (bp) between TSS and eQTL\n(strongest variant in credible set)",
           y="Predicted Change in Expression (%)\n(strongest variant in credible set)", 
           color="Seen with the TSS\nduring training")
 #+ p9.facet_wrap("~fully_seen")
 + p9.theme(legend_key_size=9,legend_text=p9.element_text(size=9),
            legend_title=p9.element_text(size=9),
            legend_position=(0.3,0.2),legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10)
           )
)
p

# %% [markdown]
# ### Distribution of encode CRE 

# %%
cres = pr.read_bed(base_path_data_gtex_fc + "GRCh38-cCREs.bed")

gene_regions = (susie_df_small_blocks[["chromosome","tss","gene_id","Strand"]]
                .rename(columns={"chromosome":"Chromosome"})
                .drop_duplicates()
               )
gene_regions["Start"] = gene_regions["tss"] - 95_000
gene_regions["End"] = gene_regions["tss"] + 95_000
gene_regions = pr.PyRanges(gene_regions)

gene_cre_df = gene_regions.join(cres,suffix="_cre",strandedness=False).df.rename(columns={"Strand_cre":"cre_type"})

# %%
gene_cre_df["is_enhancer"] = gene_cre_df["cre_type"].apply(lambda x: True if "ELS" in x else False).astype('bool')
gene_cre_df["is_promoter"] = gene_cre_df["cre_type"].apply(lambda x: True if "PLS" in x else False).astype('bool')

# %%
gene_cre_df["Strand"] = gene_cre_df["Strand"].astype("int")
gene_cre_df["cre_mid"] = ((gene_cre_df["Start_cre"] + gene_cre_df["End_cre"])/2)
gene_cre_df["tss_distance_mean"] = gene_cre_df["cre_mid"] - gene_cre_df["tss"]
gene_cre_df["abs_tss_distance_mean"] = np.abs(gene_cre_df["cre_mid"] - gene_cre_df["tss"])
gene_cre_df["tss_distance_left"] = gene_cre_df["Start_cre"] - gene_cre_df["tss"]
gene_cre_df["tss_distance_right"] = gene_cre_df["End_cre"] - gene_cre_df["tss"]
gene_cre_df["sign_tss_distance_mean"] = np.sign(gene_cre_df["tss_distance_mean"])
gene_cre_df["sign_tss_distance_left"] = np.sign(gene_cre_df["tss_distance_left"])
gene_cre_df["sign_tss_distance_right"] = np.sign(gene_cre_df["tss_distance_right"])
# once again, only consider downstream elements
# throw out blocks which are 3' of the canonical TSS
# if tss_distance > 0, then ==TSS===CRE==
# so if strand = 1, we want TSS distance to be negative: ==CRE==TSS[CDS]
# else, if strand = -1, we want TSS distance to be positive: [CDS]TSS==CRE==
# we also exclude blocks with inconsistent sign (i.e. blocks which span the TSS)
gene_cre_df = gene_cre_df.query('Strand != sign_tss_distance_mean & sign_tss_distance_left == sign_tss_distance_right')

# %%
# combine with eQTL plot
eqtl_dist_distbn = max_effects[["abs_tss_distance_mean","gene_id"]].drop_duplicates()
eqtl_dist_distbn["region_type"] = "eQTL Credible Set"
gene_dist_distbn = (gene_cre_df
                    .query('is_enhancer')
                    [["gene_id","abs_tss_distance_mean","cre_mid"]]
                    .drop_duplicates()[["abs_tss_distance_mean","gene_id"]])
gene_dist_distbn["region_type"] = "Enhancer-Like CRE"
dist_distbn = pd.concat([eqtl_dist_distbn,gene_dist_distbn])

p = (p9.ggplot(data=dist_distbn,mapping=p9.aes(x="abs_tss_distance_mean",color="region_type"))
 + p9.stat_ecdf()
 #+ p9.scale_x_log10()
 + p9.labs(x="Distance (bp) between TSS and regulatory region midpoint",
           y="Cumulative Fraction",color="Type of Region:")
 + p9.theme(legend_key_size=8,
            legend_title=p9.element_text(size=8),
            legend_text=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,
            legend_position=(0.7,0.25),
            legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10))
)
p

# %%
missing_genes = {"ADSS2":'ADSS',
                 "ADSS1":'ADSSL1',
                 "ZFTA":'C11orf95',
                 "RUSF1":'C16orf58',
                 "HROB":'C17orf53',
                 "BRME1":'C19orf57',
                 "MIR9-1HG":'C1orf61',
                 "DNAAF9":'C20orf194',
                 "BBLN":'C9orf16',
                 "CARS1":'CARS',
                 "YJU2B":'CCDC130',
                 "MIX23":'CCDC58',
                 "UVSSA":'CRIPAK', #same strand
                 "EOLA2":'CXorf40B',
                 "PABIR1":'FAM122A',
                 "NALF1":'FAM155A',
                 "SLX9":'FAM207A', #same strand
                 "CYRIA":'FAM49A',
                 "CYRIB":'FAM49B',
                 "CEP43":'FGFR1OP',
                 "CEP20":'FOPNL',
                 "H1-10":'H1FX',
                 "H2AZ2":'H2AFV',
                 "MACROH2A1":'H2AFY',
                 "H3-3A":'H3F3A',
                 "HARS1":'HARS',
                 "H1-5":'HIST1H1B',
                 "H1-2":'HIST1H1C',
                 "H1-3":'HIST1H1D',
                 "H1-4":'HIST1H1E',
                 "H2AC11":'HIST1H2AG',
                 "H2AC13":'HIST1H2AI',
                 "H2BC4":'HIST1H2BC',
                 "H2BC5":'HIST1H2BD',
                 "H2BC7":'HIST1H2BF',
                 "H2BC8":'HIST1H2BG',
                 "H2BC11":'HIST1H2BJ',
                 "H2BC12":'HIST1H2BK',
                 "H2BC13":'HIST1H2BL',
                 "H2BC15":'HIST1H2BN',
                 "H3C4":'HIST1H3D',
                 "H4C2":'HIST1H4B',
                 "H4C3":'HIST1H4C',
                 "H4C8":'HIST1H4H',
                 "H4C11":'HIST1H4J',
                 "H2AW":'HIST3H2A',
                 "IARS1":'IARS',
                 "GARRE1":'KIAA0355',
                 "ELAPOR1":'KIAA1324',
                 "ELAPOR2":'KIAA1324L',
                 "MARCHF6":'MARCH6',
                 "MARCHF9":'MARCH9',
                 "PPP5D1P":'PPP5D1',
                 "QARS1":'QARS',
                 "METTL25B":'RRNAD1',
                 "TARS3":'TARSL2',
                 "DYNLT2B":'TCTEX1D2',
                 "STING1":'TMEM173',
                 "PEDS1":'TMEM189',
                 "DYNC2I1":'WDR60',
                 "DNAAF10":'WDR92',
                 "POLR1H":'ZNRD1'}

gene_locs = gtf_df.df.query('Feature == "gene"')[["Chromosome","Start","End","Score","Strand","Frame","gene_id","gene_name"]]
gene_locs["gene"] = gene_locs["gene_name"].apply(lambda x: missing_genes[x] if x in missing_genes else x) 

ziga_df = pd.read_csv(base_path_data_fulco + "ziga_additional_columns.tsv",sep="\t")
ziga_df = (ziga_df.merge(gene_locs[["Strand","gene", "gene_id"]].drop_duplicates(),on="gene"))
ziga_df["gene_id"] = ziga_df["gene_id"].apply(lambda x: x.split('.')[0])
ziga_df["region_type"] = ziga_df["validated"].apply(lambda x: "Valid Enhancer" if x else "Non-valid Enhancer")


# %%
dist_distbn_all = pd.concat([dist_distbn,
                             (ziga_df[["actual_tss_distance","gene_id","region_type"]]
                              .rename(columns={"actual_tss_distance":"abs_tss_distance_mean"})
                             .query('abs_tss_distance_mean < 95_000'))])

# %%
p = (p9.ggplot(data=dist_distbn_all,mapping=p9.aes(x="abs_tss_distance_mean",color="region_type"))
 + p9.stat_ecdf(size=0.7)
 #+ p9.scale_x_log10()
 #+ p9.geom_vline(xintercept=10_000)
 + p9.labs(x="Distance (bp) between TSS\nand regulatory region midpoint",
           y="Cumulative Fraction",color="Type of Region:")
 + p9.theme(legend_key_size=8,
            legend_title=p9.element_text(size=8),
            legend_text=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,
            legend_position=(0.68,0.25),
            legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10))
)
p

# %%
p.save("Graphics/" + "classimba_distbn" + ".svg", width=2.9, height=3.0, dpi=300)

# %% [markdown]
# # Kircher Saturation Mutagenesis
#
# We apply a number of models to the Kircher saturation mutagenesis data, which tested many variants in different loci (mostly promoters) using a reporter plasmid.
#
# Some things are notable here:
#
# - Data are of high quality (high replicate accuracy)
# - Saturation mutagenesis: many variants!
# - Most loci are promoters
# - (Almost) all appear to be baseline expressed according to enformer (the denominator is never insane)
# - The total correlation (pooling data across loci) is considerably lower than the weighted average of per-locus correlations for CAGE but not for DNASE (miscalibration for CAGE predictions)

# %%
base_path_data = "Data/Kircher_saturation_mutagenesis/"
base_path_results = "Results/Kircher_saturation_mutagenesis/"
fasta_file_hg19 = 'Data/Genome/genome_hg19.fa'
fasta_extractor_hg19 = FastaStringExtractor(fasta_file_hg19)

# %%
locus_list = []
for path in glob.glob(base_path_data + "*/*.tsv"):
    locus = path.split("/")[-1].split("_")[-1].split(".")[0]
    locus_df = pd.read_csv(path,sep="\t", names=["Chromosome", 
                                                  "Pos_hg38",
                                                  "Ref",
                                                  "Alt",
                                                  "log2FC",
                                                  "Variance"])
    locus_df["Locus"] = locus
    locus_list.append(locus_df)
kircher_df = pd.concat(locus_list)

# %%
kircher_df = kircher_df.reset_index(drop=True).rename(columns={"Locus":"locus","Pos_hg38":"variant_pos","Chromosome":"chromosome"})


# %%
#set(kircher_df["locus"])

# %%
def get_matching_cols(cell_descriptor, suffix = "_landmark_sum_wide", expr_type="CAGE", target_df=target_df):
    keys = [x for x in 
            target_df.loc[(target_df.description.str.contains(expr_type)) & (target_df.description.str.contains(cell_descriptor, case=False))]["description"]]
    if len(keys) == 0:
        keys = [x for x in target_df.loc[(target_df.description.str.contains(expr_type))]["description"]]
    # if a key appears several times, add a numbering
    key_dict = collections.defaultdict(list)
    # create equivalence classes
    for k in keys:
        key_dict[k].append(k)
    # number each one if equi class > 1 and concat
    keys_new = []
    for k in key_dict.keys():
        equi_class = key_dict[k]
        if len(equi_class) > 1:
            equi_class = [x + "_" + str(idx+1) for idx,x in enumerate(equi_class)]
        keys_new += equi_class
    keys = keys_new
    # add suffix
    keys = [x + suffix for x in keys]
    return keys


# %%
cell_type_dict = {
    "F9":get_matching_cols("hepg2"), 
    "LDLR":get_matching_cols("hepg2"), 
    "SORT1":get_matching_cols("hepg2"),
    "GP1BB":get_matching_cols("k562"), 
    "HBB":get_matching_cols("k562"), 
    "HBG1":get_matching_cols("k562"), 
    "PKLR":get_matching_cols("k562"),
    "HNF4A":get_matching_cols("hek293"), 
    "MSMB":get_matching_cols("hek293"), 
    "TERT-HEK293T":get_matching_cols("hek293"), 
    "MYCrs6983267":get_matching_cols("hek293"),
    "ZFAND3":get_matching_cols("pancreas"),
    "TERT-GBM":get_matching_cols("glioblastoma"),
    "IRF6":get_matching_cols("keratinocyte"),
    "IRF4":get_matching_cols("mel")
}

cell_type_dict_dnase = {
    "F9":get_matching_cols("hepg2",expr_type="DNASE"), 
    "LDLR":get_matching_cols("hepg2",expr_type="DNASE"), 
    "SORT1":get_matching_cols("hepg2",expr_type="DNASE"),
    "GP1BB":get_matching_cols("k562",expr_type="DNASE"), 
    "HBB":get_matching_cols("k562",expr_type="DNASE"), 
    "HBG1":get_matching_cols("k562",expr_type="DNASE"), 
    "PKLR":get_matching_cols("k562",expr_type="DNASE"),
    "HNF4A":get_matching_cols("hek293",expr_type="DNASE"), 
    "MSMB":get_matching_cols("hek293",expr_type="DNASE"), 
    "TERT-HEK293T":get_matching_cols("hek293",expr_type="DNASE"), 
    "MYCrs6983267":get_matching_cols("hek293",expr_type="DNASE"),
    "ZFAND3":get_matching_cols("pancreas",expr_type="DNASE"),
    "TERT-GBM":get_matching_cols("glioblastoma",expr_type="DNASE"),
    "IRF6":get_matching_cols("keratinocyte",expr_type="DNASE"),
    "IRF4":get_matching_cols("mel",expr_type="DNASE")
}

# %%
cell_type_dict_basenji1 = {
    "F9":get_matching_cols("hepg2", target_df=target_df_basenji1), 
    "LDLR":get_matching_cols("hepg2", target_df=target_df_basenji1), 
    "SORT1":get_matching_cols("hepg2", target_df=target_df_basenji1),
    "GP1BB":get_matching_cols("k562", target_df=target_df_basenji1), 
    "HBB":get_matching_cols("k562", target_df=target_df_basenji1), 
    "HBG1":get_matching_cols("k562", target_df=target_df_basenji1), 
    "PKLR":get_matching_cols("k562", target_df=target_df_basenji1),
    "HNF4A":get_matching_cols("hek293", target_df=target_df_basenji1), 
    "MSMB":get_matching_cols("hek293", target_df=target_df_basenji1), 
    "TERT-HEK293T":get_matching_cols("hek293", target_df=target_df_basenji1), 
    "MYCrs6983267":get_matching_cols("hek293", target_df=target_df_basenji1),
    "ZFAND3":get_matching_cols("pancreas", target_df=target_df_basenji1),
    "TERT-GBM":get_matching_cols("glioblastoma", target_df=target_df_basenji1),
    "IRF6":get_matching_cols("keratinocyte", target_df=target_df_basenji1),
    "IRF4":get_matching_cols("mel", target_df=target_df_basenji1)
}

cell_type_dict_dnase_basenji1 = {
    "F9":get_matching_cols("hepg2",expr_type="DNASE", target_df=target_df_basenji1), 
    "LDLR":get_matching_cols("hepg2",expr_type="DNASE", target_df=target_df_basenji1), 
    "SORT1":get_matching_cols("hepg2",expr_type="DNASE", target_df=target_df_basenji1),
    "GP1BB":get_matching_cols("k562",expr_type="DNASE", target_df=target_df_basenji1), 
    "HBB":get_matching_cols("k562",expr_type="DNASE", target_df=target_df_basenji1), 
    "HBG1":get_matching_cols("k562",expr_type="DNASE", target_df=target_df_basenji1), 
    "PKLR":get_matching_cols("k562",expr_type="DNASE", target_df=target_df_basenji1),
    "HNF4A":get_matching_cols("hek293",expr_type="DNASE", target_df=target_df_basenji1), 
    "MSMB":get_matching_cols("hek293",expr_type="DNASE", target_df=target_df_basenji1), 
    "TERT-HEK293T":get_matching_cols("hek293",expr_type="DNASE", target_df=target_df_basenji1), 
    "MYCrs6983267":get_matching_cols("hek293",expr_type="DNASE", target_df=target_df_basenji1),
    "ZFAND3":get_matching_cols("pancreas",expr_type="DNASE", target_df=target_df_basenji1),
    "TERT-GBM":get_matching_cols("glioblastoma",expr_type="DNASE", target_df=target_df_basenji1),
    "IRF6":get_matching_cols("keratinocyte",expr_type="DNASE", target_df=target_df_basenji1),
    "IRF4":get_matching_cols("mel",expr_type="DNASE", target_df=target_df_basenji1)
}

# %%
# reproducibility
repro_df = pd.DataFrame([
                        {"locus":"F9","replicate_r":0.61},
                        {"locus":"GP1BB","replicate_r":0.74},
                        {"locus":"HBB","replicate_r":0.62},
                        {"locus":"HBG1","replicate_r":0.78},
                        {"locus":"HNF4A","replicate_r":0.75},
                        {"locus":"IRF4","replicate_r":0.98},
                        {"locus":"IRF6","replicate_r":0.90},
                        {"locus":"LDLR","replicate_r":0.99},
                        {"locus":"MSMB","replicate_r":0.75},
                        {"locus":"MYCrs6983267","replicate_r":0.55},
                        {"locus":"PKLR","replicate_r":0.79},
                        {"locus":"SORT1","replicate_r":0.98},
                        {"locus":"TERT-GBM","replicate_r":0.90},
                        {"locus":"TERT-HEK293T","replicate_r":0.65},
                        {"locus":"ZFAND3","replicate_r":0.72}
                    ])

# %%
## Prepare Kircher for Expecto

# remap to hg19
kircher_lift = kircher_df["chromosome"] + ":" + kircher_df["variant_pos"].astype("str") + "-" + kircher_df["variant_pos"].astype("str")
kircher_lift.to_csv(base_path_data+"kircher_lift_input.txt", index=None, header=None)

kircher_lift = pd.read_csv(base_path_data+"kircher_lift_output.txt", header=None)

kircher_df["pos_hg19"] = kircher_lift[0].apply(lambda x: x.split(":")[1].split("-")[0])

# %% [markdown]
# ## Test that liftover worked

# %%
# test that this worked
idx = 0

row0_interval = kipoiseq.Interval(chrom=kircher_df.iloc[idx]["chromosome"],
                  start=int(kircher_df.iloc[idx]["pos_hg19"])-1, 
                  end=int(kircher_df.iloc[idx]["pos_hg19"]))

row0_insert_ref = kircher_df.iloc[idx]["Ref"]
row0_insert_alt = kircher_df.iloc[idx]["Alt"]
landmark = 0
offset = compute_offset_to_center_landmark(landmark,row0_insert_ref)

modified_sequence_ref, minbin, maxbin, landmarkbin = insert_sequence_at_landing_pad(row0_insert_ref, row0_interval, fasta_extractor_hg19, mode="replace",landmark=landmark, shift_five_end=offset)
modified_sequence_alt, minbin, maxbin, landmarkbin = insert_sequence_at_landing_pad(row0_insert_alt, row0_interval, fasta_extractor_hg19, mode="replace",landmark=landmark, shift_five_end=offset)

print(modified_sequence_ref[393216//2])
print(modified_sequence_alt[393216//2])

# %%
kircher_df["ID"] = kircher_df["locus"] + "_" + kircher_df["variant_pos"].astype("str")

# %%
# write vcf
kircher_df[["chromosome", "pos_hg19", "ID", "Ref", "Alt"]].sort_values(["chromosome","pos_hg19"]).to_csv(base_path_data+"hg19_vfc.txt",index=None,header=None,sep="\t")

# %% [markdown]
# ## Test Kircher within-genome

# %%
idx = 0

row0_interval = kipoiseq.Interval(chrom=kircher_df.iloc[idx]["chromosome"],
                  start=int(kircher_df.iloc[idx]["variant_pos"])-1, 
                  end=int(kircher_df.iloc[idx]["variant_pos"]))

# %%
landmark = 0

# %%
row0_interval

# %%
row0_insert_ref = kircher_df.iloc[idx]["Ref"]
row0_insert_alt = kircher_df.iloc[idx]["Alt"]

# %%
offset = compute_offset_to_center_landmark(landmark,row0_insert)

modified_sequence_ref, minbin, maxbin, landmarkbin = insert_sequence_at_landing_pad(row0_insert_ref, row0_interval, fasta_extractor, mode="replace",landmark=landmark, shift_five_end=offset)
modified_sequence_alt, minbin, maxbin, landmarkbin = insert_sequence_at_landing_pad(row0_insert_alt, row0_interval, fasta_extractor, mode="replace",landmark=landmark, shift_five_end=offset)

# %%
modified_sequence_ref == modified_sequence_alt

# %%
modified_sequence_ref[:393216//2] == modified_sequence_alt[:393216//2]

# %%
modified_sequence_ref[393216//2]

# %%
modified_sequence_alt[393216//2]

# %%
modified_sequence_ref[393216//2 + 1:] == modified_sequence_alt[393216//2 + 1:]

# %%
# 18ba9c3f-b9d5-45d4-a578-d796b613b160
# https://hb.flatironinstitute.org/deepsea/jobs/18ba9c3f-b9d5-45d4-a578-d796b613b160

# %% [markdown]
# ## Analysis - Enformer

# %%
enformer_cols = list(set(itertools.chain.from_iterable(cell_type_dict.values()))) + list(set(itertools.chain.from_iterable(cell_type_dict_dnase.values())))
result_df = pd.read_csv(base_path_results + "kircher_ingenome-enformer-latest_results.tsv",sep="\t", usecols=["chromosome","locus","variant_pos","variant_type", "nucleotide", "offset", "orient"]+enformer_cols)

# %%
result_df = result_df.query('offset in [-43, 0, 43]').groupby(["chromosome","locus","variant_pos","variant_type", "nucleotide"])[enformer_cols].mean().reset_index()

# %% [markdown]
# ### CAGE

# %% [markdown]
# #### Reproduce paper results for LDLR

# %%
pred_col = "CAGE:hepatocellular carcinoma cell line: HepG2 ENCODE, biol__landmark_sum_wide"
#pred_col = "CAGE:hepatocellular carcinoma cell line: HepG2 ENCODE, biol__center_sum"

# %%
ldlr = result_df.query('locus == "LDLR"')[['chromosome','locus','variant_pos','variant_type','nucleotide'] + [x for x in result_df.keys() if "HepG2" in x]]

# %%
# first aggregate, then log2fc
ldlr_aggreg = ldlr.groupby(["chromosome","locus","variant_pos","variant_type", "nucleotide"])[pred_col].mean().reset_index()
ldlr_aggreg[pred_col] =  np.log2(ldlr_aggreg[pred_col] + 1) 

ldlr_ref = ldlr_aggreg.query('variant_type == "ref"').drop(columns="variant_type").rename(columns={pred_col:pred_col + "_ref","nucleotide":"Ref"})
ldlr_alt = ldlr_aggreg.query('variant_type == "alt"').drop(columns="variant_type").rename(columns={pred_col:pred_col + "_alt","nucleotide":"Alt"})

ldlr_aggreg = ldlr_ref.merge(ldlr_alt,on=["chromosome", "locus", "variant_pos"])

ldlr_aggreg[pred_col] = ldlr_aggreg[pred_col+"_alt"] - ldlr_aggreg[pred_col+"_ref"]

# %%
ldlr_merged = ldlr_aggreg.merge(kircher_df, on=["chromosome", "locus", "variant_pos","Ref","Alt"])

# %%
scipy.stats.pearsonr(ldlr_merged[pred_col], ldlr_merged["log2FC"])

# %%
# first log2fc, then aggreg
ldlr_prelog = ldlr.copy()
ldlr_prelog[pred_col] =  np.log2(ldlr_prelog[pred_col] + 1) 
ldlr_aggreg = ldlr_prelog.groupby(["chromosome","locus","variant_pos","variant_type", "nucleotide"])[pred_col].mean().reset_index()

ldlr_ref = ldlr_aggreg.query('variant_type == "ref"').drop(columns="variant_type").rename(columns={pred_col:pred_col + "_ref","nucleotide":"Ref"})
ldlr_alt = ldlr_aggreg.query('variant_type == "alt"').drop(columns="variant_type").rename(columns={pred_col:pred_col + "_alt","nucleotide":"Alt"})

ldlr_aggreg = ldlr_ref.merge(ldlr_alt,on=["chromosome", "locus", "variant_pos"])

ldlr_aggreg[pred_col] = ldlr_aggreg[pred_col+"_alt"] - ldlr_aggreg[pred_col+"_ref"]

ldlr_merged = ldlr_aggreg.merge(kircher_df, on=["chromosome", "locus", "variant_pos","Ref","Alt"])

# %%
scipy.stats.pearsonr(ldlr_merged[pred_col], ldlr_merged["log2FC"])

# %% [markdown]
# #### Overall cell-type matched results

# %%
cellmatched_aggreg = result_df[["chromosome","locus","variant_pos","variant_type", "nucleotide"] + [x for x in result_df.keys() if "CAGE" in x]]

# %%
locus_df_list = []
for locus in set(cellmatched_aggreg["locus"]):
    subset = cellmatched_aggreg.query('locus == @locus')
    cols = cell_type_dict[locus]
    subset["pred_col"] = np.mean(subset[cols],axis=1)
    subset = subset[["chromosome","locus","variant_pos","variant_type", "nucleotide", "pred_col"]]
    locus_df_list.append(subset)
cellmatched_pred = pd.concat(locus_df_list)

# %%
cellmatched_pred["log2_pred"] =  np.log2(cellmatched_pred["pred_col"] + 1) 

cellmatched_ref = cellmatched_pred.query('variant_type == "ref"').drop(columns="variant_type").rename(columns={"log2_pred":"log2_pred_ref","pred_col":"pred_col_ref","nucleotide":"Ref"})
cellmatched_alt = cellmatched_pred.query('variant_type == "alt"').drop(columns="variant_type").rename(columns={"log2_pred":"log2_pred_alt","pred_col":"pred_col_alt","nucleotide":"Alt"})

cellmatched_pred = cellmatched_ref.merge(cellmatched_alt,on=["chromosome", "locus", "variant_pos"])

cellmatched_pred["log2_pred"] = cellmatched_pred["log2_pred_alt"] - cellmatched_pred["log2_pred_ref"]

cellmatched_merged = cellmatched_pred.merge(kircher_df, on=["chromosome", "locus", "variant_pos","Ref","Alt"])

# %% [markdown]
# ##### Overall correlation

# %%
print(scipy.stats.pearsonr(cellmatched_merged["log2_pred"],cellmatched_merged["log2FC"]))
print(scipy.stats.spearmanr(cellmatched_merged["log2_pred"],cellmatched_merged["log2FC"]))

# %%
row_list = []
for locus in set(cellmatched_merged["locus"]):
    subset = cellmatched_merged.query('locus == @locus')
    row_list.append({
        "locus":locus,
        "count":len(subset.index),
        "r":scipy.stats.pearsonr(subset["log2_pred"],subset["log2FC"])[0],
        "pval":scipy.stats.pearsonr(subset["log2_pred"],subset["log2FC"])[1],
        "spearman":scipy.stats.spearmanr(subset["log2_pred"],subset["log2FC"])[0]
    })
corr_df = pd.DataFrame(row_list)

# basal expression at each locus
basal = cellmatched_merged.groupby(["locus"])["pred_col_ref"].median().reset_index().rename(columns={"pred_col_ref":"predicted_basal_expression"})
corr_df = corr_df.merge(basal,on="locus")


# variation of measured log2fc at each locus
sd_log2fc = cellmatched_merged.groupby(["locus"])["log2FC"].std().reset_index().rename(columns={"log2FC":"standard deviation of log2fc"})
corr_df = corr_df.merge(sd_log2fc,on="locus")

# %%
p9.options.figure_size = (8, (4.8/6.4)*8)

plot_df = cellmatched_merged.copy()
plot_df["locus"] = pd.Categorical(plot_df["locus"], categories = list(corr_df.sort_values("r")["locus"]))

(p9.ggplot(data=plot_df,mapping=p9.aes(x="log2_pred",y="log2FC", color="locus"))
 + p9.geom_point()
 + p9.labs(x="Predicted log2 Variant Effect on Expression", y="Measured log2 Variant Effect on Expression")
 #+ p9.geom_smooth(method="lm",color="blue")
)

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

plot_df = cellmatched_merged.copy()
plot_df["locus"] = pd.Categorical(plot_df["locus"], categories = list(corr_df.sort_values("r")["locus"]))

p = (p9.ggplot(data=plot_df,mapping=p9.aes(x="log2_pred",y="log2FC"))
 + p9.geom_bin2d(binwidth = (0.05, 0.1), raster=True)
 + p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.geom_smooth(method="lm",color="blue")
 + p9.labs(x="Predicted log 2 Variant Effect\non Expression", y="Measured log2 Variant Effect\non Expression")
 + p9.theme(legend_box_margin = 0, legend_title=p9.element_text(size=9),
            legend_text=p9.element_text(size=9),legend_key_width=5,
           axis_title=p9.element_text(size=10))
)
p

# %%
p.save("Graphics/" + "xsup_kircher_cage_all" + ".svg", width=2.6, height=3.0, dpi=300)

# %% [markdown]
# ##### Disaggregated results

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 8.0*scale)

plot_df = cellmatched_merged.copy()
plot_df["locus"] = pd.Categorical(plot_df["locus"], categories = list(corr_df.sort_values("r")["locus"]))

p = (p9.ggplot(data=plot_df,mapping=p9.aes(x="log2_pred",y="log2FC"))
 + p9.geom_point(raster=True, alpha=0.25, size=1)
 + p9.geom_vline(xintercept = 1)
 + p9.geom_vline(xintercept = -1)
 + p9.geom_hline(yintercept = 1)
 + p9.geom_hline(yintercept = -1)
 #+ p9.geom_bin2d(binwidth = (0.25, 0.1), raster=True)
 #+ p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.labs(x="Predicted log2 Variant Effect on Expression", y="Measured log2 Variant Effect on Expression")
 + p9.geom_smooth(method="lm",color="blue")
 + p9.facet_wrap('~locus', ncol=3)
 + p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=1))
)
p
#p.save("Graphics/" + "xsup_kircher_cage_disaggreg" + ".svg", width=6.4, height=8.0, dpi=300)

# %%
p.save("Graphics/" + "xsup_kircher_cage_disaggreg" + ".svg", width=6.4, height=8.0, dpi=300)

# %%
p9.options.figure_size = (6.4*scale, 2.0*scale)

#plot_df_subset = plot_df.query('locus in ["F9","PKLR","TERT-HEK293T","LDLR"]')
#plot_df_subset()
print(scipy.stats.pearsonr(plot_df.query('locus in ["F9"]')["log2_pred"],plot_df.query('locus in ["F9"]')["log2FC"]))
print(scipy.stats.pearsonr(plot_df.query('locus in ["LDLR"]')["log2_pred"],plot_df.query('locus in ["LDLR"]')["log2FC"]))
print(scipy.stats.pearsonr(plot_df.query('locus in ["F9","LDLR"]')["log2_pred"],plot_df.query('locus in ["F9","LDLR"]')["log2FC"]))

plot_df_subset = plot_df.query('locus in ["F9","LDLR"]')
plot_df_comb = plot_df_subset.copy()
plot_df_comb["locus"] = "Combined"
plot_df_comb = pd.concat([plot_df_subset,plot_df_comb])
plot_df_comb["locus"] = pd.Categorical(plot_df_comb["locus"],["F9","LDLR","Combined"])

p = (p9.ggplot(data=plot_df_comb,mapping=p9.aes(x="log2_pred",y="log2FC"))
 + p9.geom_point(raster=True, alpha=0.25, size=1)
 + p9.geom_vline(xintercept = 1)
 + p9.geom_vline(xintercept = -1)
 + p9.geom_hline(yintercept = 1)
 + p9.geom_hline(yintercept = -1)
 #+ p9.geom_bin2d(binwidth = (0.25, 0.1), raster=True)
 #+ p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.labs(x="Predicted $\mathregular{log_2}$ Variant Effect (CAGE)", y="Measured $\mathregular{log_2}$ Variant\nEffect on Expression")
 + p9.geom_smooth(method="lm",color="blue")
 + p9.facet_wrap('~locus', ncol=4)
 + p9.theme(axis_title_x=p9.element_text(size=8),axis_title_y=p9.element_text(size=10))
)
p

# %%
p.save("Graphics/" + "kircher_cage_ex" + ".svg", width=3.5, height=0.9, dpi=300)

# %%
p9.options.figure_size = (8, (4.8/6.4)*8)

plot_df = corr_df[["locus", "r"]].copy()

aggregvals = pd.DataFrame([{"locus":"Pooled", "r":scipy.stats.pearsonr(cellmatched_merged["log2_pred"],cellmatched_merged["log2FC"])[0]},
             {"locus":"Average", "r":np.abs(corr_df["r"]).mean()},
             {"locus":"Median", "r":corr_df["r"].median()}])
plot_df = pd.concat([plot_df, aggregvals])

plot_df["locus"] = pd.Categorical(plot_df["locus"], categories = list(corr_df.sort_values("r")["locus"]) + ["Pooled","Average","Median"])

(p9.ggplot(data=plot_df,mapping=p9.aes(x="locus",y="r"))
 + p9.geom_bar(stat="identity")
 + p9.labs(x="Locus", y="Pearson correlation of predicted with measured\nlog2 variant effects")
 + p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=1))
)

# %% [markdown]
# ##### Standardizing per locus

# %%
df_list = []
for locus in set(cellmatched_merged["locus"]):
    subset = cellmatched_merged.query('locus == @locus')
    subset["log2_pred_standardized"] = (subset["log2_pred"] - subset["log2_pred"].mean())/subset["log2_pred"].std()
    subset["log2_pred_standardized"] = scipy.stats.mstats.winsorize(subset["log2_pred_standardized"], limits=[0.01, 0.01])
    df_list.append(subset)
cellmatched_merged_standardized = pd.concat(df_list)

# %%
print(scipy.stats.pearsonr(cellmatched_merged_standardized["log2_pred_standardized"],cellmatched_merged_standardized["log2FC"]))
print(scipy.stats.spearmanr(cellmatched_merged_standardized["log2_pred_standardized"],cellmatched_merged_standardized["log2FC"]))

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)


plot_df = cellmatched_merged_standardized.copy()
plot_df["locus"] = pd.Categorical(plot_df["locus"], categories = list(corr_df.sort_values("r")["locus"]))

p = (p9.ggplot(data=plot_df,mapping=p9.aes(x="log2_pred_standardized",y="log2FC"))
 + p9.geom_bin2d(binwidth = (0.05, 0.1), raster=True)
 + p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.geom_smooth(method="lm",color="blue")
 + p9.labs(x="Predicted Variant Effect\non Expression (z-score)", y="Measured log2 Variant Effect\non Expression")
 + p9.geom_smooth(method="lm",color="blue")
 + p9.theme(legend_box_margin = 0, legend_title=p9.element_text(size=9),
            legend_text=p9.element_text(size=9),legend_key_width=5,
           axis_title=p9.element_text(size=10))
)
p

# %%
p.save("Graphics/" + "xsup_kircher_cage_norm" + ".svg", width=2.6, height=3.0, dpi=300)

# %%
row_list = []
for locus in set(cellmatched_merged_standardized["locus"]):
    subset = cellmatched_merged_standardized.query('locus == @locus')
    row_list.append({
        "locus":locus,
        "count":len(subset.index),
        "r":scipy.stats.pearsonr(subset["log2_pred_standardized"],subset["log2FC"])[0],
        "pval":scipy.stats.pearsonr(subset["log2_pred_standardized"],subset["log2FC"])[1],
        "spearman":scipy.stats.spearmanr(subset["log2_pred_standardized"],subset["log2FC"])[0]
    })
corr_df_std = pd.DataFrame(row_list)

# %%
corr_df_std.merge(corr_df,on="locus", suffixes=("_standardized",""))[["locus","r","r_standardized"]]

# %% [markdown]
# ### DNASE

# %%
cellmatched_aggreg_dnase = result_df[["chromosome","locus","variant_pos","variant_type", "nucleotide"]+[x for x in result_df.keys() if "DNASE" in x]]

# %%
locus_df_list = []
for locus in set(cellmatched_aggreg_dnase["locus"]):
    subset = cellmatched_aggreg_dnase.query('locus == @locus')
    cols = list(set(cell_type_dict_dnase[locus]))
    subset["pred_col"] = np.mean(subset[cols],axis=1)
    subset = subset[["chromosome","locus","variant_pos","variant_type", "nucleotide", "pred_col"]]
    locus_df_list.append(subset)
cellmatched_pred_dnase = pd.concat(locus_df_list)

# %%
cellmatched_pred_dnase["log2_pred"] =  np.log2(cellmatched_pred_dnase["pred_col"] + 1) 

cellmatched_ref_dnase = cellmatched_pred_dnase.query('variant_type == "ref"').drop(columns="variant_type").rename(columns={"log2_pred":"log2_pred_ref","pred_col":"pred_col_ref","nucleotide":"Ref"})
cellmatched_alt_dnase = cellmatched_pred_dnase.query('variant_type == "alt"').drop(columns="variant_type").rename(columns={"log2_pred":"log2_pred_alt","pred_col":"pred_col_alt","nucleotide":"Alt"})

cellmatched_pred_dnase = cellmatched_ref_dnase.merge(cellmatched_alt_dnase,on=["chromosome", "locus", "variant_pos"])

cellmatched_pred_dnase["log2_pred"] = cellmatched_pred_dnase["log2_pred_alt"] - cellmatched_pred_dnase["log2_pred_ref"]

cellmatched_merged_dnase = cellmatched_pred_dnase.merge(kircher_df, on=["chromosome", "locus", "variant_pos","Ref","Alt"])

# %%
print(scipy.stats.pearsonr(cellmatched_merged_dnase["log2_pred"],cellmatched_merged_dnase["log2FC"]))
print(scipy.stats.spearmanr(cellmatched_merged_dnase["log2_pred"],cellmatched_merged_dnase["log2FC"]))

# %%
row_list = []
for locus in set(cellmatched_merged_dnase["locus"]):
    subset = cellmatched_merged_dnase.query('locus == @locus')
    row_list.append({
        "locus":locus,
        "count":len(subset.index),
        "r":scipy.stats.pearsonr(subset["log2_pred"],subset["log2FC"])[0],
        "pval":scipy.stats.pearsonr(subset["log2_pred"],subset["log2FC"])[1],
        "spearman":scipy.stats.spearmanr(subset["log2_pred"],subset["log2FC"])[0]
    })
corr_df_dnase = pd.DataFrame(row_list)

# basal expression at each locus
basal = cellmatched_merged_dnase.groupby(["locus"])["pred_col_ref"].median().reset_index().rename(columns={"pred_col_ref":"predicted_basal_expression"})
corr_df_dnase = corr_df_dnase.merge(basal,on="locus")

# %%
corr_df_merged = corr_df[["locus","count","r"]].merge(corr_df_dnase[["locus","count","r"]], on=["locus"], suffixes=("_enformer_CAGE","_enformer_DNASE"))

# %%
p9.options.figure_size = (6.4,4.8)

plot_df = cellmatched_merged_dnase.copy()
plot_df["locus"] = pd.Categorical(plot_df["locus"], categories = list(corr_df_dnase.sort_values("r")["locus"]))

(p9.ggplot(data=plot_df,mapping=p9.aes(x="log2_pred",y="log2FC", color="locus"))
 + p9.geom_point(size=1, alpha=0.5)
 + p9.labs(x="Predicted log2 Variant Effect\non Accessibility", y="Measured log2 Variant Effect\non Expression")
 + p9.geom_smooth(method="lm",color="blue")
 + p9.geom_vline(xintercept = 0.5)
 + p9.geom_vline(xintercept = -0.5)
 + p9.geom_hline(yintercept = 1)
 + p9.geom_hline(yintercept = -1)
 + p9.theme(legend_box_margin = 0, legend_text=p9.element_text(size=9),legend_key_size=9)
)

# %%
p9.options.figure_size = (6.4,4.8)

plot_df = cellmatched_merged_dnase.copy()
plot_df["locus"] = pd.Categorical(plot_df["locus"], categories = list(corr_df_dnase.sort_values("r")["locus"]))


p = (p9.ggplot(data=plot_df,mapping=p9.aes(x="log2_pred",y="log2FC"))
 #+ p9.geom_point()
 + p9.geom_bin2d(binwidth = (0.05, 0.1), raster=True)
 + p9.labs(x="Predicted log2 Variant Effect\non Accessibility (DNASE)", y="Measured log2\nVariant Effect on Expression")
 + p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.geom_smooth(method="lm",color="blue")
 + p9.geom_vline(xintercept = 0.65, color="red")
 + p9.geom_vline(xintercept = -0.65, color="red")
 + p9.geom_vline(xintercept = 1)
 + p9.geom_vline(xintercept = -1)
 + p9.geom_hline(yintercept = 1)
 + p9.geom_hline(yintercept = -1)
 + p9.theme(legend_text=p9.element_text(size=8),legend_title=p9.element_text(size=8),
            legend_box_margin=0,legend_key_width=5,axis_title=p9.element_text(size=10))#legend_key_size=10)
 #+ p9.scale_fill_gradient(low="lightgreen",high="darkgreen")
 #+ p9.guide_colorbar()
)
p

# %%
p.save("Graphics/" + "kircher_full" + ".svg", width=2.7, height=3.0, dpi=300)

# %%
ols_kircher_var = (statsmodels.regression.linear_model.OLS
                .from_formula('log2FC ~ log2_pred',
                             data=cellmatched_merged_dnase))

res_kircher_var = ols_kircher_var.fit()

res_kircher_var.summary()

# %% [markdown]
# #### Stratify by predicted variant effects

# %%
pred_cutoff = 0.5
real_cutoff = 1

plot_df["abs_pred_log2fc"] = np.abs(plot_df["log2_pred"])
plot_df["abs_real_log2fc"] = np.abs(plot_df["log2FC"])

plot_df["pred_cutoff"] = plot_df["abs_pred_log2fc"] > pred_cutoff
plot_df["real_cutoff"] = plot_df["abs_real_log2fc"] > real_cutoff

print("Restricting by predicted variant effect size")
print(scipy.stats.pearsonr(plot_df.query('pred_cutoff')["log2_pred"], 
                           plot_df.query('pred_cutoff')["log2FC"]))
print(scipy.stats.pearsonr(plot_df.query('not pred_cutoff')["log2_pred"], 
                           plot_df.query('not pred_cutoff')["log2FC"]))
print("Restricting by measured variant effect size")
print(scipy.stats.pearsonr(plot_df.query('real_cutoff')["log2_pred"], 
                           plot_df.query('real_cutoff')["log2FC"]))
print(scipy.stats.pearsonr(plot_df.query('not real_cutoff')["log2_pred"], 
                           plot_df.query('not real_cutoff')["log2FC"]))

# %%
sklearn.metrics.PrecisionRecallDisplay.from_predictions(plot_df["real_cutoff"], plot_df["abs_pred_log2fc"])

# %%
print(sklearn.metrics.precision_score(plot_df["real_cutoff"],plot_df["pred_cutoff"]))
print(sklearn.metrics.recall_score(plot_df["real_cutoff"],plot_df["pred_cutoff"]))

# %% [markdown]
# #### Disaggregated results

# %%
p9.options.figure_size = (6.4,8.0) #(6.4, 4.8)

plot_df = cellmatched_merged_dnase.copy()
plot_df["locus"] = pd.Categorical(plot_df["locus"], categories = list(corr_df_dnase.sort_values("r")["locus"]))

p = (p9.ggplot(data=plot_df,mapping=p9.aes(x="log2_pred",y="log2FC"))
 + p9.geom_point(raster=True, alpha=0.25, size=1)
 #+ p9.geom_bin2d(binwidth = (0.25, 0.1), raster=True)
 #+ p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.labs(x="Predicted log2 Variant Effect on Accessibility (DNASE)", y="Measured log2 Variant Effect on Expression")
 + p9.geom_smooth(method="lm",color="blue")
 + p9.geom_vline(xintercept = 0.5)
 + p9.geom_vline(xintercept = -0.5)
 + p9.geom_hline(yintercept = 1)
 + p9.geom_hline(yintercept = -1)
 + p9.facet_wrap('~locus', ncol=3)
 + p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=1))
)
p

# %%
p.save("Graphics/" + "xsup_kircher_dnase_disaggreg" + ".svg", width=6.4, height=8.0, dpi=300)

# %%
p9.options.figure_size = (3.0*scale, 1.0*scale)

#plot_df_subset_dnase = plot_df.query('locus in ["F9","LDLR"]')
print(scipy.stats.pearsonr(plot_df.query('locus in ["F9"]')["log2_pred"],plot_df.query('locus in ["F9"]')["log2FC"]))
print(scipy.stats.pearsonr(plot_df.query('locus in ["LDLR"]')["log2_pred"],plot_df.query('locus in ["LDLR"]')["log2FC"]))
print(scipy.stats.pearsonr(plot_df.query('locus in ["F9","LDLR"]')["log2_pred"],plot_df.query('locus in ["F9","LDLR"]')["log2FC"]))

plot_df_subset = plot_df.query('locus in ["F9","LDLR"]')
plot_df_comb = plot_df_subset.copy()
plot_df_comb["locus"] = "Combined"
plot_df_comb = pd.concat([plot_df_subset,plot_df_comb])
plot_df_comb["locus"] = pd.Categorical(plot_df_comb["locus"],["F9","LDLR","Combined"])

p = (p9.ggplot(data=plot_df_comb,mapping=p9.aes(x="log2_pred",y="log2FC"))
 + p9.geom_point(raster=True, alpha=0.25, size=1)
 #+ p9.geom_vline(xintercept = 0.5, color="red")
 #+ p9.geom_vline(xintercept = -0.5, color="red")
 + p9.geom_hline(yintercept = 1)
 + p9.geom_hline(yintercept = -1)
 #+ p9.geom_bin2d(binwidth = (0.25, 0.1), raster=True)
 #+ p9.scale_fill_continuous(ListedColormap(cm.get_cmap('viridis', 512)(np.linspace(0.25, 1, 512))), trans="log10")
 + p9.labs(x="Predicted $\mathregular{log_2}$ Variant Effect (DNASE)", y="Measured $\mathregular{log_2}$ Variant\nEffect on Expression")
 + p9.geom_smooth(method="lm",color="blue")
 + p9.facet_wrap('~locus', ncol=4)
 + p9.scale_x_continuous(breaks=[0.5,0,-0.5])
 + p9.theme(axis_title=p9.element_text(size=8))
)
p

# %%
p.save("Graphics/" + "kircher_dnase_ex" + ".svg", width=3.5, height=0.9, dpi=300)

# %%
p9.options.figure_size = (8, (4.8/6.4)*8)

plot_df = corr_df_dnase[["locus", "r"]].copy()

aggregvals = pd.DataFrame([{"locus":"Pooled", "r":scipy.stats.pearsonr(cellmatched_merged_dnase["log2_pred"],cellmatched_merged["log2FC"])[0]},
             {"locus":"Average", "r":np.abs(corr_df_dnase["r"]).mean()},
             {"locus":"Median", "r":corr_df_dnase["r"].median()}])
plot_df = pd.concat([plot_df, aggregvals])

plot_df["locus"] = pd.Categorical(plot_df["locus"], categories = list(corr_df_dnase.sort_values("r")["locus"]) + ["Pooled","Average","Median"])

(p9.ggplot(data=plot_df,mapping=p9.aes(x="locus",y="r"))
 + p9.geom_bar(stat="identity")
 + p9.labs(x="Locus", y="Pearson correlation of predicted with measured\nlog2 variant effects")
 + p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=1))
)

# %%
print(scipy.stats.pearsonr(corr_df_dnase.merge(repro_df,on="locus")["replicate_r"],corr_df_dnase.merge(repro_df,on="locus")["r"]))

# %% [markdown]
# ### Association between predicted reference level and predicted variant effect

# %%
p9.options.figure_size = (8, (4.8/6.4)*8)

plot_df = cellmatched_merged.copy()
plot_df["abs_log2_pred"] = np.abs(plot_df["log2_pred"])

(p9.ggplot(data=plot_df,
           mapping=p9.aes(x="pred_col_ref",y="abs_log2_pred"))
 + p9.geom_point(size=0.5)
 + p9.scale_x_log10()
 #+ p9.scale_y_log10()
 + p9.geom_hline(yintercept=0.5)
 + p9.geom_smooth(method="lowess",color="blue")
 + p9.labs(x="Predicted basal expression of locus (predicted CAGE count)", y="Predicted log2 variant effect (absolute value)")
)

# %%
p9.options.figure_size = (8, (4.8/6.4)*8)

plot_df = cellmatched_merged_dnase.copy()
plot_df["abs_log2_pred"] = np.abs(plot_df["log2_pred"])

(p9.ggplot(data=plot_df,mapping=p9.aes(x="pred_col_ref",y="abs_log2_pred"))
 + p9.geom_point()
 + p9.scale_x_log10()
 + p9.scale_y_log10()
 + p9.geom_smooth(method="lowess",color="blue")
 + p9.labs(x="Predicted basal expression of locus (predicted DNASE count)", y="Predicted log2 variant effect (DNASE-proxy)")
)

# %%
p9.options.figure_size = (8, (4.8/6.4)*8)

(p9.ggplot(data=corr_df,mapping=p9.aes(x="predicted_basal_expression",y="r"))
 + p9.geom_point(alpha=0)
 + p9.geom_text(mapping=p9.aes(label="locus"))
 + p9.scale_x_log10()
 #+ p9.scale_y_log10()
 + p9.geom_hline(yintercept=corr_df["r"].mean())
 #+ p9.geom_smooth(method="lm",color="blue")
 + p9.labs(x="Predicted basal expression of locus (predicted CAGE count)", y="Pearson correlation of predicted with measured\nlog2 variant effects")
)

# %%
print(scipy.stats.pearsonr(np.log10(corr_df["predicted_basal_expression"]+1),corr_df["r"]))

# %%
p9.options.figure_size = (8, (4.8/6.4)*8)

(p9.ggplot(data=corr_df.merge(repro_df,on="locus"),mapping=p9.aes(x="replicate_r",y="r"))
 + p9.geom_point(alpha=0)
 + p9.geom_text(mapping=p9.aes(label="locus"))
 + p9.scale_x_log10()
 #+ p9.geom_smooth(method="lm",color="blue")
 + p9.labs(x="Correlation between replicates", y="Pearson correlation of predicted with measured\nlog2 variant effects")
)

# %%
print(scipy.stats.pearsonr(corr_df.merge(repro_df,on="locus")["replicate_r"],corr_df.merge(repro_df,on="locus")["r"]))

# %%
p9.options.figure_size = (8, (4.8/6.4)*8)

(p9.ggplot(data=corr_df_dnase,mapping=p9.aes(x="predicted_basal_expression",y="r"))
 + p9.geom_point(alpha=0)
 + p9.geom_text(mapping=p9.aes(label="locus"))
 + p9.scale_x_log10()
 #+ p9.geom_smooth(method="lm",color="blue")
 + p9.labs(x="Predicted basal accessibility of locus (predicted DNASE-count)", y="Pearson correlation of predicted with measured\nlog2 variant effects (DNASE-proxy)")
)

# %%
scipy.stats.pearsonr(np.log10(corr_df_dnase["predicted_basal_expression"]+1),corr_df_dnase["r"])

# %%
p9.options.figure_size = (8, (4.8/6.4)*8)

(p9.ggplot(data=corr_df_dnase.merge(repro_df,on="locus"),mapping=p9.aes(x="replicate_r",y="r"))
 + p9.geom_point(alpha=0)
 + p9.geom_text(mapping=p9.aes(label="locus"))
 + p9.scale_x_log10()
 #+ p9.geom_smooth(method="lm",color="blue")
 + p9.labs(x="Correlation between replicates", y="Pearson correlation of predicted with measured\nlog2 variant effects")
)

# %%
p9.options.figure_size = (6.4,4.8)


(p9.ggplot(data=corr_df_dnase.merge(corr_df[["locus","standard deviation of log2fc"]],on="locus"),
           mapping=p9.aes(x="standard deviation of log2fc",y="r"))
 + p9.geom_point(alpha=0.1)
 + p9.geom_text(mapping=p9.aes(label="locus"))
 #+ p9.scale_x_log10()
 #+ p9.geom_smooth(method="lm",color="blue")
 + p9.labs(x="Standard Deviation of\n Measured log2 Variant Effects", y="Pearson correlation of predicted with measured\nlog2 variant effects")
)

# %% [markdown]
# ## Analysis - Expecto

# %%
expecto_df = pd.read_csv(base_path_results + "expecto.tsv",sep="\t").rename(columns={"#chrom":"chromosome","ref_allele":"Ref","alt_allele":"Alt","position":"pos_hg19"})
expecto_df["pos_hg19"] = expecto_df["pos_hg19"].astype('str')
expecto_df = expecto_df.drop_duplicates()

expecto_df = expecto_df.rename(columns={k:k+"_effect" for k in [x for x in expecto_df.keys() if x not in ["chromosome","Ref","Alt","pos_hg19", "gene"]]})

# %%
expecto_df = kircher_df.merge(expecto_df, on=["chromosome","Ref","Alt","pos_hg19"])


# %%
def get_matching_keys(cell_descriptor):
    keys = [x for x in expecto_df.keys() if cell_descriptor in x.lower()]
    if len(keys) == 0:
        keys = [x for x in expecto_df.keys() if "_effect" in x]
    return keys

cell_type_dict_expecto = {
    "F9":get_matching_keys("hepg2"), 
    "LDLR":get_matching_keys("hepg2"), 
    "SORT1":get_matching_keys("hepg2"),
    "GP1BB":get_matching_keys("k562"), 
    "HBB":get_matching_keys("k562"), 
    "HBG1":get_matching_keys("k562"), 
    "PKLR":get_matching_keys("k562"),
    "HNF4A":get_matching_keys("hek293"), 
    "MSMB":get_matching_keys("hek293"), 
    "TERT-HEK293T":get_matching_keys("hek293"), 
    "MYCrs6983267":get_matching_keys("hek293"),
    "ZFAND3":get_matching_keys("pancreas"),
    "TERT-GBM":get_matching_keys("glioblastoma"),
    "IRF6":get_matching_keys("keratinocyte"),
    "IRF4":get_matching_keys("mel")
}

# %%
locus_df_list = []
for locus in set(expecto_df["locus"]):
    subset = expecto_df.query('locus == @locus')
    cols = cell_type_dict_expecto[locus]
    subset["pred_col"] =  np.mean(subset[cols],axis=1)# np.max(np.abs(subset[cols]),axis=1)
    subset = subset[["chromosome","locus","variant_pos","Ref", "Alt", "pred_col", "log2FC"]]
    locus_df_list.append(subset)
expecto_pred = pd.concat(locus_df_list)

# %%
print(scipy.stats.pearsonr(expecto_pred["pred_col"],expecto_pred["log2FC"]))
print(scipy.stats.spearmanr(expecto_pred["pred_col"],expecto_pred["log2FC"]))

# %%
row_list = []
for locus in set(expecto_pred["locus"]):
    subset = expecto_pred.query('locus == @locus')
    row_list.append({
        "locus":locus,
        "count":len(subset.index),
        "r":scipy.stats.pearsonr(subset["pred_col"],subset["log2FC"])[0],
        "pval":scipy.stats.pearsonr(subset["pred_col"],subset["log2FC"])[1],
        "spearman":scipy.stats.spearmanr(subset["pred_col"],subset["log2FC"])[0]
    })
corr_df_expecto = pd.DataFrame(row_list)

# %%
plot_df = expecto_pred.copy()
plot_df["locus"] = pd.Categorical(plot_df["locus"], categories = list(corr_df.sort_values("r")["locus"]))

(p9.ggplot(data=plot_df,mapping=p9.aes(x="pred_col",y="log2FC", color="locus"))
 + p9.geom_point()
 + p9.labs(x="Predicted log2 Variant Effect on Expression", y="Measured log2 Variant Effect on Expression")
 #+ p9.geom_smooth(method="lm",color="blue")
)

# %%
p9.options.figure_size = (15,7.5) #(6.4, 4.8)

plot_df = expecto_pred.copy()
plot_df["locus"] = pd.Categorical(plot_df["locus"], categories = list(corr_df.sort_values("r")["locus"]))

(p9.ggplot(data=plot_df,mapping=p9.aes(x="pred_col",y="log2FC"))
 + p9.geom_point()
 + p9.labs(x="Predicted log2 Variant Effect on Expression", y="Measured log2 Variant Effect on Expression")
 + p9.geom_smooth(method="lm",color="blue")
 + p9.facet_wrap('~locus')
)

# %% [markdown]
# ## Analysis - Basenji2

# %%
result_df_basenji2 = pd.read_csv(base_path_results + "kircher_ingenome-basenji2-latest_results.tsv",sep="\t", usecols=["chromosome","locus","variant_pos","variant_type", "nucleotide", "offset", "orient"]+enformer_cols)

# %% [markdown]
# ### CAGE

# %%
cellmatched_aggreg_basenji2 = result_df_basenji2[["chromosome","locus","variant_pos","variant_type", "nucleotide"] + [x for x in result_df_basenji2.keys() if "CAGE" in x]]

# %%
locus_df_list = []
for locus in set(cellmatched_aggreg_basenji2["locus"]):
    subset = cellmatched_aggreg_basenji2.query('locus == @locus')
    cols = cell_type_dict[locus]
    subset["pred_col"] = np.mean(subset[cols],axis=1)
    subset = subset[["chromosome","locus","variant_pos","variant_type", "nucleotide", "pred_col"]]
    locus_df_list.append(subset)
cellmatched_pred_basenji2 = pd.concat(locus_df_list)

# %%
cellmatched_pred_basenji2["log2_pred"] =  np.log2(cellmatched_pred_basenji2["pred_col"] + 1) 

cellmatched_ref_basenji2 = cellmatched_pred_basenji2.query('variant_type == "ref"').drop(columns="variant_type").rename(columns={"log2_pred":"log2_pred_ref","pred_col":"pred_col_ref","nucleotide":"Ref"})
cellmatched_alt_basenji2 = cellmatched_pred_basenji2.query('variant_type == "alt"').drop(columns="variant_type").rename(columns={"log2_pred":"log2_pred_alt","pred_col":"pred_col_alt","nucleotide":"Alt"})

cellmatched_pred_basenji2 = cellmatched_ref_basenji2.merge(cellmatched_alt_basenji2,on=["chromosome", "locus", "variant_pos"])

cellmatched_pred_basenji2["log2_pred"] = cellmatched_pred_basenji2["log2_pred_alt"] - cellmatched_pred_basenji2["log2_pred_ref"]

cellmatched_merged_basenji2 = cellmatched_pred_basenji2.merge(kircher_df, on=["chromosome", "locus", "variant_pos","Ref","Alt"])

# %%
print(scipy.stats.pearsonr(cellmatched_merged_basenji2["log2_pred"],cellmatched_merged_basenji2["log2FC"]))
print(scipy.stats.spearmanr(cellmatched_merged_basenji2["log2_pred"],cellmatched_merged_basenji2["log2FC"]))

# %%
row_list = []
for locus in set(cellmatched_merged_basenji2["locus"]):
    subset = cellmatched_merged_basenji2.query('locus == @locus')
    row_list.append({
        "locus":locus,
        "count":len(subset.index),
        "r":scipy.stats.pearsonr(subset["log2_pred"],subset["log2FC"])[0],
        "pval":scipy.stats.pearsonr(subset["log2_pred"],subset["log2FC"])[1],
        "spearman":scipy.stats.spearmanr(subset["log2_pred"],subset["log2FC"])[0]
    })
corr_df_basenji2 = pd.DataFrame(row_list)

# basal expression at each locus
basal = cellmatched_merged_basenji2.groupby(["locus"])["pred_col_ref"].median().reset_index().rename(columns={"pred_col_ref":"predicted_basal_expression"})
corr_df_basenji2 = corr_df_basenji2.merge(basal,on="locus")

# %%
corr_df_basenji2

# %%
p9.options.figure_size = (8, (4.8/6.4)*8)

(p9.ggplot(data=corr_df_basenji2,mapping=p9.aes(x="predicted_basal_expression",y="r"))
 + p9.geom_point(alpha=0)
 + p9.geom_text(mapping=p9.aes(label="locus"))
 + p9.scale_x_log10()
 #+ p9.geom_smooth(method="lm",color="blue")
 + p9.labs(x="Predicted basal expression of locus (predicted CAGE count)", y="Pearson correlation of predicted with measured\nlog2 variant effects")
)

# %% [markdown]
# ### DNASE

# %%
cellmatched_aggreg_dnase_basenji2 = result_df_basenji2[["chromosome","locus","variant_pos","variant_type", "nucleotide"]+[x for x in result_df_basenji2.keys() if "DNASE" in x]]

# %%
locus_df_list = []
for locus in set(cellmatched_aggreg_dnase_basenji2["locus"]):
    subset = cellmatched_aggreg_dnase_basenji2.query('locus == @locus')
    cols = list(set(cell_type_dict_dnase[locus]))
    subset["pred_col"] = np.mean(subset[cols],axis=1)
    subset = subset[["chromosome","locus","variant_pos","variant_type", "nucleotide", "pred_col"]]
    locus_df_list.append(subset)
cellmatched_pred_dnase_basenji2 = pd.concat(locus_df_list)

# %%
cellmatched_pred_dnase_basenji2["log2_pred"] =  np.log2(cellmatched_pred_dnase_basenji2["pred_col"] + 1) 

cellmatched_ref_dnase_basenji2 = cellmatched_pred_dnase_basenji2.query('variant_type == "ref"').drop(columns="variant_type").rename(columns={"log2_pred":"log2_pred_ref","pred_col":"pred_col_ref","nucleotide":"Ref"})
cellmatched_alt_dnase_basenji2 = cellmatched_pred_dnase_basenji2.query('variant_type == "alt"').drop(columns="variant_type").rename(columns={"log2_pred":"log2_pred_alt","pred_col":"pred_col_alt","nucleotide":"Alt"})

cellmatched_pred_dnase_basenji2 = cellmatched_ref_dnase_basenji2.merge(cellmatched_alt_dnase_basenji2,on=["chromosome", "locus", "variant_pos"])

cellmatched_pred_dnase_basenji2["log2_pred"] = cellmatched_pred_dnase_basenji2["log2_pred_alt"] - cellmatched_pred_dnase_basenji2["log2_pred_ref"]

cellmatched_merged_dnase_basenji2 = cellmatched_pred_dnase_basenji2.merge(kircher_df, on=["chromosome", "locus", "variant_pos","Ref","Alt"])

# %%
print(scipy.stats.pearsonr(cellmatched_merged_dnase_basenji2["log2_pred"],cellmatched_merged_dnase_basenji2["log2FC"]))
print(scipy.stats.spearmanr(cellmatched_merged_dnase_basenji2["log2_pred"],cellmatched_merged_dnase_basenji2["log2FC"]))

# %%
row_list = []
for locus in set(cellmatched_merged_dnase_basenji2["locus"]):
    subset = cellmatched_merged_dnase_basenji2.query('locus == @locus')
    row_list.append({
        "locus":locus,
        "count":len(subset.index),
        "r":scipy.stats.pearsonr(subset["log2_pred"],subset["log2FC"])[0],
        "pval":scipy.stats.pearsonr(subset["log2_pred"],subset["log2FC"])[1],
        "spearman":scipy.stats.spearmanr(subset["log2_pred"],subset["log2FC"])[0]
    })
corr_df_dnase_basenji2 = pd.DataFrame(row_list)

# basal expression at each locus
basal = cellmatched_merged_dnase_basenji2.groupby(["locus"])["pred_col_ref"].median().reset_index().rename(columns={"pred_col_ref":"predicted_basal_expression"})
corr_df_dnase_basenji2 = corr_df_dnase_basenji2.merge(basal,on="locus")

# %%
corr_df_dnase_basenji2

# %% [markdown]
# ## Analysis - Basenji1

# %%
basenji1_cols = list(set(itertools.chain.from_iterable(cell_type_dict_basenji1.values()))) + list(set(itertools.chain.from_iterable(cell_type_dict_dnase_basenji1.values())))
result_df_basenji1 = pd.read_csv(base_path_results + "kircher_ingenome-basenji1-latest_results.tsv",sep="\t", usecols=["chromosome","locus","variant_pos","variant_type", "nucleotide", "offset", "orient"]+basenji1_cols)

# %% [markdown]
# ### CAGE

# %%
cellmatched_aggreg_basenji1 = result_df_basenji1[["chromosome","locus","variant_pos","variant_type", "nucleotide"] + [x for x in result_df_basenji1.keys() if "CAGE" in x]]

# %%
locus_df_list = []
for locus in set(cellmatched_aggreg_basenji1["locus"]):
    subset = cellmatched_aggreg_basenji1.query('locus == @locus')
    cols = cell_type_dict_basenji1[locus]
    subset["pred_col"] = np.mean(subset[cols],axis=1)
    subset = subset[["chromosome","locus","variant_pos","variant_type", "nucleotide", "pred_col"]]
    locus_df_list.append(subset)
cellmatched_pred_basenji1 = pd.concat(locus_df_list)

# %%
cellmatched_pred_basenji1["log2_pred"] =  np.log2(cellmatched_pred_basenji1["pred_col"] + 1) 

cellmatched_ref_basenji1 = cellmatched_pred_basenji1.query('variant_type == "ref"').drop(columns="variant_type").rename(columns={"log2_pred":"log2_pred_ref","pred_col":"pred_col_ref","nucleotide":"Ref"})
cellmatched_alt_basenji1 = cellmatched_pred_basenji1.query('variant_type == "alt"').drop(columns="variant_type").rename(columns={"log2_pred":"log2_pred_alt","pred_col":"pred_col_alt","nucleotide":"Alt"})

cellmatched_pred_basenji1 = cellmatched_ref_basenji1.merge(cellmatched_alt_basenji1,on=["chromosome", "locus", "variant_pos"])

cellmatched_pred_basenji1["log2_pred"] = cellmatched_pred_basenji1["log2_pred_alt"] - cellmatched_pred_basenji1["log2_pred_ref"]

cellmatched_merged_basenji1 = cellmatched_pred_basenji1.merge(kircher_df, on=["chromosome", "locus", "variant_pos","Ref","Alt"])

# %%
print(scipy.stats.pearsonr(cellmatched_merged_basenji1["log2_pred"],cellmatched_merged_basenji1["log2FC"]))
print(scipy.stats.spearmanr(cellmatched_merged_basenji1["log2_pred"],cellmatched_merged_basenji1["log2FC"]))

# %%
row_list = []
for locus in set(cellmatched_merged_basenji1["locus"]):
    subset = cellmatched_merged_basenji1.query('locus == @locus')
    row_list.append({
        "locus":locus,
        "count":len(subset.index),
        "r":scipy.stats.pearsonr(subset["log2_pred"],subset["log2FC"])[0],
        "pval":scipy.stats.pearsonr(subset["log2_pred"],subset["log2FC"])[1],
        "spearman":scipy.stats.spearmanr(subset["log2_pred"],subset["log2FC"])[0]
    })
corr_df_basenji1 = pd.DataFrame(row_list)

# basal expression at each locus
basal = cellmatched_merged_basenji1.groupby(["locus"])["pred_col_ref"].median().reset_index().rename(columns={"pred_col_ref":"predicted_basal_expression"})
corr_df_basenji1 = corr_df_basenji1.merge(basal,on="locus")

# %%
corr_df_basenji1

# %% [markdown]
# ### DNASE

# %%
cellmatched_aggreg_dnase_basenji1 = result_df_basenji1[["chromosome","locus","variant_pos","variant_type", "nucleotide"]+[x for x in result_df_basenji1.keys() if "DNASE" in x]]

# %%
locus_df_list = []
for locus in set(cellmatched_aggreg_dnase_basenji1["locus"]):
    subset = cellmatched_aggreg_dnase_basenji1.query('locus == @locus')
    cols = list(set(cell_type_dict_dnase_basenji1[locus]))
    subset["pred_col"] = np.mean(subset[cols],axis=1)
    subset = subset[["chromosome","locus","variant_pos","variant_type", "nucleotide", "pred_col"]]
    locus_df_list.append(subset)
cellmatched_pred_dnase_basenji1 = pd.concat(locus_df_list)

# %%
cellmatched_pred_dnase_basenji1["log2_pred"] =  np.log2(cellmatched_pred_dnase_basenji1["pred_col"] + 1) 

cellmatched_ref_dnase_basenji1 = cellmatched_pred_dnase_basenji1.query('variant_type == "ref"').drop(columns="variant_type").rename(columns={"log2_pred":"log2_pred_ref","pred_col":"pred_col_ref","nucleotide":"Ref"})
cellmatched_alt_dnase_basenji1 = cellmatched_pred_dnase_basenji1.query('variant_type == "alt"').drop(columns="variant_type").rename(columns={"log2_pred":"log2_pred_alt","pred_col":"pred_col_alt","nucleotide":"Alt"})

cellmatched_pred_dnase_basenji1 = cellmatched_ref_dnase_basenji1.merge(cellmatched_alt_dnase_basenji1,on=["chromosome", "locus", "variant_pos"])

cellmatched_pred_dnase_basenji1["log2_pred"] = cellmatched_pred_dnase_basenji1["log2_pred_alt"] - cellmatched_pred_dnase_basenji1["log2_pred_ref"]

cellmatched_merged_dnase_basenji1 = cellmatched_pred_dnase_basenji1.merge(kircher_df, on=["chromosome", "locus", "variant_pos","Ref","Alt"])

# %%
print(scipy.stats.pearsonr(cellmatched_merged_dnase_basenji1["log2_pred"],cellmatched_merged_dnase_basenji1["log2FC"]))
print(scipy.stats.spearmanr(cellmatched_merged_dnase_basenji1["log2_pred"],cellmatched_merged_dnase_basenji1["log2FC"]))

# %%
row_list = []
for locus in set(cellmatched_merged_dnase_basenji1["locus"]):
    subset = cellmatched_merged_dnase_basenji1.query('locus == @locus')
    row_list.append({
        "locus":locus,
        "count":len(subset.index),
        "r":scipy.stats.pearsonr(subset["log2_pred"],subset["log2FC"])[0],
        "pval":scipy.stats.pearsonr(subset["log2_pred"],subset["log2FC"])[1],
        "spearman":scipy.stats.spearmanr(subset["log2_pred"],subset["log2FC"])[0]
    })
corr_df_dnase_basenji1 = pd.DataFrame(row_list)

# basal expression at each locus
basal = cellmatched_merged_dnase_basenji1.groupby(["locus"])["pred_col_ref"].median().reset_index().rename(columns={"pred_col_ref":"predicted_basal_expression"})
corr_df_dnase_basenji1 = corr_df_dnase_basenji1.merge(basal,on="locus")

# %%
corr_df_dnase_basenji1

# %% [markdown]
# ## Combined Plot
#
# - make boxplot for loci
# - make barplot with average, median and pooled performance for each method

# %%
corr_df_full = (corr_df_merged.merge(corr_df_expecto[["locus", "count", "r"]], on="locus", how="left")
                  .fillna(0)
                  .rename(columns={"count":"count_expecto","r":"r_expecto"})
                  .merge(corr_df_basenji2[["locus", "count", "r"]].rename(columns={"count":"count_basenji2_CAGE","r":"r_basenji2_CAGE"}), on="locus")
                  .merge(corr_df_dnase_basenji2[["locus", "count", "r"]].rename(columns={"count":"count_basenji2_DNASE","r":"r_basenji2_DNASE"}), on="locus")
                  .merge(corr_df_basenji1[["locus", "count", "r"]].rename(columns={"count":"count_basenji1_CAGE","r":"r_basenji1_CAGE"}), on="locus")
                  .merge(corr_df_dnase_basenji1[["locus", "count", "r"]].rename(columns={"count":"count_basenji1_DNASE","r":"r_basenji1_DNASE"}), on="locus")
                 )

# %%
scale = 1.0
p9.options.figure_size = (6.4*scale, 4.8*scale)

corr_df_full_molten = corr_df_full.melt('locus')
corr_df_full_molten = corr_df_full_molten.loc[corr_df_full_molten.variable.str.startswith('r')]
corr_df_full_molten["Model"] = corr_df_full_molten["variable"].apply(lambda x: x.split("_")[1].capitalize())
corr_df_full_molten["pred_type"] = corr_df_full_molten["variable"].apply(lambda x: x.split("_")[2] if len(x.split("_")) > 2 else "CAGE")
corr_df_full_molten["Prediction"] = corr_df_full_molten["pred_type"].apply(lambda x: "Expression" if x == "CAGE" else "Accessibility")

corr_df_full_molten["Model_cat"] = pd.Categorical(corr_df_full_molten["Model"], categories = ["Basenji1","Basenji2", 
                                                                    "Expecto","Enformer"])
corr_df_full_molten["Prediction"] = pd.Categorical(corr_df_full_molten["Prediction"], categories = ["Expression","Accessibility"])

corr_df_full_both_enformer = corr_df_full_molten.query('pred_type == "CAGE" or Model_cat == "Enformer"')
corr_df_full_both_enformer["Model_both"] = corr_df_full_both_enformer.apply(lambda x: x["Model"] if x["Model"] != "Enformer" else ("Enformer\n(CAGE)" if x["pred_type"] == "CAGE" else "Enformer\n(DNASE)"), 
                                                                     axis=1)
corr_df_full_both_enformer["Model_both"] = pd.Categorical(corr_df_full_both_enformer["Model_both"], categories = ["Basenji1","Basenji2", 
                                                                    "Expecto","Enformer\n(CAGE)","Enformer\n(DNASE)"])
corr_df_full_both_enformer["special_points"] = corr_df_full_both_enformer.apply(lambda x: x["locus"] if (x["locus"] in ["F9","LDLR"]) else "Other",axis=1)

p = (p9.ggplot(data=corr_df_full_both_enformer,mapping=p9.aes(x="Model_both",y='value'))
 + p9.geom_boxplot()
 + p9.geom_jitter(width=0.2,size=1,mapping=p9.aes(color="special_points"))
 #+ p9.geom_line(p9.aes(group="locus"))
 + p9.labs(x="", y="Correlation between predicted\nand measured $\mathregular{log_2}$ variant effects",
           title="Kircher",color="Locus:")
 #+ p9.scale_alpha_discrete(range=(0.7,1))
 #+ p9.scale_fill_manual(aesthetics = "fill", values = colors, labels = labels[1:2],
 #                   breaks = names(colors)[1:2], name = "First Group:")
 + p9.theme(legend_text=p9.element_text(size=8),legend_key_size=8,
            legend_box_margin=0,
            title=p9.element_text(size=10),
            legend_background=p9.element_blank(),
            legend_title=p9.element_text(size=8),
            axis_title=p9.element_text(size=10),#p9.element_rect(color="none", size=2, fill='none'),
            legend_position=(0.28,0.7),
            axis_text_x=p9.element_text(rotation=25, hjust=1))
)
p

# %%
p.save("Graphics/" + "kircher_box" + ".svg", width=2.5, height=2.8, dpi=300)

# %%
p = (p9.ggplot(data=corr_df_full_molten.query('Model == "Enformer"'),mapping=p9.aes(x="Prediction",y='value'))
 + p9.geom_boxplot()
 + p9.geom_jitter(p9.aes(color="locus"),width=0.15)
 #+ p9.geom_line(p9.aes(group="locus"))
 + p9.labs(x="", y="Correlation between predicted\nand measured log2 variant effects",
           color="Locus:",title="Kircher")
 #+ p9.scale_alpha_discrete(range=(0.7,1))
 #+ p9.scale_fill_manual(aesthetics = "fill", values = colors, labels = labels[1:2],
 #                   breaks = names(colors)[1:2], name = "First Group:")
 + p9.theme(legend_text=p9.element_text(size=8),legend_key_size=8,
            legend_box_margin=0,
            title=p9.element_text(size=10),
            legend_background=p9.element_blank(),
            legend_title=p9.element_text(size=8),
            axis_title=p9.element_text(size=10))#p9.element_rect(color="none", size=2, fill='none'),
            #legend_position=(0.23,0.7))
            #axis_text_x=p9.element_text(rotation=30, hjust=1))
)
p

# %%
p.save("Graphics/" + "xsup_kircher_box" + ".svg", width=6.4, height=4.8, dpi=300)

# %%
scale = 1
p9.options.figure_size = (6.4*scale, 4.8*scale)

corrs_cage = corr_df[["locus","r"]]
corrs_cage["Method"] = "Enformer_CAGE"
corrs_dnase = corr_df_dnase[["locus","r"]]
corrs_dnase["Method"] = "Enformer_DNASE"
corrs_expecto = corr_df_expecto[["locus","r"]]
corrs_expecto["Method"] = "Expecto"

pooled_locus = "Pooled"
avg_locus = "Average\nover loci"
med_locus = "Median\nlocus"

aggregvals_cage = pd.DataFrame([{"locus":pooled_locus, "Method":"Enformer_CAGE","r":scipy.stats.pearsonr(cellmatched_merged["log2_pred"],cellmatched_merged["log2FC"])[0]},
             {"locus":avg_locus, "Method":"Enformer_CAGE","r":np.abs(corr_df["r"]).mean()},
             {"locus":med_locus, "Method":"Enformer_CAGE","r":corr_df["r"].median()}])
aggregvals_dnase = pd.DataFrame([{"locus":pooled_locus, "Method":"Enformer_DNASE","r":scipy.stats.pearsonr(cellmatched_merged_dnase["log2_pred"],cellmatched_merged_dnase["log2FC"])[0]},
             {"locus":avg_locus, "Method":"Enformer_DNASE","r":np.abs(corr_df_dnase["r"]).mean()},
             {"locus":med_locus, "Method":"Enformer_DNASE","r":corr_df_dnase["r"].median()}])

aggregvals_expecto = pd.DataFrame([{"locus":pooled_locus, "Method":"Expecto","r":scipy.stats.pearsonr(expecto_pred["pred_col"],expecto_pred["log2FC"])[0]},
             {"locus":avg_locus, "Method":"Expecto","r":np.abs(corr_df_expecto["r"]).mean()},
             {"locus":med_locus, "Method":"Expecto","r":corr_df_expecto["r"].median()}])

aggregvals_cage_basenji2 = pd.DataFrame([{"locus":pooled_locus, "Method":"Basenji2_CAGE","r":scipy.stats.pearsonr(cellmatched_merged_basenji2["log2_pred"],cellmatched_merged_basenji2["log2FC"])[0]},
             {"locus":avg_locus, "Method":"Basenji2_CAGE","r":np.abs(corr_df_basenji2["r"]).mean()},
             {"locus":med_locus, "Method":"Basenji2_CAGE","r":corr_df_basenji2["r"].median()}])
aggregvals_dnase_basenji2 = pd.DataFrame([{"locus":pooled_locus, "Method":"Basenji2_DNASE","r":scipy.stats.pearsonr(cellmatched_merged_dnase_basenji2["log2_pred"],cellmatched_merged_dnase_basenji2["log2FC"])[0]},
             {"locus":avg_locus, "Method":"Basenji2_DNASE","r":np.abs(corr_df_dnase_basenji2["r"]).mean()},
             {"locus":med_locus, "Method":"Basenji2_DNASE","r":corr_df_dnase_basenji2["r"].median()}])

aggregvals_cage_basenji1 = pd.DataFrame([{"locus":pooled_locus, "Method":"Basenji1_CAGE","r":scipy.stats.pearsonr(cellmatched_merged_basenji1["log2_pred"],cellmatched_merged_basenji1["log2FC"])[0]},
             {"locus":avg_locus, "Method":"Basenji1_CAGE","r":np.abs(corr_df_basenji1["r"]).mean()},
             {"locus":med_locus, "Method":"Basenji1_CAGE","r":corr_df_basenji1["r"].median()}])
aggregvals_dnase_basenji1 = pd.DataFrame([{"locus":pooled_locus, "Method":"Basenji1_DNASE","r":scipy.stats.pearsonr(cellmatched_merged_dnase_basenji1["log2_pred"],cellmatched_merged_dnase_basenji1["log2FC"])[0]},
             {"locus":avg_locus, "Method":"Basenji1_DNASE","r":np.abs(corr_df_dnase_basenji1["r"]).mean()},
             {"locus":med_locus, "Method":"Basenji1_DNASE","r":corr_df_dnase_basenji1["r"].median()}])

#corrs_cage,corrs_dnase,corrs_expecto,
plot_df = pd.concat([aggregvals_cage,aggregvals_dnase,
                     aggregvals_expecto,
                     aggregvals_cage_basenji2,aggregvals_dnase_basenji2,
                     aggregvals_cage_basenji1,aggregvals_dnase_basenji1])

#plot_df["locus"] = pd.Categorical(plot_df["locus"], categories = list(corr_df_dnase.sort_values("r")["locus"]) + ["Pooled","Average","Median"])
plot_df["locus"] = pd.Categorical(plot_df["locus"], [pooled_locus,avg_locus,med_locus])

# %%
plot_df["Model"] = plot_df["Method"].apply(lambda x: x.split("_")[0])
plot_df["Prediction"] = plot_df["Method"].apply(lambda x: "Accessibility" if "DNASE" in x else "Expression")

# %%
plot_df.to_csv(base_path_results + "kircher_table.tsv",sep="\t")

# %%
plot_df["Method"] = pd.Categorical(plot_df["Method"], categories = ["Basenji1_CAGE", "Basenji1_DNASE","Basenji2_CAGE", "Basenji2_DNASE",
                                                                    "Expecto","Enformer_CAGE","Enformer_DNASE"])
plot_df["Model"] = pd.Categorical(plot_df["Model"], categories = ["Basenji1","Basenji2", 
                                                                    "Expecto","Enformer"])
plot_df["Prediction"] = pd.Categorical(plot_df["Prediction"], categories = ["Expression","Accessibility"])

p = (p9.ggplot(data=plot_df,mapping=p9.aes(x="locus",y='r',fill='Method'))
 + p9.geom_bar(stat="identity",position="dodge")
 + p9.labs(x="Predicted log2 Fold Change\nof Expression", y="Correlation between predicted\nand measured log2 variant effects")
 #+ p9.scale_alpha_discrete(range=(0.7,1))
 #+ p9.scale_fill_manual(aesthetics = "fill", values = colors, labels = labels[1:2],
 #                   breaks = names(colors)[1:2], name = "First Group:")
 + p9.theme(legend_text=p9.element_text(size=8),legend_key_size=8,
            legend_box_margin=0,
            legend_background=p9.element_blank(),
            axis_title=p9.element_text(size=10))#p9.element_rect(color="none", size=2, fill='none'),
            #legend_position=(0.23,0.7))
            #axis_text_x=p9.element_text(rotation=30, hjust=1))
)
p

# %%
p.save("Graphics/" + "kircher_bars" + ".svg", width=2.3, height=3.0, dpi=300)

# %%
scale = 1.0
p9.options.figure_size = (6.4*scale, 4.8*scale)

corrs_cage = corr_df[["locus","r"]]
corrs_cage["Method"] = "Enformer_CAGE"
corrs_dnase = corr_df_dnase[["locus","r"]]
corrs_dnase["Method"] = "Enformer_DNASE"
corrs_expecto = corr_df_expecto[["locus","r"]]
corrs_expecto["Method"] = "Expecto"
#corrs_cage_basenji2 = corr_df_basenji2[["locus","r"]]
#corrs_cage_basenji2["Method"] = "Basenji2_CAGE"
#corrs_dnase_basenji2 = corr_df_dnase_basenji2[["locus","r"]]
#corrs_dnase_basenji2["Method"] = "Basenji2_DNASE"

plot_df = pd.concat([corrs_cage,corrs_dnase,corrs_expecto])#,corrs_cage_basenji2,corrs_dnase_basenji2])

plot_df["locus"] = pd.Categorical(plot_df["locus"], categories = list(corr_df_dnase.sort_values("r")["locus"]))
plot_df["Method"] = pd.Categorical(plot_df["Method"], categories = ["Expecto","Enformer_CAGE","Enformer_DNASE"])

p = (p9.ggplot(data=plot_df.query('Method in ["Expecto","Enformer_CAGE","Enformer_DNASE"]')
           ,mapping=p9.aes(x="locus",y='r',fill='Method'))
 + p9.geom_bar(stat="identity",position="dodge")
 + p9.labs(x="Locus", y="Pearson correlation of predicted with measured\nlog2 variant effects")
 + p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=1), legend_box_margin=0,
            legend_position=(0.25,0.8),legend_background=p9.element_blank(),
           legend_title=p9.element_text(size=9), legend_key_size=9, legend_text=p9.element_text(size=9))
)
p

# %%
p.save("Graphics/" + "xsup_kircher_bars" + ".svg", width=6.4, height=4.8, dpi=300)
