"""
Adapted from Allen from:
https://github.com/AllenInstitute/ecephys_spike_sorting.git
"""
import click
import numpy as np
import os
from collections import OrderedDict

import pandas as pd

from .. import io
from .. import ksort_postprocessing
from ..wrappers import calculate_metrics
import warnings
@click.command()
@click.option('--kilosort_folder', default=None, help='kilosort_folder to read from and write to')
@click.option('--do_parallel', default=1, help='Parallel or not, 0 or 1')
@click.option('--do_silhouette', default=1, help='do_silhouette or not, 0 or 1')
@click.option('--do_drift', default=1, help='do_drift or not, 0 or 1')
@click.option('--do_pc_features', default=1, help='do_pc_features or not, 0 or 1')
def cli(kilosort_folder=None, do_parallel=True, do_pc_features=True, do_silhouette=True, do_drift=True, fs=3e4):
    """ Calculate metrics for all units on one probe"""
    # kilosort_folder = '~/res_ss_full/res_ss/tcloop_train_m022_1553627381_'
    if (np.array([do_parallel,do_pc_features, do_silhouette,do_drift])=='False'|
        np.array([do_parallel, do_pc_features, do_silhouette, do_drift]) == 'True'):
        warnings.warn('Please dont use True or False for input from command line! use 0 or 1 instead')
    if kilosort_folder is None:
        kilosort_folder = os.getcwd()
    print(f'Running cluster_quality in folder {kilosort_folder}')
    if do_pc_features:
        do_include_pcs = True
    else:
        do_include_pcs = False

    (the_spike_times, the_spike_clusters, the_spike_templates, the_templates, the_amplitudes, the_unwhitened_temps,
     the_channel_map, the_cluster_ids, the_cluster_quality,
     the_pc_features, the_pc_feature_ind) = io.load_kilosort_data(kilosort_folder,
                                                                  fs,
                                                                  False,
                                                                  include_pcs=do_include_pcs)

    try:
        (the_spike_times, the_spike_clusters, the_spike_templates,
         the_amplitudes, the_pc_features,
         the_overlap_matrix) = ksort_postprocessing.remove_double_counted_spikes(the_spike_times,
                                                                            the_spike_clusters,
                                                                            the_spike_templates,
                                                                            the_amplitudes,
                                                                            the_channel_map,
                                                                            the_templates,
                                                                            the_pc_features,
                                                                            sample_rate=fs)
    except IndexError as e: # IndexError
        print(e)
        print('Cannot remove overlapping spikes due to error above ')

    all_metrics = calculate_metrics(the_spike_times, the_spike_clusters, the_spike_templates,
                                    the_amplitudes, the_pc_features, the_pc_feature_ind,
                                    output_folder=kilosort_folder,
                                    do_pc_features=do_pc_features,
                                    do_silhouette=do_silhouette,
                                    do_drift=do_drift,
                                    do_parallel=do_parallel)


# Launch this file and drop into debug if needed
if __name__ == '__main__':
    try:
        cli()
    except SystemExit:
        pass
    # except Exception as e:
    #     print('Error. Trying to start debugger... :\n ', e)
    #     import sys, traceback, pdb
    #
    #     extype, value, tb = sys.exc_info()
    #     traceback.print_exc()
    #     pdb.post_mortem(tb)
