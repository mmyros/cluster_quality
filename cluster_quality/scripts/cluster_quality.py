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
from .. import new_wrappers
from .. import wrappers


def calculate_metrics(spike_times, spike_clusters, spike_templates, amplitudes, pc_features, pc_feature_ind,
                      output_folder=None,
                      do_parallel=True, do_pc_features=True, do_silhouette=True, do_drift=True,
                      isi_threshold=0.0015,
                      min_isi=0.000166,
                      num_channels_to_compare=5,
                      max_spikes_for_unit=1500,
                      max_spikes_for_nn=20000,
                      n_neighbors=4,
                      n_silhouette=20000,
                      drift_metrics_interval_s=51,
                      drift_metrics_min_spikes_per_interval=10
                      ):
    """ Calculate metrics for all units on one probe
    from mmy.input_output import spike_io
    ksort_folder = '~/res_ss_full/res_ss/tcloop_train_m022_1553627381_'
    spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, clusterIDs, cluster_quality, pc_features, pc_feature_ind = \
        spike_io.QualityMetrics.load_kilosort_data(ksort_folder, 3e4, False, include_pcs=True)
    metrics = QualityMetrics.calculate_metrics(spike_times, spike_clusters, amplitudes, pc_features, pc_feature_ind, ksort_folder)



    Inputs:
    ------
    spike_times : numpy.ndarray (num_spikes x 0)
        Spike times in seconds (same timebase as epochs)
    spike_clusters : numpy.ndarray (num_spikes x 0)
        Cluster IDs for each spike time
    amplitudes : numpy.ndarray (num_spikes x 0)
        Amplitude value for each spike time
    channel_map : numpy.ndarray (num_units x 0)
        Original data channel for pc_feature_ind array
    pc_features : numpy.ndarray (num_spikes x num_pcs x num_channels)
        Pre-computed PCs for blocks of channels around each spike
    pc_feature_ind : numpy.ndarray (num_units x num_channels)
        Channel indices of PCs for each unit
    epochs : list of Epoch objects
        contains information on Epoch start and stop times
    params : dict of parameters
        'isi_threshold' : minimum time for isi violations

    Outputs:
    --------
    metrics : pandas.DataFrame
        one column for each metric
        one row per unit per epoch
    """

    # ==========================================================
    # MAIN:
    # ==========================================================

    cluster_ids = np.unique(spike_clusters)
    total_units = len(np.unique(spike_clusters))
    print("Calculating isi violations")
    print(spike_clusters)
    print(total_units)
    isi_viol_rate, isi_viol_n = wrappers.calculate_isi_violations(spike_times, spike_clusters, isi_threshold, min_isi)

    print("Calculating presence ratio")
    presence_ratio = wrappers.calculate_presence_ratio(spike_times, spike_clusters, )

    print("Calculating firing rate")
    firing_rate = wrappers.calculate_firing_rate(spike_times, spike_clusters, )

    print("Calculating amplitude cutoff")
    amplitude_cutoff = wrappers.calculate_amplitude_cutoff(spike_clusters, amplitudes, )
    metrics = pd.DataFrame(data=OrderedDict((('cluster_id', cluster_ids),
                                             ('firing_rate', firing_rate),
                                             ('presence_ratio', presence_ratio),
                                             ('isi_viol_rate', isi_viol_rate),
                                             ('isi_viol_n', isi_viol_n),
                                             ('amplitude_cutoff', amplitude_cutoff),)))
    if do_pc_features:
        print("Calculating PC-based metrics")
        try:
            (isolation_distance, l_ratio,
            d_prime, nn_hit_rate, nn_miss_rate) = new_wrappers.calculate_pc_metrics(spike_clusters,
                                                                                   spike_templates,
                                                                                   total_units,
                                                                                   pc_features,
                                                                                   pc_feature_ind,
                                                                                   num_channels_to_compare,
                                                                                   max_spikes_for_unit,
                                                                                   max_spikes_for_nn,
                                                                                   n_neighbors,
                                                                                   do_parallel=do_parallel)
        except Exception:
            # Fallback
            print("Falling back to old Allen algo")
            (isolation_distance, l_ratio,
             d_prime, nn_hit_rate, nn_miss_rate) = wrappers.calculate_pc_metrics(spike_clusters,
                                                                                 spike_templates,
                                                                                 total_units,
                                                                                 pc_features,
                                                                                 pc_feature_ind,
                                                                                 num_channels_to_compare,
                                                                                 max_spikes_for_unit,
                                                                                 max_spikes_for_nn,
                                                                                 n_neighbors,
                                                                                 do_parallel=do_parallel)

        metrics0 = pd.DataFrame(data=OrderedDict((('isolation_distance', isolation_distance),
                                                  ('l_ratio', l_ratio),
                                                  ('d_prime', d_prime),
                                                  ('nn_hit_rate', nn_hit_rate),
                                                  ('nn_miss_rate', nn_miss_rate)
                                                  )))
        metrics = pd.concat([metrics, metrics0], axis=1)
    if do_silhouette:
        print("Calculating silhouette score")
        the_silhouette_score = wrappers.calculate_silhouette_score(spike_clusters,
                                                                   spike_templates,
                                                                   total_units,
                                                                   pc_features,
                                                                   pc_feature_ind,
                                                                   n_silhouette,
                                                                   do_parallel=True)
        metrics2 = pd.DataFrame(data=OrderedDict((('silhouette_score', the_silhouette_score),)),
                                index=range(len(the_silhouette_score)))

        metrics = pd.concat([metrics, metrics2], axis=1)
    if do_drift:
        print("Calculating drift metrics")
        # TODO [in_epoch] has to go. Need to modify loading function
        max_drift, cumulative_drift = wrappers.calculate_drift_metrics(spike_times,
                                                                       spike_clusters,
                                                                       spike_templates,
                                                                       pc_features,
                                                                       pc_feature_ind,
                                                                       drift_metrics_interval_s,
                                                                       drift_metrics_min_spikes_per_interval,
                                                                       do_parallel=do_parallel)

        metrics3 = pd.DataFrame(data=OrderedDict((('max_drift', max_drift),
                                                  ('cumulative_drift', cumulative_drift),
                                                  )))
        metrics = pd.concat([metrics, metrics3], axis=1)
    # write to output file if requested
    if output_folder is not None:
        metrics.to_csv(os.path.join(output_folder, 'quality_metrics.csv'), index=False)

    return metrics

from cluster_quality import ksort_postprocessing

@click.command()
@click.option('--kilosort_folder', default=None, help='kilosort_folder to read from and write to')
@click.option('--do_parallel', default=1, help='Parallel or not, 0 or 1')
@click.option('--do_silhouette', default=1, help='do_silhouette or not, 0 or 1')
@click.option('--do_drift', default=1, help='do_drift or not, 0 or 1')
@click.option('--do_pc_features', default=1, help='do_pc_features or not, 0 or 1')
def cli(kilosort_folder=None, do_parallel=True, do_pc_features=True, do_silhouette=True, do_drift=True, fs=3e4):
    """ Calculate metrics for all units on one probe"""
    # kilosort_folder = '~/res_ss_full/res_ss/tcloop_train_m022_1553627381_'
    if kilosort_folder is None:
        kilosort_folder = os.getcwd()
    if do_pc_features:
        do_include_pcs = True
    else:
        do_include_pcs = False

    (the_spike_times, the_spike_clusters, the_spike_templates, the_amplitudes, the_templates,
     the_channel_map, the_clusterIDs, the_cluster_quality,
     the_pc_features, the_pc_feature_ind) = io.load_kilosort_data(kilosort_folder,
                                                                  fs,
                                                                  False,
                                                                  include_pcs=do_include_pcs)

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
