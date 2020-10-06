import warnings
import os
import pandas as pd
import numpy as np

from collections import OrderedDict
from . import quality_metrics
from tqdm import tqdm


def calculate_isi_violations(spike_times, spike_clusters, isi_threshold, min_isi):
    cluster_ids = np.unique(spike_clusters)

    viol_rates = []
    viol_ns = []

    for idx, cluster_id in enumerate(cluster_ids):
        for_this_cluster = (spike_clusters == cluster_id)
        viol_rate, viol_n = quality_metrics.isi_violations(spike_times[for_this_cluster],
                                                           min_time=np.min(spike_times),
                                                           max_time=np.max(spike_times),
                                                           isi_threshold=isi_threshold,
                                                           min_isi=min_isi)
        viol_rates.append(viol_rate)
        viol_ns.append(viol_n)

    return np.array(viol_rates), np.array(viol_ns)


def calculate_presence_ratio(spike_times, spike_clusters):
    """

    :param spike_times:
    :param spike_clusters:
    :param total_units:
    :return:
    """
    cluster_ids = np.unique(spike_clusters)

    ratios = []

    for idx, cluster_id in enumerate(cluster_ids):
        for_this_cluster = (spike_clusters == cluster_id)
        ratios.append(quality_metrics.presence_ratio(spike_times[for_this_cluster],
                                                     min_time=np.min(spike_times),
                                                     max_time=np.max(spike_times)))

    return np.array(ratios)


def calculate_firing_rate(spike_times, spike_clusters):
    """

    :param spike_times:
    :param spike_clusters:
    :param total_units:
    :return:
    """
    cluster_ids = np.unique(spike_clusters)
    firing_rates = []

    for idx, cluster_id in enumerate(cluster_ids):
        for_this_cluster = (spike_clusters == cluster_id)
        firing_rates.append(quality_metrics.firing_rate(spike_times[for_this_cluster],
                                                        min_time=np.min(spike_times),
                                                        max_time=np.max(spike_times)))

    return np.array(firing_rates)


def calculate_amplitude_cutoff(spike_clusters, amplitudes):
    """

    :param spike_clusters:
    :param amplitudes:
    :param total_units:
    :return:
    """
    cluster_ids = np.unique(spike_clusters)

    amplitude_cutoffs = []

    for idx, cluster_id in enumerate(cluster_ids):
        for_this_cluster = (spike_clusters == cluster_id)
        amplitude_cutoffs.append(quality_metrics.amplitude_cutoff(amplitudes[for_this_cluster]))

    return np.array(amplitude_cutoffs)


def calculate_pc_metrics(spike_clusters,
                         spike_templates,
                         total_units,
                         pc_features,
                         pc_feature_ind,
                         num_channels_to_compare,
                         max_spikes_for_cluster,
                         max_spikes_for_nn,
                         n_neighbors,
                         do_parallel=True):
    """

    :param spike_clusters:
    :param total_units:
    :param pc_features:
    :param pc_feature_ind:
    :param num_channels_to_compare:
    :param max_spikes_for_cluster:
    :param max_spikes_for_nn:
    :param n_neighbors:
    :return:
    """

    assert (num_channels_to_compare % 2 == 1)
    half_spread = int((num_channels_to_compare - 1) / 2)

    cluster_ids = np.unique(spike_clusters)

    peak_channels = np.zeros((cluster_ids.max() + 1,), dtype='uint16')
    for cluster_id in cluster_ids:
        for_unit = np.squeeze(spike_clusters == cluster_id)
        pc_max = np.argmax(np.mean(pc_features[for_unit, 0, :], 0))
        peak_channels[cluster_id] = pc_feature_ind[cluster_id, pc_max]

    # Loop over clusters:
    if do_parallel:
        from joblib import Parallel, delayed
        # from joblib import wrap_non_picklable_objects
        # @delayed
        # @wrap_non_picklable_objects
        # def calculate_pc_metrics_one_cluster(**args):
        #     meas = Wrappers.calculate_pc_metrics_one_cluster(**args)
        #     return meas

        meas = Parallel(n_jobs=-1, verbose=3)(  # -1 means use all cores
            delayed(calculate_pc_metrics_one_cluster)  # Function
            (peak_channels, cluster_id, half_spread, pc_features, pc_feature_ind,  # Arguments
             spike_clusters, max_spikes_for_cluster, max_spikes_for_nn, n_neighbors
             )
            for cluster_id in cluster_ids)  # Loop
    else:
        meas = [calculate_pc_metrics_one_cluster(  # Function
            peak_channels, cluster_id, half_spread, pc_features, pc_feature_ind, spike_clusters,  # Arguments
            max_spikes_for_cluster, max_spikes_for_nn, n_neighbors)
            for cluster_id in tqdm(cluster_ids, desc='Calculating isolation metrics')]  # Loop

    # Unpack:
    isolation_distances = []
    l_ratios = []
    d_primes = []
    nn_hit_rates = []
    nn_miss_rates = []
    for mea in meas:
        isolation_distance, d_prime, nn_miss_rate, nn_hit_rate, l_ratio = mea
        isolation_distances.append(isolation_distance)
        d_primes.append(d_prime)
        nn_miss_rates.append(nn_miss_rate)
        nn_hit_rates.append(nn_hit_rate)
        l_ratios.append(l_ratio)

    return (np.array(isolation_distances), np.array(l_ratios), np.array(d_primes),
            np.array(nn_hit_rates), np.array(nn_miss_rates))


def calculate_pc_metrics_one_cluster(peak_channels, cluster_id, half_spread, pc_features, pc_feature_ind,
                                     spike_clusters, max_spikes_for_cluster, max_spikes_for_nn, n_neighbors):
    # HELPERS:
    def make_index_mask(spike_clusters, unit_id, min_num, max_num):
        """ Create a mask for the spike index dimensions of the pc_features array
        Inputs:
        -------
        spike_clusters : numpy.ndarray (num_spikes x 0)
            Contains cluster IDs for all spikes in pc_features array
        unit_id : Int
            ID for this unit
        min_num : Int
            Minimum number of spikes to return; if there are not enough spikes for this unit, return all False
        max_num : Int
            Maximum number of spikes to return; if too many spikes for this unit, return a random subsample
        Output:
        -------
        index_mask : numpy.ndarray (boolean)
            Mask of spike indices for pc_features array
        """

        index_mask = spike_clusters == unit_id

        inds = np.where(index_mask)[0]

        if len(inds) < min_num:
            index_mask = np.zeros((spike_clusters.size,), dtype='bool')
        else:
            index_mask = np.zeros((spike_clusters.size,), dtype='bool')
            order = np.random.permutation(inds.size)
            index_mask[inds[order[:max_num]]] = True

        return index_mask

    def make_channel_mask(unit_id, pc_feature_ind, channels_to_use, these_inds=None):
        """ Create a mask for the channel dimension of the pc_features array
        Inputs:
        -------
        unit_id : Int
            ID for this unit
        pc_feature_ind : np.ndarray
            Channels used for PC calculation for each unit
        channels_to_use : np.ndarray
            Channels to use for calculating metrics
        Output:
        -------
        channel_mask : numpy.ndarray
            Channel indices to extract from pc_features array

        """
        if these_inds is None:
            these_inds = pc_feature_ind[unit_id, :]
        channel_mask = [np.argwhere(these_inds == i)[0][0] for i in channels_to_use]

        # channel_mask = [np.argwhere(these_inds == i)[0][0] for i in available_to_use]

        return np.array(channel_mask)

    def get_unit_pcs(these_pc_features, index_mask, channel_mask):
        """ Use the index_mask and channel_mask to return PC features for one unit
        Inputs:
        -------
        these_pc_features : numpy.ndarray (float)
            Array of pre-computed PC features (num_spikes x num_PCs x num_channels)
        index_mask : numpy.ndarray (boolean)
            Mask for spike index dimension of pc_features array
        channel_mask : numpy.ndarray (boolean)
            Mask for channel index dimension of pc_features array
        Output:
        -------
        unit_PCs : numpy.ndarray (float)
            PCs for one unit (num_spikes x num_PCs x num_channels)
        """

        unit_PCs = these_pc_features[index_mask, :, :]

        unit_PCs = unit_PCs[:, :, channel_mask]

        return unit_PCs

    def features_intersect(pc_feature_ind, these_channels):
        """
        # Take only the channels that have calculated features out of the ones we are interested in:
        # This should reduce the occurence of 'except IndexError' below

        Args:
            these_channels: channels_to_use or units_for_channel

        Returns:
            channels_to_use: intersect of what's available in PCs and what's needed
        """
        intersect = set(pc_feature_ind[these_channels[0], :])  # Initialize
        for cluster_id2 in these_channels:
            # Make a running intersect of what is available and what is needed
            intersect = intersect & set(pc_feature_ind[cluster_id2, :])
        return np.array(list(intersect))

    # HELPERS OVER
    peak_channel = peak_channels[cluster_id]

    half_spread_down = peak_channel \
        if peak_channel < half_spread \
        else half_spread

    half_spread_up = np.max(pc_feature_ind) - peak_channel \
        if peak_channel + half_spread > np.max(pc_feature_ind) \
        else half_spread

    units_for_channel, channel_index = np.unravel_index(
        np.where(pc_feature_ind.flatten() == peak_channel)[0],
        pc_feature_ind.shape)

    # Skip peak_channels if they are not present in unit_list:
    units_for_channel = units_for_channel[np.in1d(units_for_channel, peak_channels)]

    units_in_range = (peak_channels[units_for_channel] >= peak_channel - half_spread_down) * \
                     (peak_channels[units_for_channel] <= peak_channel + half_spread_up)

    units_for_channel = units_for_channel[units_in_range]

    channels_to_use = np.arange(peak_channel - half_spread_down, peak_channel + half_spread_up + 1)

    # Use channels that are available in PCs:
    channels_to_use = features_intersect(pc_feature_ind, channels_to_use)
    # If this yields nothing, use units_for_channel:
    if len(channels_to_use) < 1:
        channels_to_use = features_intersect(pc_feature_ind, units_for_channel)

    spike_counts = np.zeros(units_for_channel.shape)

    for idx2, cluster_id2 in enumerate(units_for_channel):
        spike_counts[idx2] = np.sum(spike_clusters == cluster_id2)

    this_unit_idx = np.where(units_for_channel == cluster_id)[0]

    if spike_counts[this_unit_idx] > max_spikes_for_cluster:
        relative_counts = spike_counts / spike_counts[this_unit_idx] * max_spikes_for_cluster
    else:
        relative_counts = spike_counts

    all_pcs = np.zeros((0, pc_features.shape[1], channels_to_use.size))
    all_labels = np.zeros((0,))
    for idx2, cluster_id2 in enumerate(units_for_channel):

        try:
            channel_mask = make_channel_mask(cluster_id2, pc_feature_ind, channels_to_use)
        except IndexError:
            # Occurs when pc_feature_ind does not contain all channels of interest
            # In that case, we will exclude this unit for the calculation
            pass
        else:
            subsample = int(relative_counts[idx2])
            index_mask = make_index_mask(spike_clusters, cluster_id2, min_num=0, max_num=subsample)
            pcs = get_unit_pcs(pc_features, index_mask, channel_mask)
            labels = np.ones((pcs.shape[0],)) * cluster_id2

            all_pcs = np.concatenate((all_pcs, pcs), 0)
            all_labels = np.concatenate((all_labels, labels), 0)

    all_pcs = np.reshape(all_pcs, (all_pcs.shape[0], pc_features.shape[1] * channels_to_use.size))
    if ((all_pcs.shape[0] > 10)
            and (cluster_id in all_labels)
            and (len(channels_to_use) > 0)
            and not (all_labels == cluster_id).all()
    ):

        isolation_distance, l_ratio = quality_metrics.mahalanobis_metrics(all_pcs, all_labels, cluster_id)

        d_prime = quality_metrics.lda_metrics(all_pcs, all_labels, cluster_id)

        nn_hit_rate, nn_miss_rate = quality_metrics.nearest_neighbors_metrics(all_pcs, all_labels,
                                                                              cluster_id,
                                                                              max_spikes_for_nn,
                                                                              n_neighbors)
    else:  # Too few spikes or cluster doesnt exist
        isolation_distance = np.nan
        d_prime = np.nan
        nn_miss_rate = np.nan
        nn_hit_rate = np.nan
        l_ratio = np.nan
    return isolation_distance, d_prime, nn_miss_rate, nn_hit_rate, l_ratio


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
    isi_viol_rate, isi_viol_n = calculate_isi_violations(spike_times, spike_clusters, isi_threshold, min_isi)

    print("Calculating presence ratio")
    presence_ratio = calculate_presence_ratio(spike_times, spike_clusters, )

    print("Calculating firing rate")
    firing_rate = calculate_firing_rate(spike_times, spike_clusters, )

    print("Calculating amplitude cutoff")
    amplitude_cutoff = calculate_amplitude_cutoff(spike_clusters, amplitudes, )
    metrics = pd.DataFrame(data=OrderedDict((('cluster_id', cluster_ids),
                                             ('firing_rate', firing_rate),
                                             ('presence_ratio', presence_ratio),
                                             ('isi_viol_rate', isi_viol_rate),
                                             ('isi_viol_n', isi_viol_n),
                                             ('amplitude_cutoff', amplitude_cutoff),)))
    if do_pc_features:
        print("Calculating PC-based metrics")
        # try:
        (isolation_distance, l_ratio,
         d_prime, nn_hit_rate, nn_miss_rate) = calculate_pc_metrics(spike_clusters,
                                                                    spike_templates,
                                                                    total_units,
                                                                    pc_features,
                                                                    pc_feature_ind,
                                                                    num_channels_to_compare,
                                                                    max_spikes_for_unit,
                                                                    max_spikes_for_nn,
                                                                    n_neighbors,
                                                                    do_parallel=do_parallel)
        # except Exception:
        #     # Fallback
        #     print("Falling back to old Allen algo")
        #     (isolation_distance, l_ratio,
        #      d_prime, nn_hit_rate, nn_miss_rate) = calculate_pc_metrics(spike_clusters,
        #                                                                          spike_templates,
        #                                                                          total_units,
        #                                                                          pc_features,
        #                                                                          pc_feature_ind,
        #                                                                          num_channels_to_compare,
        #                                                                          max_spikes_for_unit,
        #                                                                          max_spikes_for_nn,
        #                                                                          n_neighbors,
        #                                                                          do_parallel=do_parallel)

        metrics0 = pd.DataFrame(data=OrderedDict((('isolation_distance', isolation_distance),
                                                  ('l_ratio', l_ratio),
                                                  ('d_prime', d_prime),
                                                  ('nn_hit_rate', nn_hit_rate),
                                                  ('nn_miss_rate', nn_miss_rate)
                                                  )))
        metrics = pd.concat([metrics, metrics0], axis=1)
    if do_silhouette:
        print("Calculating silhouette score")
        the_silhouette_score = quality_metrics.calculate_silhouette_score(spike_clusters,
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
        max_drift, cumulative_drift = quality_metrics.calculate_drift_metrics(spike_times,
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
