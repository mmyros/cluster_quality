import warnings

import numpy as np
from sklearn.metrics import silhouette_score

from . import quality_metrics


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
        from tqdm import tqdm
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
    units_for_channel = units_for_channel[np.in1d(units_for_channel,peak_channels)]

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
        and not (all_labels== cluster_id).all()
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


def calculate_silhouette_score(spike_clusters,
                               spike_templates,
                               total_units,
                               pc_features,
                               pc_feature_ind,
                               total_spikes,
                               do_parallel=True):
    """

    :param spike_clusters:
    :param pc_features:
    :param pc_feature_ind:
    :param total_spikes:
    :return:
    """
    import warnings
    cluster_ids = np.unique(spike_clusters)
    random_spike_inds = np.random.permutation(spike_clusters.size)
    random_spike_inds = random_spike_inds[:total_spikes]
    num_pc_features = pc_features.shape[1]
    num_channels = np.max(pc_feature_ind) + 1
    all_pcs = np.zeros((total_spikes, num_channels * num_pc_features))

    for idx, i in enumerate(random_spike_inds):
        unit_id = spike_templates[i]
        channels = pc_feature_ind[unit_id, :]

        for j in range(0, num_pc_features):
            all_pcs[idx, channels + num_channels * j] = pc_features[i, j, :]

    cluster_labels = spike_clusters[random_spike_inds]

    SS = np.empty((total_units, total_units))
    SS[:] = np.nan
    """

for idx1, i in enumerate(cluster_ids):
    for idx2, j in enumerate(cluster_ids):
        
        if j > i:
            inds = np.in1d(cluster_labels, np.array([i,j]))
            X = all_pcs[inds,:]
            labels = cluster_labels[inds]
            
            if len(labels) > 2 and len(np.unique(labels)) > 1:
                SS[idx1,idx2] = silhouette_score(X, labels)                        
    """

    def score_inner_loop(i, cluster_ids):
        """
        Helper to loop over cluster_ids in one dimension. We dont want to loop over both dimensions in parallel-
        that will create too much worker overhead
        Args:
            i: index of first dimension
            cluster_ids: iterable of cluster ids

        Returns: scores for dimension j

        """
        scores_1d = []
        for j in cluster_ids:
            if j > i:
                inds = np.in1d(cluster_labels, np.array([i, j]))
                X = all_pcs[inds, :]
                labels = cluster_labels[inds]
                # len(np.unique(labels))=1 Can happen if total_spikes is low:
                if (len(labels) > 2) and (len(np.unique(labels)) > 1):
                    scores_1d.append(silhouette_score(X, labels))
                else:
                    scores_1d.append(np.nan)
            else:
                scores_1d.append(np.nan)
        return scores_1d

    # Build lists
    if do_parallel:
        from joblib import Parallel, delayed
        scores = Parallel(n_jobs=-1, verbose=2)(delayed(score_inner_loop)(i, cluster_ids) for i in cluster_ids)
    else:
        scores = [score_inner_loop(i, cluster_ids) for i in cluster_ids]

    # Fill the 2d array
    for i, col_score in enumerate(scores):
        for j, one_score in enumerate(col_score):
            SS[i, j] = one_score

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = np.nanmin(SS, 0)
        b = np.nanmin(SS, 1)
    return np.array([np.nanmin([a, b]) for a, b in zip(a, b)])


def calculate_drift_metrics(spike_times,
                            spike_clusters,
                            spike_templates,
                            pc_features,
                            pc_feature_ind,
                            interval_length,
                            min_spikes_per_interval,
                            do_parallel=True):
    """

    :param spike_times:
    :param spike_clusters:
    :param total_units:
    :param pc_features:
    :param pc_feature_ind:
    :param interval_length:
    :param min_spikes_per_interval:
    :return:
    """

    def get_spike_depths(spike_clusters, pc_features, pc_feature_ind):

        """
        Calculates the distance (in microns) of individual spikes from the probe tip
        This implementation is based on Matlab code from github.com/cortex-lab/spikes
        Input:
        -----
        spike_clusters : numpy.ndarray (N x 0)
            Cluster IDs for N spikes
        pc_features : numpy.ndarray (N x channels x num_PCs)
            PC features for each spike
        pc_feature_ind  : numpy.ndarray (M x channels)
            Channels used for PC calculation for each unit
        Output:
        ------
        spike_depths : numpy.ndarray (N x 0)
            Distance (in microns) from each spike waveform from the probe tip
        """

        pc_features_copy = np.copy(pc_features)
        pc_features_copy = np.squeeze(pc_features_copy[:, 0, :])
        pc_features_copy[pc_features_copy < 0] = 0
        pc_power = pow(pc_features_copy, 2)

        spike_feat_ind = pc_feature_ind[spike_clusters, :]
        spike_depths = np.sum(spike_feat_ind * pc_power, 1) / np.sum(pc_power, 1)

        return spike_depths * 10

    def calc_one_cluster(cluster_id):
        """
        Helper to calculate drift for one cluster
        Args:
            cluster_id:

        Returns:
            max_drift, cumulative_drift
        """
        in_cluster = spike_clusters == cluster_id
        times_for_cluster = spike_times[in_cluster]
        depths_for_cluster = depths[in_cluster]

        median_depths = []

        for t1, t2 in zip(interval_starts, interval_ends):

            in_range = (times_for_cluster > t1) * (times_for_cluster < t2)

            if np.sum(in_range) >= min_spikes_per_interval:
                median_depths.append(np.median(depths_for_cluster[in_range]))
            else:
                median_depths.append(np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # RuntimeWarning: All-NaN slice encountered
            median_depths = np.array(median_depths)
            max_drift = np.around(np.nanmax(median_depths) - np.nanmin(median_depths), 2)
            cumulative_drift = np.around(np.nansum(np.abs(np.diff(median_depths))), 2)
        return max_drift, cumulative_drift

    max_drifts = []
    cumulative_drifts = []

    depths = get_spike_depths(spike_templates, pc_features, pc_feature_ind)

    interval_starts = np.arange(np.min(spike_times), np.max(spike_times), interval_length)
    interval_ends = interval_starts + interval_length

    cluster_ids = np.unique(spike_clusters)

    if do_parallel:
        from joblib import Parallel, delayed
        meas = Parallel(n_jobs=-1, verbose=2)(delayed(calc_one_cluster)(cluster_id)
                                              for cluster_id in cluster_ids)
    else:
        meas = [calc_one_cluster(cluster_id) for cluster_id in cluster_ids]

    for max_drift, cumulative_drift in meas:
        max_drifts.append(max_drift)
        cumulative_drifts.append(max_drift)
    return np.array(max_drifts), np.array(cumulative_drifts)