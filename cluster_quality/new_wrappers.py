import numpy as np

from . import quality_metrics


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
    template_ids = np.unique(spike_templates)

    template_peak_channels = np.zeros((len(template_ids),), dtype='uint16')
    cluster_peak_channels = np.zeros((len(cluster_ids),), dtype='uint16')

    for idx, template_id in enumerate(template_ids):
        for_template = np.squeeze(spike_templates == template_id)
        pc_max = np.argmax(np.mean(pc_features[for_template, 0, :], 0))
        template_peak_channels[idx] = pc_feature_ind[template_id, pc_max]

    for idx, cluster_id in enumerate(cluster_ids):
        for_unit = np.squeeze(spike_clusters == cluster_id)
        templates_for_unit = np.unique(spike_templates[for_unit])
        template_positions = np.where(np.isin(template_ids, templates_for_unit))[0]
        cluster_peak_channels[idx] = np.median(template_peak_channels[template_positions])

    # Loop over clusters:
    if do_parallel:
        from joblib import Parallel, delayed
        meas = Parallel(n_jobs=-1, verbose=3)(  # -1 means use all cores
            # delayed(Wrappers.calculate_pc_metrics_one_cluster_old)  # Function
            # (template_peak_channels, cluster_id, half_spread, pc_features, pc_feature_ind, spike_clusters,  # Inputs
            #  max_spikes_for_cluster, max_spikes_for_nn, n_neighbors)
            delayed(calculate_pc_metrics_one_cluster)  # Function
            (cluster_peak_channels, idx, cluster_id, cluster_ids,
             half_spread, pc_features, pc_feature_ind,
             spike_clusters, spike_templates,
             max_spikes_for_cluster, max_spikes_for_nn, n_neighbors)
            for idx, cluster_id in enumerate(cluster_ids))  # Loop
    else:
        from tqdm import tqdm
        meas = []
        for idx, cluster_id in tqdm(enumerate(cluster_ids), total=cluster_ids.max(), desc='PC metrics'):  # Loop
            # meas.append(Wrappers.calculate_pc_metrics_one_cluster_old(
            #     template_peak_channels, cluster_id, half_spread, pc_features, pc_feature_ind, spike_clusters,
            #     max_spikes_for_cluster, max_spikes_for_nn, n_neighbors))
            meas.append(calculate_pc_metrics_one_cluster(  # Function
                cluster_peak_channels, idx, cluster_id, cluster_ids,
                half_spread, pc_features, pc_feature_ind,
                spike_clusters, spike_templates,
                max_spikes_for_cluster, max_spikes_for_nn, n_neighbors))

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


def calculate_pc_metrics_one_cluster(cluster_peak_channels, idx, cluster_id, cluster_ids,
                                     half_spread, pc_features, pc_feature_ind,
                                     spike_clusters, spike_templates,
                                     max_spikes_for_cluster, max_spikes_for_nn, n_neighbors):

    # def make_index_mask(spike_clusters, unit_id, min_num, max_num):
    #     """ Create a mask for the spike index dimensions of the pc_features array
    #     Inputs:
    #     -------
    #     spike_clusters : numpy.ndarray (num_spikes x 0)
    #         Contains cluster IDs for all spikes in pc_features array
    #     unit_id : Int
    #         ID for this unit
    #     min_num : Int
    #         Minimum number of spikes to return; if there are not enough spikes for this unit, return all False
    #     max_num : Int
    #         Maximum number of spikes to return; if too many spikes for this unit, return a random subsample
    #     Output:
    #     -------
    #     index_mask : numpy.ndarray (boolean)
    #         Mask of spike indices for pc_features array
    #     """
    #
    #     index_mask = spike_clusters == unit_id
    #
    #     inds = np.where(index_mask)[0]
    #
    #     if len(inds) < min_num:
    #         index_mask = np.zeros((spike_clusters.size,), dtype='bool')
    #     else:
    #         index_mask = np.zeros((spike_clusters.size,), dtype='bool')
    #         order = np.random.permutation(inds.size)
    #         index_mask[inds[order[:max_num]]] = True
    #
    #     return index_mask
    #
    # def make_channel_mask(unit_id, pc_feature_ind, channels_to_use, these_inds=None):
    #     """ Create a mask for the channel dimension of the pc_features array
    #     Inputs:
    #     -------
    #     unit_id : Int
    #         ID for this unit
    #     pc_feature_ind : np.ndarray
    #         Channels used for PC calculation for each unit
    #     channels_to_use : np.ndarray
    #         Channels to use for calculating metrics
    #     Output:
    #     -------
    #     channel_mask : numpy.ndarray
    #         Channel indices to extract from pc_features array
    #
    #     """
    #     if these_inds is None:
    #         these_inds = pc_feature_ind[unit_id, :]
    #     channel_mask = [np.argwhere(these_inds == i)[0][0] for i in channels_to_use]
    #
    #     # channel_mask = [np.argwhere(these_inds == i)[0][0] for i in available_to_use]
    #
    #     return np.array(channel_mask)
    #
    # def get_unit_pcs_old(these_pc_features, index_mask, channel_mask):
    #     """ Use the index_mask and channel_mask to return PC features for one unit
    #     Inputs:
    #     -------
    #     these_pc_features : numpy.ndarray (float)
    #         Array of pre-computed PC features (num_spikes x num_PCs x num_channels)
    #     index_mask : numpy.ndarray (boolean)
    #         Mask for spike index dimension of pc_features array
    #     channel_mask : numpy.ndarray (boolean)
    #         Mask for channel index dimension of pc_features array
    #     Output:
    #     -------
    #     unit_PCs : numpy.ndarray (float)
    #         PCs for one unit (num_spikes x num_PCs x num_channels)
    #     """
    #
    #     unit_PCs = these_pc_features[index_mask, :, :]
    #
    #     unit_PCs = unit_PCs[:, :, channel_mask]
    #
    #     return unit_PCs

    def get_unit_pcs(unit_id,
                                        spike_clusters,
                                        spike_templates,
                                        pc_feature_ind,
                                        pc_features,
                                        channels_to_use,
                                        subsample):

        """ Return PC features for one unit

        Inputs:
        -------
        unit_id : Int
            ID for this unit
        spike_clusters : np.ndarray
            Cluster labels for each spike
        spike_templates : np.ndarry
            Template labels for each spike
        pc_feature_ind : np.ndarray
            Channels used for PC calculation for each unit
        pc_features : np.ndarray
            Array of all PC features
        channels_to_use : np.ndarray
            Channels to use for calculating metrics
        subsample : Int
            maximum number of spikes to return

        Output:
        -------
        unit_PCs : numpy.ndarray (float)
            PCs for one unit (num_spikes x num_PCs x num_channels)

        """

        inds_for_unit = np.where(spike_clusters == unit_id)[0]

        spikes_to_use = np.random.permutation(inds_for_unit)[:subsample]

        unique_template_ids = np.unique(spike_templates[spikes_to_use])

        unit_PCs = []

        for template_id in unique_template_ids:

            index_mask = spikes_to_use[np.squeeze(spike_templates[spikes_to_use]) == template_id]
            these_inds = pc_feature_ind[template_id, :]

            pc_array = []

            for i in channels_to_use:

                if np.isin(i, these_inds):
                    channel_index = np.argwhere(these_inds == i)[0][0]
                    pc_array.append(pc_features[index_mask, :, channel_index])
                else:
                    return None

            unit_PCs.append(np.stack(pc_array, axis=-1))

        if len(unit_PCs) > 0:

            return np.concatenate(unit_PCs)
        else:
            return None

    def features_intersect(pc_feature_ind, these_channels):
        """
        # Take only the channels that have calculated features out of the ones we are interested in:
        # This should reduce the occurence of 'except IndexError' below

        Args:
            these_channels: channels_to_use or units_for_channel

        Returns:
            channels_to_use: intersect of what's available in PCs and what's needed
        """
        intersect = set(channels_to_use)  # Initialize
        for cluster_id2 in these_channels:
            # Make a running intersect of what is available and what is needed
            intersect = intersect & set(pc_feature_ind[cluster_id2, :])
        return np.array(list(intersect))

    peak_channel = cluster_peak_channels[idx]
    num_spikes_in_cluster = np.sum(spike_clusters == cluster_id)

    half_spread_down = peak_channel \
        if peak_channel < half_spread \
        else half_spread

    half_spread_up = np.max(pc_feature_ind) - peak_channel \
        if peak_channel + half_spread > np.max(pc_feature_ind) \
        else half_spread

    channels_to_use = np.arange(peak_channel - half_spread_down, peak_channel + half_spread_up + 1)
    units_in_range = cluster_ids[np.isin(cluster_peak_channels, channels_to_use)]

    # Use channels that are available in PCs:
    channels_to_use = features_intersect(pc_feature_ind, channels_to_use)
    # If this yields nothing, use units_for_channel:
    if len(channels_to_use) < 1:
        channels_to_use = features_intersect(pc_feature_ind, units_in_range)

    spike_counts = np.zeros(units_in_range.shape)

    for idx2, cluster_id2 in enumerate(units_in_range):
        spike_counts[idx2] = np.sum(spike_clusters == cluster_id2)

    if num_spikes_in_cluster > max_spikes_for_cluster:
        relative_counts = spike_counts / num_spikes_in_cluster * max_spikes_for_cluster
    else:
        relative_counts = spike_counts

    all_pcs = np.zeros((0, pc_features.shape[1], channels_to_use.size))
    all_labels = np.zeros((0,))

    # for idx2, cluster_id2 in enumerate(units_in_range):
    #
    #     try:
    #         channel_mask = make_channel_mask(cluster_id2, pc_feature_ind, channels_to_use)
    #     except IndexError:
    #         # Occurs when pc_feature_ind does not contain all channels of interest
    #         # In that case, we will exclude this unit for the calculation
    #         pass
    #     else:
    #         subsample = int(relative_counts[idx2])
    #         index_mask = make_index_mask(spike_clusters, cluster_id2, min_num=0, max_num=subsample)
    #         pcs = get_unit_pcs(pc_features, index_mask, channel_mask)
    #         # pcs = get_unit_pcs(cluster_id2, spike_clusters, spike_templates,
    #         #                                           pc_feature_ind, pc_features, channels_to_use,
    #         #                                           subsample)
    #         labels = np.ones((pcs.shape[0],)) * cluster_id2
    #
    #         all_pcs = np.concatenate((all_pcs, pcs), 0)
    #         all_labels = np.concatenate((all_labels, labels), 0)

    # New Allen implementation still misses neurons that are not on many channels, eg stereotrodes
    for idx2, cluster_id2 in enumerate(units_in_range):

        subsample = int(relative_counts[idx2])

        pcs = get_unit_pcs(cluster_id2, spike_clusters, spike_templates,
                           pc_feature_ind, pc_features, channels_to_use,
                           subsample)

        if pcs is not None and len(pcs.shape) == 3:
            labels = np.ones((pcs.shape[0],)) * cluster_id2

            all_pcs = np.concatenate((all_pcs, pcs), 0)
            all_labels = np.concatenate((all_labels, labels), 0)

    all_pcs = np.reshape(all_pcs, (all_pcs.shape[0], pc_features.shape[1] * channels_to_use.size))
    if ((all_pcs.shape[0] > 10)
            and (cluster_id in all_labels)
            and (len(channels_to_use) > 0)):
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