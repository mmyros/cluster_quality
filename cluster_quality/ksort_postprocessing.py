"""

Clean up Kilosort outputs by removing putative double-counted spikes.

Kilosort occasionally fits a spike template to the residual of another spike. See this discussion for more information.

This module aims to correct for this by removing spikes from the same unit or neighboring units that occur within 5 samples (0.16 ms) of one another. This is not ideal, since it can potentially remove legitimate spike times, but on the whole it seems worth it to avoid having spurious zero-time-lag correlation between units.

We are not currently taking into account spike amplitude when removing spikes; the module just deletes one spike from an overlapping pair that occurs later in time.
"""
import numpy as np


def find_within_unit_overlap(spike_train, overlap_window=5):
    """
    Finds overlapping spikes within a single spike train.
    Parameters
    ----------
    spike_train : numpy.ndarray
        Spike times (in samples)
    overlap_window : int
        Number of samples to search for overlapping spikes
    Outputs
    -------
    spikes_to_remove : numpy.ndarray
        Indices of overlapping spikes in spike_train
    """

    spikes_to_remove = np.where(np.diff(spike_train) < overlap_window)[0]

    return spikes_to_remove


def find_between_unit_overlap(spike_train1, spike_train2, overlap_window=5):
    """
    Finds overlapping spikes between two spike trains
    Parameters
    ----------
    spike_train1 : numpy.ndarray
        Spike times (in samples)
    spike_train2 : numpy.ndarray
        Spike times (in samples)
    overlap_window : int
        Number of samples to search for overlapping spikes
    Outputs
    -------
    spikes_to_remove1 : numpy.ndarray
        Indices of overlapping spikes in spike_train1
    spikes_to_remove2 : numpy.ndarray
        Indices of overlapping spikes in spike_train2
    """

    spike_train = np.concatenate((spike_train1, spike_train2))
    original_inds = np.concatenate((np.arange(len(spike_train1)), np.arange(len(spike_train2))))
    cluster_ids = np.concatenate((np.zeros((len(spike_train1),)), np.ones((len(spike_train2),))))

    order = np.argsort(spike_train)
    sorted_train = spike_train[order]
    sorted_inds = original_inds[order][1:]
    sorted_cluster_ids = cluster_ids[order][1:]

    spikes_to_remove = np.diff(sorted_train) < overlap_window

    spikes_to_remove1 = sorted_inds[spikes_to_remove * (sorted_cluster_ids == 0)]
    spikes_to_remove2 = sorted_inds[spikes_to_remove * (sorted_cluster_ids == 1)]

    return spikes_to_remove1, spikes_to_remove2


def remove_spikes(spike_times, spike_clusters, spike_templates, amplitudes, pc_features, spikes_to_remove):
    """
    Removes spikes from Kilosort outputs
    Inputs:
    ------
    spike_times : numpy.ndarray (num_spikes x 0)
        Spike times in samples
    spike_clusters : numpy.ndarray (num_spikes x 0)
        Cluster IDs for each spike time
    spike_templates : numpy.ndarray (num_spikes x 0)
        Template IDs for each spike time
    amplitudes : numpy.ndarray (num_spikes x 0)
        Amplitude value for each spike time
    pc_features : numpy.ndarray (num_spikes x num_pcs x num_channels)
        Pre-computed PCs for blocks of channels around each spike
    spikes_to_remove : numpy.ndarray
        Indices of spikes to remove
    Outputs:
    --------
    spike_times : numpy.ndarray (num_spikes - spikes_to_remove x 0)
    spike_clusters : numpy.ndarray (num_spikes - spikes_to_remove x 0)
    spike_templates : numpy.ndarray (num_spikes - spikes_to_remove x 0)
    amplitudes : numpy.ndarray (num_spikes - spikes_to_remove x 0)
    pc_features : numpy.ndarray (num_spikes - spikes_to_remove x num_pcs x num_channels)
    """
    spikes_to_remove = spikes_to_remove.astype(int)
    spike_times = np.delete(spike_times, spikes_to_remove, 0)
    spike_clusters = np.delete(spike_clusters, spikes_to_remove, 0)
    spike_templates = np.delete(spike_templates, spikes_to_remove, 0)
    amplitudes = np.delete(amplitudes, spikes_to_remove, 0)
    pc_features = np.delete(pc_features, spikes_to_remove, 0)

    return spike_times, spike_clusters, spike_templates, amplitudes, pc_features


def remove_double_counted_spikes(spike_times, spike_clusters, spike_templates, amplitudes, channel_map, templates,
                                 pc_features, sample_rate, within_unit_overlap_window=0.000166,
                                 between_unit_overlap_window=0.000166, between_unit_channel_distance=5
                                 ):
    """ Remove putative double-counted spikes from Kilosort outputs
    Inputs:
    ------
    spike_times : numpy.ndarray (num_spikes x 0)
        Spike times in samples
    spike_clusters : numpy.ndarray (num_spikes x 0)
        Cluster IDs for each spike time
    spike_templates : numpy.ndarray (num_spikes x 0)
        Template IDs for each spike time
    amplitudes : numpy.ndarray (num_spikes x 0)
        Amplitude value for each spike time
    channel_map : numpy.ndarray (num_units x 0)
        Original data channel for pc_feature_ind array
    templates : numpy.ndarray (num_units x num_channels x num_samples)
        Spike templates for each unit
    pc_features : numpy.ndarray (num_spikes x num_pcs x num_channels)
        Pre-computed PCs for blocks of channels around each spike
    pc_feature_ind : numpy.ndarray (num_units x num_channels)
        Channel indices of PCs for each unit
    sample_rate : Float
        Sample rate of spike times
    'within_unit_overlap_window' :
        time window for removing overlapping spikes
    'between_unit_overlap_window' :
        time window for removing overlapping spikes
    'between_unit_channel_distance' :
         number of channels over which to search for overlapping spikes

    Outputs:
    --------
    spike_times : numpy.ndarray (num_spikes x 0)
        Spike times in seconds (same timebase as epochs)
    spike_clusters : numpy.ndarray (num_spikes x 0)
        Cluster IDs for each spike time
    spike_templates : numpy.ndarray (num_spikes x 0)
        Template IDs for each spike time
    amplitudes : numpy.ndarray (num_spikes x 0)
        Amplitude value for each spike time
    pc_features : numpy.ndarray (num_spikes x num_pcs x num_channels)
        Pre-computed PCs for blocks of channels around each spike
    overlap_matrix : numpy.ndarray (num_clusters x num_clusters)
        Matrix indicating number of spikes removed for each pair of clusters
    """

    unit_list = np.arange(np.max(spike_clusters) + 1)

    peak_channels = np.squeeze(channel_map[np.argmax(np.max(templates, 1) - np.min(templates, 1), 1)])

    order = np.argsort(peak_channels)
    # Skip order if they are not present in unit_list:
    order = order[np.in1d(order, unit_list)]
    overlap_matrix = np.zeros((peak_channels.size, peak_channels.size))

    within_unit_overlap_samples = int(within_unit_overlap_window * sample_rate)
    between_unit_overlap_samples = int(between_unit_overlap_window * sample_rate)
    print('Removing within-unit overlapping spikes...')
    spikes_to_remove = np.zeros((0,))

    for idx1, unit_id1 in enumerate(unit_list[order]):
        for_unit1 = np.where(spike_clusters == unit_id1)[0]
        to_remove = find_within_unit_overlap(spike_times[for_unit1], within_unit_overlap_samples)
        overlap_matrix[idx1, idx1] = len(to_remove)
        spikes_to_remove = np.concatenate((spikes_to_remove, for_unit1[to_remove]))

    spike_times, spike_clusters, spike_templates, amplitudes, pc_features = remove_spikes(spike_times,
                                                                                          spike_clusters,
                                                                                          spike_templates,
                                                                                          amplitudes,
                                                                                          pc_features,
                                                                                          spikes_to_remove)

    print('Removing between-unit overlapping spikes...')
    spikes_to_remove = np.zeros((0,))

    for idx1, unit_id1 in enumerate(unit_list[order]):

        for_unit1 = np.where(spike_clusters == unit_id1)[0]

        for idx2, unit_id2 in enumerate(unit_list[order]):

            if idx2 > idx1 and (np.abs(peak_channels[unit_id1] - peak_channels[unit_id2]) <
                                between_unit_channel_distance):
                for_unit2 = np.where(spike_clusters == unit_id2)[0]

                to_remove1, to_remove2 = find_between_unit_overlap(spike_times[for_unit1], spike_times[for_unit2],
                                                                   between_unit_overlap_samples)

                overlap_matrix[idx1, idx2] = len(to_remove1) + len(to_remove2)

                spikes_to_remove = np.concatenate((spikes_to_remove, for_unit1[to_remove1], for_unit2[to_remove2]))

    print(f'Overlapping spikes to remove: {np.unique(spikes_to_remove)}')
    try:
        spike_times, spike_clusters, spike_templates, amplitudes, pc_features = remove_spikes(spike_times,
                                                                                          spike_clusters,
                                                                                          spike_templates,
                                                                                          amplitudes,
                                                                                          pc_features,
                                                                                          np.unique(spikes_to_remove))
    except IndexError as e:
        print(e)
        print('Cannot remove overlapping spikes due to error above ')

    return spike_times, spike_clusters, spike_templates, amplitudes, pc_features, overlap_matrix
