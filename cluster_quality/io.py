import os

import numpy as np



"""
Adapted from Allen from:
https://github.com/AllenInstitute/ecephys_spike_sorting.git
"""


def load_kilosort_data(folder,
                       sample_rate=None,
                       convert_to_seconds=True,
                       use_master_clock=False,
                       include_pcs=False,
                       template_zero_padding=21):

    """
    Loads Kilosort output files from a directory
    Inputs:
    -------
    folder : String
        Location of Kilosort output directory
    sample_rate : float (optional)
        AP band sample rate in Hz
    convert_to_seconds : bool (optional)
        Flags whether to return spike times in seconds (requires sample_rate to be set)
    use_master_clock : bool (optional)
        Flags whether to load spike times that have been converted to the master clock timebase
    include_pcs : bool (optional)
        Flags whether to load spike principal components (large file)
    template_zero_padding : int (default = 21)
        Number of zeros added to the beginning of each template
    Outputs:
    --------
    spike_times : numpy.ndarray (N x 0)
        Times for N spikes
    spike_clusters : numpy.ndarray (N x 0)
        Cluster IDs for N spikes
    spike_templates : numpy.ndarray (N x 0)
        Template IDs for N spikes
    amplitudes : numpy.ndarray (N x 0)
        Amplitudes for N spikes
    unwhitened_temps : numpy.ndarray (M x samples x channels)
        Templates for M units
    channel_map : numpy.ndarray
        Channels from original data file used for sorting
    cluster_ids : Python list
        Cluster IDs for M units
    cluster_quality : Python list
        Quality ratings from cluster_group.tsv file
    pc_features (optinal) : numpy.ndarray (N x channels x num_PCs)
        PC features for each spike
    pc_feature_ind (optional) : numpy.ndarray (M x channels)
        Channels used for PC calculation for each unit
    """
    folder = os.path.expanduser(folder)

    def load(folder, filename):

        """
        Loads a numpy file from a folder.
        Inputs:
        -------
        folder : String
            Directory containing the file to load
        filename : String
            Name of the numpy file
        Outputs:
        --------
        data : numpy.ndarray
            File contents
        """

        return np.load(os.path.join(folder, filename))

    def read_cluster_group_tsv(filename):

        """
        Reads a tab-separated cluster_group.tsv file from disk
        Inputs:
        -------
        filename : String
            Full path of file
        Outputs:
        --------
        IDs : list
            List of cluster IDs
        quality : list
            Quality ratings for each unit (same size as IDs)
        """

        info = np.genfromtxt(filename, dtype='str')
        cluster_ids = info[1:, 0].astype('int')
        cluster_quality = info[1:, 1]

        return cluster_ids, cluster_quality

    if use_master_clock:
        spike_times = load(folder, 'spike_times_master_clock.npy')
    else:
        spike_times = load(folder, 'spike_times.npy')

    spike_clusters = load(folder, 'spike_clusters.npy')
    spike_templates = load(folder, 'spike_templates.npy')
    amplitudes = load(folder, 'amplitudes.npy')
    templates = load(folder, 'templates.npy')
    unwhitening_mat = load(folder, 'whitening_mat_inv.npy')
    channel_map = load(folder, 'channel_map.npy')

    if include_pcs:
        pc_features = load(folder, 'pc_features.npy')
        pc_feature_ind = load(folder, 'pc_feature_ind.npy')

    templates = templates[:, template_zero_padding:, :]  # remove zeros
    spike_clusters = np.squeeze(spike_clusters)  # fix dimensions
    spike_times = np.squeeze(spike_times)  # fix dimensions
    spike_templates = np.squeeze(spike_templates)  # fix dimensions

    if convert_to_seconds and sample_rate is not None:
        spike_times = spike_times / sample_rate

    unwhitened_temps = np.zeros((templates.shape))

    for temp_idx in range(templates.shape[0]):
        unwhitened_temps[temp_idx, :, :] = np.dot(np.ascontiguousarray(templates[temp_idx, :, :]),
                                                  np.ascontiguousarray(unwhitening_mat))

    try:
        cluster_ids, cluster_quality = read_cluster_group_tsv(os.path.join(folder, 'cluster_group.tsv'))
    except OSError:
        cluster_ids = np.unique(spike_clusters)
        cluster_quality = ['unsorted'] * cluster_ids.size

    if not include_pcs:
        return spike_times, spike_clusters, spike_templates, amplitudes, unwhitened_temps, channel_map, cluster_ids, cluster_quality
    else:
        return spike_times, spike_clusters, spike_templates, amplitudes, unwhitened_temps, channel_map, cluster_ids, cluster_quality, pc_features, pc_feature_ind