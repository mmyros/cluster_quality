from sklearn.metrics import silhouette_score
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from scipy.stats import chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NearestNeighbors
import warnings


def isi_violations(spike_train, min_time, max_time, isi_threshold, min_isi=0):
    """Calculate ISI violations for a spike train.
    Based on metric described in Hill et al. (2011) J Neurosci 31: 8699-8705
    modified by Dan Denman from cortex-lab/sortingQuality GitHub by Nick Steinmetz
    Inputs:
    -------
    spike_train : array of spike times
    min_time : minimum time for potential spikes
    max_time : maximum time for potential spikes
    isi_threshold : threshold for isi violation
    min_isi : threshold for duplicate spikes
    Outputs:
    --------
    fpRate : rate of contaminating spikes as a fraction of overall rate
        A perfect unit has a fpRate = 0
        A unit with some contamination has a fpRate < 0.5
        A unit with lots of contamination has a fpRate > 1.0
    num_violations : total number of violations
    """
    duplicate_spikes = np.where(np.diff(spike_train) <= min_isi)[0]

    spike_train = np.delete(spike_train, duplicate_spikes + 1)
    isis = np.diff(spike_train)

    num_spikes = len(spike_train)
    num_violations = sum(isis < isi_threshold)
    violation_time = 2 * num_spikes * (isi_threshold - min_isi)
    total_rate = firing_rate(spike_train, min_time, max_time)
    violation_rate = num_violations / violation_time
    fpRate = violation_rate / total_rate

    return fpRate, num_violations


def presence_ratio(spike_train, min_time, max_time, num_bins=100):
    """Calculate fraction of time the unit is present within an epoch.
    Inputs:
    -------
    spike_train : array of spike times
    min_time : minimum time for potential spikes
    max_time : maximum time for potential spikes
    Outputs:
    --------
    presence_ratio : fraction of time bins in which this unit is spiking
    """

    h, b = np.histogram(spike_train, np.linspace(min_time, max_time, num_bins))

    return np.sum(h > 0) / num_bins


def firing_rate(spike_train, min_time=None, max_time=None):
    """Calculate firing rate for a spike train.
    If no temporal bounds are specified, the first and last spike time are used.
    Inputs:
    -------
    spike_train : numpy.ndarray
        Array of spike times in seconds
    min_time : float
        Time of first possible spike (optional)
    max_time : float
        Time of last possible spike (optional)
    Outputs:
    --------
    fr : float
        Firing rate in Hz
    """

    if min_time is not None and max_time is not None:
        duration = max_time - min_time
    else:
        duration = np.max(spike_train) - np.min(spike_train)

    fr = spike_train.size / duration

    return fr


def amplitude_cutoff(amplitudes, num_histogram_bins=500, histogram_smoothing_value=3):
    """ Calculate approximate fraction of spikes missing from a distribution of amplitudes
    Assumes the amplitude histogram is symmetric (not valid in the presence of drift)
    Inspired by metric described in Hill et al. (2011) J Neurosci 31: 8699-8705
    Input:
    ------
    amplitudes : numpy.ndarray
        Array of amplitudes (don't need to be in physical units)
    Output:
    -------
    fraction_missing : float
        Fraction of missing spikes (0-0.5)
        If more than 50% of spikes are missing, an accurate estimate isn't possible
    """

    h, b = np.histogram(amplitudes, num_histogram_bins, density=True)

    pdf = gaussian_filter1d(h, histogram_smoothing_value)
    support = b[:-1]

    peak_index = np.argmax(pdf)
    G = np.argmin(np.abs(pdf[peak_index:] - pdf[0])) + peak_index

    bin_size = np.mean(np.diff(support))
    fraction_missing = np.sum(pdf[G:]) * bin_size

    fraction_missing = np.min([fraction_missing, 0.5])

    return fraction_missing


def mahalanobis_metrics(all_pcs, all_labels, this_unit_id):
    # def mahalanobis_metrics(pcs_for_this_unit, pcs_for_other_units):
    """ Calculates isolation distance and L-ratio (metrics computed from Mahalanobis distance)
    Based on metrics described in Schmitzer-Torbert et al. (2005) Neurosci 131: 1-11
    Inputs:
    -------
    all_pcs : numpy.ndarray (num_spikes x PCs)
        2D array of PCs for all spikes
    all_labels : numpy.ndarray (num_spikes x 0)
        1D array of cluster labels for all spikes
    this_unit_id : Int
        number corresponding to unit for which these metrics will be calculated
    Outputs:
    --------
    isolation_distance : float
        Isolation distance of this unit
    l_ratio : float
        L-ratio for this unit
    """

    pcs_for_this_unit = all_pcs[all_labels == this_unit_id, :]
    pcs_for_other_units = all_pcs[all_labels != this_unit_id, :]
    mean_value = np.expand_dims(np.mean(pcs_for_this_unit, 0), 0)

    try:
        VI = np.linalg.inv(np.cov(pcs_for_this_unit.T))
    except np.linalg.linalg.LinAlgError:  # case of singular matrix
        return np.nan, np.nan

    mahalanobis_other = np.sort(cdist(mean_value,
                                      pcs_for_other_units,
                                      'mahalanobis', VI=VI)[0])

    mahalanobis_self = np.sort(cdist(mean_value,
                                     pcs_for_this_unit,
                                     'mahalanobis', VI=VI)[0])

    n = np.min([pcs_for_this_unit.shape[0], pcs_for_other_units.shape[0]])  # number of spikes

    if np.isnan(mahalanobis_other.any()):
        warnings.warn(f'Nans in mahalanobis_other for cluster {this_unit_id}. l_ratio will be nan')
    if n >= 2:

        dof = pcs_for_this_unit.shape[1]  # number of features

        l_ratio = np.sum(1 - chi2.cdf(pow(mahalanobis_other, 2), dof)) / mahalanobis_self.shape[0]
        isolation_distance = pow(mahalanobis_other[n - 1], 2)

    else:
        l_ratio = np.nan
        isolation_distance = np.nan

    return isolation_distance, l_ratio


def lda_metrics(all_pcs, all_labels, this_unit_id):
    """ Calculates d-prime based on Linear Discriminant Analysis
    Based on metric described in Hill et al. (2011) J Neurosci 31: 8699-8705
    Inputs:
    -------
    all_pcs : numpy.ndarray (num_spikes x PCs)
        2D array of PCs for all spikes
    all_labels : numpy.ndarray (num_spikes x 0)
        1D array of cluster labels for all spikes
    this_unit_id : Int
        number corresponding to unit for which these metrics will be calculated
    Outputs:
    --------
    d_prime : float
        Isolation distance of this unit
    l_ratio : float
        L-ratio for this unit
    """

    X = all_pcs

    y = np.zeros((X.shape[0],), dtype='bool')
    y[all_labels == this_unit_id] = True

    lda = LDA(n_components=1)

    X_flda = lda.fit_transform(X, y)

    flda_this_cluster = X_flda[np.where(y)[0]]
    flda_other_cluster = X_flda[np.where(np.invert(y))[0]]

    d_prime = (np.mean(flda_this_cluster) - np.mean(flda_other_cluster)) / np.sqrt(
        0.5 * (np.std(flda_this_cluster) ** 2 + np.std(flda_other_cluster) ** 2))

    return d_prime


def nearest_neighbors_metrics(all_pcs, all_labels, this_unit_id, max_spikes_for_nn, n_neighbors):
    """ Calculates unit contamination based on NearestNeighbors search in PCA space
    Based on metrics described in Chung, Magland et al. (2017) Neuron 95: 1381-1394
    Inputs:
    -------
    all_pcs : numpy.ndarray (num_spikes x PCs)
        2D array of PCs for all spikes
    all_labels : numpy.ndarray (num_spikes x 0)
        1D array of cluster labels for all spikes
    this_unit_id : Int
        number corresponding to unit for which these metrics will be calculated
    max_spikes_for_nn : Int
        number of spikes to use (calculation can be very slow when this number is >20000)
    n_neighbors : Int
        number of neighbors to use
    Outputs:
    --------
    hit_rate : float
        Fraction of neighbors for target cluster that are also in target cluster
    miss_rate : float
        Fraction of neighbors outside target cluster that are in target cluster
    """

    total_spikes = all_pcs.shape[0]
    ratio = max_spikes_for_nn / total_spikes
    this_unit = all_labels == this_unit_id

    X = np.concatenate((all_pcs[this_unit, :], all_pcs[np.invert(this_unit), :]), 0)

    n = np.sum(this_unit)

    if ratio < 1:
        inds = np.arange(0, X.shape[0] - 1, 1 / ratio).astype('int')
        X = X[inds, :]
        n = int(n * ratio)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    this_cluster_inds = np.arange(n)

    this_cluster_nearest = indices[:n, 1:].flatten()
    other_cluster_nearest = indices[n:, 1:].flatten()

    hit_rate = np.mean(this_cluster_nearest < n)
    miss_rate = np.mean(other_cluster_nearest < n)

    return hit_rate, miss_rate


def silhouette_score_inner_loop(i, cluster_ids, cluster_labels, all_pcs):
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

    # Build lists
    if do_parallel:
        from joblib import Parallel, delayed
        scores = Parallel(n_jobs=-1, verbose=12)(delayed(silhouette_score_inner_loop)
                                                 (i, cluster_ids, cluster_labels, all_pcs) for i in cluster_ids)
    else:
        scores = [silhouette_score_inner_loop(i, cluster_ids, cluster_labels, all_pcs) for i in cluster_ids]

    # Fill the 2d array
    for i, col_score in enumerate(scores):
        if np.isnan(col_score).all():
            warnings.warn(f'Cluster {i} has all-nan col_score. It will have nan silhouette score')
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
        if np.isnan(pc_power).any():
            nan_clusters = spike_clusters[np.isnan(pc_power)]
            warnings.warn(f'pc_power for {nan_clusters} are nan')

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
