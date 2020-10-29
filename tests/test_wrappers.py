from cluster_quality import io
from importlib import reload

reload(io)
from tests import test_dependencies

reload(test_dependencies)
from cluster_quality import wrappers
from pathlib import Path
import pandas as pd
import numpy as np
np.random.seed(1000)


# from ecephys_spike_sorting.modules.quality_metrics import metrics


def test_calculate_isi_violations():
    """
    The simplest metric
    Returns
    -------

    """
    ### SinglePhase3
    base_path, files = test_dependencies.download_test_data(
        base_url='http://data.cortexlab.net/singlePhase3/data/',
        base_path='test_data/', download_features=False)
    (spike_times, spike_clusters, spike_templates, templates, amplitudes,
     unwhitened_temps, channel_map, cluster_ids, cluster_quality, pc_features, pc_feature_ind
     ) = io.load_kilosort_data(base_path, include_pcs=False, sample_rate=3e4)
    isi_viol_rate, isi_viol_n = wrappers.calculate_isi_violations(spike_times, spike_clusters,
                                                                  isi_threshold=0.0015,
                                                                  min_isi=0.000166,
                                                                  )

    assert np.isclose(isi_viol_rate.mean(), 2.455, atol=.001)
    assert np.isclose(isi_viol_n.mean(), 91.135, atol=.00001)


def download_and_load(include_pcs=True, subsample=50):
    # SinglePhase3
    base_path, files = test_dependencies.download_test_data(base_url='http://data.cortexlab.net/singlePhase3/data/',
                                                            base_path='test_data/',
                                                            download_features=True)
    path_expected = Path(f'expected_output/{base_path.parts[-1]}')
    if not path_expected.parent.exists():
        path_expected = Path(f'tests/expected_output/{base_path.parts[-1]}')
    path_expected.mkdir(parents=False, exist_ok=True)
    (spike_times, spike_clusters, spike_templates, templates, amplitudes, unwhitened_temps,
     channel_map, cluster_ids, cluster_quality, pc_features, pc_feature_ind
     ) = io.load_kilosort_data(base_path, include_pcs=include_pcs, sample_rate=3e4)

    # Subsample for speed
    i = np.arange(0, spike_clusters.shape[0], subsample)  # last number is subsampling factor
    # i = np.random.randint(0, spike_clusters.shape[0], int(spike_clusters.shape[0] / 50))
    (spike_times, spike_clusters, spike_templates, amplitudes) = (
        spike_times[i], spike_clusters[i], spike_templates[i], amplitudes[i])
    if include_pcs:
        pc_features = pc_features[i]

    return (base_path, path_expected, spike_times, spike_clusters, spike_templates, templates, amplitudes,
            unwhitened_temps, channel_map, cluster_ids, cluster_quality, pc_features, pc_feature_ind)


def test_calculate_timing_metrics():
    (base_path, path_expected, spike_times, spike_clusters, spike_templates, templates, amplitudes,
     unwhitened_temps, channel_map, cluster_ids, cluster_quality, pc_features, pc_feature_ind
     ) = download_and_load(include_pcs=False)

    # Test separate stages by turning `do_*` flags on or off
    df = wrappers.calculate_metrics(spike_times, spike_clusters, spike_templates, amplitudes, pc_features,
                                    pc_feature_ind,
                                    output_folder=None, do_parallel=False,
                                    do_pc_features=False, do_silhouette=False, do_drift=False)
    # df.to_csv(path_expected / 'timing_metrics.csv', index=False)  # Uncomment this if results must change
    df1 = pd.read_csv(path_expected / 'timing_metrics.csv')
    pd.testing.assert_frame_equal(df.round(1), df1.round(1), check_dtype=False)

    for col in df.columns:
        assert not df[col].isna().all(), f' Column {col} is all nan'


def test_calculate_pc_features():
    (base_path, path_expected, spike_times, spike_clusters, spike_templates, templates, amplitudes,
     unwhitened_temps, channel_map, cluster_ids, cluster_quality, pc_features, pc_feature_ind
     ) = download_and_load(subsample=1000)

    df = wrappers.calculate_metrics(spike_times, spike_clusters, spike_templates, amplitudes, pc_features,
                                    pc_feature_ind,
                                    output_folder=None, do_parallel=True,
                                    do_pc_features=True, do_silhouette=False, do_drift=False)
    # df.to_csv(path_expected / 'pc_features.csv', index=False)  # Uncomment this if results must change
    # pd.testing.assert_frame_equal(df.round(1), pd.read_csv(path_expected / 'pc_features.csv').round(1),
    #                               check_dtype=False)

    for col in df.columns:
        assert not df[col].isna().all(), f' Column {col} is all nan'


def test_calculate_silhouette():
    (base_path, path_expected, spike_times, spike_clusters, spike_templates, templates, amplitudes,
     unwhitened_temps, channel_map, cluster_ids, cluster_quality, pc_features, pc_feature_ind
     ) = download_and_load(subsample=1000)

    df = wrappers.calculate_metrics(spike_times, spike_clusters, spike_templates, amplitudes, pc_features,
                                    pc_feature_ind,
                                    output_folder=None, do_parallel=True,
                                    do_pc_features=False, do_silhouette=True, do_drift=False)
    # df.to_csv(path_expected / 'silhouette.csv', index=False)  # Uncomment this if results must change
    pd.testing.assert_frame_equal(df.round(1), pd.read_csv(path_expected / 'silhouette.csv').round(1),
                                  check_dtype=False)
    for col in df.columns:
        assert not df[col].isna().all(), f' Column {col} is all nan'


def test_calculate_drift():
    (base_path, path_expected, spike_times, spike_clusters, spike_templates, templates, amplitudes,
     unwhitened_temps, channel_map, cluster_ids, cluster_quality, pc_features, pc_feature_ind
     ) = download_and_load(subsample=1000)

    df = wrappers.calculate_metrics(spike_times, spike_clusters, spike_templates, amplitudes, pc_features,
                                    pc_feature_ind,
                                    output_folder=None, do_parallel=True,
                                    do_pc_features=False, do_silhouette=False, do_drift=True)
    # df.to_csv(path_expected / 'drift.csv', index=False)  # Uncomment this if results must change
    pd.testing.assert_frame_equal(df.round(1), pd.read_csv(path_expected / 'drift.csv').round(1), check_dtype=False)
    for col in df.columns:
        assert not df[col].isna().all(), f' Column {col} is all nan'
