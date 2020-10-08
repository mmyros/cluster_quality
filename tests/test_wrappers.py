from cluster_quality import io
from importlib import reload
reload(io)
from tests import test_dependencies
reload(test_dependencies)
from cluster_quality import wrappers
from pathlib import Path
import pandas as pd
import numpy as np


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
        base_path='tests/test_data/', download_features=False)
    (spike_times, spike_clusters, spike_templates, templates, amplitudes,
     unwhitened_temps, channel_map, cluster_ids, cluster_quality, pc_features, pc_feature_ind
     ) = io.load_kilosort_data(base_path, include_pcs=False, sample_rate=3e4)
    isi_viol_rate, isi_viol_n = wrappers.calculate_isi_violations(spike_times, spike_clusters,
                                                                  isi_threshold=0.0015,
                                                                  min_isi=0.000166,
                                                                  )

    assert np.isclose(isi_viol_rate.mean(), 2.455, atol=.001)
    assert np.isclose(isi_viol_n.mean(), 91.135, atol=.00001)



def test_calculate_metrics():
    ### SinglePhase3
    base_path, files = test_dependencies.download_test_data(base_url='http://data.cortexlab.net/singlePhase3/data/',
                                                            base_path='tests/test_data/',
                                                            download_features=True)
    path_expected = Path(f'tests/expected_output/{base_path.parts[-1]}')
    path_expected.mkdir(parents=True, exist_ok=True)

    (spike_times, spike_clusters, spike_templates, templates, amplitudes,
     unwhitened_temps, channel_map, cluster_ids, cluster_quality, pc_features, pc_feature_ind
     ) = io.load_kilosort_data(base_path, include_pcs=False, sample_rate=3e4)

    # Subsample for speed
    i = np.arange(0, spike_clusters.shape[0], 50)  # last number is subsampling factor
    # i = np.random.randint(0, spike_clusters.shape[0], int(spike_clusters.shape[0] / 50))
    (spike_times, spike_clusters, spike_templates, amplitudes) = (
        spike_times[i], spike_clusters[i], spike_templates[i], amplitudes[i])

    # Test separate stages by turning `do_*` flags on or off
    df = wrappers.calculate_metrics(spike_times, spike_clusters, spike_templates, amplitudes, pc_features,
                                    pc_feature_ind,
                                    output_folder=None, do_parallel=False,
                                    do_pc_features=False, do_silhouette=False, do_drift=False)
    df.to_csv(path_expected / 'timing_metrics.df', index=False)
    df1 = pd.read_csv(path_expected / 'timing_metrics.df')
    pd.testing.assert_frame_equal(df, df1, check_dtype=False)

    (spike_times, spike_clusters, spike_templates, _, amplitudes, _, _, _, _, pc_features, pc_feature_ind
     ) = io.load_kilosort_data(base_path, include_pcs=True, sample_rate=3e4)

    # Subsample for speed
    i = np.arange(0, spike_clusters.shape[0], 50)  # last number is subsampling factor
    # i = np.random.randint(0, spike_clusters.shape[0], int(spike_clusters.shape[0] / 50))
    (spike_times, spike_clusters, spike_templates, amplitudes, pc_features) = (
        spike_times[i], spike_clusters[i], spike_templates[i], amplitudes[i], pc_features[i])

    df = wrappers.calculate_metrics(spike_times, spike_clusters, spike_templates, amplitudes, pc_features,
                                    pc_feature_ind,
                                    output_folder=None, do_parallel=True,
                                    do_pc_features=True, do_silhouette=False, do_drift=False)
    df.to_csv(path_expected / 'pc_features.df', index=False)
    pd.testing.assert_frame_equal(df, pd.read_csv(path_expected / 'pc_features.df'), check_dtype=False)

    df = wrappers.calculate_metrics(spike_times, spike_clusters, spike_templates, amplitudes, pc_features,
                                    pc_feature_ind,
                                    output_folder=None, do_parallel=True,
                                    do_pc_features=False, do_silhouette=True, do_drift=False)
    df.to_csv(path_expected / 'silhouette.df', index=False)
    pd.testing.assert_frame_equal(df, pd.read_csv(path_expected / 'silhouette.df'), check_dtype=False)

    df = wrappers.calculate_metrics(spike_times, spike_clusters, spike_templates, amplitudes, pc_features,
                                    pc_feature_ind,
                                    output_folder=None, do_parallel=True,
                                    do_pc_features=False, do_silhouette=False, do_drift=True)
    df.to_csv(path_expected / 'drift.df', index=False)
    pd.testing.assert_frame_equal(df, pd.read_csv(path_expected / 'drift.df'), check_dtype=False)


# test_calculate_metrics()
