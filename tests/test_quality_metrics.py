from cluster_quality import io
from tests.test_dependencies import download_test_data


def test_isi_violations():
    ### SinglePhase3
    base_path, files = download_test_data(base_url='http://data.cortexlab.net/singlePhase3/data/',
                                          base_path='tests/test_data/')

    (spike_times, spike_clusters, spike_templates, templates, amplitudes,
     unwhitened_temps, channel_map, cluster_ids, cluster_quality, pc_features, pc_feature_ind
     ) = io.load_kilosort_data(base_path, include_pcs=True)

    isi_violations(spike_train, min_time, max_time, isi_threshold, min_isi=0)
    assert True
