from cluster_quality import io
from tests.test_dependencies import download_test_data

def test_load_kilosort_data():
    ### SinglePhase3
    base_path, files = download_test_data(base_url='http://data.cortexlab.net/singlePhase3/data/',
                                          base_path='tests/test_data/')
    outputs = io.load_kilosort_data(base_path, include_pcs=True)
    assert sum([output is None for output in outputs]) == 0, f'Missing some files? Some outputs are None: {outputs}'
    (spike_times, spike_clusters, spike_templates, templates, amplitudes,
     unwhitened_temps, channel_map, cluster_ids, cluster_quality, pc_features, pc_feature_ind) = outputs
    assert len(set(spike_clusters)) == 675
    assert sum(cluster_quality == 'good') == 242

    ### doublePhase3 (has no feature files available)
    base_path, files = download_test_data(base_url='http://data.cortexlab.net/dualPhase3/data/frontal/',
                                          base_path='tests/test_data/')
    outputs = io.load_kilosort_data(base_path, include_pcs=False)
    assert sum([output is None for output in outputs[:-2]]) == 0, f'Missing some files? Some outputs are None: {outputs}'
    (spike_times, spike_clusters, spike_templates, templates, amplitudes,
     unwhitened_temps, channel_map, cluster_ids, cluster_quality, pc_features, pc_feature_ind) = outputs
