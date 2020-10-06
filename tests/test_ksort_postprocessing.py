from cluster_quality import io
from tests import test_dependencies
from cluster_quality import ksort_postprocessing


def test_remove_double_counted_spikes():
    ### SinglePhase3
    base_path, files = test_dependencies.download_test_data(
        base_url='http://data.cortexlab.net/singlePhase3/data/',
        base_path='tests/test_data/', download_features=False)

    (spike_times, spike_clusters, spike_templates, templates, amplitudes,
     unwhitened_temps, channel_map, cluster_ids, cluster_quality, pc_features, pc_feature_ind
     ) = io.load_kilosort_data(base_path, include_pcs=False, sample_rate=3e4)

    (spike_times, spike_clusters, spike_templates, amplitudes, pc_features, overlap_matrix
     ) = ksort_postprocessing.remove_double_counted_spikes(spike_times,
                                                           spike_clusters,
                                                           spike_templates,
                                                           amplitudes,
                                                           channel_map,
                                                           templates,
                                                           pc_features,
                                                           sample_rate=3e4)
    assert sum(spike_times)>0
    import numpy as np
    assert np.isclose(sum(spike_times), 12437925816.246)
    assert spike_times.shape[0]==6941125
