import urllib.request
from pathlib import Path
from tqdm import tqdm
from cluster_quality import io
from importlib import reload

reload(io)


def download_test_data(base_url='http://data.cortexlab.net/singlePhase3/data/',
                       base_path='tests/test_data/'
                       ):
    # Add name of dataset:
    base_path=Path(base_path+"".join(base_url.split('/')[-3:]))
    base_path.mkdir(parents=True, exist_ok=True)
    files = ['amplitudes.npy', 'channel_map.npy', 'channel_positions.npy', 'cluster_groups.csv', 'spike_clusters.npy',
             'spike_templates.npy', 'spike_times.npy', 'templates.npy', 'templates_ind.npy', 'whitening_mat_inv.npy',
             'pc_features.npy', 'pc_feature_ind.npy']
    for file in tqdm(files):
        fname = base_path / Path(file)
        if not fname.exists():
            http = urllib3.PoolManager()
            r = http.request('GET', base_url + file)
            with open(fname, 'wb') as out:
                out.write(r.read())
            #urllib.request.urlretrieve(base_url + file, fname)
    return base_path, files


def test_load_kilosort_data():
    ### SinglePhase3
    base_path, files = download_test_data(base_url='http://data.cortexlab.net/singlePhase3/data/',
                       base_path='tests/test_data/')
    outputs = io.load_kilosort_data(base_path, include_pcs=True)
    assert sum([output is None for output in outputs]) == 0, f'Missing some files? Some outputs are None: {outputs}'
    (spike_times, spike_clusters, spike_templates, templates, amplitudes,
     unwhitened_temps, channel_map, cluster_ids, cluster_quality, pc_features, pc_feature_ind) = outputs
    assert len(set(spike_clusters)) == 675
    assert sum(cluster_quality=='good') == 242

