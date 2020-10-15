from pathlib import Path
from tqdm import tqdm
import urllib3


def download_test_data(base_url='http://data.cortexlab.net/singlePhase3/data/',
                       base_path='test_data/',
                       download_features=True
                       ):
    # Add name of dataset:
    url_split = base_url.split('/')
    if len(url_split) == 6:
        base_path = Path(base_path + "".join(url_split[-3:]))
    elif len(url_split) > 6:
        base_path = Path(base_path + "".join(url_split[-4:]))
    base_path.mkdir(parents=True, exist_ok=True)
    print(f'Downloading test data to {base_path.absolute()}')
    files = ['amplitudes.npy', 'channel_map.npy', 'channel_positions.npy', 'cluster_groups.csv', 'spike_clusters.npy',
             'spike_templates.npy', 'spike_times.npy', 'templates.npy', 'templates_ind.npy', 'whitening_mat_inv.npy',
             ]
    if download_features:
        files += ['pc_features.npy', 'pc_feature_ind.npy']
    for file in tqdm(files):
        fname = base_path / Path(file)
        if not fname.exists():
            http = urllib3.PoolManager()
            r = http.request('GET', base_url + file)
            with open(fname, 'wb') as out:
                out.write(r.data)
            r.release_conn()
    return base_path, files
