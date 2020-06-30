# cluster_quality
Quality metrics based on Allen Institute's 
# Setup
In Anaconda prompt:
`pip install -U git+https://github.com/mmyros/cluster_quality.git`

OR 

Download/clone this repo and run `python setup.py install` or `python setup.py develop`

# Usage from command line
```bash
cd path/to/sorting
cluster_quality 
OR (from any path):
cluster_quality --do_parallel=0 --do_drift=0 --kilosort-folder=path/to/sorting 
```
cluster_quality -h for more options


# Usage from python/Spyder/PyCharm:
```python
from cluster_quality.scripts import cluster_quality
path='C:\my_path_to_files'
cluster_quality.cli(kilosort_folder='path/to/sorting', do_drift=0,do_parallel=1)
```