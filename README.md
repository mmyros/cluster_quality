[![Build Status](https://travis-ci.com/mmyros/cluster_quality.svg?branch=master)](https://travis-ci.com/mmyros/cluster_quality)
[![codecov](https://codecov.io/gh/mmyros/cluster_quality/branch/master/graph/badge.svg?token=Y4K6ADRTXR)](undefined)
[![Coverage Status](https://coveralls.io/repos/github/mmyros/cluster_quality/badge.svg)](https://coveralls.io/github/mmyros/cluster_quality)
# cluster_quality
Quality metrics based on Allen Institute's 
# Setup
In Anaconda prompt:
`pip install -U git+https://github.com/mmyros/cluster_quality.git`

OR 

Download/clone this repo and run `python setup.py install` or `python setup.py develop`

## Note for Windows users
If for some reason dependencies do not get installed automatically and you run into missing packages problems, use:
``` bash
pip install click numpy scipy scikit-learn pandas joblib
```

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

If all goes well, a comma-separated file `quality_metrics.csv` should appear in the same folder. 
To inspect it, open in Excel or:
```python
import pandas as pd
result=pd.read_csv('quality_metrics.csv')
print(result.head())
```