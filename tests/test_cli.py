from cluster_quality import io
from importlib import reload

reload(io)
from tests import test_dependencies

reload(test_dependencies)
from cluster_quality import wrappers
from pathlib import Path
import pandas as pd
import numpy as np
from click.testing import CliRunner
from cluster_quality.scripts import cluster_quality
import os

def test_cli():
    runner = CliRunner()
    base_path, files = test_dependencies.download_test_data(
        base_url='http://data.cortexlab.net/singlePhase3/data/',
        base_path='tests/test_data/', download_features=False)
    os.chdir(base_path)
    result = runner.invoke(cluster_quality.cli, '--do_pc_features=0 --do_silhouette=0 --do_drift=0')
    assert result.exit_code == 0

# def test_cli_explicit_path():
#     runner = CliRunner()
#     base_path, files = test_dependencies.download_test_data(
#         base_url='http://data.cortexlab.net/singlePhase3/data/',
#         base_path='tests/test_data/', download_features=False)
#     os.chdir(base_path)
#
#     result = runner.invoke(
#         cluster_quality.cli,
#         f'--kilosort-folder={str(base_path.absolute())}/ ')
#     # --do_pc_features=0 --do_silhouette=0 --do_drift=0
#     assert result.exit_code == 0
