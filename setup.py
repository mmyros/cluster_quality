from setuptools import setup, find_packages

setup(
    name='cluster_quality',
    version='0.0.1',
    install_requires=[
        'click',
        'numpy',
        'scipy',
        'scikit-learn',
        'pandas',
        'joblib'
    ],
    packages=find_packages(),
    include_package_data=True,
    py_modules=['cluster_quality'],
    entry_points='''
    [console_scripts]
    cluster_quality=cluster_quality.scripts.cluster_quality:cli
    ''',

    url='',
    license='MIT',
    author='Maxym Myroshnychenko',
    author_email='mmyros@gmail.com',
    description='extras for spikesorting curation'
)
