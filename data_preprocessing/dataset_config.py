"""Official benchmark schema; feature order is part of the experiment contract."""
from dataclasses import dataclass

@dataclass(frozen=True)
class DatasetConfig:
    name: str
    rows: int
    features: tuple[str, ...]
    excluded: tuple[str, ...]
    source: str
    uci_id: int | None = None
    raw_files: tuple[str, ...] = ()
    version: str = 'Official source snapshot; SHA256 recorded at preparation'
    cleaning: str = 'Keep duplicate observations; drop rows with nonnumeric, NaN or infinite selected values; reject constant features. No clipping or transforms.'

_properties = ('atomic_mass', 'fie', 'atomic_radius', 'Density', 'ElectronAffinity', 'FusionHeat', 'ThermalConductivity', 'Valence')
_summaries = ('mean', 'wtd_mean', 'gmean', 'wtd_gmean', 'entropy', 'wtd_entropy', 'range', 'wtd_range', 'std', 'wtd_std')
DATASETS = {
    'credit_card': DatasetConfig('Default of Credit Card Clients', 30000,
        ('LIMIT_BAL', 'AGE') + tuple(f'BILL_AMT{i}' for i in range(1, 7)) + tuple(f'PAY_AMT{i}' for i in range(1, 7)),
        ('ID', 'default payment next month', 'SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'),
        'https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients', 350, ('source.csv',)),
    'california_housing': DatasetConfig('California Housing', 20640,
        ('MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'),
        ('MedHouseVal',), 'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html', raw_files=('source.csv',)),
    'superconductivity': DatasetConfig('Superconductivty Data', 21263,
        ('number_of_elements',) + tuple(f'{s}_{p}' for p in _properties for s in _summaries),
        ('critical_temp', 'unique_m.csv (alternative representation, not additional rows)'),
        'https://archive.ics.uci.edu/dataset/464/superconductivty+data', 464, ('source.csv',)),
    'gas_turbine': DatasetConfig('Gas Turbine CO and NOx Emission Data Set', 36733,
        ('AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'CO', 'NOX'), ('year',),
        'https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set', 551,
        tuple(f'gt_{y}.csv' for y in range(2011, 2016))),
}

def config_path(root, dataset, mode, rate, seed):
    from pathlib import Path
    if dataset not in DATASETS or mode not in ('quick', 'full'):
        raise ValueError('Unknown dataset or mode')
    if not 0 < rate < 1:
        raise ValueError('Benchmark rate must be strictly between 0 and 1')
    return Path(root) / dataset / mode / f'rate_{float(rate)}_seed_{seed}'
