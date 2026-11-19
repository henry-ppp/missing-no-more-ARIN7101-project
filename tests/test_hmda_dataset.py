"""Offline tests for the HMDA Rhode Island 2023 dataset: no network, synthetic Data Browser responses only."""
import gzip
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from data_preprocessing.clean import clean
from data_preprocessing.dataset_config import DATASETS, OPTIONAL, RETIRED
from data_preprocessing.download import HMDA_QUERY, download, hmda_csv, load_raw

FEATURES = OPTIONAL['hmda_ri_2023'].features
FILE_URL = 'https://files.ffiec.cfpb.gov/data-browser/datasets/2023/filtered-queries/one-year/abc.csv'


def data_browser_csv():
    """A small Data Browser-like CSV: extra columns, NA and Exempt entries, and a tract column to be ignored."""
    rows = [
        dict(activity_year=2023, lei='A', loan_amount=235000, loan_to_value_ratio='80.0', interest_rate='6.5',
             rate_spread='0.3', loan_term='360', property_value='295000', income='110', tract_population=4000),
        dict(activity_year=2023, lei='B', loan_amount=105000, loan_to_value_ratio='Exempt', interest_rate='7.0',
             rate_spread='1.1', loan_term='360', property_value='Exempt', income='70', tract_population=4100),
        dict(activity_year=2023, lei='C', loan_amount=455000, loan_to_value_ratio='95.0', interest_rate='NA',
             rate_spread='NA', loan_term='360', property_value='485000', income='180', tract_population=3900),
        dict(activity_year=2023, lei='D', loan_amount=55000, loan_to_value_ratio='60.5', interest_rate='8.25',
             rate_spread='-0.2', loan_term='240', property_value='95000', income='-5', tract_population=4000),
    ]
    return pd.DataFrame(rows).to_csv(index=False).encode()


class Response(io.BytesIO):
    def __init__(self, body, url):
        super().__init__(body)
        self.url = url

    def geturl(self):
        return self.url


class HMDADataset(unittest.TestCase):
    def test_schema_keeps_hmda_optional(self):
        self.assertEqual(list(DATASETS), ['credit_card', 'california_housing', 'superconductivity', 'gas_turbine'])
        self.assertEqual(list(OPTIONAL), ['hmda_ri_2023'])
        self.assertEqual(RETIRED, {})
        self.assertEqual(FEATURES, ('loan_amount', 'loan_to_value_ratio', 'interest_rate', 'rate_spread', 'loan_term',
                                    'property_value', 'income'))

    def test_follows_a_gzipped_redirect_notice(self):
        notice = gzip.compress(f'This and all future requests should be directed to <a href="{FILE_URL}">this URI</a>.'
                               .encode())
        responses = [Response(notice, HMDA_QUERY), Response(gzip.compress(data_browser_csv()), FILE_URL)]
        with patch('data_preprocessing.download.urlopen', side_effect=lambda *a, **k: responses.pop(0)) as network:
            body, url = hmda_csv()
        self.assertEqual(network.call_count, 2)
        self.assertEqual(url, FILE_URL)
        self.assertEqual(body, data_browser_csv())

    def test_follows_an_http_redirect(self):
        with patch('data_preprocessing.download.urlopen', return_value=Response(data_browser_csv(), FILE_URL)):
            body, url = hmda_csv()
        self.assertEqual((body, url), (data_browser_csv(), FILE_URL))

    def test_rejects_an_unexpected_response(self):
        with patch('data_preprocessing.download.urlopen', return_value=Response(b'<html>error</html>', HMDA_QUERY)):
            with self.assertRaises(ValueError):
                hmda_csv()

    def test_download_stores_bytes_and_cleaning_keeps_complete_loans(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            with patch('data_preprocessing.download.urlopen', return_value=Response(data_browser_csv(), FILE_URL)):
                folder = download('hmda_ri_2023', temp)
            self.assertEqual((folder / 'source.csv').read_bytes(), data_browser_csv())
            metadata = json.loads((folder / 'source_metadata.json').read_text())
            self.assertEqual((metadata['query'], metadata['file_url'], metadata['rows']), (HMDA_QUERY, FILE_URL, 4))
            raw, _ = load_raw('hmda_ri_2023', temp)
            cleaned, summary = clean('hmda_ri_2023', raw)
        self.assertEqual(tuple(cleaned.columns), FEATURES)
        self.assertEqual(cleaned.index.tolist(), ['source.csv:0', 'source.csv:3'])   # Exempt and NA rows dropped
        self.assertEqual(summary['dropped_invalid_rows'], 2)
        self.assertEqual(cleaned.loc['source.csv:3', 'income'], -5)                  # negative income kept


if __name__ == '__main__':
    unittest.main()
