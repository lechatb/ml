import os

import pandas as pd

import download_news
import pytest


def test_passing():
    assert True

def test_failure():
   assert False


def test_download_rss():
    def check_dataframe(df):
        if isinstance(df, pd.DataFrame):
            if df.shape[0] > 1:
                if set(df.columns) == set(['title',
                      'description',
                      'link',
                      'published',
                      'summary']):
                    return True
        return False
    res_df = download_news.download_rss()
    assert check_dataframe(res_df)
