import os
import zipfile

import numpy as np

import preprocess_uncertain_text as put
import pytest

def test_preprocess_uncertain_text():
    assert put.edit_distance_numba('городокккк', 'гоoрдокккк') == 2
    assert put.edit_distance('городокккк', 'гоoрдок') == 5
