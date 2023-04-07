import os
import zipfile

import numpy as np

import preprocess_texts
import pytest


def test_preprocess_tfidf_stem_texts():
    texts = np.array(["Уникальная информация",
                      "Текст на русском языке, со знаками препинания!",
                      "Другой текст на русском языке.",
                      "Если друг оказался вдруг и не друг, и не враг, а так..."])
    encoded_texts, vocabulary = preprocess_texts.preprocess_tfidf_texts(
        texts, proc_it=preprocess_texts.stem_it)
    res_voc = {'уникальн': 9, 'информац': 4, 'текст': 8, 'русск': 7,
               'язык': 10, 'знак': 3, 'препинан': 6, 'друг': 1,
               'есл': 2, 'оказа': 5, 'враг': 0}
    res_text = np.array([[0., 0., 0., 0., 0.70710678, 0., 0., 0., 0., 0.70710678, 0.],
                        [0., 0., 0., 0.50867187, 0., 0., 0.50867187,
                            0.40104275, 0.40104275, 0., 0.40104275],
                        [0., 0.5, 0., 0., 0., 0., 0., 0.5, 0.5, 0., 0.5],
                        [0.42693074, 0.6731942, 0.42693074, 0., 0., 0.42693074,  0., 0., 0., 0., 0.]])
    assert vocabulary == res_voc
    assert np.allclose(encoded_texts, res_text, 0.01)


def test_preprocess_tfidf_lem_texts():
    texts = np.array(["Уникальная информация",
                      "Текст на русском языке, со знаками препинания!",
                      "Другой текст на русском языке.",
                      "Если друг оказался вдруг и не друг, и не враг, а так..."])
    encoded_texts, vocabulary = preprocess_texts.preprocess_tfidf_texts(
        texts, proc_it=preprocess_texts.lem_it)
    #print( encoded_texts)
    #print( vocabulary)
    res_text = np.array([[0., 0., 0., 0.70710678, 0., 0., 0., 0., 0.70710678, 0.],
                         [0., 0., 0.50867187, 0., 0., 0.50867187,
                             0.40104275, 0.40104275,  0., 0.40104275],
                         [0., 0., 0., 0., 0., 0., 0.57735027,
                             0.57735027, 0., 0.57735027],
                         [0.40824829, 0.81649658, 0., 0., 0.40824829, 0., 0., 0., 0., 0.]])
    res_voc = {'уникальный': 8, 'информация': 3, 'текст': 7, 'русский': 6, 'язык': 9,
               'знак': 2, 'препинание': 5, 'друг': 1, 'оказаться': 4, 'враг': 0}
    assert vocabulary == res_voc
    assert np.allclose(encoded_texts, res_text, 0.01)


def test_existance_and_structure_of_vocabulary_file():
    filename = "unigrams.cyr.lc.zip"
    assert os.path.isfile(filename), f"File '{filename}' not found"
    assert zipfile.is_zipfile(
        filename), f"File '{filename}' is not a valid ZIP archive"

    with zipfile.ZipFile(filename, "r") as zip_file:
        file_list = zip_file.namelist()
        assert len(
            file_list) == 1, f"ZIP archive '{filename}' must contain exactly one file"
        file_name = file_list[0]
        assert file_name.endswith(
            ".lc"), f"ZIP archive '{filename}' must contain an lc file"
        with zip_file.open(file_name, "r") as text_file:
            for line in text_file:
                parts = line.decode("utf-8").strip().split("\t")
                assert len(
                    parts) == 3, f"Line '{line}' in file '{file_name}' does not have 3 tab-separated parts"
                assert all(part.isnumeric(
                ) for part in parts[1:]), f"Line '{line}' in file '{file_name}' has non-numeric counts"


def test_get_idf_lem_vocabulary():
    voc = preprocess_texts.get_idf_vocabulary(proc_it=preprocess_texts.lem_it)
    assert len(voc) > 0, "too short lem vocabulary"
    assert "рогатка" in voc.keys(), "Wrong content of lem vocabulary"


def test_get_idf_stem_vocabulary():
    voc = preprocess_texts.get_idf_vocabulary(proc_it=preprocess_texts.stem_it)
    assert len(voc) > 0, "too short stem vocabulary"
    assert "рогат" in voc.keys(), "Wrong content of stem vocabulary"
