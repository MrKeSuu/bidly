# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import json
import importlib
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import converter
importlib.reload(converter)

YOLO_JSON_FILEPATH = pathlib.Path('fixtures/deal1-result-md.json')
YOLO_JSON_FILEPATH = pathlib.Path('fixtures/deal3-result-sm.json')

# %% [markdown]
# ## `DealConverter` whiteboard

# %%
converter = converter.DealConverter()
converter.read_yolo(YOLO_JSON_FILEPATH)

# %%
converter.card.name.unique()

# %%
converter.dedup()
converter.report_missing_and_fp()


# %% [markdown]
# ## Dedup EDA

# %%
def read_yolo(path):
    with open(path) as f:
        return pd.json_normalize(json.load(f)[0]['objects'], sep='__')

res = read_yolo(YOLO_JSON_FILEPATH)
res.shape

# %%
midx = pd.MultiIndex.from_product([res.name, res.name], names=['n1', 'n2'])
midx


# %%
def _make_pair_wise(df: pd.DataFrame):
    midx = pd.MultiIndex.from_product([df.index, df.index], names=['n1', 'n2'])
    dist = pd.concat(
        [df.add_suffix('_1').reindex(midx, level='n1'),
         df.add_suffix('_2').reindex(midx, level='n2')], axis=1)
    return dist.loc[
        midx[midx.get_level_values(0) < midx.get_level_values(1)]]


# %%
def _euclidean_dist(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


# %%
(
    res[['name', 'confidence',
         '__relative_coordinates__center_x', '__relative_coordinates__center_y']]
        .rename(columns=lambda s: s.split('_')[-1])
        .query('confidence > 0.99')  # debug
        .assign(group_rank=lambda df:
                    df.groupby('name')
                        .transform(lambda s: s.rank(method='first')).x)
        .assign(uniq_name=lambda df:
                    df.name.str.cat(
                        [df.group_rank.astype(int).astype(str)],
                        sep='_'))
        .drop(columns=['group_rank']).set_index('uniq_name')
        .pipe(_make_pair_wise)
        .query('name_1 == name_2').drop(columns=['name_1', 'name_2'])
        .assign(dist=lambda df: df.apply(
                    lambda df: _euclidean_dist(df.x_1, df.y_1, df.x_2, df.y_2),
                    axis=1))
        .sort_index()
)

# %%

# %% [markdown]
# ## Plot detected cards

# %%
import pathlib

FILENAME = 'deal3-manual-edit.json'
FILENAME = 'deal1-result-md.json'
YOLO_JSON_EDITTED_FILEPATH = pathlib.Path('fixtures') / FILENAME

def locate_detected_classes(res, min_conf=0.7):
    __, ax = plt.subplots(figsize=(10,10))

    for __, row in res.iterrows():
        if row['confidence'] < min_conf:
            continue
        ax.annotate(
            row['name'],
            (row['__relative_coordinates__center_x'],
             1-row['__relative_coordinates__center_y']))


# %%
res = read_yolo(YOLO_JSON_EDITTED_FILEPATH)
res.shape

# %%
locate_detected_classes(res)
# manual edit looks good

# %%
