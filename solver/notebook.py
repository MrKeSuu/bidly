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

def optimize_cell_width():
    from IPython.display import display, HTML
    display(HTML("<style>.container { width:100% !important; }</style>"))
optimize_cell_width()

YOLO_JSON_FILEPATH_1M = pathlib.Path('fixtures/deal1-result-md.json')
YOLO_JSON_FILEPATH_2M = pathlib.Path('fixtures/deal2-result-md.json')
YOLO_JSON_FILEPATH_2S = pathlib.Path('fixtures/deal2-result-sm.json')
YOLO_JSON_FILEPATH_3S = pathlib.Path('fixtures/deal3-result-sm.json')
YOLO_JSON_FILEPATH_3I = pathlib.Path('fixtures/deal3-manual-edit.json')

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

res = read_yolo(YOLO_JSON_FILEPATH_3S)
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
pair = (
    res[['name', 'confidence',
         '__relative_coordinates__center_x', '__relative_coordinates__center_y']]
        .rename(columns=lambda s: s.split('_')[-1])
        .query('confidence >= 0.7')  # debug
        .assign(group_rank=lambda df:
                    df.groupby('name')
                        .transform(lambda s: s.rank(method='first'))
                        .x)
        .assign(uniq_name=lambda df:
                    df.name.str
                        .cat([df.group_rank.astype(int).astype(str)], sep='_'))
        .drop(columns=['group_rank']).set_index('uniq_name')
        .pipe(_make_pair_wise)
        .query('name_1 == name_2').drop(columns=['name_1', 'name_2'])
        .assign(dist_=lambda df: df.apply(
            lambda df: _euclidean_dist(df.x_1, df.y_1, df.x_2, df.y_2), axis=1))
        .sort_index()
)
pair.shape

# %%
X_dist = (
    pair.query('0.1 <= dist_ <= 0.3')
        .dist_
        .values.reshape(-1, 1)
)
X_dist.shape

# %%
import sklearn.cluster

# %%
# agg_clt = sklearn.cluster.AgglomerativeClustering(n_clusters=19)
# clt_id = agg_clt.fit(X_dist).labels_

# pd.DataFrame(dict(dist_=X_dist.ravel(), clt_id_=clt_id)).sort_values('dist_')
# # hier clt not so effective, when not filtered by dist_: 0.1 <= dist_ <= 0.3

# %%
db_clt = sklearn.cluster.DBSCAN(eps=0.01, min_samples=3)  # trial-n-error eps
clt_id = db_clt.fit(X_dist).labels_

pd.DataFrame(dict(dist_=X_dist.ravel(), clt_id_=clt_id)).sort_values('dist_')


# %% [markdown]
# ## Plot detected cards

# %%
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
res = read_yolo(YOLO_JSON_FILEPATH_3I)
res.shape

# %%
locate_detected_classes(res)
# manual edit looks good

# %%
