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
        return pd.json_normalize(json.load(f)[0]['objects'])

res = read_yolo(YOLO_JSON_FILEPATH_1M)
res.shape


# %%
def _make_pair_wise(df: pd.DataFrame):
    midx = pd.MultiIndex.from_product([df.index, df.index], names=['n1', 'n2'])
    pair = pd.concat(
        [df.add_suffix('_1').reindex(midx, level='n1'),
         df.add_suffix('_2').reindex(midx, level='n2')], axis=1)
    return pair.loc[
        midx[midx.get_level_values(0) < midx.get_level_values(1)]]

def _euclidean_dist(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


# %%
# calc pair dist
dist = (
    res[['name', 'confidence',
         'relative_coordinates.center_x', 'relative_coordinates.center_y']]
        .rename(columns=lambda s: s.split('_')[-1])
        .query('confidence >= 0.7')  # debug
        # make uniq names for pair-wise dists
        .assign(group_rank=lambda df:
                    df.groupby('name').x
                        .transform(lambda s: s.rank(method='first')))
        .assign(uniq_name=lambda df:
                    df.name.str
                        .cat([df.group_rank.astype(int).astype(str)], sep='_'))
        .drop(columns=['group_rank']).set_index('uniq_name')
        .pipe(_make_pair_wise)
        .query('name_1 == name_2')
        .assign(dist_=lambda df: df.apply(
            lambda df: _euclidean_dist(df.x_1, df.y_1, df.x_2, df.y_2),
            axis=1))
        .sort_index()
)
dist.shape

# %%
X_dist = (
    dist.query('0.1 <= dist_ <= 0.3')
        .dist_
        .values.reshape(-1, 1)
)
X_dist.shape

# %%
import sklearn.cluster

def find_densest(dists, min_size):
    """Find densest subset of dists.

    As part of smart dedup, the idea is to find the 'mode' of
    all pair-wise distances, in order to filter out bad pairs.

    - `eps` tuned for dist between two symbols on the same card
    - only works for 1-d array currently"""
    X = np.array(dists).reshape(-1, 1)
    dbscan = sklearn.cluster.DBSCAN(eps=0.01, min_samples=min_size)
    clt_id = dbscan.fit(X).labels_

    clustered = pd.DataFrame(dict(dist_=X.ravel(), clt_id_=clt_id))
    if clustered.clt_id_.max() > 0:
        print("WARNING: more than one cluster found")
    densest = clustered.loc[clustered.clt_id_ == 0, 'dist_']
    return densest

densest_dist = find_densest(X_dist, min_size=3)
densest_dist.shape

# %%
# good dup
good_pair = dist.loc[dist.dist_.isin(densest_dist)].filter(regex='^(name|x|y)')
good_dup = pd.concat([
    good_pair.filter(regex='_1$').rename(columns=lambda s: s.replace('_1', '')),
    good_pair.filter(regex='_2$').rename(columns=lambda s: s.replace('_2', ''))
], ignore_index=True)
good_dup.shape

# %%
# all dup
all_dup = res.loc[res.name.isin(res.name.value_counts().loc[lambda s: s > 1].index)]
all_dup.shape

# %%
res.pipe(locate_detected_classes, min_conf=0)

res.sort_values('confidence').drop_duplicates('name', keep='last').append(
    all_dup
        .merge(good_dup.set_index(['name', 'x', 'y']),
               left_on=['name',
                        'relative_coordinates.center_x',
                        'relative_coordinates.center_y'],
               right_index=True)
).drop_duplicates().pipe(locate_detected_classes)


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
            (row['relative_coordinates.center_x'],
             1-row['relative_coordinates.center_y']))


# %%
res = read_yolo(YOLO_JSON_FILEPATH_3I)
res.shape

# %%
locate_detected_classes(res)
# manual edit looks good

# %%
