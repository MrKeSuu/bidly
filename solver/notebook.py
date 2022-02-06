# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import importlib

import converter
importlib.reload(converter)

converter = converter.DealConverter()
converter.read_yolo('fixtures/deal1-result-md.json')

# %%
converter.card.name.unique()

# %%
converter.dedup()
converter.report_missing_and_fp()

# %%
