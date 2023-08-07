# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: bidly
#     language: python
#     name: bidly
# ---

# %%
import json
import pathlib

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches
import pandas as pd

# %matplotlib inline

# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Plot images with labels

# %%
TEST_DEAL_DIRPATH = pathlib.Path('evaluation/test-deals/')

IMG1_FILEPATH = TEST_DEAL_DIRPATH / 'deal1-md-sq.jpg'
IMG2_FILEPATH = TEST_DEAL_DIRPATH / 'deal2-md-sq.jpg'
IMG3_FILEPATH = TEST_DEAL_DIRPATH / 'deal3-md-sq.jpg'
IMG4_FILEPATH = TEST_DEAL_DIRPATH / 'deal4-md-sq.jpg'
IMG5_FILEPATH = TEST_DEAL_DIRPATH / 'deal5-md-sq.jpg'
IMG6_FILEPATH = TEST_DEAL_DIRPATH / 'deal6-md-sq.jpg'
IMG7_FILEPATH = TEST_DEAL_DIRPATH / 'deal7-md-sq.jpg'
# 'labels' were based on predictions of md-sq-net1280
IMG1_LABELS_FILEPATH = TEST_DEAL_DIRPATH / 'deal1-md-sq-net1280-labels.json'
IMG2_LABELS_FILEPATH = TEST_DEAL_DIRPATH / 'deal2-md-sq-net1280-labels.json'
IMG3_LABELS_FILEPATH = TEST_DEAL_DIRPATH / 'deal3-md-sq-net1280-labels.json'
IMG4_LABELS_FILEPATH = TEST_DEAL_DIRPATH / 'deal4-md-sq-net1280-labels.json'
IMG5_LABELS_FILEPATH = TEST_DEAL_DIRPATH / 'deal5-md-sq-net1280-labels.json'
IMG6_LABELS_FILEPATH = TEST_DEAL_DIRPATH / 'deal6-md-sq-net1280-labels.json'
IMG7_LABELS_FILEPATH = TEST_DEAL_DIRPATH / 'deal7-md-sq-net1280-labels.json'

IMAGE_RESIZE_FACTOR = 1
TEXT_Y_OFFEST = -50

def load_img(path):
    image = cv2.imread(str(path))
    height, width = image.shape[:2]
    resized_image = cv2.resize(
        image, (IMAGE_RESIZE_FACTOR*width, IMAGE_RESIZE_FACTOR*height),
        interpolation=cv2.INTER_CUBIC)
    converted_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    return converted_image

def overlay_labels(img, label_info, selected_classes=None):
    if selected_classes:
        label_info = [d for d in label_info if d['name'] in selected_classes]

    __, ax = plt.subplots(figsize=(12, 12))

    marker_xs, marker_ys, text_ys = [], [], []
    bb_ws, bb_hs = [], []
    label_names = []
    label_confids = []
    img_w, img_h = img.shape[:2]

    for lbl in label_info:
        label_names.append(lbl['name'])
        label_confids.append(round(lbl['confidence'], 3))
        marker_xs.append(img_w*lbl["relative_coordinates"]["center_x"])
        marker_ys.append(img_h*lbl["relative_coordinates"]["center_y"])
        text_ys.append(img_h*lbl["relative_coordinates"]["center_y"] + TEXT_Y_OFFEST)
        bb_ws.append(img_w*lbl["relative_coordinates"]["width"])
        bb_hs.append(img_h*lbl["relative_coordinates"]["height"])

    # ax.scatter(marker_xs, marker_ys, c='g', s=20)
    for x, y, bb_w, bb_h in zip(marker_xs, marker_ys, bb_ws, bb_hs):
        rect = matplotlib.patches.Rectangle(
            (x - bb_w/2, y - bb_h/2),
            bb_w, bb_h,
            linewidth=1, edgecolor='g', facecolor='none', alpha=0.8,
        )
        ax.add_patch(rect)

    for name, confid, text_x, text_y in zip(label_names, label_confids, marker_xs, text_ys):
        ax.text(
            text_x, text_y, f"{name}, {confid}",
            ha="center", va="center",
            bbox=dict(boxstyle="round", ec=(0.5, 1, 0.5), fc=(0.8, 1, 0.8), alpha=0.3),
        )

    return ax


# %%
img = load_img(IMG3_FILEPATH)
with open(IMG3_LABELS_FILEPATH,'r') as fi:
    label_info = json.load(fi)

ax = overlay_labels(img, label_info, selected_classes={'2d', '9c', '2c'})
ax.imshow(img)

# %% [markdown]
# #### Some refs

# %%
# def imShow(path):
#     image = cv2.imread(str(path))
#     height, width = image.shape[:2]
#     resized_image = cv2.resize(image,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)

#     fig = plt.gcf()
#     fig.set_size_inches(24, 16)
#     plt.axis("off")
#     plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
#     plt.show()

# def display(self):
#     fig, ax = plt.subplots(1, figsize=(imgW/100,imgH/100), dpi=100)
#     ax.imshow(self.final)
#     for bb in self.listbba:
#         rect = patches.Rectangle(
#             (bb.x1, bb.y1),
#             bb.x2 - bb.x1,
#             bb.y2 - bb.y1,
#             linewidth=1, edgecolor='b', facecolor='none')
#         ax.add_patch(rect)

#     print(f"With cards: {self.class1}, {self.class2}, {self.class3}")

# def locate_detected_classes(res, min_conf=0.7):
#     __, ax = plt.subplots(figsize=(10,10))

#     for __, row in res.iterrows():
#         if row['confidence'] < min_conf:
#             continue
#         ax.annotate(row['name'], (row['center_x'], 1-row['center_y']))

#     # quadrant guide lines
#     ax.plot([0, 1], [0, 1], ls='--', c='grey', alpha=0.5)
#     ax.plot([0, 1], [1, 0], ls='--', c='grey', alpha=0.5)

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true
# ## label for golden test set

# %%
img1 = load_img(IMG1_FILEPATH)
with open(IMG1_LABELS_FILEPATH,'r') as fi:
    lbl1 = json.load(fi)
len(lbl1)

# %%
ax = overlay_labels(img1, [d for d in lbl1 if 'c' in d['name']], selected_classes=[])
ax.imshow(img1)

# %%

# %%
img2 = load_img(IMG2_FILEPATH)
with open(IMG2_LABELS_FILEPATH,'r') as fi:
    lbl2 = json.load(fi)
len(lbl2)

# %%
ax = overlay_labels(img2, [d for d in lbl2 if 'c' in d['name']], selected_classes=[])
ax.imshow(img2)

# %%

# %%
img3 = load_img(IMG3_FILEPATH)
with open(IMG3_LABELS_FILEPATH,'r') as fi:
    lbl3 = json.load(fi)
len(lbl3)

# %%
ax = overlay_labels(img3, [d for d in lbl3 if 'c' in d['name']], selected_classes=[])
ax.imshow(img3)

# %%

# %%

# %%

# %%
img4 = load_img(IMG4_FILEPATH)
with open(IMG4_LABELS_FILEPATH,'r') as fi:
    lbl4 = json.load(fi)
len(lbl4)

# %%
ax = overlay_labels(img4, [d for d in lbl4 if 'c' in d['name']], selected_classes=[])
ax.imshow(img4)

# %%

# %%
img5 = load_img(IMG5_FILEPATH)
with open(IMG5_LABELS_FILEPATH,'r') as fi:
    lbl = json.load(fi)
# with open(PREDICTION_FILEPATH,'r') as fi:
#     lbl = json.load(fi)[0]['objects']
len(lbl)

# %%
ax = overlay_labels(img5, [d for d in lbl if 'c' in d['name']], selected_classes=[])
# ax = overlay_labels(img5, lbl, selected_classes=['3h'])
ax.imshow(img5)

# %%

# %%
img6 = load_img(IMG6_FILEPATH)
with open(IMG6_LABELS_FILEPATH,'r') as fi:
    lbl = json.load(fi)
# with open(PREDICTION_FILEPATH,'r') as fi:
#     lbl = json.load(fi)[0]['objects']
len(lbl)


# %%
# ax = overlay_labels(img6, [d for d in lbl if 's' in d['name']], selected_classes=[''])
ax = overlay_labels(img6, lbl, selected_classes=['6s'])
ax.imshow(img6)

# %%

# %%
img7 = load_img(IMG7_FILEPATH)
with open(IMG7_LABELS_FILEPATH,'r') as fi:
    lbl = json.load(fi)
with open(PREDICTION_FILEPATH,'r') as fi:
    lbl = json.load(fi)[0]['objects']
len(lbl)


# %%
# ax = overlay_labels(img7, [d for d in lbl if 'c' in d['name']], selected_classes=[])
ax = overlay_labels(img7, lbl, selected_classes=['2s', '8c', '7d', '9d', '8d', '6h', '6s', 'Qd', 'Ks', '10c'])
ax.imshow(img7)

# %%

# %% [markdown] tags=[]
# ### class value counts

# %% tags=[]
pd.concat([
    pd.read_json(IMG1_LABELS_FILEPATH),
    pd.read_json(IMG2_LABELS_FILEPATH),
    pd.read_json(IMG3_LABELS_FILEPATH),
    pd.read_json(IMG4_LABELS_FILEPATH),
    pd.read_json(IMG5_LABELS_FILEPATH),
    pd.read_json(IMG6_LABELS_FILEPATH),
    pd.read_json(IMG7_LABELS_FILEPATH),
]).groupby(['class_id', 'name']).size()

# %%

# %% [markdown]
# ## Eval metrics
# - [x] ,basic counts, e.g. FP
# - [x] ,mAP50, mAP70
# - [x] ,mA4AP70: YL mean over A and 4 classes only
# - [ ] *mean Prec70 @ 0.981 recall
#     - min 52 positives, at least 51 TP to ensure full hand
# - [x] :gather into proper methods
# - [x] plot_fp_fn
#   - check `overlay_labels`
#   - should labels top, pred down
#   - https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.invert_yaxis.html
#   - probably update plot_bbox

# %%
import evaluation.core as eval_core
import evaluation.util as eval_util

GROUND_TRUTH_IMG_FILEPATH = IMG7_FILEPATH
GROUND_TRUTH_IMG_FILEPATH = IMG6_FILEPATH
GROUND_TRUTH_IMG_FILEPATH = IMG5_FILEPATH
# GROUND_TRUTH_IMG_FILEPATH = IMG4_FILEPATH
# GROUND_TRUTH_IMG_FILEPATH = IMG3_FILEPATH
# GROUND_TRUTH_IMG_FILEPATH = IMG2_FILEPATH

GROUND_TRUTH_FILEPATH = IMG7_LABELS_FILEPATH
GROUND_TRUTH_FILEPATH = IMG6_LABELS_FILEPATH
GROUND_TRUTH_FILEPATH = IMG5_LABELS_FILEPATH
# GROUND_TRUTH_FILEPATH = IMG4_LABELS_FILEPATH
# GROUND_TRUTH_FILEPATH = IMG3_LABELS_FILEPATH
# GROUND_TRUTH_FILEPATH = IMG2_LABELS_FILEPATH

PREDICTION_FILEPATH = '/home/yiqian/Downloads/rawpred-deal7-md-sq-net1280.json'
PREDICTION_FILEPATH = '/home/yiqian/Downloads/rawpred-deal6-md-sq-net1280.json'
PREDICTION_FILEPATH = '/home/yiqian/Downloads/rawpred-deal5-md-sq-net1280.json'
# PREDICTION_FILEPATH = pathlib.Path('evaluation') / 'rawpred-deal4-sm-sq-net1280.json'
# PREDICTION_FILEPATH = pathlib.Path('evaluation') / 'rawpred-deal4-sm-sq-net800.json'
# PREDICTION_FILEPATH = pathlib.Path('evaluation') / 'rawpred-deal3-md-sq-net1600.json'
# PREDICTION_FILEPATH = pathlib.Path('evaluation') / 'rawpred-deal2-md-sq-net800.json'

# %% [markdown]
# ### yolo4

# %%
y4_reader = eval_core.Yolo4PredReader()
evl = eval_core.Evaluator(GROUND_TRUTH_FILEPATH, PREDICTION_FILEPATH, pred_reader=y4_reader)

# %%
evl.report_precision_metrics()

# %%
evl.report_main_metrics()

# %%
eval_core.plot_misclf(list(evl.paired_objs(min_iou=0.5)), GROUND_TRUTH_IMG_FILEPATH, classes={}, thresh=0.5)

# %% tags=[]
# pd.DataFrame(evl.report_clf_metrics())

# %%
# import sklearn.metrics

# # y_true = [t[0] for t in gt_n_probas]
# # y_proba = [t[1] for t in gt_n_probas]
# y_true = [1, 1, 1, 1, 1, 1, 1, 1]
# y_proba = [0.931964, 0.947206, 0.943143, 0.909146, 0.943796, 0.946635, 0.944464, 0]
# sklearn.metrics.PrecisionRecallDisplay.from_predictions(y_true, y_proba)
# # **YL: PR curve does not capture FNs**

# %%

# %% [markdown]
# ### yolo5

# %%
Y5_PRED_FILEPATH = "/home/yiqian/repos/bidly/detector/yolov5/runs/detect/exp5/labels/deal3-md-sq.txt"  # default 640
Y5_PRED_FILEPATH = "/home/yiqian/repos/bidly/detector/yolov5/runs/detect/exp6/labels/deal3-md-sq.txt"  # M img 1184
Y5_PRED_FILEPATH = "/home/yiqian/repos/bidly/detector/yolov5/runs/detect/exp7/labels/deal3-md-sq.txt"  # + --conf-thres 0.3
# Y5_PRED_FILEPATH = "/home/yiqian/repos/bidly/detector/yolov5/runs/detect/exp8/labels/deal4-md-sq.txt"
# @ 165 epoch
Y5_PRED_FILEPATH = "/home/yiqian/repos/bidly/detector/yolov5/runs/detect/exp9/labels/deal4-md-sq.txt"
# Y5_PRED_FILEPATH = "/home/yiqian/repos/bidly/detector/yolov5/runs/detect/exp10/labels/deal3-md-sq.txt"
# Y5_PRED_FILEPATH = "/home/yiqian/repos/bidly/detector/yolov5/runs/detect/exp11/labels/deal7-md-sq.txt"
# Y5_PRED_FILEPATH = "/home/yiqian/repos/bidly/detector/yolov5/runs/detect/exp12/labels/deal6-md-sq.txt"
# @ 168
# Y5_PRED_FILEPATH = "/home/yiqian/repos/bidly/detector/yolov5/runs/detect/exp13/labels/deal6-md-sq.txt"  # - --conf-thres 0.3

y5_reader = eval_core.Yolo5PredReader()
evl = eval_core.Evaluator(
    GROUND_TRUTH_FILEPATH,
    Y5_PRED_FILEPATH,
    pred_reader=y5_reader
)

# %%
evl.report_precision_metrics()

# %%
evl.report_main_metrics()

# %%
eval_core.plot_misclf(evl.paired_objs(min_iou=0.5), GROUND_TRUTH_IMG_FILEPATH, thresh=0.5)

# %%

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ### yolo5 @ 75epoch

# %%
Y5_PRED_FILEPATH = "/home/yiqian/repos/bidly/detector/yolov5/runs/detect/exp7/labels/deal3-md-sq.txt"  # + --conf-thres 0.3 @~75 epoch
Y5_PRED_FILEPATH = "/home/yiqian/repos/bidly/detector/yolov5/runs/detect/exp8/labels/deal4-md-sq.txt"

y5_reader = eval_core.Yolo5PredReader()
evl = eval_core.Evaluator(
    GROUND_TRUTH_FILEPATH,
    Y5_PRED_FILEPATH,
    pred_reader=y5_reader
)

# %%
evl.report_precision_metrics()

# %%
eval_core.plot_misclf(evl.paired_objs(min_iou=0.5), GROUND_TRUTH_IMG_FILEPATH, thresh=0.5)

# %% [markdown] tags=[]
# ### yolo5 with ONNX+OpenCV

# %% tags=[]
DETECTION = [{'x': 0.31272367528966954,
  'y': 0.7226960465714738,
  'w': 0.03251164990502435,
  'h': 0.05107031641779719,
  'confidence': 0.9684845,
  'class_id': 46},
 {'x': 0.5753957903062975,
  'y': 0.16204801765648094,
  'w': 0.03224717281960152,
  'h': 0.047206698237238706,
  'confidence': 0.9679887,
  'class_id': 18},
 {'x': 0.67303044087178,
  'y': 0.705046473322688,
  'w': 0.046479482908506654,
  'h': 0.03671236618145092,
  'confidence': 0.9677404,
  'class_id': 22},
 {'x': 0.5698497875316724,
  'y': 0.49751173483358846,
  'w': 0.03230919709076752,
  'h': 0.04713666761243666,
  'confidence': 0.9676906,
  'class_id': 19},
 {'x': 0.5762367764034787,
  'y': 0.6641482791385136,
  'w': 0.03082653960666141,
  'h': 0.04281623943431957,
  'confidence': 0.9675868,
  'class_id': 5},
 {'x': 0.788282652158995,
  'y': 0.4700956086854677,
  'w': 0.029823010032241408,
  'h': 0.04789329541696084,
  'confidence': 0.96626353,
  'class_id': 47},
 {'x': 0.06880084888355152,
  'y': 0.48360489510201116,
  'w': 0.03449598518577782,
  'h': 0.04908966373752903,
  'confidence': 0.9661804,
  'class_id': 48},
 {'x': 0.601293151443069,
  'y': 0.90740544087178,
  'w': 0.04634436401160988,
  'h': 0.03593855935174066,
  'confidence': 0.96539414,
  'class_id': 22},
 {'x': 0.5393858729182064,
  'y': 0.15410077894056165,
  'w': 0.033113611711038125,
  'h': 0.04934918236088108,
  'confidence': 0.96526283,
  'class_id': 20},
 {'x': 0.33933879233695363,
  'y': 0.439424772520323,
  'w': 0.03748000956870414,
  'h': 0.04282370773521629,
  'confidence': 0.9650733,
  'class_id': 44},
 {'x': 0.3943926579243428,
  'y': 0.48282793405893687,
  'w': 0.04448568176578831,
  'h': 0.035193443298339844,
  'confidence': 0.96487045,
  'class_id': 15},
 {'x': 0.30806577527845225,
  'y': 0.23630629359064875,
  'w': 0.0289896855483184,
  'h': 0.04902203340788145,
  'confidence': 0.96470904,
  'class_id': 25},
 {'x': 0.9092165972735431,
  'y': 0.5158079508188609,
  'w': 0.04146661307360675,
  'h': 0.03778299769839725,
  'confidence': 0.9644358,
  'class_id': 28},
 {'x': 0.2448698765522725,
  'y': 0.42295894107303106,
  'w': 0.027342003745001717,
  'h': 0.048266449490109005,
  'confidence': 0.96389174,
  'class_id': 45},
 {'x': 0.5358541333997572,
  'y': 0.5297941774935335,
  'w': 0.041834405950597814,
  'h': 0.03804124368203653,
  'confidence': 0.96372163,
  'class_id': 30},
 {'x': 0.6317218574317726,
  'y': 0.6799880362845756,
  'w': 0.03723940978179107,
  'h': 0.03793353325611836,
  'confidence': 0.963702,
  'class_id': 3},
 {'x': 0.5039744506011138,
  'y': 0.15975633827415672,
  'w': 0.02607199630221805,
  'h': 0.047708914086625386,
  'confidence': 0.9636323,
  'class_id': 9},
 {'x': 0.6051559448242188,
  'y': 0.6628256101866026,
  'w': 0.03505389432649355,
  'h': 0.04010847452524546,
  'confidence': 0.9636201,
  'class_id': 2},
 {'x': 0.6263271537986962,
  'y': 0.39366036492425044,
  'w': 0.03581027726869326,
  'h': 0.048241270555032266,
  'confidence': 0.96345675,
  'class_id': 14},
 {'x': 0.3545039151165936,
  'y': 0.45784383206754115,
  'w': 0.03907783611400707,
  'h': 0.041139563998660526,
  'confidence': 0.963354,
  'class_id': 42},
 {'x': 0.03607560815037908,
  'y': 0.5238761901855469,
  'w': 0.030702484620584024,
  'h': 0.04164865210249617,
  'confidence': 0.96331704,
  'class_id': 36},
 {'x': 0.41056671658077754,
  'y': 0.5103899466024863,
  'w': 0.04662183168772105,
  'h': 0.03139875708399592,
  'confidence': 0.96317965,
  'class_id': 21},
 {'x': 0.614223170924831,
  'y': 0.17875246099523595,
  'w': 0.03576163987855654,
  'h': 0.0481907032631539,
  'confidence': 0.963127,
  'class_id': 14},
 {'x': 0.9549881187645165,
  'y': 0.540244231352935,
  'w': 0.04281243118079933,
  'h': 0.0345826986673716,
  'confidence': 0.96273285,
  'class_id': 27},
 {'x': 0.8446743939373945,
  'y': 0.48573097022804057,
  'w': 0.03091665860768911,
  'h': 0.04318879746101998,
  'confidence': 0.96261895,
  'class_id': 34},
 {'x': 0.8698842847669447,
  'y': 0.7410505655649545,
  'w': 0.043904114413905786,
  'h': 0.034740554319845664,
  'confidence': 0.962495,
  'class_id': 27},
 {'x': 0.2772679457793365,
  'y': 0.25562000274658203,
  'w': 0.031533573124859784,
  'h': 0.04574022099778459,
  'confidence': 0.9624101,
  'class_id': 37},
 {'x': 0.2717934428034602,
  'y': 0.7566218505034576,
  'w': 0.030405521392822266,
  'h': 0.044871894088951314,
  'confidence': 0.9620773,
  'class_id': 49},
 {'x': 0.7609225608207084,
  'y': 0.45443514231089,
  'w': 0.02688434478398916,
  'h': 0.05004422406892519,
  'confidence': 0.96168864,
  'class_id': 40},
 {'x': 0.32059089557544607,
  'y': 0.183815285966203,
  'w': 0.027184412286088273,
  'h': 0.04697311568904568,
  'confidence': 0.9615742,
  'class_id': 33},
 {'x': 0.8720320624274176,
  'y': 0.5006367966935441,
  'w': 0.0324066651834024,
  'h': 0.03973259796967377,
  'confidence': 0.9609655,
  'class_id': 10},
 {'x': 0.48557905248693517,
  'y': 0.6685963708001214,
  'w': 0.022043513285147177,
  'h': 0.04775708108334928,
  'confidence': 0.9608193,
  'class_id': 6},
 {'x': 0.667011003236513,
  'y': 0.4639371923498205,
  'w': 0.018309233961878595,
  'h': 0.04834036891524856,
  'confidence': 0.9605948,
  'class_id': 7},
 {'x': 0.3438503677780564,
  'y': 0.1826827977154706,
  'w': 0.02377704671911291,
  'h': 0.047449814306723105,
  'confidence': 0.9600733,
  'class_id': 12},
 {'x': 0.5360524976575697,
  'y': 0.6653468157794025,
  'w': 0.028128971924652923,
  'h': 0.04405297459782781,
  'confidence': 0.9599684,
  'class_id': 29},
 {'x': 0.8174967894683013,
  'y': 0.47732703750197947,
  'w': 0.029120690113789326,
  'h': 0.045747328448939965,
  'confidence': 0.9599438,
  'class_id': 39},
 {'x': 0.3363342027406435,
  'y': 0.7065057496766786,
  'w': 0.02641895816132829,
  'h': 0.04846214925920641,
  'confidence': 0.9596662,
  'class_id': 41},
 {'x': 0.2758083085756044,
  'y': 0.42649073214144323,
  'w': 0.029453506340851653,
  'h': 0.046481921866133404,
  'confidence': 0.958734,
  'class_id': 13},
 {'x': 0.2065016256796347,
  'y': 0.4202179779877534,
  'w': 0.022105105825372645,
  'h': 0.052313923835754395,
  'confidence': 0.9584518,
  'class_id': 17},
 {'x': 0.4668625496529244,
  'y': 0.14315673467275258,
  'w': 0.01982887854447236,
  'h': 0.05109180630864324,
  'confidence': 0.9580672,
  'class_id': 23},
 {'x': 0.4130789782549884,
  'y': 0.6595698176203547,
  'w': 0.017041836235974286,
  'h': 0.04984948441788957,
  'confidence': 0.9574559,
  'class_id': 51},
 {'x': 0.17456581785872177,
  'y': 0.41464921590444204,
  'w': 0.01825349878620457,
  'h': 0.0485039691667299,
  'confidence': 0.9569216,
  'class_id': 11},
 {'x': 0.15129371591516444,
  'y': 0.42055867169354416,
  'w': 0.018309214630642452,
  'h': 0.048501066259435704,
  'confidence': 0.95658046,
  'class_id': 38},
 {'x': 0.7324600219726562,
  'y': 0.44114396378800674,
  'w': 0.024891838834092423,
  'h': 0.05070653799417857,
  'confidence': 0.9554379,
  'class_id': 50},
 {'x': 0.09870698645308211,
  'y': 0.46536626042546453,
  'w': 0.022556366147221747,
  'h': 0.04393512493855244,
  'confidence': 0.95500726,
  'class_id': 26},
 {'x': 0.12024282764744115,
  'y': 0.4364073985331767,
  'w': 0.018879025369077117,
  'h': 0.04694948647473309,
  'confidence': 0.95468307,
  'class_id': 0},
 {'x': 0.6254761154587204,
  'y': 0.4935584197173247,
  'w': 0.017528906061842636,
  'h': 0.047887389724319045,
  'confidence': 0.95418495,
  'class_id': 8},
 {'x': 0.438633377487595,
  'y': 0.6548253652211783,
  'w': 0.0218295922150483,
  'h': 0.045483518291164086,
  'confidence': 0.9529449,
  'class_id': 32},
 {'x': 0.7087925060375316,
  'y': 0.4504547634640256,
  'w': 0.02193691279437091,
  'h': 0.0478096427144231,
  'confidence': 0.9515933,
  'class_id': 4},
 {'x': 0.36967883238921295,
  'y': 0.16867119557148702,
  'w': 0.020848623804143956,
  'h': 0.04630359443458351,
  'confidence': 0.95125735,
  'class_id': 31},
 {'x': 0.38563645852578654,
  'y': 0.660771859658731,
  'w': 0.017619381079802644,
  'h': 0.04607136507292051,
  'confidence': 0.9504919,
  'class_id': 35},
 {'x': 0.3976744316719674,
  'y': 0.16767383266139677,
  'w': 0.01671921884691393,
  'h': 0.04920741996249637,
  'confidence': 0.9495728,
  'class_id': 24},
 {'x': 0.4188962369351774,
  'y': 0.15700733339464343,
  'w': 0.018697603328807932,
  'h': 0.05145237574706207,
  'confidence': 0.94619304,
  'class_id': 16},
 {'x': 0.4444975466341586,
  'y': 0.15319433727779905,
  'w': 0.01951982846131196,
  'h': 0.04711716239516799,
  'confidence': 0.94182295,
  'class_id': 1},
 {'x': 0.3651229239798881,
  'y': 0.6751047082849451,
  'w': 0.020919084548950195,
  'h': 0.05268162650031012,
  'confidence': 0.90438396,
  'class_id': 43}]

# 1056x1056
# DETECTION = [{'x': 0.27726196520256274, 'y': 0.25551423159512604, 'w': 0.03177948792775472, 'h': 0.045781247543566155, 'confidence': 0.972484, 'class_id': 37}, {'x': 0.27183483586166846, 'y': 0.7566002932461825, 'w': 0.03080903400074352, 'h': 0.044470353560014206, 'confidence': 0.97196704, 'class_id': 49}, {'x': 0.57634821805087, 'y': 0.6641515096028646, 'w': 0.030871546629703407, 'h': 0.043007702538461395, 'confidence': 0.9717343, 'class_id': 5}, {'x': 0.844770604913885, 'y': 0.48583377491344104, 'w': 0.031211029399525036, 'h': 0.04326102950356223, 'confidence': 0.96978295, 'class_id': 34}, {'x': 0.0983554811188669, 'y': 0.46533931385387073, 'w': 0.022328528490933506, 'h': 0.04376782792987245, 'confidence': 0.96914846, 'class_id': 26}, {'x': 0.6141437761711351, 'y': 0.1789244305003773, 'w': 0.03562674377903794, 'h': 0.047985571803468643, 'confidence': 0.9689386, 'class_id': 14}, {'x': 0.5696879300204191, 'y': 0.4973939837831439, 'w': 0.0320898691813151, 'h': 0.04726330800489946, 'confidence': 0.96840984, 'class_id': 19}, {'x': 0.631914369987719, 'y': 0.6797651233095111, 'w': 0.037343534556302155, 'h': 0.03891031669847893, 'confidence': 0.96775734, 'class_id': 3}, {'x': 0.6729944402521307, 'y': 0.7047968777743253, 'w': 0.046381198998653526, 'h': 0.036806525606097595, 'confidence': 0.9674876, 'class_id': 22}, {'x': 0.5361009655576764, 'y': 0.6652724526145242, 'w': 0.028346332636746494, 'h': 0.043792139400135384, 'confidence': 0.9671829, 'class_id': 29}, {'x': 0.03603709466529615, 'y': 0.5239057829885772, 'w': 0.03059993368206602, 'h': 0.04164206259178393, 'confidence': 0.96640736, 'class_id': 36}, {'x': 0.320185256726814, 'y': 0.18370689045299182, 'w': 0.026056152401548443, 'h': 0.04713877403374874, 'confidence': 0.96605796, 'class_id': 33}, {'x': 0.24482383150042908, 'y': 0.4228178082090436, 'w': 0.027637026526711204, 'h': 0.04869877930843469, 'confidence': 0.96490943, 'class_id': 45}, {'x': 0.3942962415290602, 'y': 0.4827442169189453, 'w': 0.044488256627863106, 'h': 0.035580049861561165, 'confidence': 0.964766, 'class_id': 15}, {'x': 0.575231898914684, 'y': 0.16192024404352362, 'w': 0.03267050511909254, 'h': 0.04807883681672992, 'confidence': 0.9646633, 'class_id': 18}, {'x': 0.601163979732629, 'y': 0.9071901494806462, 'w': 0.04661284432266698, 'h': 0.03643466487075343, 'confidence': 0.9637179, 'class_id': 22}, {'x': 0.9550247770367246, 'y': 0.5401159344297467, 'w': 0.042909246502500595, 'h': 0.03455766403313839, 'confidence': 0.9636115, 'class_id': 27}, {'x': 0.31266695080381446, 'y': 0.7224481178052498, 'w': 0.03200191440004291, 'h': 0.05127411177664092, 'confidence': 0.9632104, 'class_id': 46}, {'x': 0.5393467527447325, 'y': 0.1540698427142519, 'w': 0.03326215888514663, 'h': 0.04929747003497499, 'confidence': 0.9630709, 'class_id': 20}, {'x': 0.4853705203894413, 'y': 0.668685566295277, 'w': 0.022347757310578316, 'h': 0.047699046857429275, 'confidence': 0.962899, 'class_id': 6}, {'x': 0.06866293242483429, 'y': 0.4836730668039033, 'w': 0.03390168782436487, 'h': 0.04868574937184652, 'confidence': 0.96261156, 'class_id': 48}, {'x': 0.2757948384140477, 'y': 0.4265485532356031, 'w': 0.02931530186624238, 'h': 0.04676782723629114, 'confidence': 0.9626048, 'class_id': 13}, {'x': 0.41056202397201996, 'y': 0.5099289056026575, 'w': 0.04704491658644243, 'h': 0.031898397387880265, 'confidence': 0.96255744, 'class_id': 21}, {'x': 0.8717675064549302, 'y': 0.5005110538367069, 'w': 0.032386187351111206, 'h': 0.03986819585164388, 'confidence': 0.9622914, 'class_id': 10}, {'x': 0.5038945169159861, 'y': 0.15986568277532404, 'w': 0.025822103023529053, 'h': 0.048121470393556534, 'confidence': 0.962124, 'class_id': 9}, {'x': 0.6049421483820134, 'y': 0.6628909255519058, 'w': 0.03508063157399496, 'h': 0.040018164750301476, 'confidence': 0.96207964, 'class_id': 2}, {'x': 0.43854375319047406, 'y': 0.6547045274214311, 'w': 0.021985825264092648, 'h': 0.0463961326714718, 'confidence': 0.96183175, 'class_id': 32}, {'x': 0.6261954452052261, 'y': 0.39350495193943835, 'w': 0.03596693819219416, 'h': 0.048231804009639855, 'confidence': 0.9614858, 'class_id': 14}, {'x': 0.7885046872225675, 'y': 0.47004754615552496, 'w': 0.030185811447374748, 'h': 0.04803993846430923, 'confidence': 0.9604632, 'class_id': 47}, {'x': 0.8697646169951467, 'y': 0.740956450953628, 'w': 0.04361688729488489, 'h': 0.03499881787733598, 'confidence': 0.96040636, 'class_id': 27}, {'x': 0.33930292996493255, 'y': 0.43931568030155066, 'w': 0.03757820346138694, 'h': 0.0431373625090628, 'confidence': 0.9601978, 'class_id': 44}, {'x': 0.3083056825580019, 'y': 0.23621908823649088, 'w': 0.02897534587166526, 'h': 0.04937053810466419, 'confidence': 0.96007246, 'class_id': 25}, {'x': 0.760703115752249, 'y': 0.45438613313617127, 'w': 0.026547258550470524, 'h': 0.05002891656124231, 'confidence': 0.95998424, 'class_id': 40}, {'x': 0.7323048331520774, 'y': 0.44152323404947913, 'w': 0.02441291736833977, 'h': 0.04965478001218854, 'confidence': 0.9598858, 'class_id': 50}, {'x': 0.5358130715110085, 'y': 0.5298621437766335, 'w': 0.04205276388110536, 'h': 0.038551988023700134, 'confidence': 0.95987976, 'class_id': 30}, {'x': 0.3548268693866152, 'y': 0.4579601865826231, 'w': 0.03961452932068796, 'h': 0.04193301273114754, 'confidence': 0.95979375, 'class_id': 42}, {'x': 0.3435902451023911, 'y': 0.1824764482902758, 'w': 0.02369174270918875, 'h': 0.04817517959710323, 'confidence': 0.9585533, 'class_id': 12}, {'x': 0.17474233742916223, 'y': 0.4149293899536133, 'w': 0.017867137085307728, 'h': 0.04902697693217884, 'confidence': 0.9582925, 'class_id': 11}, {'x': 0.20644167697790897, 'y': 0.42034556648947974, 'w': 0.021880736856749565, 'h': 0.05201305042613636, 'confidence': 0.95705116, 'class_id': 17}, {'x': 0.3694458585796934, 'y': 0.16881380659161194, 'w': 0.020795511476921314, 'h': 0.04631857438520952, 'confidence': 0.95673233, 'class_id': 31}, {'x': 0.33643682075269293, 'y': 0.7063106768059009, 'w': 0.026996630610841694, 'h': 0.04875523032564105, 'confidence': 0.9566743, 'class_id': 41}, {'x': 0.7088572184244791, 'y': 0.4500789931326201, 'w': 0.022036664413683342, 'h': 0.048498410167116104, 'confidence': 0.9565275, 'class_id': 4}, {'x': 0.15156907746286102, 'y': 0.42062265222722833, 'w': 0.018573771823536266, 'h': 0.04789094491438432, 'confidence': 0.955422, 'class_id': 38}, {'x': 0.9090722401936849, 'y': 0.5158108797940341, 'w': 0.041262662771976356, 'h': 0.037700038967710556, 'confidence': 0.9545315, 'class_id': 28}, {'x': 0.46699691541267163, 'y': 0.14333242358583392, 'w': 0.019603461930246063, 'h': 0.05126928921901819, 'confidence': 0.9530833, 'class_id': 23}, {'x': 0.8179654786081025, 'y': 0.4773896824229847, 'w': 0.029663230433608547, 'h': 0.04518620534376665, 'confidence': 0.9526329, 'class_id': 39}, {'x': 0.6254908243815104, 'y': 0.4936180692730528, 'w': 0.017521052649526886, 'h': 0.04797390374270352, 'confidence': 0.9525635, 'class_id': 8}, {'x': 0.1202629551743016, 'y': 0.43619005607836175, 'w': 0.018830790664210464, 'h': 0.04647476745374275, 'confidence': 0.9519648, 'class_id': 0}, {'x': 0.6671466249408143, 'y': 0.46378531600489764, 'w': 0.0180850372169957, 'h': 0.048882816777084816, 'confidence': 0.95176864, 'class_id': 7}, {'x': 0.4447817657933091, 'y': 0.15271761923125296, 'w': 0.01993853576255567, 'h': 0.048744974714336975, 'confidence': 0.9517294, 'class_id': 1}, {'x': 0.413007707306833, 'y': 0.6595022028142756, 'w': 0.017384944540081604, 'h': 0.05023773872491085, 'confidence': 0.94615537, 'class_id': 51}, {'x': 0.3854594375147964, 'y': 0.660541245431611, 'w': 0.017674854307463676, 'h': 0.04547718799475468, 'confidence': 0.9460745, 'class_id': 35}, {'x': 0.41903414870753436, 'y': 0.15785442699085583, 'w': 0.018609789284792812, 'h': 0.05105712919524222, 'confidence': 0.9447884, 'class_id': 16}, {'x': 0.3647426258433949, 'y': 0.676232944835316, 'w': 0.020355083725669167, 'h': 0.05200489362080892, 'confidence': 0.9422059, 'class_id': 43}, {'x': 0.3976755720196348, 'y': 0.16783821221553918, 'w': 0.016457097096876663, 'h': 0.0505309213291515, 'confidence': 0.9344879, 'class_id': 24}]

# %%
len(DETECTION)


# %%
class OnnxPredReader(eval_core.ILabelReader):
    def read(self, src: list):
        pred_info = pd.DataFrame(src)

        with open(eval_core.FILE_PATH.parent.parent/'yolo-cfg'/'obj.names', 'r') as f:
            card_names = [l.strip() for l in f.readlines() if l]

        objs = (
            pred_info
                .assign(card_name=pred_info.class_id.map(card_names.__getitem__))
                .assign(yolo_obj=lambda df: df.apply(self._make_yolo_obj, axis=1))
                .yolo_obj.tolist()
        )
        return objs
    
    @staticmethod
    def _make_yolo_obj(row: pd.Series):
        return eval_core.YoloObject(
            name=row.card_name,
            x=row.x,
            y=row.y,
            w=row.w,
            h=row.h,
            confid=row.confidence,
        )


# %%
onnx_reader = OnnxPredReader()
evl = eval_core.Evaluator(
    GROUND_TRUTH_FILEPATH,
    DETECTION,
    pred_reader=onnx_reader
)

# %%
evl.report_main_metrics()

# %%
eval_core.plot_misclf(evl.paired_objs(min_iou=0.5), GROUND_TRUTH_IMG_FILEPATH, thresh=0)
# all D and H class id was wrong? bad color conversion: COLOR_BGR2RGB

# %% [markdown]
# ## YOLO5 CLI commands

# %%
python train.py --img 1184 --epochs 300 --data ../yolo-cfg/yolov5-dataset.yaml --weights yolov5s.pt --cache disk --batch-size -1 --resume

# %%
python detect.py --weights runs/train/exp/weights/best.pt --img 1184 --source ../evaluation/test-deals/deal3-md-sq.jpg --save-txt --save-conf --conf-thres 0.3

# %%
python train.py --img 1184 --epochs 100 --data ../yolo-cfg/yolov5-dataset.yaml --weights runs/train/r2/weights/best.pt --cache disk --batch-size -1 --name r3 --resume

# %%
python export.py --img 1056 --weights runs/train/r3/weights/best.pt --include onnx --opset=12

# %% [markdown]
# ## YOLO5 training results in progress

# %%
import seaborn as sns

# %%
eval_core.report_baseline()

# %%

# %% [markdown]
# ### R1

# %%
EXP_NAME = 'exp'
weight_path = f'yolov5/runs/train/{EXP_NAME}/weights/best.pt'
result_path = f'yolov5/runs/train/{EXP_NAME}/results.csv'

# %%
gold_res = eval_core.report_gold_test(weight_path)
gold_res

# %% tags=[]
eval_core.plot_gold_misclf(weight_path)

# %% [markdown]
# ### R2

# %%
EXP_NAME = 'r2'
weight_path = f'yolov5/runs/train/{EXP_NAME}/weights/best.pt'
result_path = f'yolov5/runs/train/{EXP_NAME}/results.csv'

# %%
gold_res = eval_core.report_gold_test(weight_path)
gold_res

# %% tags=[]
eval_core.plot_gold_misclf(weight_path, thresh=0.8)

# %% [markdown]
# ### R3

# %%
EXP_NAME = 'r3'
weight_path = f'yolov5/runs/train/{EXP_NAME}/weights/best.pt'
result_path = f'yolov5/runs/train/{EXP_NAME}/results.csv'

# %%
gold_res = eval_core.report_gold_test(weight_path)
gold_res

# %% tags=[]
eval_core.plot_gold_misclf(weight_path, thresh=0.5)

# %%
# model = torch.hub.load('yolov5', 'custom', path=WEIGHT_PATH, source='local')

# _res = model(GROUND_TRUTH_IMG_FILEPATH, size=1184)

# %%
y5_pd_reader = eval_core.Yolo5PredPandasReader()
evl = eval_core.Evaluator(
    GROUND_TRUTH_FILEPATH,
    _res.pandas().xywhn[0],
    pred_reader=y5_pd_reader
)

# %%
evl.report_main_metrics()

# %%

# %% [markdown]
# ### loss etc.

# %%
res = pd.read_csv(result_path)
res.shape

# %%
res.tail()

# %%
(
    res.rename(columns=str.strip)
        .melt(id_vars=["epoch"],
              value_vars=['train/box_loss', 'train/obj_loss', 'train/cls_loss', 'val/box_loss', 'val/obj_loss', 'val/cls_loss'])
        .assign(loss=lambda df: df.variable.str.slice(-8),
                kind=lambda df: df.variable.str.slice(None, 3))
        # .query('epoch >= 65')
        .pipe((sns.relplot, 'data'), kind='line',
              x='epoch', y='value', facet_kws=dict(sharey=False),
              hue='kind', col='loss')
)

# %%
res.loc[:, 'metrics/mAP_0.5:0.95'].plot(kind='line')

# %%

# %%
# %cd yolov5

# %%
# import utils.plots

# utils.plots.plot_results('runs/train/exp/results.csv')

# %%
# %cd ..

# %%

# %%

# %% [markdown]
# ## misc

# %%
import random
one_pair = random.choice(list(evl.paired_objs()))[:2]
eval_core.plot_paired_boxes(*one_pair)
