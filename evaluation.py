'''
Evaluate:
	call python evaluation.py [LABEL_FOLDER] [PRED_BOX_FOLDER]

LABEL_FOLDER is the folder of true labels for the input images. The name of images should match the name of label files (for example 1.jpg matches 1.txt).

PRED_BOX_FOLDER is the ‘boxes’ folder generated after calling run.py or clicking ‘Inference’ button in the GUI. 
'''

import sys
label_folder = sys.argv[1]
pred_folder = sys.argv[2]

from glob import glob
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score, f1_score, accuracy_score, mean_squared_error, confusion_matrix

labels = list(glob(os.path.join(label_folder, "*.txt")))
counter = 0
Y = []
data = []
for label in labels:
    iid = int(os.path.basename(label).split(".")[0])
    bboxes = open(os.path.join(label_folder, "{}.txt".format(iid)), "r").read().splitlines()[1:]
    bboxes = [([int(f) for f in bbox.split()[:4]], bbox.split()[4].strip().lower()=="interested") for bbox in bboxes]
    for bbox in bboxes:
        (x_min,y_min,x_max,y_max) = bbox[0]
        label = bbox[1]
        Y.append(label)
        data.append({"image":iid,"x_min":x_min,"y_min":y_min,"x_max":x_max,"y_max":y_max,"label":label})
        counter += 1
Y = np.array(Y, dtype=np.bool)
label = pd.DataFrame(data).sort_values("image").reset_index().drop("index", axis=1)

dfs = []
for pred in list(glob(os.path.join(pred_folder, "*.txt"))):
    iid = int(os.path.basename(pred).split(".")[0])
    df = pd.read_csv(pred)
    df['label']=df['cls'].str.strip()=="interested"
    df = df.rename(columns={" cood1":"x_min"," cood2":"y_min"," cood3":"x_max"," cood4":"y_max"," conf":"conf"})
    #df = df.drop(["cls"], axis=1)
    df['image'] = iid
    df = df[df.conf>=0.5]
    dfs.append(df)
pred = pd.concat(dfs).sort_values("image").reset_index().drop("index", axis=1)

def take_max(arr, v):
    return np.where(arr>v,arr,v)
def take_min(arr, v):
    return np.where(arr<v,arr,v)

label_work = label.copy()
label_work['matched'] = False
label_work['matched_to'] = None

pred_work = pred.copy()
pred_work['matched_x_min'] = 0
pred_work['matched_x_max'] = 0
pred_work['matched_y_min'] = 0
pred_work['matched_y_max'] = 0
pred_work['matched_to'] = 0

actual = []
predicted = []
predicted_conf = []

dual_match = 0
acc = 0
tot = 0

li = 0

mis_match_lis = []

mis_matched_labels = []
for i,row in pred.iterrows():
    cands = label_work[label_work.image==row['image']]
    
    # calculate overlap size
    xmin=take_max(cands['x_min'], row['x_min'])
    xmax=take_min(cands['x_max'], row['x_max'])
    ymin=take_max(cands['y_min'], row['y_min'])
    ymax=take_min(cands['y_max'], row['y_max'])
    overlap = (ymax-ymin)*(xmax-xmin)
    overlap[(xmax<xmin)]=0
    
    # remove non-overlapping cands
    cands = cands[overlap>0]
    if len(cands) == 0:
        mis_matched_labels.append(row.label)
        if row.label:
            li = i
            mis_match_lis.append(li)
        continue
    
    # get most overlapped cand
    cands = cands.copy()
    cands['overlap'] = overlap[overlap>0]
    cands.sort_values("overlap",ascending=False)
    to_match = cands.iloc[0]
    
    if to_match.matched:
        dual_match += 1
        label_work.loc[to_match.name, 'matched_to'].append(row.name)
    else:
        label_work.loc[to_match.name, 'matched_to'] = [row.name]
        label_work.loc[to_match.name, 'matched']=True
    
    # for debugging
    pred_work.loc[i, 'matched_to'] = to_match.name
    pred_work.loc[i, 'matched_x_min'] = to_match.x_min
    pred_work.loc[i, 'matched_y_min'] = to_match.y_min
    pred_work.loc[i, 'matched_x_max'] = to_match.x_max
    pred_work.loc[i, 'matched_y_max'] = to_match.y_max
    
    actual.append(to_match.label)
    predicted.append(row.label)
    if row.label:
        predicted_conf.append(row.conf)
    else:
        predicted_conf.append(1-row.conf)
actual = np.array(actual)
predicted = np.array(predicted)
predicted_conf = np.array(predicted_conf)

labeled_but_not_pred = label_work[(label_work['matched_to'] == None)]
pred_but_not_label = mis_matched_labels

detection_rm = pd.DataFrame([
    {
        'Labeled':len(actual),
        'Not Labeled': "{} ({} interested)".format(len(mis_matched_labels), len(mis_match_lis))
    },
    
    {
        'Labeled':len(labeled_but_not_pred),
        'Not Labeled':None
    }
], index=['Detected', 'Not Detected'])

print("Detection result matrix:")
print(detection_rm)

print("".join(['=']*100))

print("Classification confusion matrix:")

cls_cm = pd.DataFrame(confusion_matrix(actual, predicted),
                      columns=['Predicted Non-Intrested', "Predicted Intrested"],
                      index=['Labeled Non-Intrested', "Labeled Intrested"])
print(cls_cm)

print("".join(['-']*100))

auc,f1,acc = roc_auc_score(actual, predicted_conf), f1_score(actual, predicted), accuracy_score(actual, predicted)
print("Classification scores:")
print("AUC:\t\t{}".format(auc))
print("F1:\t\t{}".format(f1))
print("Accuracy:\t{}".format(acc))