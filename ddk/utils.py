import json
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt

def gen_df(jsons):
    dict_list = []
    for idx,json_file in enumerate(jsons):
        tmp_dict = {}
        with open(json_file) as f :
            tmp_json = json.load(f)
            for key in tmp_json.keys():            
                tmp_dict.update(tmp_json[key])  
        tmp_dict['filename']=str((json_file.parents[1]/json_file.name).with_suffix(".wav"))
        dict_list.append(tmp_dict)    
    return pd.DataFrame(dict_list)


def drop_df_list(df_list):

    drop_list = ['device','deviceUsedAge','comunicationTool','comunicationTool','rehabilitation','rehabilitation',
                'numberOfRecordings','recordingSystem','hearingLoss','script','category','recordingQuality',
                'sentenceType','recordingDevice','recordingDate','hospital','diagnostics']
    df_list_ext = *map(lambda x : x.drop(drop_list,axis=1),df_list),
    
    return df_list_ext

def cm_analysis(y_true, y_pred, filename, labels, classes, ymap=None, figsize=(17,17)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      classes:   aliases for the labels. String array to be shown in the cm plot.
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    sns.set(font_scale=2.8)

    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.2f%%\n%d/%d' % (p, c, s)
            #elif c == 0:
            #    annot[i, j] = ''
            else:
                annot[i, j] = '%.2f%%\n%d' % (p, c)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm = cm * 100
    cm.index.name = 'True Label'
    cm.columns.name = 'Predicted Label'
    fig, ax = plt.subplots(figsize=figsize)
    plt.yticks(va='center')

    sns.heatmap(cm, annot=annot, fmt='', ax=ax, xticklabels=classes, cbar=True, cbar_kws={'format':PercentFormatter()}, yticklabels=classes, cmap="Blues")
    #plt.savefig(filename,  bbox_inches='tight')
    
           
    