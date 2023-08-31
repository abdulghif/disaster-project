import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, make_scorer

def get_score(y_true,y_pred):
    """Get Score for Each Column"""
    
    accuracy = accuracy_score(y_true,y_pred)
    precision =round( precision_score(y_true,y_pred,average='micro'))
    recall = recall_score(y_true,y_pred,average='micro')
    f1 = f1_score(y_true,y_pred,average='micro')
    
    return {'Precision':precision, 'Recall':recall,'F1-score':f1,'Accuracy':accuracy}

def custom_multiclf_score(y_true,y_pred):
    list_results = []
    for i,column in enumerate(y_true.columns):
        result = f1_score(y_true.loc[:,column].values,y_pred[:,i],average='micro')
        list_results.append(result)
    list_results = np.array(list_results)
    return list_results.mean()

score = make_scorer(custom_multiclf_score)