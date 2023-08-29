from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

def get_score(y_true,y_pred):
    """Get Score for Each Column"""
    
    accuracy = accuracy_score(y_true,y_pred)
    precision =round( precision_score(y_true,y_pred,average='micro'))
    recall = recall_score(y_true,y_pred,average='micro')
    f1 = f1_score(y_true,y_pred,average='micro')
    
    return {'Precision':precision, 'Recall':recall,'F1-score':f1,'Accuracy':accuracy}