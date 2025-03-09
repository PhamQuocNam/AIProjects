from sklearn.metrics import f1_score


def compute_metric(preds,labels):
    true_predictions= [
        [p for (p,l) in zip(pred,label) if l !=-100]
        for pred,label in zip(preds,labels)
    ]
    
    true_labels= [
        [l for (p,l) in zip(pred,label) if l !=-100]
        for pred,label in zip(preds,labels)
    ]
    
    true_predictions_flat = [p for sublist in true_predictions for p in sublist]
    true_labels_flat = [l for sublist in true_labels for l in sublist]
    results = f1_score(true_labels_flat, true_predictions_flat, average="macro")  
    return results


