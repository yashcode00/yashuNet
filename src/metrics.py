#####################################################
## @author : Yash Sharma
## @email: yashuvats.42@gmail.com
#####################################################
import torch

"""Function for computation of the metrics"""

def jaccard_index(predicted, true):
    intersection = torch.logical_and(predicted, true)
    union = torch.logical_or(predicted, true)
    return torch.sum(intersection).float() / torch.sum(union).float()
 
def dice_coefficient(predicted, true):
    intersection = torch.logical_and(predicted, true)
    return 2. * torch.sum(intersection).float() / (torch.sum(predicted).float() + torch.sum(true).float())
 
def calculate_metrics(predicted, true):
    """Calculate precision and recall for binary images."""
    TP = torch.sum(torch.logical_and(predicted == 1, true == 1)).float()
    FP = torch.sum(torch.logical_and(predicted == 1, true == 0)).float()
    FN = torch.sum(torch.logical_and(predicted == 0, true == 1)).float()
    TN = torch.sum(torch.logical_and(predicted == 0, true == 0)).float()
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
 
    return precision, recall, accuracy


def compute_metric(loader, model):
    model.eval()
    jaccard_mean = 0
    dice_mean = 0
    precision_mean = 0
    recall_mean = 0
    accuracy_mean = 0
    
    with torch.no_grad():
        for x, y in loader:
            x = x
            y = y.unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            jaccard = jaccard_index(preds, y)
            dice = dice_coefficient(preds, y)
            precision, recall, accuracy = calculate_metrics(preds, y)
 
            jaccard_mean += jaccard
            dice_mean += dice
            precision_mean += precision
            recall_mean += recall
            accuracy_mean += accuracy

    return {
        "Jaccard": jaccard_mean/len(loader),
        "Dice": dice_mean/len(loader),
        "Precision": precision_mean/len(loader),
        "Recall": recall_mean/len(loader),
        "Accuracy": accuracy_mean/len(loader)
    }