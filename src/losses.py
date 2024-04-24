import torch.nn as nn
from torch.functional import F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice
    
class JaccardLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        jaccard = (intersection + self.smooth) / (union + self.smooth)
        return 1 - jaccard
    
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()
    
class ComboLoss(nn.Module):
    def __init__(self, factors=None, losses=None):
        super(ComboLoss, self).__init__()
        
        if factors is None:
            factors = [1.0]
        if losses is None:
            losses = [nn.BCEWithLogitsLoss()]

        assert len(factors) == len(losses), "Factors and losses must have the same number of elements."
        self.factors = factors
        self.losses = losses

    def forward(self, inputs, targets):
        total_loss = 0,0
        total_factors = sum(self.factors)
        for factor, loss_function in zip(self.factors, self.losses):
            loss = loss_function(inputs, targets)
            weighted_loss = factor * loss
            total_loss += weighted_loss
        return total_loss/total_factors
