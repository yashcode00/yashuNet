import torch
import torch.nn.functional as F

class HybridLoss:
    def __init__(self, weight_bce=1, weight_dice=0.01, weight_jaccard=0.01):
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.weight_jaccard = weight_jaccard
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def dice_coeff(self, y_true, y_pred):
        intersection = torch.sum(y_true * y_pred)
        union = torch.sum(y_true) + torch.sum(y_pred)
        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        return dice

    def jaccard_similarity(self, y_true, y_pred):
        intersection = torch.sum(y_true * y_pred)
        union = torch.sum(y_true) + torch.sum(y_pred) - intersection
        jaccard = (intersection + 1e-5) / (union + 1e-5)
        return jaccard

    def __call__(self, y_true, y_pred):
        bce_loss = self.bce_loss(y_pred, y_true)
        dice_loss = 1 - self.dice_coeff(y_true, y_pred)
        jaccard_loss = 1 - self.jaccard_similarity(y_true, y_pred)

        hybrid_loss = (
            self.weight_bce * bce_loss +
            self.weight_dice * dice_loss +
            self.weight_jaccard * jaccard_loss
        ) 

        return hybrid_loss
