##################################################################################################
## Important Intializations
##################################################################################################

class ModelConfig:
    def __init__(self) -> None:
        self.paths = {}
        self.paths["labelled"] = {}
        self.paths["unlabelled"] = {}

        # Paths for labelled data
        self.paths["labelled"]["train_images"] = "/path/to/labelled/train/images"
        self.paths["labelled"]["train_masks"] = "/path/to/labelled/train/masks"
        self.paths["labelled"]["val_images"] = "/path/to/labelled/val/images"
        self.paths["labelled"]["val_masks"] = "/path/to/labelled/val/masks"

        # Paths for unlabelled data
        self.paths["unlabelled"]["train"] = "/path/to/unlabelled/train/images"
        self.paths["unlabelled"]["val"] = "/path/to/unlabelled/val/images"

        self.n_epochs_supervised = 1
        self.n_epochs_unsupervised = 1 

        self.stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ### remove this later or change this
        self.IMAGE_HEIGHT = 256
        self.IMAGE_WIDTH  = 256

        self.batch_size_selfSupervised = 32
        self.batch_size = 32

        self.lr1= 1e-4 ## for self-supervised training
        self.lr2 = 1e-6 ## for complete training full network for segmentation
        