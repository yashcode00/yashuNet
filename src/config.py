##################################################################################################
## Important Intializations
##################################################################################################

class ModelConfig:
    def __init__(self) -> None:
        self.paths = {}
        self.paths["labelled"] = {}
        self.paths["unlabelled"] = {}

        # Paths for labelled data
        self.paths["labelled"]["train_images"] = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/labelled/train_image"
        self.paths["labelled"]["train_masks"] = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/labelled/train_mask"
        self.paths["labelled"]["val_images"] = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/labelled/val_image"
        self.paths["labelled"]["val_masks"] = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/labelled/val_mask"

        # Paths for unlabelled data
        self.paths["unlabelled"]["train"] = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/unlabelled/train_image"
        self.paths["unlabelled"]["val"] = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/unlabelled/val_image"

        self.n_epochs_supervised = 100
        self.n_epochs_unsupervised = 500 

        self.stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ### remove this later or change this

        self.IMAGE_HEIGHT = 256
        self.IMAGE_WIDTH  = 256

        self.batch_size_selfSupervised = 8
        self.batch_size = 128

        self.lr = 1e-3
        self.lr1= 1e-4 ## for self-supervised training
        self.lr2 = 1e-6 ## for complete training full network for segmentation
        self.preload = None
        self.preload2= "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/models/colorization_HistologySegmentation_20240423_162019/pthFiles/model_epoch_14"
        # Define the layers you want to exclude
        self.layers_to_exclude = ['inc.double_conv.0.weight', 'outc.conv.weight', 'outc.conv.bias']


        ## for the final train after train
        self.encoder_lr = 1e-6
        self.decoder_lr = 5e-4
        