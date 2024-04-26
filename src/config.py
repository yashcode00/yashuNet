##################################################################################################
## Important Intializations
##################################################################################################


class  ModelConfig_afterPretext:
    def __init__(self) -> None:
        self.paths = {}
        self.paths["labelled"] = {}
        self.paths["unlabelled"] = {}
        self.img_dir = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/labelled/images"
        self.mask_dir = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/labelled/masks"

        # Paths for labelled data
        self.paths["labelled"]["train_images"] = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/labelled/train_image"
        self.paths["labelled"]["train_masks"] = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/labelled/train_mask"
        self.paths["labelled"]["val_images"] = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/labelled/val_image"
        self.paths["labelled"]["val_masks"]  = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/labelled/val_mask"

        # Paths for unlabelled data
        self.paths["unlabelled"]["train"] = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/unlabelled/train_image"
        self.paths["unlabelled"]["val"] = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/unlabelled/val_image"

        self.n_epochs_supervised = 100
        self.n_epochs_unsupervised = 500 

        self.stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ### remove this later or change this

        self.IMAGE_HEIGHT = 256
        self.IMAGE_WIDTH  = 256

        self.batch_size_selfSupervised = 8
        self.batch_size = 80

        self.lr = 1e-3
        self.lr1= 1e-4 ## for self-supervised training
        self.lr2 = 1e-6 ## for complete training full network for segmentation
        self.preload = None
        self.preload2= "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/models/CL-HistologySegmentation_20240424_140528/pthFiles/model_epoch_5"
        # Define the layers you want to exclude
        self.layers_to_exclude = None #['inc.double_conv.0.weight', 'outc.conv.weight', 'outc.conv.bias']


        ## for the final train after train
        self.encoder_lr = 1e-6
        self.decoder_lr = 5e-4
        self.margin = 5
        self.k = 10

class ModelConfig_VineNet:
    def __init__(self) -> None:
        self.paths = {}
        self.paths["labelled"] = {}
        self.paths["unlabelled"] = {}
        self.img_dir = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/VineNet/images"
        self.mask_dir = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/VineNet/masks"

        # Paths for labelled data
        self.paths["labelled"]["train_images"] = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/labelled/train_image"
        self.paths["labelled"]["train_masks"] = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/labelled/train_mask"
        self.paths["labelled"]["val_images"] = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/labelled/val_image"
        self.paths["labelled"]["val_masks"]  = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/labelled/val_mask"

        # Paths for unlabelled data
        self.paths["unlabelled"]["train"] = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/unlabelled/train_image"
        self.paths["unlabelled"]["val"] = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/unlabelled/val_image"

        self.n_epochs_supervised = 70
        self.n_epochs_unsupervised = 500 

        self.stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ### remove this later or change this

        self.IMAGE_HEIGHT = 1920
        self.IMAGE_WIDTH  = 960

        self.batch_size = 64

        self.lr = 1e-3
        self.lr1= 1e-4 ## for self-supervised training
        self.lr2 = 1e-6 ## for complete training full network for segmentation
        self.preload = None
        self.preload2= None
        # Define the layers you want to exclude
        self.layers_to_exclude = None # ['inc.double_conv.0.weight', 'outc.conv.weight', 'outc.conv.bias']


        ## for the final train after train
        self.encoder_lr = 5e-4
        self.decoder_lr = 5e-4
        self.margin = 5
        self.k = 10


class ModelConfig:
    def __init__(self) -> None:
        self.paths = {}
        self.paths["labelled"] = {}
        self.paths["unlabelled"] = {}
        self.img_dir = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/labelled/images"
        self.mask_dir = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/labelled/masks"

        # Paths for labelled data
        self.paths["labelled"]["train_images"] = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/labelled/train_image"
        self.paths["labelled"]["train_masks"] = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/labelled/train_mask"
        self.paths["labelled"]["val_images"] = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/labelled/val_image"
        self.paths["labelled"]["val_masks"]  = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/labelled/val_mask"

        # Paths for unlabelled data
        self.paths["unlabelled"]["train"] = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/unlabelled/train_image"
        self.paths["unlabelled"]["val"] = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet/unlabelled/val_image"

        self.n_epochs_supervised = 70
        self.n_epochs_unsupervised = 500 

        self.stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ### remove this later or change this

        self.IMAGE_HEIGHT = 256
        self.IMAGE_WIDTH  = 256

        self.batch_size_selfSupervised = 8
        self.batch_size = 80

        self.lr = 1e-3
        self.lr1= 1e-4 ## for self-supervised training
        self.lr2 = 1e-6 ## for complete training full network for segmentation
        self.preload = None
        self.preload2= "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/models/CL-HistologySegmentation_20240424_140528/pthFiles/model_epoch_5"
        # Define the layers you want to exclude
        self.layers_to_exclude = None # ['inc.double_conv.0.weight', 'outc.conv.weight', 'outc.conv.bias']


        ## for the final train after train
        self.encoder_lr = 1e-6
        self.decoder_lr = 5e-4
        self.margin = 5
        self.k = 10
        