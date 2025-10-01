import os

DIR_PATH = os.path.dirname(os.path.abspath(__file__))


DATASET_NAME = "user dataset"

# write original filename for training and testing
FILE_NAME = "new_data.log"

# whether preprocessing stage should be executed
PREPROCESS = True

# whether split stage should be executed
SPLIT = True

# options: default, three  
SPLIT_MODE = "default" 

SPLIT_RATIO = 0.2


# whether feature extraction should be done (for both training and testing)
FEATURE_EXTRACTION = True

# Options for FEATURE_EXTRACTOR : Kitnet, PixNet, Stat
FEATURE_EXTRACTOR = 'PixNet'

# MODEL_NAME must include any one of these : Densenet161, RandomForest, MLP, DecisionTree, ResNet
MODEL_NAME = "Resnet_demo"

# Options for mode : train and test (Results will be saved in "Results" folder)
MODE = "train"
EPOCHS = 3

# The folder will be created inside train and test folders
TRAIN_DATASET_DIR = "target_train_images"

TEST_DATASET_DIR = "target_test_images"

# Options for this : FGSM, PGD, C&W, None
ADV_ATTACK = None







