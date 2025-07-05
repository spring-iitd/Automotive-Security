import os

DIR_PATH = os.path.dirname(os.path.abspath(__file__))



DATASET_NAME = "CARLA"

# write filename for training and testing
FILE_NAME = "data.log"

# whether preprocessing stage should be executed
PREPROCESS = False

# whether split stage should be executed
SPLIT = False

# whether feature extraction should be done
FEATURE_EXTRACTION = True


# MODEL_NAME must include any one of these : Desnsenet161, RandomForest, MLP, DecisionTree, ResNet
MODEL_NAME = "MLP_Carla"


# Options for FEATURE_EXTRACTOR : Kitnet, PixNet, Stat
FEATURE_EXTRACTOR = 'Stat'

# Options for mode : train and test
MODE = "train"


EPOCHS = 6


# If using images as features and FEATURE_EXTRACTION is True

TRAIN_DATASET_DIR = "target_train_images"
TRAIN_DATASET_LABEL = "target_train_labels.txt"

TEST_DATASET_DIR = "target_test_images"
TEST_DATASET_LABEL = "target_test_labels.txt"


# Options for adv_attack : Blackbox , Whitebox, None
ADV_ATTACK = "blackbox"

# If adv_attack is not None
SURROGATE_MODEL = "Densenet161_Carla_test2"
TARGET_MODEL =  "ResNet_Carla_test2"

# Options for this : FGSM, PGD, C&W
# ADV_ATTACK_TYPE = "FGSM"

# Attack parameters 
# EPSILON = 
# MAX_INJECTION_LIMIT = 
# Options for original_attack : DOS, SPOOF, FUZZING
# ORIGINAL_ATTACK = 






