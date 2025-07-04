import os

DIR_PATH = os.path.dirname(os.path.abspath(__file__))


# DATASET_NAME = "CANIntrusion Dataset"
# DATASET_NAME = "CarHacking Dataset"
# DATASET_NAME = "ExampleDataset"
# DATASET_NAME = "MIRGU"
DATASET_NAME = "CARLA"

# write filename for training and testing
FILE_NAME = "data.log"

# PREPROCESS = True
PREPROCESS = False

# SPLIT = True
SPLIT = False

# FEATURE_EXTRACTION = False
FEATURE_EXTRACTION = True


# MODEL_NAME must include any one of these : Desnsenet161, RandomForest, MLP, DecisionTree, ResNet
MODEL_NAME = "MLP_Carla"
# MODEL_NAME = "DecisionTree_Carla"
# MODEL_NAME = "RandomForest_Carla"

# Options for FEATURE_EXTRACTOR : Kitnet, PixNet, Stat

FEATURE_EXTRACTOR = 'Stat'

# MODE = "test"
MODE = "train"

EPOCHS = 6


# If using images as features and FEATURE_EXTRACTION is True

TRAIN_DATASET_DIR = "target_train_images"
TRAIN_DATASET_LABEL = "target_train_labels.txt"

TEST_DATASET_DIR = "target_test_images"
TEST_DATASET_LABEL = "target_test_labels.txt"

# Options for adv_attack : Blackbox , Whitebox, None
# ADV_ATTACK = True
ADV_ATTACK = False

# Options for this : FGSM, PGD, C&W
# ADV_ATTACK_TYPE = "FGSM"

# Attack parameters 
# EPSILON = 
# MAX_INJECTION_LIMIT = 
# Options for original_attack : DOS, SPOOF, FUZZING
# ORIGINAL_ATTACK = 




SURROGATE_MODEL = "Densenet161_Carla_test2"
TARGET_MODEL =  "ResNet_Carla_test2"




