
# Options for adv_attack : Blackbox , Whitebox, None
ADV_ATTACK_TYPE = "blackbox"

# If adv_attack_type is Blackbox, then choose different surrogate and target models
SURROGATE_MODEL = "Densenet161_demo"
TARGET_MODEL =  "Resnet_demo"

# conventional attack parameters for string matching in case of modification (type : string)
# ID : 11 bits and DLC : 4 bits
ID = "00000000000"
DLC = "1000"

# Attack parameters 
EPSILON = 1
MAX_INJECTION_LIMIT = 30  # Maximum number of messages that can be injected

# Options for original_attack : DOS, SPOOF, FUZZING
# ORIGINAL_ATTACK = 

