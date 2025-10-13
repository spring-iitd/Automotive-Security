import os
import sys
from config import *
# from attacks import attack_handler
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import attacks.attack_handler as attack_handler

def get_attack(attack_name):
    if(attack_name is None or attack_name.lower() == "none"):
        return None
    
    for attack_class in attack_handler.__all_classes__:
        if attack_class.__name__.lower() in attack_name.lower():
            print("found attack : ", attack_class.__name__)
            attack_class()
            return attack_class().apply()
            
    raise Exception(f"{attack_name} not yet implemented")