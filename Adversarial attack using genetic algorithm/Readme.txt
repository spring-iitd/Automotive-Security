First download the all the files from car hacking dataset, and save it inside a folder named CAN_DATA in the same directory as of train.py and run it. 

python3 train.py --model Inception_Resnet

It will create train and test .npz files, train the models, and save all the models as well (for each attack from corresponding CSV or text files).

After that, make sure all the npz files are present in CAN_DATA directory, just run each attack separately for generating adversarial attack and obtaining all the results.

python3 Spoofing_attack.py

python3 DoS_attack.py

python3 Fuzzy_attack.py