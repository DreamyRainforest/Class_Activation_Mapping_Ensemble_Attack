# Class_Activation_Mapping_Ensemble_Attack

# Requirements:
Python 3.7.16     
torch 1.8.0+cu111  
torchvision 0.9.0+cu111    
tqdm 4.65.0    
numpy 1.21.6     
pillow 9.5.0     
# Experiments:
The code consists of three Python scripts. Before running the code, you need to complete the following two steps:     

Download Data: Download the data from the provided link (https://pan.baidu.com/s/1NlenXev0cN1l55ZSVQ-_nw; password: d6tn) and place it in the benign_image/ directory.    

Calculate Class Activation Maps (CAM): Compute the class activation maps and place them in the CAM/ directory.    

# Running the code

untaregt_attack_example.py：Non-targeted attack     
taregt_attack_example.py：targeted attack     
victim_one.py: test    

# Acknowledgments:  
Code refer to     
https://github.com/Harry24k/adversarial-attacks-pytorch    
https://github.com/frgfm/torch-cam    
https://github.com/erbloo/dr_cvpr20    
https://github.com/RobustBench/robustbench
