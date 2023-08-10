# Class_Activation_Mapping_Ensemble_Attack

# Requirements:
Python 3.7.16     
torch 1.8.0+cu111  
torchvision 0.9.0+cu111    
tqdm 4.65.0    
numpy 1.21.6     
pillow 9.5.0     
# Experiments:
The code consists of 3 Python scripts. Before running the code, please download the data (https://pan.baidu.com/s/1MEjNh6evha2hcdrQXjNv8w?pwd=yzza). The data is then placed in benign_image/. Please calculate the class activation map (https://github.com/frgfm/torch-cam) and place it in CAM/

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
