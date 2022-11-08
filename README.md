# HiRAND
This is the code of paper: HiRAND: A novel semi-supervised deep learning framework for predicting drug response to therapy in cancer.

# Requirements
Python 3.7.3  
Please install other pakeages by pip install -r requirements.txt

# Usage Example
Runing HiRAND on the drug response data: python train_drug.py  
Runing HiRAND on simulation data: python train_simulation.py  
Runing HiRAND for 5 times on data: sh submit.sh  

# Code Base Structure
**train_drug.py**: main script for drug reponse prediction.  
**train_simulation.py**: main script for simulation data prediction.  
**until.py**: contains definitions for aample similarity graph construction and etc.  
**model.py**: contains PyTorch model definitions for HiRAND.  
**generate_simulation_data.py**: generates the simulation data.  
**submit.sh**: complete the multiple prediction task using train_drug.py or train_simulation.py.   

# The drug related data
The drug data is too large to upload. If you need them, please download [drug data](https://pan.baidu.com/s/1KFEx11_jHQDQTuXMjTwcgQ?pwd=bu2d). (retrieve password:bu2d).

# Contact
For questions or comments about the code please contact: likang@ems.hrbmu.edu.cn
