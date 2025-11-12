----------------------------------------------------------------------------------------------------------
DISCLAIMER -- The codes in this repository are modified codes from "https://github.com/csccm-iitd/Sp2GNO"


----------------------------------------------------------------------------------------------------------
Set up conda environment

  conda create -n sp2gno4dfn_env python=3.8

  conda activate sp2gno4dfn_env

Install (I have used cpu version) 

  pip3 install torch --index-url https://download.pytorch.org/whl/cu118
  
  pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv  -f https://data.pyg.org/whl/torch-2.4.1+cu118.html
  
  conda install -c conda-forge fenics=2019.1.0 -y
  
  pip3 install torch-geometric==2.4.0 matplotlib pandas prettytable
  
Save the U_sol.csv file in the same folder as cloudpoint_generator.py file

Run the cloud point generator file and generator the dataset

Save the data set file 'dfn_4frac_prob.mat'  as in the following directory, and the code in 'Sp2GNO4DFN' directory.


\Code_Folder

  \data
    \dfn-
      dfn_4frac_prob.mat
  \Sp2GNO4DFN


Run train_dfn.py to train the operator
