## Basic rules
- All `.m` files start with "my" is subfunction, otherwise the main function.   
- Folder `fitting` is for interpolation part. Folder `svm predict` is used for prediction work, including svm, kriging and neutral network. Results and figures used in thesis are saved in folder `final_results`.  
- SVM and Kriging are used by `MATLAB`; NN are runable in `Python3` (and ploted in `MATLAB`). 
- `.ipynb` files are file for `Jupyter Lab`. Strongly recommend for debugging. 
## Code path
MATLAB `.m` file direction of final_data folder:
- NNvsSVMvsKrig_dimension: 
  - `FA/fitting/fit_nd_time_nn.m`
- rmse_time_dimension_05_04: 
  - `FA/fitting/fit_nd_time.m`
- rmse_time_number: 
  - `FA/fitting/fit_nd_time.m`
- NNvsSVMvsKrig_dataNumber: 
  - `FA/fitting/fit_nd_nn.m`
- method_1: 
  - `FA/svm predict/m1_svm_krig.m`
- method_2: 
  - `FA/svm predict/krig_m2.m`  
  - `FA/svm predict/svm_m2.m`

Python `.py` file path: 
- method_1 and method_2: 
  - `FA/nn_m1m2.py`
- method_3:
  - `FA/nn_m3.py`