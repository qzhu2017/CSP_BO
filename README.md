# CSP_BO
Crystal Structure Prediction with Bayesian Optimization

To run the code
```
$ python setup.py install
$ cd examples
$ python example_validate.py models/test_2.json ../database/PtHO.db

------Gaussian Process Regression------
Kernel: 18.555**2 *Dot(length=0.010) 1 energy (0.005) 11 forces (0.150)

load the GP model from  models/test_2.json
Train Energy [   1]: R2 0.9480 MAE  0.000 RMSE  0.000
Train Forces [  33]: R2 0.9824 MAE  0.006 RMSE  0.011
   1 E: -5.265 -> -5.265  F_MSE:  0.363 
   2 E: -5.265 -> -5.265  F_MSE:  0.363 
   3 E: -5.265 -> -5.265  F_MSE:  0.363 
   4 E: -5.265 -> -5.265  F_MSE:  0.363 
   5 E: -5.265 -> -5.265  F_MSE:  0.363 
   6 E: -5.265 -> -5.265  F_MSE:  0.363 
   7 E: -5.265 -> -5.265  F_MSE:  0.363 
   8 E: -5.265 -> -5.265  F_MSE:  0.363 
   9 E: -5.265 -> -5.265  F_MSE:  0.363 
  10 E: -5.265 -> -5.265  F_MSE:  0.363 
Test Energy [  10]: R2 0.5782 MAE  0.000 RMSE  0.000
Test Forces [5760]: R2 0.2158 MAE  0.229 RMSE  0.363
35.767 seconds elapsed
save the figure to  E.png
save the figure to  F.png
```
