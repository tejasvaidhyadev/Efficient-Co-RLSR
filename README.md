# Efficient-Co-RLSR
Contains  implementation of  Efficent Co-RLSR

### Training 
run
```
python train.py --dataset <path_to_dataset> --epochs <no of epochs>
```
--outputDim is use to specify the number of output dimension

The model will take last ```outputDim``` as target labels

### Baseline 
Implements the linear RSE

### TODO
-[ ] Model stroing in restor_dir 
-[ ] Making M parametric and randomly division of attributes
-[ ] adding other kernals 

