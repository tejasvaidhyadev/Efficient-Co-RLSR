# Efficient-Co-RLSR
Contains  implementation of  Efficent Co-RLSR  
Paper: [link](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.7014&rep=rep1&type=pdf)  
Processed dataset provided at: ./dataset/  

### Training 
Run the below line
```
python train.py --dataset <path_to_dataset> --epochs <no of epochs>
```
--outputDim is use to specify the number of output dimension

The model will take last ```outputDim``` as target labels

### Baseline 
Implements the RSE (also known as Ridge regression)

### Example
Step1: keep the processed dataset file at ./dataset with last column corresponds to target label  

step2:
```

$ python train.py --dataset ./dataset/pollution.data.txt --epochs 9000 --batch_size 16 --outputDim 1

split of attribute at
9
Loading the datasets...
model...
File name pollution.data.txt
Starting training for 9000 epoch(s)
Training completed for printing loss uncomment 75 and 76 linn in train.
Starting testing for epoch(s)
combine-rms on test set: 51.64
done
Training of Baseline starting
Training of Baseline successful. To print losses uncomment the line 67 and 68
Baseline testing Starting
baseline loss on test: 89.166
```

### Results
Results: [link](https://docs.google.com/document/d/17BtTTLGRWsAMNoZkYNj13UZ1iDSJ96VNql0TteO5lFA/edit?usp=sharing)



