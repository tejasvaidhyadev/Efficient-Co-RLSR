# Co-Regularised Least Squares Regression
This Repo Contains python implementation of paper Efficient [Co-Regularised Least Squares Regression](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.7014&rep=rep1&type=pdf)  
### Instructions
Run the below line
```
python train.py --dataset <path_to_dataset> --epochs <no of epochs>
```
```--outputDim``` is use to specify the number of output dimension

- The model will take last ```outputDim``` column of dataset as target labels

### Baseline 
For comparison, we are implementing RSE as baseline (also known as Ridge regression)

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
### Miscellanous
- **Lisence**:  MIT
- You may contact me by opening an issue on this repo. Please allow 2-3 days of time to address the issue.
