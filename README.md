# Identifying value of Brazilian Real coins by image using Convolutional Neural Networks (CNN)

This project used the dataset from Kaggle: Brazilian Coins - Classification set. https://www.kaggle.com/lgmoneda/br-coins
To fit the dataset to this code I separated the images randomly into two folders (training set and test set) and inside each folder, I separated it by value, keeping 2500 images on training set (500 each category) and 500 for test set (100 each category).

The configuration of this CNN is:
- Convolutional input Layer with 32 5x5 feature maps in rectifier activation function.
- Max Pool layer of 2x2
- Convolutional input Layer with 32 3x3 feature maps in rectifier activation function.
- Max Pool layer of 2x2
- Convolutional input Layer with 64 3x3 feature maps in rectifier activation function.
- Max Pool layer of 2x2
- Convolutional input Layer with 64 3x3 feature maps in rectifier activation function.
- Max Pool layer of 2x2
- Dropout set to 20%
- Fully connected layer with 128 units and rectifier activation function
- Fully connected output layer 5 units and a softmax activation function.
- Learning rate of 0.001
- Optimizer = "Adam"


### Results
We got an accuracy of around 66% with this configuration.

An example of a picture for training set is below and the conditions of the picture are not good, so it might confuse the algorithm. 
<img src="https://github.com/amyoshino/Identifying-Brazilin-Coins-with-CNN/blob/master/50_cents.jpg" width="250">  
<img src="https://github.com/amyoshino/Identifying-Brazilin-Coins-with-CNN/blob/master/1_real.jpg" width="250">

The ground truth test is to use different views of the coin, bigger ones and different angles. Example of coins used to this purpose is below.
<img src="https://github.com/amyoshino/Identifying-Brazilin-Coins-with-CNN/blob/master/Coins_to_test/moeda025.jpg" width="300">
<img src="https://github.com/amyoshino/Identifying-Brazilin-Coins-with-CNN/blob/master/Coins_to_test/moeda100.jpg" width="300">


In the ground truth test, 3 out of 5 coins were correctly rocognized. Tuning (adding more layers or changing hyperparameters) better this model might improve the results.

