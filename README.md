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

The results as we will see are acceptable. Tuning (adding more layers or changing hyperparameters) better this model can definitely improve the results.
