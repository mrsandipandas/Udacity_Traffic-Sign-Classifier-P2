# **Traffic Sign Recognition Solution** 


---

**The goals of the project is as follows:**

* The data set used can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip) and you can find the solution to sign classification problem [here](https://github.com/s-a-n-d-y/Udacity_Traffic-Sign-Classifier-P2/blob/master/Traffic_Sign_Classifier.ipynb)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./traffic-signs-data/data_summary.jpg "Dataset summary"
[image2]: ./traffic-signs-data/data_augmented.jpg "Grayscaling and random noise"
[image3]: ./traffic-signs-classification/extra_crops.jpg "Traffic sign crop"
[image4]: ./traffic-signs-classification/extra_marked.jpg "Traffic sign crop region of interest"
[image5]: ./traffic-signs-classification/extra_predictions.jpg "Cropped traffic sign prediction"
[image6]: ./traffic-signs-classification/visualize_image.png "Random traffic sign"
[image7]: ./traffic-signs-classification/visualize_image_CNN_1.png "Activation map for first CNN layer"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training set examples = 31799
* Number of validation set examples = 3000
* Number of testing set examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43
* Number of training set examples (after flip) = 54674
* Number of augmented images = 43000

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across various classes for the training set.

![Dataset summary][image1]

Similar representation is available for the test set in the html file or in the jupyter notebook.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the three channels are not required for classification of the trafffic signs. Then I did some more flipping and rotation of the original image to aument to the training dataset. I decided to generate additional data because it would greatly increse the robustness og the training and generate a more accurate model with the weights.

Here is an example of the training dataset of the traffic sign claases. The first coloumn represents the original image.

![Grayscaling and random noise][image2]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|  |     Description	        					| 
|:---------------------:|:--:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| |
| Preprocess     	| | 	| 
|                    | Batch Normalization | 
|                    | Convolution | Kernal size 3x3, Padding same
|                    | Prelu |                                  |
|                    | Convolution | Kernal size 3x3, Padding same
|                    | Prelu |  				|
| Dense Block	     |       |       Kernal size 5x5, Padding same     |
|                    | Maxpool |     Kernal size 2x2, Padding same     |
| Dense Block	     |       |       Kernal size 3x3, Padding same     |
|                    | Dropout |     Keep probability = 0.9     |
|                    | Maxpool |     Kernal size 2x2, Padding same     |
| Dense Block	     |       |       Kernal size 3x3, Padding same     |
|                    | Dropout |     Keep probability = 0.9     |
|                    | Maxpool |     Kernal size 1x1, Padding same     |


The dense block contains of batch normalization, Relu operation and convolution.



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameters:

* Batch size = 128 per iteration. 
* The solver used is SGD, with momentum = 0.9.
* Cross entropy loss is used
* The learning rate policy is “step”, starting from 0.1, then to 0.01, 0.001 and 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


* What architecture was chosen?<br/>
My model is based on [Densenet](https://github.com/liuzhuang13/DenseNet) architecture. 

* Why did you believe it would be relevant to the traffic sign application?<br/>
For each layer, the feature maps of all preceding layers are treated as separate inputs whereas its own feature maps are passed on as inputs to all subsequent layers. This connectivity pattern yields state-of-the-art accuracies on CIFAR10/100 (with or without data augmentation) and SVHN. On the large scale ILSVRC 2012 (ImageNet) dataset, DenseNet achieves a similar accuracy as ResNet, but using less than half the amount of parameters and roughly half the number of FLOPs. Since, it performs well on imagenet with half of the amount of parameters, I expect it would have similar results on the traffic sign classification problem as well.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?<br/>
  * Training loss: 0.056222
  * Training accuracy: 0.992188
  * Validation loss: 0.010780
  * Validation accuracy: 0.998000
  * Testing loss: 0.042439
  * Testing accuracy:  0.987648
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Random traffic signs][image4] 

In the beginning I cropped out the region of interest based on the traffic signs in the image and it looked as follows:

![Cropped traffic signs][image3] 

The last image would be difficult as it does not have any labelled data.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric). Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here are the results of the prediction:

![Cropped traffic sign prediction][image5]

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.987648. In the random set there was one traffic sign which didn'tbelong to any of the 43 classes of the test set. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. 

The code for making predictions on my final model is located in the section 'Analyze Performance'.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

One image was randomly selected from test set for visualtion of the activation maps. The image which was selected was:

![Original image][image6] 


After running through the saved tensors the output of the first layer of the network shows the folloing activation map:

![Activation maps][image7] 

As shown, inthe intial layer the activation maps in general ties to learn the lines and other generalistic structure.

