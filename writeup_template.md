#**Traffic Sign Recognition**


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./intermediate_images/download.png "image count per class"
[image2]: ./intermediate_images/grayscale-1.png "Grayscaling"
[image3]: ./intermediate_images/historgram-gray.png "historgram-gray"
[image4]: ./german-web-test-images/30speed.jpg "Traffic Sign 1 - 30"
[image5]: ./german-web-test-images/Caution.jpg "Traffic Sign 2"
[image6]: ./german-web-test-images/Do-Not-Enter.jpg "Traffic Sign 3"
[image7]: ./german-web-test-images/PriorityRoad.jpg "Traffic Sign 4"
[image8]: ./german-web-test-images/roadWorks-2.jpg "Traffic Sign 5"
[image9]: ./german-web-test-images/stop.jpg "Traffic Sign 6"

[image10]: ./intermediate_images/do-not-enter-vis1.png "visualize 1 "
[image11]: ./intermediate_images/do-not-enter-vis2.png "visualize 2 "
[image12]: ./intermediate_images/do-not-enter-vis3.png "visualize 2 "


---
***github project archive***

You're reading it! and here is a link to my [project code] (https://github.com/peterbarry/udacity-self-driving-ND-traffic-sign-recognition/blob/master/Traffic_Sign_Classifier.ipynb)

***Data Set Summary & Exploration***


The code for this step is contained in the second  code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:
*  Number of training examples = 34799
*  Number of testing examples = 12630
*  Number of classes = 43
*  Shape of Y= (34799,)



The code for this step is contained in the fourth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data the numeber of samples of date in the training class for each of the classes.
As you can see there is a wide variation in sample count per class. Some classes have around 2000 wheres other have just a vw underd. This is not an ideal data set and could result in errors. It is better to have approximiatly the same number of training samples per class. A folloup up ste would be to
* Get more data for the classes with low data count
* Enrich the low count date samples by generating new samples fo rthe class with random noise, minor titing and scaling.


![alt text][image1]

***Data Set processing***

As a first step, I decided to convert the images to grayscale image, to test the premis that there was sufficent shape information in the date set to perform well.
Overall it did perform quite well but I think in a subsequent work I would not convert to grayscale and leave the data set as 3channel. This is becuse there is a lot of colour informaton in the signs too.
The code for this step is contained in the fourth code cell of the IPython notebook.


Here is an example of a traffic sign image after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because to facilitate training of the network, CNNs require normalised data as referenced in hte lectures.

I did review the data in the cell 3 and it showed a large varation in brigtness, I did not try to process the inages to normalize the brightness as the performance was good on the images, in future i would consider normalizing this.  The picure below is a historgram of the normalized data

![alt text][image3]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets was provided in the origional data.
* Trainng set size 34799
* validation set size 4410

The validaiton set size is jsut about adquate and 10-20% size of training set size is a good balance.



***CNN Model***

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 10 cell of the ipython notebook, called model architecture
The mode is the LeNet model - unmodifed.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution      	| 32x32x1 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 28x28x6 2x2 stride,  outputs 14x14x6 				|
| Convolution 	    | 14x14x6, 1x1 stride outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 10x10x16 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| 3 fully connected layers.        									|
| Softmax				| on hot encoded - number of classes=sign types.        									|




####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 15th cell of the ipython notebook.

To train the model, I used an adam optimiser.

I tried varing epoch/batch and learning rates


| Epochs         		|     Batch size | Learning rate | Final Validatin accuracy | comment|
|:---------------------:|:-----------------:|:----------------:|:--------------:|
| 10         		| 32 							| 0.001 | 0.93 | good initial start|
| 3500        		| 256 							| 0.00001 | 0.91  | Perhaps overfit too slow|
|1000| 128|0.001| 0.92| Very slow convergence |
|1000|16|0.001|0.95 | good perfmance|
|100|16|0.001|0.954| some jitter on validation performance|


It should be noteed that the validation accuracy has a level of jitter and is not monotonically increasing.
the following images shows the validation accuracy for each epoch in the trainined model used.

![alt text][image10]


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 15th  cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.951


I choose the well known LeNet model:
* Looking at the shapes and features of hand written letter and those found in signs, it looked like a reasonable starting point. It performed aboce the required level for the class. Developing models can be a long trail and error process where some emperical/intuition  expereince is built up.
* The model peromred at 0.951 on validation data on the last epoch, and classified  100% of my test images


***Test a Model on New Images***

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8] ![alt text][image9]

All images where of a different scale but all quite clear and the sign occupied a large portion of the image.

*** Model Predictions for new test data source on line ***

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 30 Speed     		| 30 Speed    									|
| Caution    			|Caution										|
| Do not enter 					| Do not enter											|
| Priority Road    		| Priority Road 					 				|
| Roadworks			| Roadworks      							|
| Stop| Stop|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably (better, but small data set) to the accuracy on the test set of
* Test Accuracy = 0.941



| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| 30 Speed   									|
| 1.00     				| Caution 										|
| 1.00					| Do not enter											|
| 1.00	      			| Priority Road				 				|
| 1.00				    | Roadworks      							|
| 1.00| Stop |


The sign type for test images are
[ 1 18 17 12 25 14]

Top Predictions [
 [ **1** 25 13 35 33]
 [**18**  0  1  2  3]
 [**17**  0  1  2  3]
 [**12**  0  1  2  3]
 [**25** 18  5  2  1]
 [**14**  0  1  2  3]]


**** Visualisation of the network ****

The notepad shows the activation in the first and second convolution layers of the network. I have selected the "do no enter" sign to make an assesment to how the network is behaving

![alt text][image10]

* The first activation layers
![alt text][image11]

* The second activatin layers
![alt text][image12]

In the frist layer, one of the darkest areas is the horizontal line, a distinguishing aspect of this sign. The following convolution layer shows activations for the horizontal line and circular segments of the sign outline.
