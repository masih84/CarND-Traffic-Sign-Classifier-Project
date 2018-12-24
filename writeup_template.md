# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./data_sample/Training_set_Histogram.png "Training Set Visualization"
[image2]: ./data_sample/Validation_set_Histogram.png "Validation Set Visualization"
[image3]: ./data_sample/Test_set_Histogram.png "Test Set Visualization"
[image4]: ./data_sample/Test_set.png "Training Set"
[image5]: ./data_sample/Validation_set.png "Validation Set"
[image6]: ./data_sample/Test_set.png "Test Set"
[image7]: ./samples/Real_Test.png "Final Test Set"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/masih84/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) and [project html](https://github.com/masih84/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html).

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410 
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the Hisogram of Training, Validation and Test data sets.

##### Training Set: 
![alt text][image1] 
![alt text][image4] 

##### Validation Set: 

![alt text][image2]
![alt text][image5]

##### Test Set: 
![alt text][image3]
![alt text][image6]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? 

I decided to Not to convert the images to grayscale because Color is very impt of traffic signs and converting to grayscale will lose this factor.

I Only normalized the image data because in optimization convergence speed of NN faster. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs  10x10x64			|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				|
| Fully connected	1	| outputs 1000  		        									|
| Drop OUT 1	| keep_prob = 50%  		        									|
| Fully connected	2	| outputs 1000  		        									|
| Drop OUT 2	| keep_prob = 50%  		        									|
| Softmax				| softmax_cross_entropy_with_logits        									|
| regularization_cost | weights, REGULARIZATION_PARAM   |
|	reduce_mean					|			cross_entropy+reg_cost									|
|	optimizer					|				AdamOptimizer								|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a Bach size of 100, learning rate of 0.0005. Also, I used REGULARIZATION PARAM of 1e-5 to penalize large weights.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 97.6% 
* test set accuracy of 95.7%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
 I started by using parameters we used in example of course. then increased the size of filter, added drop out option and include weight regularization cost.
 
* What were some problems with the initial architecture?
The main problem was the validation accuracy was in 90%. adding all options improve it to 97%.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Not using drop out made the model over-fitted.

* Which parameters were tuned? How were they adjusted and why?
Size of layer, learning rate, bachsize are tunned by Trial and error.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Not using drop out made the model over-fitted.

If a well known architecture was chosen:
* What architecture was chosen? No spesific architecture used.
* Why did you believe it would be relevant to the traffic sign application? AN
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 The high accuracy percentage of validation and test shows the model is working.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7]
The images might be difficult to classify because they are in different size, with water mark and clear shape with no shadow.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Slippery road      		| Slippery road   									| 
| Speed limit (50km/h)     			| Speed limit (50km/h) 										|
| Stop					| Stop											|
|No passing	      		| Speed limit (30km/h)					 				|
| Children crossing			| Speed limit (30km/h)      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Slippery road (probability of 1.0), and the image does contain a Slippery road sign. The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Slippery road   									| 
| .00     				| Beware of ice/snow 										|
| .00					| Right-of-way at the next intersection											|


For the second image, the model is relatively sure that this is a Speed limit (50km/h) (probability of 0.99), and the image does contain a Speed limit (50km/h) sign. The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Speed limit (50km/h)   									| 
| .00     				| Speed limit (80km/h) 										|
| .00					| Speed limit (100km/h)											|


For the third image, the model is relatively sure that this is a Stop (probability of 1.00), and the image does contain a Stop sign. The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Stop  | 
| .00     				| No vehicles|
| .00					| Speed limit (80km/h)											|

 For the forth and fifth image, the model is compeletly wrong. The signs are No passing and Children crossing. The top three soft max probabilities were
 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Stop  | 
| .00     				| No vehicles |
| .00					| Speed limit (80km/h)											|


For the third image, the model is relatively sure that this is a Stop (probability of 1.00), and the image does contain a Stop sign. The top three soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|0.72        			| speed limit (30km/h)  | 
| 0.25    				| No vehicles |
| 0.02 				| Stop |


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.00        			| Speed limit (30km/h)  | 
| 0.00    				| Speed limit (20km/h) |
| 0.00 				| Bicycles crossing |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


