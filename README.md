# Architectural Basics
## 1.  Image Normalization

 Image Normalization is a part of image preprocessing. The idea to standardize/normalize the input to network as much as possible            because of following reasons: 
 
**a)** learning is more stable (by reducing variability across the training data)                                                        
**b)** your network will generalize better to novel data (because the normalization reduces the variability between your training               and test data)                                                                                                                    
**c)** the inputs will fall within the useful range of your nonlinearities                                                                

  there are three common techniques for image normalization: 
  
  **i.** Bring Pixel values in range of 0 to 1                                                                                            
  **ii.**  Bring Pixel values in range of -1 to 1                                                                                        
  **iii.** Bring Pixel values in range of 0 mean and  1 std dev 

## 2.  Receptive Field
Our objective is to have the final global receptive field (at the final prediction layer or output layer) to be equal to the size of the object. This is important as the network needs to "see" the whole image before it can predict exactly what the image is all about. 
This would mean that we need to add as many layers are required to reach the final receptive field equal to the size of the object.
 
![Receptive Field](images/RF.gif)

Here in first layer as a 5x5 image, convolving  with a kernel of size 3x3, and hence the resulting output resolution will be a channel with 3x3 pixels/values. In second layer when we convolve on this 3x3 channel(last result) with a kernel of size 3x3, we will get only 1 output. 
The final output of 1 or 1x1, we could have used a 5x5 kernel directly. 
This means that using a 3x3 kernel twice is equivalent to using a 5x5 kernel. This also means that two layers of 3x3 have a resulting receptive field of 5x5. 

### 3.  MaxPooling
MaxPooling helps to reduce the number of layers count significantly along with doubling the receptive field than previous layer. In our MNIST dataset example to reach from input image 28x28 to 1x1 size we need 14 layers. But with 2 maxpooling layers we need only 8 layers. 
We prefer 2x2 maxpooling instead 3x3 to keep data loss minimum level. Following images tells how maxpooling operation works.

![Maxpool1](maxp_new.JPG)

### 4.  How many layers 
   Number of layers need to include to have whole input image object view or global receptive field of object at final prediction layer.
   The no of layers are required without maxpooling is more than with maxpooling.        

### 5.  Kernels and how do we decide the number of kernels?
Kernels are feature extractors, used to extract different features from input image. The number of kernels required to have atleast more than number of classes count. We gradually increase number of kernels as we proceed from begining to prediction layer. Sometimes we learned new features with kernels or combine large learned kernels to small kernels count.

### 6.  3x3 Convolutions
3x3 convolutions are used to learn new features. They are symmetric about origin which is good. 3x3 convolution looking at a 3x3 area, we are looking at 9 multiplications and the sum of the resulting 9 multiplications being passed on to the output. 

![3x3 conv](3x3conv.gif)

By changing 3x3 convolution values, we can obtain different results.(Horizontal/vertical/diagonal lines)
![3x3 conv-edge](conv-line-detection.jpg)

### 7.  Position of MaxPooling
Our Network should learn **edges and gradients -> texture and patterns -> parts of objects  -> objects**, as we proceed from begining to prediction layer. Maxpooling is used only after network learned above things completely to avoid loss of important learned features.
Also its recommended not to use maxpooling close to begining and end of network to avoid unrecoverable data loss.

### 8.  Concept of Transition Layers
Transition layers are used in between two convolution blocks. It consists of maxpooling and 1x1 convolutions. So that we can reduce image channels resolution and channels depth which is at the input of transition layer, before passing to next convolution block.

### 9.  1x1 Convolutions
Instead of learning new features, combining large number of features into complex small number, is achieved by 1x1 convolution.
1x1 Covolution benefits:                                                                                                                  
**a)** 1x1 is computation less expensive.                                                                                                
**b)** 1x1 is not even a proper convolution, as we can, instead of convolving each pixel separately, multiply the whole channel with          just 1 number                                                                                                                        
**c)** 1x1 is merging the pre-existing feature extractors, creating new ones, keeping in mind that those features are found together (like edges/gradients which make up an eye)                                                                                              
**d** 1x1 is performing a weighted sum of the channels, so it can so happen that it decides not to pick a particular feature which defines the background and not a part of the object.

![1x1 conv](1x1conv.png)

### 10. Position of Transition Layer
Transition layers are placed between convolution blocks. Only quality required features are selected from previous convolution block and feed to next convolution block.

### 11. The distance of MaxPooling from Prediction
The position of maxpooling is a way from prediction layer to avoid loss of high quality features learned.
### 12. SoftMax
Softmax is not probability, but probability like! It's used in last prediction layer to maximize distance between different classes score and represent in terms of numbers. All class values prediction score sum to 1. Generally in real life critical situation softmax is not prefered. Its just to make us happy to see prediction score. :)

![softmax](https://github.com/avadhutc/Session4/blob/master/images/softmax.png)
### 13. Learning Rate
Learning rate or step size is a hyper-parameter that controls how much we are adjusting the weights of our network with respect the loss gradient. See effects of differnt learning rates.

![lr](lr.png)
### 14. Batch Size, and effects of batch size
We can't feed the entire dataset into the neural net at once. So, we divide dataset into Number of Batches or sets or parts.
Batchsize should not be either too small or too high. It should be optimum, to have all datapoints variances in each batch. Selection of batchsize also depends on GPU capacity. See how training and test accuracies vary with different batchsizes.
![batchsize](batchsize.png)
### 15. When to add validation checks
We should have validation check per epoch. So that we can have entire record of validation checks.

### 16. Number of Epochs and when to increase them
We can start with some x number of epochs. Understanding overfitting or underfitting scenarios we can either reduce or increase the number of epochs.

### 17. How do we know our network is not going well, comparatively, very early
Once training begins, by observing either training and validation accuracy or training and validation loss in few epochs, we can find there is no improvements in training, validation accuarcy/loss or both.

### 18. Batch Normalization
We normalize the input layer by adjusting and scaling the activations. For example, when we have features from 0 to 1 and some from 1 to 1000, we should normalize them to speed up learning. How about hidden layers. Batch normalization reduces the amount by what the hidden unit values shift around (covariance shift). To explain covariance shift, let’s have a deep network on cat detection. We train our data on only black cats’ images. So, if we now try to apply this network to data with colored cats, it is obvious; we’re not going to do well. The training set and the prediction set are both cats’ images but they differ a little bit. In other words, if an algorithm learned some X to Y mapping, and if the distribution of X changes, then we might need to retrain the learning algorithm by trying to align the distribution of X with the distribution of Y.
![bn1](batchnorm1.JPG)
![bn2](batchnorm2.JPG)
### 19. The distance of Batch Normalization from Prediction
We don't use Batch Normalization close to Prediction. Because we want pass on information network learned to prediction layer without any changes.
### 20. DropOut
Dropout generally used when gap between training and validation increasing. To reduce that gap dropout is helpful. It gives regularization effect by randomly turning off few neurons.

![dropout](dropout.gif)
![dropout1](dropout1.JPG)
![dropout2](dropouteffect.JPG)
### 21. When do we introduce DropOut, or when do we know we have some overfitting
When training accuracy improving, but not the validation accuracy. So gap between them is increasing, then we falls into overfitting situation.
### 22. LR schedule and concept behind it
Optimizer objective is to converge faster. Some independent variables(features) have strong correlation and other have weak correlation 
with dependent variable(output). So different learning rates are required for different features to converge faster. With LR schedule we can adjust learning rate at different scenarios and achive faster convergence.
![lrscheduler](lrscheduler.png)

### 23. Adam vs SGD
Stochastic Gradient Descent (SGD),is a variant of gradient descent. Instead of performing computations on the whole dataset — which is redundant and inefficient — SGD only computes on a small subset or random selection of data examples. SGD produces the same performance as regular gradient descent when the learning rate is low. SGD has trouble navigating ravines, i.e. areas where the surface curves much more steeply in one dimension than in another, which are common around local optima. In these scenarios, SGD oscillates across the slopes of the ravine while only making hesitant progress along the bottom towards the local optimum.
Adaptive Moment Estimation (Adam),is an algorithm for gradient-based optimization of stochastic objective functions. It combines the advantages of two SGD extensions — Root Mean Square Propagation (RMSProp) and Adaptive Gradient Algorithm (AdaGrad) — and computes individual adaptive learning rates for different parameters. Check different optimizers convergence path

![adamvssgd](sgd1.gif)
![adamvssgd1](sgd2.gif)

On MNIST dataset performance of different optimizers.
![adamvssgd3](mnistadamsgd.JPG)

[Reference: Differentially private optimization algorithms for deep neural networks](https://ieeexplore.ieee.org/document/8355063/figures#figures)
### 24. When do we stop convolutions and go ahead with a larger kernel or some other   	alternative (which we have not yet covered)
When we form object in the image or reach the receptive field of object at prediction layer. Now we don't want to relearn features with convolutions, then we simply use larger kernels or Global Average Pooling(GAP) to adjust loud features into number of classes, before feeding for prediction.
