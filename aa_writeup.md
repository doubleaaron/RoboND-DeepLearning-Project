[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project ##

[image_0]: ./docs/misc/sim_screenshot.png
![alt text][image_0] 


The Udacity Robotics Software Engineer Nanodegree :: Deep Learning Project is a Follow Me Quadcopter Drone Project. It utilizes a Fully Connected Convolutional Neural Network (FCN) in Tensorflow and Keras to build a model that identifies, targets and tracks a person from a Simulation Drone Camera feed built in Unity3D. The simulated Drone must acquire and follow a target person while ignoring other people that are randomly spawned around the target person.

Below is a video of how the final target tracking runs in the QuadSim simulator. It has an average IoU of 42%.

[![Follow Me!](./images/youtube_screen.jpg)](https://www.youtube.com/watch?v=LM8i6oglozw)

### Architecture ###

The model has to be able to segment out objects within a live video stream which means that every pixel in the still frame image needs to have a label. Semantic Segmentation is a technique used with a Fully Convolutional Network to achieve this result. At the end of the process every pixel will be colored in one of the segmentation colors.


[image_5]: ./images/FCN.png
![alt text][image_5]
Image Credit: http://cvlab.postech.ac.kr/research/deconvnet/


#### Encoder ####
Convnets learn from local 2D windows of information to get small patterns on the inputs. These patterns are translation independent. Once it learns a pattern it can recognize it anywhere. They can also learn spatial hierarchies of patterns as well. One layer can learn edges, the next can learn larger patterns built from the first layer and so on. In this way convnets can increasingly learn more complex visual concepts. They operate over 3D Tensors which are called feature maps with height, width and depth (HxWxD). Depth is also called a Channel Axis. For RGB images the Depth channel is 3 for Red, Green and Blue.

A convolution works by sliding a window of size 3x3, 5x5, etc. over the 3D feature map, stopping at each location and extracting features (HxWxD). Each window is transformed via a tensor into a 1D vector of shape(output_depth).

This Encoding is then followed by a decoding process that is a transposed encoding process (or deconvolution) with an upsampling process. This is also called a fractionally strided convolution. This operation goes in the opposite direction to a convolution and translates the activations into meaningful information that scales up the activation to the same image size.

I trained my FCN locally with tensorflow-gpu on a Quadro M1200 and the speed were adequate enough to get an acceptable result without going to the AWS instance with Tesla K80. It required the batch size to not be too high and get an out of memory error but still yieled an acceptable result. When I have more time I will try to get a higher final result and IoU with a faster workstation.

Batch size was a balancing act as higher sizes could overflow CPU and GPU cache and writing to system RAM which slows the process down.

Number of epochs increases model accuracy but after 10 the accuracy didn't improve as much.

I found a good balance in the steps per epoch and validation steps in 200 and 50. If it was doubled the training time doubled but marginal increases in accuracy were achieved. If it was 100 and 25 (/2) there was a huge decrease in accuracy.

Here are my training epochs from these Hyperparameters:

```python
learning_rate = 0.005
batch_size = 30
num_epochs = 10
steps_per_epoch = 200
validation_steps = 50
workers = 2
```

[image_2]: ./images/sem_seg_epochs_01.jpg
![alt text][image_2]

[image_3]: ./images/sem_seg_epochs_02.jpg
![alt text][image_3]

[image_4]: ./images/sem_seg_epochs_03.jpg
![alt text][image_4]


### Prediction ###

Once the model is trained it's time to make Predictions. There are three types of images available from the validation set:

```
patrol_with_targ: Test how well the network can detect the hero from a distance.
patrol_non_targ: Test how often the network makes a mistake and identifies the wrong person as the target.
following_images: Test how well the network can identify the target while following them.
```

#### following_images ####

[image_6]: ./images/following_01.png
![alt text][image_6]

[image_7]: ./images/following_02.png
![alt text][image_7]

[image_8]: ./images/following_03.png
![alt text][image_8]

#### patrol_with_targ ####

[image_9]: ./images/following_withtarget_01.png
![alt text][image_9]

[image_10]: ./images/following_withtarget_02.png
![alt text][image_10]

[image_11]: ./images/following_withtarget_03.png
![alt text][image_11]

#### patrol_non_targ ####

[image_12]: ./images/following_notarget_01.png
![alt text][image_12]

[image_13]: ./images/following_notarget_02.png
![alt text][image_13]

[image_14]: ./images/following_notarget_03.png
![alt text][image_14]


### Evaluation ###

The final score of my model was 0.424, and the final IoU without the target was 0.550.

[image_15]: ./images/evaluation.jpg
![alt text][image_15]

### Future Enhancements ###

A few methods could be utilized to improve the final score:

1. Increase the resolution of the images: Increasing the resolution of the images would help especially in cases where the target is far away. This would significantly increase the training time and performance so it may not be ideal for using in a real time situation like a drone where the compute power per watt is an issue.

2. Increase the number of images in the dataset: Increasing the number of images in the dataset would likely help, but there could be a tradeoff between under and overfitting. This would need to be tested by taking more images from the drone for training and validation.

3. Increase the batch size in the hyperparameters for each single pass: I didn't have enough memory to increase the Batch Size per pass, but it would probably help in accuracy up to a point. This would also need to be tested on a machine with more power.

Additional enhancements: For this example, the model was trained on humans but it could be repurposed to recognize things like vehicles or animals. It wouldn't work very well if you just fed it new images, the model would need to be trained and validated on images from the ground up with a large, good dataset such as CIFAR-10.
