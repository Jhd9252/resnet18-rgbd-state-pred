A dump repo for a project on state prediction of a robot hand. Specifically the tips of each finger. 

The images are given as RGBD (RGB + depth) images, and the data is structured such that 

- trainX.pt
	- Comprises of (RGB images, depth images, file ids)
	- Each RGB image has the dimensions (num_data_samples, num_camera_views, num_channels_height, width)
	- Each depth image has (num data samples, num camera views, height, width)

- trainY.pt
	- consists of (num data samples, 12) where 12 corresponds to the (x, y, z) coordinates of each of the four fingertips

- testX.pt

The process involves lazy loading the dataset from persistent memory. Each epoch of training, each batch loaded reference paths are used to pull the images. 

The model used was PyTorch’s implementation of ResNet-18 with pre-trained weights.

ReduceLROnPlateau scheduler for the learning rate was used to step according to loss for each training epoch.
