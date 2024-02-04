# Photo Colorization Project
This project aims to colorize black & white photos using a convolutional neural network (CNN) and a StreamLit application. The input of the neural network will be black & white photos and the output will be colorized photos.

## Project Overview
The project consists of the following steps:

- Data collection and preprocessing: We will use a dataset of face images from GitHub and our own photos to train the network. We will convert the images to LAB color space and split them into L (lightness) and AB (color) channels.
- Model building and training: We will use Keras and TensorFlow to build and train a CNN model that takes the L channel as input and predicts the AB channel as output. We will use various layers such as Conv2D, UpSampling2D, BatchNormalization, Activation, etc. to construct the network architecture.
- Model evaluation and testing: We will use skimage to convert the predicted AB channel and the original L channel back to RGB color space and save the colorized images. We will also compare the colorized images with the original images and evaluate the model performance using metrics such as mean squared error (MSE) and peak signal-to-noise ratio (PSNR).
- Application development and deployment: We will use StreamLit to create a web application that allows users to upload their own black & white photos and see the colorized results. We will also deploy the application to a cloud platform such as Heroku or AWS.

## Project Requirements
To run this project, you will need the following software and libraries:

- Python 3.7 or higher
- Keras 2.6.0 or higher
- TensorFlow 2.6.0 or higher
- StreamLit 1.2.0 or higher
- skimage 0.18.3 or higher
- numpy 1.21.2 or higher
- matplotlib 3.4.3 or higher
- PIL 8.3.2 or higher

## Usage
- Import and preprocess black and white photos using the code provided in the colorization_app.ipynb file.
- Build the model architecture using the code provided in the  file.
- Train the model using the code provided in the colorization_app.ipynb file.
- Test the model on new black and white photos using the code provided in the colorization_app.ipynb file.
- Use the StreamLit library to develop an application for colorizing black and white photos.

## License
This project is licensed under the MIT License.

## Used Software and Libraries (and Hardware if it exists)
- From `keras.layers` we will use:
  - `Conv2D`
  - `UpSampling2D`
  - `InputLayer`
  - `Conv2DTranspose`
  - `Activation`
  - `Dense`
  - `Dropout`
  - `Flatten`
- From `tensorflow.keras.layers` we will use:
  - `BatchNormalization`
- From `keras.models` we will use:
  - `Sequential`
- From `keras_preprocessing.image` we will use:
  - `ImageDataGenerator`
  - `img_to_array`
  - `array_to_img`
  - `load_img`
- From `skimage.color` we will use:
  - `rgb2lab`
  - `lab2rgb`
  - `rgb2gray`
  - `xyz2lab`
- From `skimage.io` we will use:
  - `Imsave`
- `Tensorflow`
- Note: These libraries might be changed in case we use GANs instead of convolution.

## Used Datasets
- https://github.com/2014mchidamb/DeepColorization/tree/master/face_images

