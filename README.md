# A Semi Automatic Image Annotation Tool

Semi Automatic Image Annotation Toolbox for groud detection. 

## Installation

1) Clone this repository.

2) In the repository, execute `pip install -r requirements.txt`.
   Note that due to inconsistencies with how `tensorflow` should be installed,
   this package does not define a dependency on `tensorflow` as it will try to install that (which at least on Arch Linux results in an incorrect installation).
   Please make sure `tensorflow` is installed as per your systems requirements.
   Also, make sure Keras 2.1.3 or higher and OpenCV 3.x is installed.

### Dependencies

1) Tensorflow >= 1.7.0

2) OpenCV = 3.x

3) Keras >= 2.1.3

For, Python >= 3.5

### Instructions

1) Select the pedestrians' ID for which you need suggestions add draw a box at the position you want.

2) When annotating manually, select the pedestrians who are in a group, "add" it in to data file.

3) The final annotations can be found in the file `annotations.csv` in ./annotations/

### Usage
```
python main.py
```

Tested on:

Windows 10

### Acknowledgments

1) [Meditab Software Inc.](https://www.meditab.com/)

2) [Keras implementation of RetinaNet object detection](https://github.com/fizyr/keras-retinanet)

3) [Computer Vision Group](https://cvgldce.github.io/), L.D. College of Engineering

4) [virajmavani](https://github.com/virajmavani/semi-auto-image-annotation-tool)
