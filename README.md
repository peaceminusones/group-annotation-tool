# A Semi Automatic Image Annotation Tool

Semi Automatic Image Annotation Toolbox for groud detection. 

## Installation

1) Clone this repository.

2) In the repository, execute `pip install -r requirements.txt`.
   Also, make sure OpenCV 3.x is installed.

### Dependencies

1) OpenCV = 3.x

2) Python >= 3.5

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
