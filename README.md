# id-matching

A face recognition application that takes a photo and checks for a match in a folder of identity documents.

## Usage

```bash
py idMatch.py <input_image> <documents_folder>
```
## Examples

```bash
py .\idMatch.py .\Test_Data\sample2.png .\Test_Data\documentfolder\
.\Test_Data\sample2.png  not matched in  .\Test_Data\documentfolder\
```
```bash
py .\idMatch.py .\Test_Data\sample1.png .\Test_Data\documentfolder\
.\Test_Data\sample1.png  matched  .\Test_Data\documentfolder\document1.png  in  .\Test_Data\documentfolder\
```

## Requirements

Use the package manager [pip](https://pypi.org/) to install the following packages.

```bash
pip install tensorflow
```
```bash
pip install Keras
```
```bash
pip install opencv-python
```
```bash
pip install mtcnn
```
```bash
pip install matplotlib
```
```bash
pip install keras_vggface
```

## Description

The executable named idMatch accepts as parameters:
1. The path to the input image.
2. A path to a folder containing identity documents. 
 
The user can type in a command line “py idMatch <input_image> <documents_folder>” and receive as output:
e.g. “photo1 matched document1 in folder1” if the code matched photo1 to document1 in folder1 or “photo1 not matched in folder1”.


## References

1. [MTCNN face detection implementation for TensorFlow](https://github.com/ipazc/mtcnn)
2. [VGGFace implementation with Keras](https://github.com/rcmalli/keras-vggface)