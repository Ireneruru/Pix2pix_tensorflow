#  Pix2pix_tensorflow
Implement Image-to-Image Translation with Conditional Adversarial Networks   https://github.com/phillipi/pix2pix on tensorflow.

## Setup

### Prerequisites
- Python with numpy
- NVIDIA GPU + CUDA 8.0 + CuDNNv5.1
- TensorFlow 0.12

### Getting Started
- Download the dataset (script borrowed from [torch code](https://github.com/phillipi/pix2pix/blob/master/datasets/download_dataset.sh)):
```bash
bash ./download_dataset.sh facades
```
- Train the model
```bash
python main.py train.py
```
## Results
 ---to be updated

## Train
Current code supports [CMP Facades](http://cmp.felk.cvut.cz/~tylecr1/facade/) dataset.




