# Underlying surfaces segmentation

> A deep learning-based implementation of an underlying surfaces segmentation algorithm

## Installation

First of all, make sure you have all pre requirements:

- Git
- Python 3.7

Then, execute the next commands in your Git Bash:

```sh
git clone https://github.com/djbelyak/ussegmentation.git
cd ussegmentation
pip install -r requirements_cpu.txt
# or if CUDA available
pip install -r requirements_gpu.txt
pip install -e .
```

## Usage

Before usage you need to obtain pre-trained models:

```sh
ussegmentation get models
```

To run enet inference on a video from webcam:

```sh
ussegmentation inference enet --model-file 'data\models\enet_cityscapes.pth'
```

To run enet inference on a video with preview:

```sh
ussegmentation inference enet --model-file 'data\models\enet_cityscapes.pth' --input-file 'data\datasets\copter\video.mp4' --output-file 'data\datasets\copter\empty_inference.mp4'
```

To run enet inference on an image without preview:

```sh
ussegmentation inference enet --model-file 'data\models\enet_cityscapes.pth' --input-type image --input-file 'data\datasets\2019\frame_200.png' --output-file 'data\datasets\1.png' --no-show
```

To obtain needed datasets execute:

```sh
ussegmentation get datasets
```

To run training on copter dataset:

```sh
ussegmentation train enet --dataset copter --model-file 'data\models\enet_copter.pth'
```

To run training on cityscapes dataset:

```sh
ussegmentation train enet --dataset cityscapes --model-file 'data\models\enet_cityscapes.pth'
```

## Acknowledgments

Dataset provided by [Cityscapes Dataset](https://www.cityscapes-dataset.com/).

Enet architecture was proposed by [Adam Paszke and others](https://arxiv.org/abs/1606.02147).

Pytorch layout was developed by [David Silva](https://github.com/davidtvs/PyTorch-ENet).

The research was supported by the Russian Foundation for Basic Research (Grant No. 18-47-400003).
