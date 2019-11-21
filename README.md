# Underlying surfaces segmentation

> A deep learning-based implementation of an underlying surfaces segmentation algorithm

## Installation

First of all, make sure you have all pre requirements:

- Git
- Python

Then, execute the next commands in your Git Bash:

```sh
git clone https://github.com/djbelyak/ussegmentation.git
cd ussegmentation
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install -e .
```

## Usage

Before usage you need to obtain pre-trained models:

```sh
ussegmentation get models
```

To run empty inference on a video with preview:

```sh
ussegmentation inference empty --model-file 'data\models\empty_copter.pth' --input-file 'data\datasets\copter\video.mp4' --output-file 'data\datasets\copter\empty_inference.mp4'
```

To run empty inference on an image without preview:

```sh
ussegmentation inference empty --input-type image --input-file 'data\datasets\2019\frame_200.png' --output-file 'data\datasets\1.png' --no-show
```

To obtain needed datasets execute:

```sh
ussegmentation get datasets
```

To run training on copter dataset:

```sh
ussegmentation train empty --dataset copter --model-file 'data\models\empty_copter.pth'
```

To run training on cityscapes dataset:

```sh
ussegmentation train empty --dataset cityscapes --model-file 'data\models\empty_cityscapes.pth'
```

## Acknowledgments

The research was supported by the Russian Foundation for Basic Research (Grant No. 18-47-400003)
