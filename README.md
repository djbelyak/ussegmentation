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
pip install -r requirements.txt
pip install -e .
```

## Usage

Before usage you need to obtain pre-trained models:

```sh
ussegmentation get models
```

To run empty inference on a video with preview:

```sh
ussegmentation inference empty --input-file 'datasets\copter\DJI_0007(1).mp4' --output-file 'datasets\copter\empty_inference.mp4'
```

To run empty inference on an image without preview:

```sh
ussegmentation inference empty --input-type image --input-file 'datasets\2019\frame_200.png' --output-file 'datasets\1.png' --no-show
```

To obtain needed datasets execute:

```sh
ussegmentation get datasets
```

## Acknowledgments

The research was supported by the Russian Foundation for Basic Research (Grant No. 18-47-400003)
