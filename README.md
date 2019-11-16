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

To run a default model run:

```sh
ussegmentation inference
```

To obtain needed datasets execute:

```sh
ussegmentation get datasets
```

## Acknowledgments

The research was supported by the Russian Foundation for Basic Research (Grant No. 18-47-400003)
