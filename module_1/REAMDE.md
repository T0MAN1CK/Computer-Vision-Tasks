# Pan-Sharpening Project
## build docker image

stand in directory where you see module_1/ and run:
`docker build -t pan-sharpen -f module_1/Dockerfile .`

## run docker container

`docker run --rm -v "${PWD}/module_1:/app/module_1" pan-sharpen`

Output will be: /pan_sharpened_outputs/

## Requirements (for manual local run)

Python 3.10+

opencv-python

numpy

pre-commit

ruff

## Rund

`python main.py`
