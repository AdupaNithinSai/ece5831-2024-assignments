# Rock-Paper-Scissors with Teachable Machine
This project Repo implements a Rock-Paper-Scissors game using a machine learning model trained with Google's Teachable Machine. The model can classify images of rock, paper, and scissors and provides predictions through a Python script. Additionally, a live webcam version is available to play the game interactively.

## Requirements
To run this project, you need the following:
- Python 3.x
- OpenCV
- NumPy
- TensorFlow
- Matplotlib

You can install the required packages using:
```bash
pip install opencv-python numpy tensorflow matplotlib
```

## Usage
### Classifying a Single Image
To classify an image of rock, paper, or scissors, use the following command:
```bash
python rock-paper-scissors.py --image path/to/your/image.jpg
```

### Example
```bash
python rock-paper-scissors.py --image paper.jpg
```
The output will display the class of the image and the confidence score.

### Real-time Classification with Webcam
To use your webcam for real-time classification, run:
```bash
python rock-paper-scissors-live.py
```

## Screen Recording
A demonstration video of the live classification can be found on my YouTube channel: [YouTube Link](https://youtu.be/GMAd2sv_FR8)

You can also watch the video here:
<iframe width="560" height="315" src="https://www.youtube.com/embed/GMAd2sv_FR8?si=Ruma8aNSS6IgTUwi" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>