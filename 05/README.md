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
A demonstration video of the live classification can be found on my YouTube channel: 
######### Video recorded with code in rock-paper-scissors-live.py: [YouTube Link](https://youtu.be/GMAd2sv_FR8)
Video recorded with third party app:[![Rock-Paper-Scissors Demo](https://img.youtube.com/vi/GMAd2sv_FR8/0.jpg)](https://youtu.be/GMAd2sv_FR8)

