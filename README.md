# computerVisionFootball

## Intro
Project uses object detection model & processes it's output to analyze the video of a football match & detect/calculate:
* player, referee and ball positions & track each object
* estimate camera movement
* estimate player speed

## Input
Input video is 08fd33_4.mp4

## Processing the input
Main script where all the code is executed is football_object_detection.ipynb, all other .py scripts contain classes so the code is better organized.

## Fine tuned model
Model used to do initial object detection is a model trained in my other repo https://github.com/bcicvaric/computerVisionModelTuning

## Output
Main script will create an output file in outputs/proc_08fd33_4.mp4

## Reference
https://www.youtube.com/watch?v=neBZ6huolkg&t=79s
