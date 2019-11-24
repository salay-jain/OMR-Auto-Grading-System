# OMR-Auto-Grading-System

The major goal of our project is to find the result of the candidate by matching their response with the ideal responses of the questions provided.

Team Members : 
Salay Jain
Harsh Sharma

Working:

1) Download the zip file of the repo or clone the repo from the link : https://github.com/salay-jain/OMR-Auto-Grading-System

2) Some of the python3 libraries required are : numpy, matplotlib, PIL, imutils and argparse. Make sure to run these commands witout any error : -\
	a) import numpy -\
	b) import matplotlib.pyplot -\
	c) from PIL import Image -\
	d) import cv2 -\
	e) import imutils -\
	f) import argparse -\

3) Run the command: python3 main.py --inp1 Path to template image --inp2 Path to ideal answerkey --inp3 Path to the to be tested OMR sheet. For default you can use i1.jpeg i2.jpeg and test.jpeg in respective order.

4) The program will give outputs as : 
	a) Points Matching for the Ideal answer sheet
	b) Points Matching for the Students answer sheet
	c) Checked OMR - Right Answers by Green & Wrong answers by Red and unresponded are left blank
	d) Total result

5) To see our other works and observations refer to the python notebook (.ipynb) file where our algoritm is analysed for many diiferet types of input and output.

6) We have created our own dataset and it can be downloaded from the link : abcd. So to analyse our code on new images just specify the path as argument to the main.py file.
While in our .ipynb file we have use the structuring like all our images are kept in the Src folder and we can acces them from there.

7) Hope we are able to solve your problem of OMR grading and made the solution to this problem easier and cheap that can be used by small instituions/coaching centers etc. easily with efficient outputs.
