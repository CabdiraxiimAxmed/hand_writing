import numpy as np
from tkinter import *
import time
import math
from PIL import Image
import random
from predictions import make_predictions, get_accuracy, make_prediction
from get_dataset import x_train, y_train, x_dev, y_dev
from get_test_dataset import x_test, y_test
from image_process import process_image

parameters = np.load('parameters.npy', allow_pickle = True).tolist()


SEEDS = 1
root = Tk()

text = Text(root, height=2, width=30)
text.pack()

root.geometry('500x500')
canvas = Canvas(root, bg = 'white', height = '400', width = '500')

def paint(event):
    x = event.x
    y = event.y
    canvas = event.widget
    for i in range(SEEDS):
        canvas.create_line(x, y, x+6, y+6, fill='black', width = 4)

def save_image(canvs, fileName):
    print("creating image")
    canvs.postscript(file = fileName + '.eps')
    img = Image.open(fileName + '.eps')
    print('saving image')
    img.save(fileName + '.jpg', 'jpeg')

def get_image(preprocessed_digits):
    for i, digit in enumerate(preprocessed_digits):
        if i == len(preprocessed_digits) - 1:
            image = digit.reshape(28*28, 1)
    return image
def submit():
    save_image(canvas, 'my_image')
    time.sleep(10)
    print("image processing .....")
    preprocessed_digits = process_image()
    image = get_image(preprocessed_digits)
    print(image.T)
    result = make_prediction(image, parameters)
    text.delete(1.0, END)
    text.insert(END, f'Numberka waa {result}')
def reset_canvas():
    canvas.delete('all')
submit = Button(root, text = 'submit', command = submit)
delete = Button(root, text = 'reset', command = reset_canvas)

root.bind('<B1-Motion>', paint)

canvas.pack()
submit.pack()
delete.pack()
root.mainloop()
# test_predictions = make_predictions(x_test, parameters)
# accuracy = get_accuracy(test_predictions, y_test)
# print(accuracy)
