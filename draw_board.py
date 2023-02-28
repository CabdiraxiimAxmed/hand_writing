from tkinter import *
from PIL import Image
import math
import random

RADIUS = 10
SEEDS = 1
root = Tk()
root.geometry('500x500')
canvas = Canvas(root, bg = 'white', height = '400', width = '500')
def paint(event):
    x = event.x
    y = event.y
    canvas = event.widget
    for i in range(SEEDS):
        canvas.create_line(x, y, x+6, y+6, fill='black', width = 1)

def save_image(canvs, fileName):
    canvs.postscript(file = fileName + '.eps')
    img = Image.open(fileName + '.eps')
    img.save(fileName + '.jpg', 'jpeg')

def submit():
    save_image(canvas, 'my_image')

button = Button(root, text = 'submit', command = submit)

root.bind('<B1-Motion>', paint)

canvas.pack()
button.pack()
root.mainloop()
