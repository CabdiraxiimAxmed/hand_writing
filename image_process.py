import numpy as np
import cv2
import matplotlib.pyplot as plt

def process_image():
    img = cv2.imread('my_image.jpg')
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # NOTE: values_changed: 125 -> 75, THRESH_BINARY -> THRESH_BINARY_INV
    ret,thresh = cv2.threshold(grayscale,75,255,cv2.THRESH_BINARY_INV) # image, lower, max, type
    # NOTE: values_changed: RETR_TREE -> RERT_EXTERNAL
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # image, mode, method
    cv2.drawContours(img, contours, -1, (0,255,0), 3) # image, contour, start, color, thickness.
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    preprocessed_digits = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # creating rectangle around the digit.
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness = 2)
        # croping out the digit from the image corresponding to the current contours in the for loop
        digit = thresh[y:y+h, x:x+h] # NOTE: try removing y and x
        # resize the digit to (18, 18)
        resized_digit = cv2.resize(digit, (18, 18))
        #padding the digit with 5 pixel of black color (zeros)
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values = 0)
        preprocessed_digits.append(padded_digit)
    return preprocessed_digits

# digits = process_image()
# for i, digit in enumerate(digits):
#     if i == len(digits) - 1:
#         plt.imshow(digit, 'gray')
#         plt.show()
#         image = digit.reshape(28*28, 1)





