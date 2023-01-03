import cv2 as cv
import numpy as np

blank = np.zeros((500,500), dtype="uint8")

rect = cv.rectangle(blank.copy(), (50,50),(450,450), 225, -1)
circle = cv.circle(blank.copy(), (250,250), 250, 225, -1)

cv.imshow("rect", rect)
cv.imshow("circle", circle)

# bit wise OR
bitwise_or = cv.bitwise_or(rect, circle)
cv.imshow("OR", bitwise_or)

# bit wise AND
bitwise_and = cv.bitwise_and(rect, circle)
cv.imshow("AND", bitwise_and)

# bit wise Xor 
bitwise_xor = cv.bitwise_xor(rect, circle)
cv.imshow("xor", bitwise_xor)

# bit wise not
bitwise_not = cv.bitwise_not(rect)
cv.imshow("not", bitwise_not)



cv.waitKey(00)