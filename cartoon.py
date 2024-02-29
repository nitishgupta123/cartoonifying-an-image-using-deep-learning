import cv2

import matplotlib.pyplot as plt

image=cv2.imread('/content/c2.jpg',cv2.IMREAD_UNCHANGED)

im=cv2.resize(image,(400,600))

plt.imshow(image)
plt.show()


from google.colab.patches import cv2_imshow

cv2_imshow(image)

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

gray=cv2.blur(gray,(1,1))

edges=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)

plt.imshow(edges,cmap="gray")

im=cv2.resize(edges,(400,500))

cv2_imshow(edges)

color=cv2.bilateralFilter(image,9,300,250)

cartoon=cv2.bitwise_and(color,color,mask=edges)

im=cv2.resize(cartoon,(400,500))

cv2_imshow(cartoon)