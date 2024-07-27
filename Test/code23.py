import numpy as np
import cv2
original_image = cv2.imread(filename='Free_Image_6.jpg', flags=cv2.IMREAD_COLOR)
kernel1 = np.ones(shape=(3,3), dtype=np.float32) / 9
kernel2 = np.ones(shape=(9,9), dtype=np.float32) / 81
# 필터(weighted value)
average_image_3_by_3=cv2.filter2D(src=original_image,ddepth=-1,kernel=kernel1)
average_image_9_by_9 = cv2.filter2D(src=original_image, ddepth=-1, kernel=kernel2)
cv2.imshow(winname="Original Image", mat=original_image)
cv2.imshow(winname="3by3 filtered", mat=average_image_3_by_3)
cv2.imshow(winname="9by9 filtered", mat=average_image_9_by_9)
cv2.waitKey(delay=0)
cv2.destroyAllWindows()
