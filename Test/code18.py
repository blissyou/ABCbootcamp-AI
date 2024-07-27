import cv2

img = cv2.imread(
    filename='/datas/Rose.jpeg'
    , flags=cv2.IMREAD_GRAYSCALE)
# cv2.imshow(winname = 'Lenna',mat = img)
# cv2.waitKey()
# cv2.destroyAllWindows()

"""
fiter = [[-1,-1,0]],
        [-1 ,0 ,1 ],  -> 매트릭스 형태에서 -1 과 1에 차이를 찾아내겠다 라는 뜻
        [0, -1, -1]]
"""
import numpy as np
kernel = np.array((-1,-1,0,-1,0,1,0,1,1))
dst = cv2.filter2D(src=img ,ddepth= -1,kernel = kernel)
cv2.imshow(winname='FILTER',mat=dst)
cv2.waitKey()
cv2.destroyAllWindows()
