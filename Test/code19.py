import cv2
img = cv2.imread(filename='/datas/Lenna_512x512.png', flags=cv2.IMREAD_GRAYSCALE)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img)
print(minVal, maxVal, minLoc, maxLoc)
dst = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)

(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(dst)
print(minVal, maxVal, minLoc, maxLoc)
cv2.imshow('img',img)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
