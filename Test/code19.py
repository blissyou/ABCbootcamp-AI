import cv2
img = cv2.imread(filename='/Users/choewonhyeong/Desktop/my_fucking_project/ABCbootcamp/AItest/Test/datas/Lenna_512x512.png',
                 flags=cv2.IMREAD_GRAYSCALE)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(src=img)
print(minVal, maxVal, minLoc, maxLoc)
dst = cv2.normalize(src=img,dst=None,alpha=100,beta=200,norm_type=cv2.NORM_MINMAX)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(src=dst)
print(minVal, maxVal, minLoc, maxLoc)
cv2.imshow('LENNA',mat=img)
cv2.imshow('NORM',mat=dst)
cv2.waitKey()
cv2.destroyAllWindows()
