import cv2
original_image = cv2.imread('Free_Image_10.jpg')
image_gray = cv2.imread(filename= 'Free_Image_10.jpg'
                        ,flags=cv2.IMREAD_GRAYSCALE)
edge_image = cv2.adaptiveThreshold(src = image_gray,maxValue=255,
                                   adaptiveMethod= cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresholdType= cv2.THRESH_BINARY,
                                   blockSize= 9,
                                   C= 0) # c는 -5 ~5
cv2.imshow(winname='Original',mat= original_image)
cv2.imshow(winname='EDGE',mat=edge_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
