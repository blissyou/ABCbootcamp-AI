import cv2

color_image = cv2.imread('Free_Image_12.jpg', cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)


def changing_threshold(x: int) -> None:
    (_, threshold_image) = cv2.threshold(src=gray_image,
                                         thresh=x,
                                         maxval=255,
                                         type=cv2.THRESH_BINARY)
    cv2.imshow(winname='THRESHOLD Image', mat=threshold_image)
    return None


cv2.namedWindow('THRESHOLD Image')
cv2.createTrackbar('THRESHOLD', 'THRESHOLD Image', 0, 255, changing_threshold)

# cv2.imshow('Threshold',gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
