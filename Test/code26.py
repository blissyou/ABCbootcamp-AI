import cv2
import numpy as np

front_image = cv2.imread("Front_Image.jpg",cv2.IMREAD_COLOR)
back_image = cv2.imread("Background_Image.jpg",cv2.IMREAD_COLOR)

# cv2.imshow("Front_Image",mat=front_image)
# cv2.imshow("Background_Image",mat=back_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

image_hsv = cv2.cvtColor(src = front_image,code = cv2.COLOR_BGR2HSV)
green_low_bound = np.array([40,100,50]) #녹색의 하한선
green_upper_bound = np.array([80,255,255]) # 녹색의 상한선

# mask image
mask_image = cv2.inRange(image_hsv, green_low_bound, green_upper_bound)
cv2.imshow("Image",mask_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

inverse_mask_image = cv2.bitwise_not(mask_image)
cv2.imshow('inverse mask image',inverse_mask_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 녹색픽셀들만 추출하기
extract_green_pixels = cv2.bitwise_and(src1=front_image,src2=front_image,mask=mask_image)
cv2.imshow('extract_green pixels',extract_green_pixels)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 녹색 아닌 픽셀만 추출하기
not_extract_green_pixels = cv2.bitwise_and(src1=front_image,src2=front_image,mask=inverse_mask_image)
cv2.imshow('extract_green pixels',not_extract_green_pixels)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 녹색과 겹치는 배경 추출하기
extract_green_background_image = cv2.bitwise_and(src1=back_image,src2=back_image,mask=mask_image)
cv2.imshow('extract_green background',extract_green_background_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 합치기
merge_image = cv2.bitwise_or(not_extract_green_pixels,extract_green_background_image)
cv2.imshow('add_image', merge_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
