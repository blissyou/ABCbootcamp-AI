import cv2

# 이미지 읽기
image_1 = cv2.imread('./Free_Image_1.jpg')
image_2 = cv2.imread('./Free_Image_2.jpg')

# 두 이미지의 크기를 동일하게 맞춤
height, width = image_1.shape[:2]
image_2_resized = cv2.resize(image_2, (width, height))

def changing_weight_value(x: int) -> None:
    weight = x / 100
    merged_image = cv2.addWeighted(src1=image_1, alpha=1-weight, src2=image_2_resized, beta=weight, gamma=0)
    cv2.imshow(winname="Display", mat=merged_image)
    return None

# 윈도우 생성 및 트랙바 추가
cv2.namedWindow(winname='Display')
cv2.createTrackbar('weight', 'Display', 0, 100, changing_weight_value)

# 트랙바의 기본 값에 해당하는 이미지 출력
changing_weight_value(0)

# 키 입력 대기
cv2.waitKey(delay=0)
cv2.destroyAllWindows()
