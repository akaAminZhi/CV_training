import cv2

# 读取图像
image = cv2.imread("detect_images/lsb/full_floor_plan/level1_0001.png")

# 转为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("detect_images/lsb/full_floor_plan/level1_0001_out.png", gray)

# 设定灰色的范围，假设灰色值在180~220之间（你可以调整）
lower_gray = 100
upper_gray = 250

# 创建灰色掩码
mask = cv2.inRange(gray, lower_gray, upper_gray)

# 将掩码区域设为黑色
image[mask > 0] = [255, 255, 255]

# 显示和保存结果
# cv2.imshow("Result", image)
cv2.imwrite("detect_images/lsb/full_floor_plan/level1_0001_out2.png", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
