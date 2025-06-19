import cv2
import os

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

marker_size = 400
marker_id_0 = 0
marker_image_0 = cv2.aruco.generateImageMarker(aruco_dict, marker_id_0, marker_size)

marker_id_1 = 1
marker_image_1 = cv2.aruco.generateImageMarker(aruco_dict, marker_id_1, marker_size)

marker_id_2 = 2
marker_image_2 = cv2.aruco.generateImageMarker(aruco_dict, marker_id_2, marker_size)

marker_id_3 = 3
marker_image_3 = cv2.aruco.generateImageMarker(aruco_dict, marker_id_3, marker_size)

marker_id_4 = 4
marker_image_4 = cv2.aruco.generateImageMarker(aruco_dict, marker_id_4, marker_size)

marker_id_5 = 5
marker_image_5 = cv2.aruco.generateImageMarker(aruco_dict, marker_id_5, marker_size)

marker_id_6 = 6
marker_image_6 = cv2.aruco.generateImageMarker(aruco_dict, marker_id_6, marker_size)

marker_id_7 = 7
marker_image_7 = cv2.aruco.generateImageMarker(aruco_dict, marker_id_7, marker_size)

marker_id_8 = 8
marker_image_8 = cv2.aruco.generateImageMarker(aruco_dict, marker_id_8, marker_size)

marker_id_9 = 9
marker_image_9 = cv2.aruco.generateImageMarker(aruco_dict, marker_id_9, marker_size)

# Simpan sebagai file
cv2.imwrite(os.path.join('aruco', 'aruco_robot_0.png'), marker_image_0)
cv2.imwrite(os.path.join('aruco', 'aruco_robot_1.png'), marker_image_1)
cv2.imwrite(os.path.join('aruco', 'aruco_robot_2.png'), marker_image_2)
cv2.imwrite(os.path.join('aruco', 'aruco_robot_3.png'), marker_image_3)
cv2.imwrite(os.path.join('aruco', 'aruco_robot_4.png'), marker_image_4)
cv2.imwrite(os.path.join('aruco', 'aruco_robot_5.png'), marker_image_5)
cv2.imwrite(os.path.join('aruco', 'aruco_robot_6.png'), marker_image_6)
cv2.imwrite(os.path.join('aruco', 'aruco_robot_7.png'), marker_image_7)
cv2.imwrite(os.path.join('aruco', 'aruco_robot_8.png'), marker_image_8)
cv2.imwrite(os.path.join('aruco', 'aruco_robot_9.png'), marker_image_9)