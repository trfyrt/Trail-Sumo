import cv2
import os

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

marker_size = 200
marker_id_0 = 0
marker_image_0 = cv2.aruco.generateImageMarker(aruco_dict, marker_id_0, marker_size)

marker_id_1 = 1
marker_image_1 = cv2.aruco.generateImageMarker(aruco_dict, marker_id_1, marker_size)

# Simpan sebagai file
cv2.imwrite(os.path.join('aruco', 'aruco_robot_0.png'), marker_image_0)
cv2.imwrite(os.path.join('aruco', 'aruco_robot_1.png'), marker_image_1)