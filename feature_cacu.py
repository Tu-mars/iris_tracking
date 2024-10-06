import numpy as np
import math
import cv2

class feature_calculate:

    def calculate_iris_center(iris_landmarks, image_width, image_height):
                    x_coords = [landmark.x * image_width for landmark in iris_landmarks]  # 转换为像素坐标
                    y_coords = [landmark.y * image_height for landmark in iris_landmarks]
                    center_x = int(sum(x_coords) / len(x_coords))  # 取x的平均值
                    center_y = int(sum(y_coords) / len(y_coords))  # 取y的平均值
                    return center_x, center_y

    def rotation_matrix_to_angles(rotation_matrix):
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return np.array([x, y, z]) * 180. / math.pi

    def calculate_eye_mid(inner_x, inner_y, outer_x, outer_y):
        mid_x = (inner_x + outer_x) / 2
        mid_y = (inner_y + outer_y) / 2
        return mid_x, mid_y

    def calculate_vector_to_what(x1, y1, x2, y2):
        vector_x = x1- x2
        vector_y = y1 - y2
        return vector_x, vector_y

    def calculate_angle_between_vectors(x1, y1, x2, y2):
        angle_vector = np.degrees(np.arctan2(y1, x1) - np.arctan2(y2, x2))
        angle_vector = (angle_vector + 360) % 360
        return angle_vector
    def calculate_distance(x1, y1, x2, y2):
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return distance
    def ratio_calculate_distance(x1, y1, x2, y2, normal):
        normalize = normal / 2
        distance = (np.sqrt((x1 - x2)**2 + (y1 - y2)**2)) / normalize
        return distance

    def draw_results_on_image(image, rg_distance, lf_distance, rg_angle, lf_angle):
        # 顯示角度
        # 顯示距離
        cv2.putText(image, f"Right iris Distance : {rg_distance:.2f}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(image, f"Left iris Distance : {lf_distance:.2f}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        cv2.putText(image, f"Right Eye Angle: {rg_angle:.2f}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(image, f"Left Eye Angle: {lf_angle:.2f}", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)