import os
import glob
import math
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import itertools
from feature_cacu import feature_calculate as fc

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

input_folder = r'C:\coding\eye_tracking\videoset\ADHD\26228945謝采玹'
output_folder = r'C:\coding\eye_tracking\dataset\detection_data\Control'
output_folder2 = r'C:\coding\eye_tracking\dataset\feature_data\Control'
video_extensions = ['*.mp4', '*.mkv']

input_file_list = []
for extension in video_extensions:
    input_file_list.extend(glob.glob(os.path.join(input_folder, '**', extension), recursive=True))
print("找到的文件:", input_file_list)

for index, input_file in enumerate(input_file_list):
    filename = os.path.basename(input_file)
    output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + '.xlsx')
    output_file2 = os.path.join(output_folder2, os.path.splitext(filename)[0] + '_feature.xlsx')
    cap = cv2.VideoCapture(input_file)
    
    LEFT_EYE_INNER_CORNER_INDEX = 463
    RIGHT_EYE_INNER_CORNER_INDEX = 243
    LEFT_EYE_OUTER_CORNER_INDEX = 263
    RIGHT_EYE_OUTER_CORNER_INDEX = 130
    
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    
    LEFT_BLINK = []
    RIGHT_BLINK = []
    BLINK_COUNT = 0
    LABEL = []
    RIGHT_IRIS = []
    LEFT_IRIS = []
    LEFT_EYE_INNER_CORNERS = []
    RIGHT_EYE_INNER_CORNERS = []
    LEFT_EYE_OUTER_CORNER = []
    RIGHT_EYE_OUTER_CORNER = []
    DISTANCE = []
    IRIS_ANGLE = []
    PITCH = []
    YAW = []
    ROLL = []


    with mp_face_mesh.FaceMesh(
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.85, 
        min_tracking_confidence=0.85) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            else:    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                imgHeight = image.shape[0]
                imgWidth = image.shape[1]
                results = face_mesh.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                face_coordination_in_real_world = np.array([
                    [285, 528, 200],
                    [285, 371, 152],
                    [202, 574, 128],
                    [173, 425, 108],
                    [355, 574, 128],
                    [391, 425, 108]
                ], dtype=np.float64)
                
                blank_image = np.zeros((imgHeight, imgWidth, 3), dtype=np.uint8)
                face_coordination_in_image = []
                
                if results.multi_face_landmarks:
                    iris_xy = []  
                    for face_landmarks in results.multi_face_landmarks:
                            for idx, lm in enumerate(face_landmarks.landmark):
                                if idx in [1, 9, 57, 130, 287, 359]:
                                    x, y = int(lm.x * imgWidth), int(lm.y * imgHeight)
                                    face_coordination_in_image.append([x, y])
                                    
                            
                            face_coordination_in_image = np.array(face_coordination_in_image,
                                                        dtype=np.float64)
                            focal_length = 1 * imgWidth
                            cam_matrix = np.array([[imgWidth, 0, imgWidth / 2],
                                        [0, imgHeight, imgHeight / 2],
                                        [0, 0, 1]])
                            # The Distance Matrix
                            dist_matrix = np.zeros((4, 1), dtype=np.float64)
                            # Use solvePnP function to get rotation vector
                            success, rotation_vec, transition_vec = cv2.solvePnP(
                                face_coordination_in_real_world, face_coordination_in_image,
                                cam_matrix, dist_matrix)
                            # Use Rodrigues function to convert rotation vector to matrix
                            rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)

                            result = fc.rotation_matrix_to_angles(rotation_matrix)

                            for i, info in enumerate(zip(('pitch', 'yaw', 'roll'), result)):
                                k, v = info
                                text = f'{k}: {int(v)}'
                                cv2.putText(image, text, (20, i*30 + 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
                            pitch,yaw,roll = result
                            
                             
                            PITCH.append(pitch)
                            YAW.append(yaw)
                            ROLL.append(roll)
                            
                            #左內眼角座標
                            left_eye_inner_corner_x = int(face_landmarks.landmark[LEFT_EYE_INNER_CORNER_INDEX].x * imgWidth)
                            left_eye_inner_corner_y = int(face_landmarks.landmark[LEFT_EYE_INNER_CORNER_INDEX].y * imgHeight - 2)
                            # left_eye_inner_corner_visibility = float(face_landmarks.landmark[LEFT_EYE_INNER_CORNER_INDEX].visibility)
                            LEFT_EYE_INNER_CORNERS.append([left_eye_inner_corner_x, left_eye_inner_corner_y])
                
                            #捕捉右眼內眼角坐標
                            right_eye_inner_corner_x = int(face_landmarks.landmark[RIGHT_EYE_INNER_CORNER_INDEX].x * imgWidth)
                            right_eye_inner_corner_y = int(face_landmarks.landmark[RIGHT_EYE_INNER_CORNER_INDEX].y * imgHeight - 2)
                            # right_eye_inner_corner_visibility = float(face_landmarks.landmark[RIGHT_EYE_INNER_CORNER_INDEX].visibility)
                            RIGHT_EYE_INNER_CORNERS.append([right_eye_inner_corner_x, right_eye_inner_corner_y])
                            
                            
                            #捕捉左外眼角座標
                            left_eye_outer_corner_x = int(face_landmarks.landmark[LEFT_EYE_OUTER_CORNER_INDEX].x * imgWidth)
                            left_eye_outer_corner_y = int(face_landmarks.landmark[LEFT_EYE_OUTER_CORNER_INDEX].y * imgHeight)
                            #left_eye_outer_corner_visibility = float(face_landmarks.landmark[LEFT_EYE_OUTER_CORNER_INDEX].visibility)
                            LEFT_EYE_OUTER_CORNER.append([left_eye_outer_corner_x, left_eye_outer_corner_y])
                            
                            #捕捉右外眼座標
                            right_eye_outer_corner_x = int(face_landmarks.landmark[RIGHT_EYE_OUTER_CORNER_INDEX].x * imgWidth)
                            right_eye_outer_corner_y = int(face_landmarks.landmark[RIGHT_EYE_OUTER_CORNER_INDEX].y * imgHeight)
                            # right_eye_outer_corner_visibility = float(face_landmarks.landmark[RIGHT_EYE_OUTER_CORNER_INDEX].visibility)
                            RIGHT_EYE_OUTER_CORNER.append([right_eye_outer_corner_x, right_eye_outer_corner_y])
                            
                            right_iris_landmarks = {
                                "right_iris": [face_landmarks.landmark[i] for i in [469, 470, 471, 472]],  
                            }
                            for iris, landmarks in right_iris_landmarks.items():
                                RIGHT_IRIS_X, RIGHT_IRIS_Y = fc.calculate_iris_center(landmarks, imgWidth, imgHeight)
                                #cv2.circle(image, (RIGHT_IRIS_X, RIGHT_IRIS_Y), 1, (0, 0, 255), -1)
                                RIGHT_IRIS.append([RIGHT_IRIS_X, RIGHT_IRIS_Y])

                            left_iris_landmarks = {
                                "left_iris": [face_landmarks.landmark[i] for i in [474, 475, 476, 477]]
                                
                            }
                            for iris, landmarks in left_iris_landmarks.items():
                                LEFT_IRIS_X, LEFT_IRIS_Y = fc.calculate_iris_center(landmarks, imgWidth, imgHeight)
                                #cv2.circle(image, (LEFT_IRIS_X, LEFT_IRIS_Y), 1, (0, 0, 255), -1)
                                LEFT_IRIS.append([LEFT_IRIS_X, LEFT_IRIS_Y])
                            left_blink = fc.detect_blink(face_landmarks.landmark, LEFT_EYE_INDICES, imgWidth, imgHeight)
                            right_blink = fc.detect_blink(face_landmarks.landmark, RIGHT_EYE_INDICES, imgWidth, imgHeight)

                            LEFT_BLINK.append(left_blink)
                            RIGHT_BLINK.append(right_blink)
                            
                                
                    cv2.circle(image,(int(left_eye_inner_corner_x), int(left_eye_inner_corner_y)), 2, (0, 0, 255), cv2.FILLED) #left_eye_corner
                    
                    cv2.circle(image,(int(right_eye_inner_corner_x), int(right_eye_inner_corner_y)), 2, (0, 0, 255), cv2.FILLED) #left_eye_corner
                    cv2.circle(image,(int(left_eye_outer_corner_x), int(left_eye_outer_corner_y)), 2, (0, 0, 255), cv2.FILLED) #left_eye_outer
                    cv2.circle(image,(int(right_eye_outer_corner_x), int(right_eye_outer_corner_y)), 2, (0, 0, 255), cv2.FILLED) #left_eye_outer
                    
                    # 計算右眼的中點
                    right_mid_x, right_mid_y = fc.calculate_eye_mid(right_eye_inner_corner_x, right_eye_inner_corner_y, 
                                                                 right_eye_outer_corner_x, right_eye_outer_corner_y)
                    
                    #cv2.circle(image, (int(right_mid_x), int(right_mid_y)), 1, (255, 0, 0), cv2.FILLED)
                    right_top_left = (int(right_mid_x - 3), int(right_mid_y - 4))  # 方框左上角
                    right_bottom_right = (int(right_mid_x + 5), int(right_mid_y + 1))  # 方框右下角
                    #cv2.rectangle(image, right_top_left, right_bottom_right, (0, 255, 0), 1)  # 绿色方框
                    
                    # 計算左眼的中點
                    left_mid_x, left_mid_y = fc.calculate_eye_mid(left_eye_inner_corner_x, left_eye_inner_corner_y, 
                                                               left_eye_outer_corner_x, left_eye_outer_corner_y)
                    #cv2.circle(image, (int(left_mid_x), int(left_mid_y)), 1, (255, 0, 0), cv2.FILLED)

                    left_top_left = (int(left_mid_x - 5), int(left_mid_y - 4))  # 方框左上角
                    left_bottom_right = (int(left_mid_x + 5), int(left_mid_y + 1))  # 方框右下角
                    #cv2.rectangle(image, left_top_left, left_bottom_right, (0, 255, 0), 1)  # 绿色方框

                    rg_v_mid_in_x, rg_v_mid_in_y = fc.calculate_vector_to_what(
                        right_eye_inner_corner_x, right_eye_inner_corner_y, 
                        right_mid_x, right_mid_y)
                    lf_v_mid_in_x, lf_v_mid_in_y = fc.calculate_vector_to_what(
                        left_eye_inner_corner_x, left_eye_inner_corner_y,
                        left_mid_x, left_mid_y)
                    
                    #vector mid to iris
                    rg_v_mid_iris_x, rg_v_mid_iris_y = fc.calculate_vector_to_what(
                        RIGHT_IRIS_X, RIGHT_IRIS_Y,
                        right_mid_x, right_mid_y)
                    lf_v_mid_iris_x, lf_v_mid_iris_y = fc.calculate_vector_to_what(
                        LEFT_IRIS_X, LEFT_IRIS_Y,
                        left_mid_x, left_mid_y )
                    
                    #vector mid to inner angle ##
                    rg_angle_vector_inner = fc.calculate_angle_between_vectors(rg_v_mid_iris_x, rg_v_mid_iris_y, 
                                                                            rg_v_mid_in_x, rg_v_mid_in_y )
                    lf_angle_vector_inner = fc.calculate_angle_between_vectors(lf_v_mid_iris_x, lf_v_mid_iris_y,
                                                                             lf_v_mid_in_x, lf_v_mid_in_y )
                    
                    IRIS_ANGLE.append([rg_angle_vector_inner, lf_angle_vector_inner])
                    # distance inner to outer 
                    rg_distance_inner_outer = fc.calculate_distance(right_eye_inner_corner_x, right_eye_inner_corner_y, right_eye_outer_corner_x, right_eye_outer_corner_y)
                    lf_distance_inner_outer = fc.calculate_distance(left_eye_inner_corner_x, left_eye_inner_corner_y, left_eye_outer_corner_x, left_eye_outer_corner_y)

                    #distance mid to iris ##
                    rg_distance_mid_iris = fc.ratio_calculate_distance(right_mid_x, right_mid_y, RIGHT_IRIS_X, RIGHT_IRIS_Y, rg_distance_inner_outer)
                    lf_distance_mid_iris = fc.ratio_calculate_distance(left_mid_x, left_mid_y, LEFT_IRIS_X, LEFT_IRIS_Y, lf_distance_inner_outer)
                    DISTANCE.append([rg_distance_mid_iris, lf_distance_mid_iris])

                    #put_text_to_image
                    fc.draw_results_on_image(image,rg_distance_mid_iris, lf_distance_mid_iris,
                                          rg_angle_vector_inner, lf_angle_vector_inner)
                    
                    if left_blink and right_blink:
                        LABEL.append(3)
                        cv2.putText(image, "Blink Detected", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)    
                    else:
                        if left_top_left[0] <= LEFT_IRIS_X <= left_bottom_right[0] and left_top_left[1] <= LEFT_IRIS_Y <= left_bottom_right[1] and right_top_left[0] <= RIGHT_IRIS_X <= right_bottom_right[0] and right_top_left[1] <= RIGHT_IRIS_Y <= right_bottom_right[1]:
                            key = cv2.waitKey(0) & 0xFF  # 等待手动按键
                            if key == ord('1'):
                                LABEL.append(1)  # 手动标记为 "focus"
                            if key == ord('2'):
                                LABEL.append(2)  #  "non_focus"
                            if key == ord('3'):
                                LABEL.append(3)  # "idk"
                        else:
                            LABEL.append(2) #non_focus

                else:

                    RIGHT_IRIS_X = ""
                    RIGHT_IRIS_Y = ""
                    LEFT_IRIS_X = ""
                    LEFT_IRIS_Y = ""
                    nose_x = ""
                    nose_y = ""
                    left_eye_inner_corner_x = ""
                    left_eye_inner_corner_y = ""
                    right_eye_inner_corner_x = ""
                    right_eye_inner_corner_y = ""
                    left_eye_outer_corner_x = ""
                    left_eye_outer_corner_y = ""
                    right_eye_outer_corner_x = ""
                    right_eye_outer_corner_y = ""

                    pitch = "" 
                    yaw = ""
                    roll = ""
                    
                    PITCH.append(pitch)
                    YAW.append(yaw)
                    ROLL.append(roll)
                    
                    RIGHT_IRIS.append([RIGHT_IRIS_X, RIGHT_IRIS_Y])
                    LEFT_IRIS.append ([LEFT_IRIS_X,  LEFT_IRIS_Y])
                    LABEL.append(None)
                    IRIS_ANGLE.append([None, None])
                    DISTANCE.append([None, None])
                    
                    LEFT_EYE_INNER_CORNERS.append([left_eye_inner_corner_x, left_eye_inner_corner_y])
                    RIGHT_EYE_INNER_CORNERS.append([right_eye_inner_corner_x, right_eye_inner_corner_y])

                    LEFT_EYE_OUTER_CORNER.append([left_eye_outer_corner_x, left_eye_outer_corner_y])
                    RIGHT_EYE_OUTER_CORNER.append([right_eye_outer_corner_x, right_eye_outer_corner_y ])

            cv2.namedWindow('MediaPipe Face Mesh', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('MediaPipe Face Mesh', 1920, 1080) 
            cv2.imshow('MediaPipe Face Mesh',image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Space bar key for pause control
                break

        cap.release()
        cv2.destroyAllWindows()
# 获取所有数组的最小长度
    min_length = min(len(RIGHT_IRIS), len(LEFT_IRIS), len(LEFT_EYE_INNER_CORNERS), 
                    len(RIGHT_EYE_INNER_CORNERS), len(LEFT_EYE_OUTER_CORNER), 
                    len(RIGHT_EYE_OUTER_CORNER), len(PITCH), len(YAW), len(ROLL), 
                    len(LABEL), len(DISTANCE), len(IRIS_ANGLE))
                    

    # 截取到最小的长度以确保可以拼接
    RIGHT_IRIS = RIGHT_IRIS[:min_length]
    LEFT_IRIS = LEFT_IRIS[:min_length]
    LEFT_EYE_INNER_CORNERS = LEFT_EYE_INNER_CORNERS[:min_length]
    RIGHT_EYE_INNER_CORNERS = RIGHT_EYE_INNER_CORNERS[:min_length]
    LEFT_EYE_OUTER_CORNER = LEFT_EYE_OUTER_CORNER[:min_length]
    RIGHT_EYE_OUTER_CORNER = RIGHT_EYE_OUTER_CORNER[:min_length]
    PITCH = PITCH[:min_length]
    YAW = YAW[:min_length]
    ROLL = ROLL[:min_length]
    LABEL = LABEL[:min_length]
    DISTANCE = DISTANCE[:min_length]
    IRIS_ANGLE = IRIS_ANGLE[:min_length]
    LEFT_BLINK = LEFT_BLINK[:min_length]
    RIGHT_BLINK = RIGHT_BLINK[:min_length]

    IRIS_Data = np.hstack((np.array(RIGHT_IRIS), np.array(LEFT_IRIS),
                           np.array(LEFT_EYE_INNER_CORNERS), np.array(RIGHT_EYE_INNER_CORNERS),
                           np.array(LEFT_EYE_OUTER_CORNER), np.array(RIGHT_EYE_OUTER_CORNER),
                           np.array(PITCH)[:, np.newaxis], np.array(YAW)[:, np.newaxis], np.array(ROLL)[:, np.newaxis]))
                           
    feature_data = np.hstack((np.array(DISTANCE), np.array(IRIS_ANGLE), np.array(PITCH)[:, np.newaxis], np.array(YAW)[:, np.newaxis], np.array(ROLL)[:, np.newaxis],
                           np.array(LABEL)[:, np.newaxis]
                           ))   
    df = pd.DataFrame(IRIS_Data, columns=['RIGHT_IRIS_X', 'RIGHT_IRIS_Y', 'LEFT_IRIS_X', 'LEFT_IRIS_Y',
                                          'LEFT_EYE_INNER_X', 'LEFT_EYE_INNER_Y',
                                          'RIGHT_EYE_INNER_X', 'RIGHT_EYE_INNER_Y',
                                          'LEFT_EYE_OUTER_X', 'LEFT_EYE_OUTER_Y',
                                          'RIGHT_EYE_OUTER_X', 'RIGHT_EYE_OUTER_Y',
                                          'pitch', 'yaw', 'roll'
                                          ])
    df['LEFT_BLINK'] = LEFT_BLINK
    df['RIGHT_BLINK'] = RIGHT_BLINK
    df['BLINK_COUNT'] = BLINK_COUNT
    
    df2 = pd.DataFrame(feature_data, columns=['right_distance','left_distance', 
                                              'right_angle','left_angle', 
                                              'pitch', 'yaw', 'roll', 
                                              'Label'
                                              ])
    
    df.to_excel(output_file, index=False)
    df2.to_excel(output_file2, index=False)