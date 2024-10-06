import pandas as pd
import numpy as np
import glob
import os
from scipy.ndimage import median_filter

group_label = 'Non_focus' #Control ADHD
input_folder = r'C:\coding\eye_tracking\dataset\feature_data\\' + group_label
output_folder = r'C:\coding\eye_tracking\dataset\filter_data\\'+ group_label
# 支持的视频文件扩展名

merged_file = r'C:\coding\eye_tracking\dataset\feature_data\\' + group_label + '_merged_data.xlsx'
merged_data = pd.DataFrame()

def check_and_interpolate(data):
    # 將無窮大或超過閾值的數據標記為 NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 只對數字型的數據進行處理
    data = data.apply(pd.to_numeric, errors='coerce')  # 將非數字的值轉換為 NaN

    # 將極端值設為 NaN
    data[data > 1e10] = np.nan
    data[data < -1e10] = np.nan

    # 使用內插法補全 NaN 值，按列進行線性插值
    return data.interpolate(method='linear', axis=0, limit_direction='both')

def convert_angle_to_sin_cos(angle_deg):
    # 將角度轉換為弧度
    angle_rad = np.deg2rad(angle_deg)
    # 計算正弦和餘弦值
    sin_val = np.sin(angle_rad)
    cos_val = np.cos(angle_rad)
    return sin_val, cos_val

video_extensions = ['*xlsx']
# 获取所有视频文件路径
input_file_list = []
for extension in video_extensions:
    input_file_list.extend(glob.glob(os.path.join(input_folder, '**', extension), recursive=True))
print("找到的文件:", input_file_list)

for input_file in input_file_list :
    filename = os.path.basename(input_file)
    file_id = os.path.splitext(filename)[0]
    output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + '_filter.xlsx')
    # 直接drop空直
    df = pd.read_excel(input_file)
    df_cleaned = df.dropna() #先計算 空值特徵
    pitch_column = 'pitch'
    yaw_column = 'yaw'
    roll_column = 'roll'

    # 設定每個數據的門檻值範圍 (根據實際需求調整)
    pitch_threshold_lower = -40
    pitch_threshold_upper = 40
    yaw_threshold_lower = -70
    yaw_threshold_upper = 70
    roll_threshold_lower = -90
    roll_threshold_upper = 90

    #### 3軸 filter   
    # 設定各個數據列的名稱


    # 偵測異常值
    df_cleaned['is_pitch_anomaly'] = (df_cleaned[pitch_column] < pitch_threshold_lower) | (df_cleaned[pitch_column] > pitch_threshold_upper)
    df_cleaned['is_yaw_anomaly'] = (df_cleaned[yaw_column] < yaw_threshold_lower) | (df_cleaned[yaw_column] > yaw_threshold_upper)
    df_cleaned['is_roll_anomaly'] = (df_cleaned[roll_column] < roll_threshold_lower) | (df_cleaned[roll_column] > roll_threshold_upper)

    # 使用內插法來替換 Pitch 的異常值
    df_cleaned[pitch_column] = df_cleaned[pitch_column].mask(df_cleaned['is_pitch_anomaly']).interpolate(method='linear')

    # 使用內插法來替換 Yaw 的異常值
    df_cleaned[yaw_column] = df_cleaned[yaw_column].mask(df_cleaned['is_yaw_anomaly']).interpolate(method='linear')

    # 使用內插法來替換 Roll 的異常值
    df_cleaned[roll_column] = df_cleaned[roll_column].mask(df_cleaned['is_roll_anomaly']).interpolate(method='linear')

    # 刪除標記異常值的欄位
    df_cleaned = df_cleaned.drop(columns=['is_pitch_anomaly', 'is_yaw_anomaly', 'is_roll_anomaly'])
    
    df_cleaned['rg_sin_val'], df_cleaned['rg_cos_val'] = zip(*df_cleaned['right_angle'].apply(convert_angle_to_sin_cos))
    df_cleaned['lf_sin_val'], df_cleaned['lf_cos_val'] = zip(*df_cleaned['left_angle'].apply(convert_angle_to_sin_cos))



    df_cdata = df_cleaned[['right_distance', 'left_distance', 
                        'rg_sin_val', 'rg_cos_val',
                        'lf_sin_val', 'lf_cos_val', 
                        'pitch', 'yaw', 'roll', 
                        'Label'
                        ]]
    df_cdata = check_and_interpolate(df_cdata)
    
    df_cdata['ID'] = file_id  # 加入文件ID
    df_cdata['Group'] = group_label  # 加入群組標籤

    #df_cdata.to_excel(output_file, index=False)
    filtered_data = df_cdata[df_cdata['Label'].isin([1, 2])]
    #Save the updated dataframe with the distance converted to 0 or 1 to a new excel file

    filtered_data.to_excel(output_file, index=False)
    if 'ID' in filtered_data.columns and 'Group' in filtered_data.columns:
    # 合併資料，並保留所有欄位
        merged_data = pd.concat([merged_data, filtered_data], ignore_index=True)
    else:
        print("ID 或 GroupLabel 欄位不存在於 filtered_data 中")

    print(f"已經存在: {output_file}")


merged_data.to_excel(merged_file, index=False)
print(f"所有資料已成功合併、檢查並內插處理，並儲存到 {merged_file}")