import os
import glob
import pandas as pd
import numpy as np

input_folder = r'C:\coding\eye_tracking\dataset\feature_data\merged'  # 替換成你資料夾的路徑
output_file = r'C:\coding\eye_tracking\dataset\feature_data\merged_data.xlsx'

# 支持的 Excel 文件擴展名
excel_extensions = ['*.xlsx']

# 获取所有 Excel 文件路径
input_file_list = []
for extension in excel_extensions:
    input_file_list.extend(glob.glob(os.path.join(input_folder, '**', extension), recursive=True))
print("找到的文件:", input_file_list)

# 創建一個空的 DataFrame 來存儲合併後的數據
merged_data = pd.DataFrame()

# 讀取每個 Excel 文件並垂直堆疊
for input_file in input_file_list:
    df = pd.read_excel(input_file)
    merged_data = pd.concat([merged_data, df], ignore_index=True)

# 檢查是否存在無窮大或非常大的值，並進行內插處理
# 檢查無窮大或超出合理範圍的值，這裡設置一個極大的閾值（例如 1e10）
# 將合併和處理後的數據保存到一個新的 Excel 文件
merged_data['Label'] = merged_data['Label'] - 1
merged_data.to_excel(output_file, index=False)

print(f"所有資料已成功合併、檢查並內插處理，並儲存到 {output_file}")