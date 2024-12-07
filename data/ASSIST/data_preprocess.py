import pandas as pd
import numpy as np
import os
import csv
from tqdm import tqdm

"""
# Verify the data characteristics of the item cold start scenario
# 加载CSV文件
train_df = pd.read_csv('D:/GitRepository/IFedCD/data/CiteULike/train.csv')
test_df = pd.read_csv('D:/GitRepository/IFedCD/data/CiteULike/test.csv')

# 验证test.csv中的uid在train.csv中都出现过
uid_in_train = test_df['uid'].isin(train_df['uid']).all()
if uid_in_train:
    print("all the uid values of the data in test.csv have appeared in train.csv")
else:
    print("all the uid values of the data in test.csv haven't appeared in train.csv")

# 验证test.csv中的iid在train.csv中都没有出现过
iid_in_train = test_df['iid'].isin(train_df['iid']).any()
if not iid_in_train:
    print("all the iid values of the data in test.csv haven't appeared in train.csv")
else:
    print("all the iid values of the data in test.csv haven appeared in train.csv")
"""
    

def extract_columns(csv_file_path, output_file_path, columns, min_samples, sorted_attribute):
    
    df = pd.read_csv(csv_file_path)
    if not all(column in df.columns for column in columns):
        print("please check the column_name")
        return
    extracted_df = df[columns]
    
    # 找出样本数大于或等于min_samples的user_id
    user_id_counts = extracted_df['user_id'].value_counts()
    valid_user_ids = user_id_counts[user_id_counts >= min_samples].index
    filtered_df = extracted_df[extracted_df['user_id'].isin(valid_user_ids)]
    
    if sorted_attribute not in columns:
        print("please check the sorted_attribute")
        return
    
    sorted_df = filtered_df.sort_values(by=sorted_attribute, ascending=True)
    sorted_df['correct'] = sorted_df['correct'].apply(lambda x: 0 if x != 1 else x)
    sorted_df.to_csv(output_file_path, index=False) 

def get_data_info(data_path, info_path):
    
    df = pd.read_csv(data_path)
    
    unique_skill_ids = set()
    for skill_ids in df['skill_id']:
    # 如果skill_id是字符串，分割成单独的分量
        if isinstance(skill_ids, str):
            # 分割字符串并更新集合
            unique_skill_ids.update(skill_ids.split(','))
            # 如果skill_id已经是单个值，直接添加到集合
        else:
            unique_skill_ids.add(skill_ids)
            
    num_unique_skill_ids = len(unique_skill_ids)
    user_counts = df['user_id'].nunique()
    problem_counts = df['problem_id'].nunique()

    # 打印每个属性列的不同取值个数
    print(f"Total number of users: {user_counts}")
    print(f"Total number of problems: {problem_counts}")
    print(f"Total number of skill: {num_unique_skill_ids}")
    
    with open(info_path, 'w') as file:
        file.write(f"Total number of users: {user_counts}")
        file.write(f"Total number of problems: {problem_counts}")
        file.write(f"Total number of skill: {num_unique_skill_ids}")
        

def map_and_replace_id(input_file_path, output_file_path, map_matrix_path):
    # 读取CSV文件
    df = pd.read_csv(input_file_path)
    
    # 根据problem_id的值进行升序排列，并获取索引
    problem_id_sorted = df['problem_id'].unique()
    problem_id_sorted.sort()
    problem_id_to_index = {id: idx for idx, id in enumerate(problem_id_sorted)}
    # 替换原数据集中的problem_id为映射后的索引
    df['problem_id'] = df['problem_id'].map(problem_id_to_index)
    
    # 对skill_id同样操作
    # 存储所有skill_id的集合，用于排序和分配索引
    all_skill_ids = set()
    
    # 遍历每一行，分割skill_id并更新集合
    for skill_id in df['skill_id'].apply(lambda x: x.split(',')):
        all_skill_ids.update(skill_id)
    
    # 对所有skill_id进行排序，并分配索引
    sorted_skill_ids = sorted(all_skill_ids)
    skill_id_index = {skill_id: index for index, skill_id in enumerate(sorted_skill_ids)}
    
    def replace_with_index(skill_id):
        return skill_id_index.get(skill_id.strip(), -1)  # 如果skill_id不在索引中，返回-1
    
    # 逐行读取，并逐个替换skill_id的值
    df['skill_id'] = df['skill_id'].apply(lambda x: ','.join(map(str, map(replace_with_index, x.split(',')))))
    
    skill_id_index_list = [(v, k) for v, k in skill_id_index.items()]  # 将字典转换为列表，格式为(index, skill_id)
    np.save(map_matrix_path, skill_id_index_list)
    
    skill_id_map_matrix = list(sorted_skill_ids)
    np.save(map_matrix_path, skill_id_map_matrix) # 保存skill_id的映射关系
    df.to_csv(output_file_path, index=False)
        
def check_data(file_path):
    invalid_indices = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # 第一行为属性名，跳过
        for index, row in enumerate(reader):
            # 检查记录是否符合规范
            if not check_record(row):
                invalid_indices.append(index + 1)  # 保存行索引，加1是因为索引从0开始，而行号从1开始

    # 输出所有不合规记录的行索引
    if invalid_indices:
        print("irregular records:", invalid_indices)
    else:
        print("no irregular record")

def check_record(record):
    # 检查第一个属性是否为非负整数
    if not record[0].isdigit() or int(record[0]) < 0:
        return False
    
    # 检查第二个属性是否为[0, 6803]的正整数
    if not record[1].isdigit() or int(record[1]) < 0 or int(record[1]) > 6803:
        return False
    
    # 检查第三个属性是否为{0, 1}
    if record[2] not in ['0.0', '1.0']:
        return False
    
    # 检查第四个属性
    if ',' in record[3]:  # 多值情况
        values = record[3].split(',')
        for value in values:
            if not value.isdigit() or not 0 <= int(value) <= 199:
                return False
    else:  # 单值情况
        if not record[3].isdigit() or not 0 <= int(record[3]) <= 199:
            return False
    
    return True

def generate_and_save_feature_matrix(data_file_path, feature_file_path):
    df = pd.read_csv(data_file_path)
    
    problem_num = len(df['problem_id'].unique())
    skill_ids = []
    for skill_id in df['skill_id']:
        if ',' in skill_id:  
            skill_ids.extend(skill_id.split(','))  
        else:  
            skill_ids.append(skill_id)
    skill_num = len(set(skill_ids))
    print('problem num is:{}\n'.format(problem_num))
    print('skill num is:{}\n'.format(skill_num))
    
    feature_matrix = np.zeros((problem_num, skill_num), dtype=int)
    
    # 预处理，创建索引映射
    problem_id_to_index = {pid: i for i, pid in enumerate(df['problem_id'].unique())}
    
    # 使用tqdm显示进度条
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Processing'):
        i = problem_id_to_index[row['problem_id']]
        
        skill_idx = row['skill_id'].split(',') if ',' in row['skill_id'] else [row['skill_id']]
        for j in skill_idx:
            j = int(j)
            feature_matrix[i, j] = 1
    
    print('\nthe shape of item_feature:', feature_matrix.shape)
    
    np.save(feature_file_path, feature_matrix)
    print(f"Feature matrix saved as {feature_file_path}")


# generate_and_save_feature_matrix('D:/GitRepository/IFedCD/data/ASSIST/mapped_data.csv', 'D:/GitRepository/IFedCD/data/ASSIST/item_features.npy')

# check_data('D:/GitRepository/IFedCD/data/ASSIST/mapped_data.csv')

# 使用函数
#input_file_path = 'D:/GitRepository/IFedCD/data/ASSIST/extracted_data.csv'  # 替换为你的输入文件路径
#output_file_path = 'D:/GitRepository/IFedCD/data/ASSIST/mapped_data.csv'  # 替换为你的输出文件路径
#map_and_replace_id('D:/GitRepository/IFedCD/data/ASSIST/extracted_data.csv', 
 #                  'D:/GitRepository/IFedCD/data/ASSIST/mapped_data.csv', 
 #                  'D:/GitRepository/IFedCD/data/ASSIST/skill_mapped_matrix.npy') 

# 假设CSV文件路径为'data.csv'，输出文件路径为'binary_matrix.npy'
#csv_file_path = 'D:/GitRepository/IFedCD/data/ASSIST/extracted_data.csv'
#output_file_path = 'D:/GitRepository/IFedCD/data/ASSIST/item_features.npy'
#generate_and_save_binary_feature_matrix(csv_file_path, output_file_path)

# 假设.npy文件的路径为'binary_matrix.npy'
#npy_file_path = 'D:/GitRepository/IFedCD/data/ASSIST/item_features.npy'

# 加载.npy文件
#binary_matrix = np.load(npy_file_path)
#print('Matrix shape:', binary_matrix.shape)
# 计算矩阵中0和1的数量
#num_zeros = np.count_nonzero(binary_matrix == 0)
#num_ones = np.count_nonzero(binary_matrix == 1)
#print('Number of zeros:', num_zeros)
#print('Number of ones:', num_ones)

"""
orig_file_path = 'D:/GitRepository/IFedCD/data/ASSIST/ASSIST2009.csv'  
extract_file_path = 'D:/GitRepository/IFedCD/data/ASSIST/extracted_data.csv'  
columns = ['user_id', 'problem_id', 'correct', 'skill_id'] 
min_samples = 30
sorted_attribute = 'problem_id' 
extract_columns(orig_file_path, extract_file_path, columns, min_samples, sorted_attribute)

get_data_info('D:/GitRepository/IFedCD/data/ASSIST/extracted_data.csv', 'D:/GitRepository/IFedCDF/data/ASSIST/info.txt')
"""
