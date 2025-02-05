import os
import shutil
from sklearn.model_selection import train_test_split

class_1_path = 'C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Alluvial soil'
class_2_path = 'C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils'
class_3_path = 'C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil'

# class_4_path = 'C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Loamy soil'
# class_5_path = 'C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Sandy loam'
# class_6_path = 'C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Sandy soil'


train_path = 'C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\train'
valid_path = 'C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\valid'

for class_path in [train_path, valid_path]:
    # for soil_class in ['Alluvial Soil', 'Clayey Soil', 'Laterite Soil', 'Loamy Soil', 'Sandy loam', 'Sandy soil']:
    for soil_class in ['Alluvial Soil', 'Clayey Soil', 'Laterite Soil']:
        os.makedirs(os.path.join(class_path, soil_class), exist_ok=True)

split_ratio = 0.8

def copy_files(file_list, src_path, dest_path):
    for file_name in file_list:
        src_file = os.path.join(src_path, file_name)
        dest_file = os.path.join(dest_path, file_name)
        shutil.copy(src_file, dest_file)

train_class_1, valid_class_1 = train_test_split(os.listdir(class_1_path), train_size=split_ratio)
train_class_2, valid_class_2 = train_test_split(os.listdir(class_2_path), train_size=split_ratio)
train_class_3, valid_class_3 = train_test_split(os.listdir(class_3_path), train_size=split_ratio)

# train_class_4, valid_class_4 = train_test_split(os.listdir(class_4_path), train_size=split_ratio)
# train_class_5, valid_class_5 = train_test_split(os.listdir(class_5_path), train_size=split_ratio)
# train_class_6, valid_class_6 = train_test_split(os.listdir(class_6_path), train_size=split_ratio)

copy_files(train_class_1, class_1_path, os.path.join(train_path, 'Alluvial Soil'))
copy_files(valid_class_1, class_1_path, os.path.join(valid_path, 'Alluvial Soil'))

copy_files(train_class_2, class_2_path, os.path.join(train_path, 'Clayey Soil'))
copy_files(valid_class_2, class_2_path, os.path.join(valid_path, 'Clayey Soil'))

copy_files(train_class_3, class_3_path, os.path.join(train_path, 'Laterite Soil'))
copy_files(valid_class_3, class_3_path, os.path.join(valid_path, 'Laterite Soil'))

# copy_files(train_class_4, class_4_path, os.path.join(train_path, 'Loamy Soil'))
# copy_files(valid_class_4, class_4_path, os.path.join(valid_path, 'Loamy Soil'))

# copy_files(train_class_5, class_5_path, os.path.join(train_path, 'Sandy loam'))
# copy_files(valid_class_5, class_5_path, os.path.join(valid_path, 'Sandy loam'))

# copy_files(train_class_6, class_6_path, os.path.join(train_path, 'Sandy soil'))
# copy_files(valid_class_6, class_6_path, os.path.join(valid_path, 'Sandy soil'))