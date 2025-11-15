import os
import shutil

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def move_files_recursive(src_folder, dst_folder_json, dst_folder_jpg):
    for root, _, files in os.walk(src_folder):
        for fname in files:
            src_path = os.path.join(root, fname)
            if fname.endswith('.json'):
                shutil.copy2(src_path, os.path.join(dst_folder_json, fname))
            elif fname.endswith('.jpg') or fname.endswith('.jpeg') or fname.endswith('.png'):
                shutil.copy2(src_path, os.path.join(dst_folder_jpg, fname))

def main():
    base_src = 'train_valid'
    base_dst = 'dataset'

    folders = ['train_json', 'train_jpg', 'val_json', 'val_jpg']
    for folder in folders:
        ensure_dir(os.path.join(base_dst, folder))

    move_files_recursive(
        os.path.join(base_src, 'train'),
        os.path.join(base_dst, 'train_json'),
        os.path.join(base_dst, 'train_jpg')
    )

    move_files_recursive(
        os.path.join(base_src, 'valid'),
        os.path.join(base_dst, 'val_json'),
        os.path.join(base_dst, 'val_jpg')
    )

    print("모든 파일을 dataset/에 정리 완료!")

if __name__ == '__main__':
    main()
