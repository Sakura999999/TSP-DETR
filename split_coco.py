import json
import random

def split_coco_json(full_json_path, train_json_path, val_json_path, split_ratio=0.9):
    print(f"正在读取完整的 JSON: {full_json_path}")
    with open(full_json_path, 'r') as f:
        data = json.load(f)
        
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    
    # 1. 随机打乱图片
    random.seed(42) # 固定随机种子，保证每次切分结果一样
    random.shuffle(images)
    
    # 2. 计算切分点
    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # 3. 提取图片对应的 ID 集合
    train_img_ids = set(img['id'] for img in train_images)
    val_img_ids = set(img['id'] for img in val_images)
    
    # 4. 把标注框分发给对应的集合
    train_anns = [ann for ann in annotations if ann['image_id'] in train_img_ids]
    val_anns = [ann for ann in annotations if ann['image_id'] in val_img_ids]
    
    # 5. 保存 Train JSON
    train_data = {'images': train_images, 'annotations': train_anns, 'categories': categories}
    with open(train_json_path, 'w') as f:
        json.dump(train_data, f)
    print(f"✅ 生成训练集: {len(train_images)} 张图, {len(train_anns)} 个框 -> {train_json_path}")
        
    # 6. 保存 Val JSON
    val_data = {'images': val_images, 'annotations': val_anns, 'categories': categories}
    with open(val_json_path, 'w') as f:
        json.dump(val_data, f)
    print(f"✅ 生成验证集: {len(val_images)} 张图, {len(val_anns)} 个框 -> {val_json_path}")

if __name__ == '__main__':
    # 你的完整 JSON 路径
    full_json = './data/dota/annotations/instances_train2017.json'
    # 临时改个名当作备份
    import os
    backup_json = full_json + '.bak'
    if not os.path.exists(backup_json):
        os.rename(full_json, backup_json)
        
    train_out = './data/dota/annotations/instances_train2017.json'
    val_out = './data/dota/annotations/instances_val2017.json'
    
    split_coco_json(backup_json, train_out, val_out, split_ratio=0.9)