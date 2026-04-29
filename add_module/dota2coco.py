import os
import json
from PIL import Image

# DOTA 1.0 的 15 个经典类别
DOTA_CLASSES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank',  'soccer-ball-field',
    'roundabout', 'harbor', 'swimming-pool', 'helicopter'
]

def dota2coco(dataset_dir, out_json_path):
    print(f"正在转换目录: {dataset_dir}")
    img_dir = os.path.join(dataset_dir, 'images')
    # label_dir = os.path.join(dataset_dir, 'annfiles')
    # 自动寻找标签文件夹
    label_dir = None
    for name in ['labelTxt', 'labels', 'annfiles']:
        temp_dir = os.path.join(dataset_dir, name)
        if os.path.exists(temp_dir):
            label_dir = temp_dir
            break
            
    if not label_dir:
        print(f"⚠️ 警告: 在 {dataset_dir} 中找不到任何标签文件夹！")
        return
        
    coco = {"images": [], "annotations": [], "categories": []}
    class_to_id = {cls: i + 1 for i, cls in enumerate(DOTA_CLASSES)}
    for cls, idx in class_to_id.items():
        coco["categories"].append({"id": idx, "name": cls, "supercategory": "none"})

    img_names = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    ann_id = 1
    
    for img_id, img_name in enumerate(img_names):
        # 1. 写入图片信息
        img_path = os.path.join(img_dir, img_name)
        with Image.open(img_path) as img:
            width, height = img.size
            
        coco["images"].append({
            "file_name": img_name, "id": img_id,
            "width": width, "height": height
        })

        # 2. 读取并写入对应的标签信息
        txt_name = img_name.replace('.png', '.txt')
        txt_path = os.path.join(label_dir, txt_name)
        
        if not os.path.exists(txt_path):
            continue  # 如果没有标签文件，说明是纯背景图，跳过标注
            
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9: continue # 跳过不合法或表头的行
            
            # DOTA 格式: x1 y1 x2 y2 x3 y3 x4 y4 classname diff
            try:
                coords = [float(x) for x in parts[:8]]
                cls_name = parts[8]
            except ValueError:
                continue
                
            if cls_name not in class_to_id: continue
            
            # 将 8点旋转框 转换为 4点水平框 (x_min, y_min, width, height)
            xs, ys = coords[0::2], coords[1::2]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            w, h = xmax - xmin, ymax - ymin
            
            # 极其重要：过滤掉面积为 0 或极小的无效框，防止 NaN 梯度爆炸！
            if w <= 1.0 or h <= 1.0: continue
                
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_to_id[cls_name],
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1

    print(f"✅ 转换完成: {out_json_path}")
    print(f"👉 包含图片数: {len(coco['images'])}, 生成有效标注框数: {len(coco['annotations'])}\n")
    
    with open(out_json_path, 'w') as f:
        json.dump(coco, f)

if __name__ == '__main__':
    base_dir = './data/dota'
    out_dir = os.path.join(base_dir, 'annotations')
    os.makedirs(out_dir, exist_ok=True)
    
    # 转换训练集
    trainval_dir = '/home/xd/dataset/dota-v10/versions/6/split_ss_dota1_0/trainval'
    train_json = os.path.join(out_dir, 'instances_train2017.json')
    dota2coco(trainval_dir, train_json)
    
    # 转换验证集
    val_dir = '/home/xd/dataset/dota-v10/versions/6/split_ss_dota1_0/val'
    val_json = os.path.join(out_dir, 'instances_val2017.json')
    dota2coco(val_dir, val_json)