import json
import matplotlib.pyplot as plt

# 1. 准备存放数据的列表
epochs = []
train_losses = []
test_losses = []
mAP_all = []  # 总体 mAP
mAP_50 = []   # mAP@0.50

# 2. 读取 log.txt 文件
log_path = 'output/dab_detr_r50/log.txt'  # 请确保路径正确，如果在同级目录直接写 'log.txt'
# log_path = 'output/dota_baseline/log.txt'

try:
    with open(log_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            epochs.append(data['epoch'])
            train_losses.append(data['train_loss'])
            test_losses.append(data['test_loss'])
            
            # test_coco_eval_bbox 是一个包含12个COCO指标的数组
            # 第0个是整体 mAP (0.50:0.95)，第1个是 mAP@0.50
            if 'test_coco_eval_bbox' in data:
                mAP_all.append(data['test_coco_eval_bbox'][0] * 100) # 乘以100换算成百分比
                mAP_50.append(data['test_coco_eval_bbox'][1] * 100)
except FileNotFoundError:
    print(f"找不到文件: {log_path}，请检查路径。")
    exit()

# 3. 开始画图
plt.figure(figsize=(12, 5))

# 左边的图：Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, test_losses, label='Test Loss', marker='s')
plt.title('Training and Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 右边的图：mAP 精度曲线
plt.subplot(1, 2, 2)
if mAP_all:
    plt.plot(epochs, mAP_all, label='mAP (0.5:0.95)', marker='o', color='green')
    plt.plot(epochs, mAP_50, label='mAP@0.50', marker='s', color='orange')
    plt.title('mAP Evaluation')
    plt.xlabel('Epoch')
    plt.ylabel('mAP (%)')
    plt.legend()
    plt.grid(True)

plt.tight_layout()

# 4. 保存图片
save_name = 'training_curve.png'
# save_name = 'training_curve_dota.png'
plt.savefig(save_name, dpi=300)
print(f"图表已成功保存为: {save_name} ！请在 VS Code 中双击打开查看。")