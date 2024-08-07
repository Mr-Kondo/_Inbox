# -*- coding: utf-8 -*-
# ゴミファイルの処理
import os, shutil
if os.path.exists('../axial_t1wce_2_class'):
    shutil.rmtree('../axial_t1wce_2_class')
if os.path.exists('./fixed_path.yaml'):
   os.remove('./fixed_path.yaml')
if os.path.exists('./runs'):
    shutil.rmtree('./runs')
if os.path.exists('./test.yaml'):
   os.remove('./test.yaml')
   
# 必要なライブラリのインストール（ターミナル上で以下のコマンドを実施）
"""
# !pip install ultralytics
# !pip install 'git+https://github.com/facebookresearch/segment-anything.git'
# !{sys.executable} -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'
# !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
"""

# ライブラリのインポート
import ultralytics
from ultralytics import YOLO
import torch
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# GPUデバイスが認識されていることの確認
print(torch.backends.mps.is_available())

# 学習画像データをコード直下に解凍
import shutil
shutil.unpack_archive('../01_input/axial_t1wce_2_class.zip', '../')

files = os.listdir('../axial_t1wce_2_class/images/train')
random_file = random.choice(files)
random_file = os.path.splitext(random_file)[0]
print(random_file)

train_images = os.listdir('../axial_t1wce_2_class/images/train')
test_images = os.listdir('../axial_t1wce_2_class/images/test')
train_labels = os.listdir('../axial_t1wce_2_class/labels/train')
test_labels = os.listdir('../axial_t1wce_2_class/labels/test')

num_train_images = len(train_images)
num_test_images = len(test_images)
num_train_labels = len(train_labels)
num_test_labels = len(test_labels)

os.path.splitext(random_file)[0]
unlabelled = set(train_images) - set([os.path.splitext(file)[0] + '.jpg' for file in train_labels])

for image in unlabelled:
    os.remove(os.path.join('../axial_t1wce_2_class/images/train', image))

print(num_train_images, num_test_images)
print(num_train_labels, num_test_labels)

with open(os.path.join('../axial_t1wce_2_class/labels/train', f'{random_file}.txt'),'r') as f:
    labels = f.readlines()
    labels = labels[0].split(' ')
    f.close()

img = cv2.imread(os.path.join('../axial_t1wce_2_class/images/train', f'{random_file}.jpg'), 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ラベルデータからROI座標を取得
tumor_class, x, y, w, h = int(labels[0]), float(labels[1]), float(labels[2]), float(labels[3]), float(labels[4])
x_pt1 = int((x - w/2) * img.shape[1])
y_pt1 = int((y - h/2) * img.shape[0])
x_pt2 = int((x + w/2) * img.shape[1])
y_pt2 = int((y + h/2) * img.shape[0])

if tumor_class == 1:
    colour = (255, 0, 0)
else:
    colour = (0, 255, 0)
cv2.rectangle(img, (x_pt1, y_pt1), (x_pt2, y_pt2), colour, 1)
plt.savefig(img)

# 検証用データは訓練データから２割取り出す
val_split = int(num_train_images * 0.2)
val_images = random.sample(train_images, val_split)

if os.path.exists('../axial_t1wce_2_class/images/val'):
    shutil.rmtree('../axial_t1wce_2_class/images/val')
if os.path.exists('../axial_t1wce_2_class/labels/val'):
    shutil.rmtree('../axial_t1wce_2_class/labels/val')

os.makedirs('../axial_t1wce_2_class/images/val')
os.makedirs('../axial_t1wce_2_class/labels/val')

for image in val_images:
    shutil.move(os.path.join('../axial_t1wce_2_class/images/train', image), '../axial_t1wce_2_class/images/val')

for image in val_images:
    label = os.path.splitext(image)[0] + '.txt'
    shutil.move(os.path.join('../axial_t1wce_2_class/labels/train', label), '../axial_t1wce_2_class/labels/val')

#yamlデータを、学習データへのパスを記入して、作成する。
text ="""
path: /Volumes/ML_Mac/Programming/Python/Brain_Tumor/axial_t1wce_2_class
train: images/train
val: images/val

# Classes
nc: 2 # 脳（腫瘍部）と背景
names: ['negative','positive']
"""

with open('./fixed_path.yaml', 'w') as file:
    file.write(text)

model = YOLO('yolov8n.pt')

#early_stopping = EarlyStopping()
results = model.train(data='./fixed_path.yaml', batch=8, epochs=50, device='mps')
results=YOLO('./runs/detect/train/weights/best.pt') # 最も検証結果のよかったモデルを呼び出す

training_save_dir = './' + str(results.save_dir)

#評価指標のグラフを出力させる
plt.figure(figsize=(20, 10))
img = cv2.imread(os.path.join(training_save_dir, 'results.png'))
plt.savefig(img)

plt.figure(figsize=(20, 10))
img = cv2.imread(os.path.join(training_save_dir, 'confusion_matrix.png'))
plt.savefig(img)

plt.figure(figsize=(20, 10))
img = cv2.imread(os.path.join(training_save_dir, 'val_batch0_pred.jpg'))
plt.savefig(img)

plt.figure(figsize=(20, 10))
img = cv2.imread(os.path.join(training_save_dir, 'val_batch1_pred.jpg'))
plt.savefig(img)

#試験データに学習モデルを適用する
trained_model = YOLO(training_save_dir + '/weights/best.pt')
predictions = trained_model.predict(
    source="../axial_t1wce_2_class/images/test",
    conf=0.4, save_txt=True, save_conf=True)

predictions_save_dir = './' + predictions[0].save_dir + '/labels'

def draw_bbox(file_path, filename, img):
    with open(os.path.join(file_path, f'{filename}.txt'),'r') as f:
        labels = f.readlines()
        labels = labels[0].split(' ')
        print(labels)
        f.close()

    tumor_class, x, y, w, h = int(labels[0]), float(labels[1]), float(labels[2]), float(labels[3]), float(labels[4])
    x_pt1 = int((x - w/2) * img.shape[1])
    y_pt1 = int((y - h/2) * img.shape[0])
    x_pt2 = int((x + w/2) * img.shape[1])
    y_pt2 = int((y + h/2) * img.shape[0])

    if tumor_class == 0:
        colour = (255, 0, 0)
        label = 'Negative'
    else:
        colour = (0, 255, 0)
        label = 'Positive'
    if len(labels) > 5:
        prob = float(labels[5])
        prob = round(prob, 1)
        prob = str(prob)
        label = label + ' ' + prob

    cv2.rectangle(img, (x_pt1, y_pt1), (x_pt2, y_pt2), colour, 1)
    cv2.putText(img, label, (x_pt1, y_pt1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

files = os.listdir(predictions_save_dir)
random_file = random.choice(files)
random_file = os.path.splitext(random_file)[0]

img_pred = cv2.imread(os.path.join('../axial_t1wce_2_class/images/test', f'{random_file}.jpg'), 1)
img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
draw_bbox(predictions_save_dir, random_file, img_pred)

img_real = cv2.imread(os.path.join('../axial_t1wce_2_class/images/test', f'{random_file}.jpg'), 1)
img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)
draw_bbox('../axial_t1wce_2_class/labels/test', random_file, img_real)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img_pred)
axes[0].set_title('Predicted Image')
axes[0].axis('off')

axes[1].imshow(img_real)
axes[1].set_title('Real Image')
axes[1].axis('off')

plt.tight_layout()
plt.savefig()

text = """
path: /Volumes/ML_Mac/Programming/Python/Brain_Tumor/axial_t1wce_2_class
train: images/train
val: images/test

# Classes
nc: 2
names: ['negative','positive']
"""
with open("./test.yaml", 'w') as file:
    file.write(text)

metrics = trained_model.val(data="./test.yaml")

test_save_dir = './' + str(metrics.save_dir)

plt.figure(figsize=(20, 10))
img = cv2.imread(os.path.join(test_save_dir, 'confusion_matrix.png'))
plt.savefig(img)

plt.figure(figsize=(20, 10))
img = cv2.imread(os.path.join(test_save_dir, 'val_batch0_pred.jpg'))
plt.savefig(img)

plt.figure(figsize=(20, 10))
img = cv2.imread(os.path.join(test_save_dir, 'val_batch1_pred.jpg'))
plt.savefig(img)

image_path = os.path.join(test_save_dir, 'val_batch0_pred.jpg')

# Create a figure with a specified size
plt.figure(figsize=(20, 10))
# Read the image using OpenCV
img = cv2.imread(image_path)
# Display the image using matplotlib
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct display
# Add a title to the plot (optional)
plt.title('Predicted Image')
# Show the plot
plt.axis('off')  # Hide axis ticks and labels
plt.savefig(img)

"""## [SAM(Segment Anything Model)](https://segment-anything.com/)"""
real_path = os.path.join('../axial_t1wce_2_class/images/test', f'{random_file}.jpg')

import locale
locale.getpreferredencoding = lambda: "UTF-8"

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

image = cv2.cvtColor(cv2.imread(real_path), cv2.COLOR_BGR2RGB)
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
device = "mps"
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(image)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

#上で作成した学習モデルでの腫瘍の位置を検出し、それをSAMモデルに読み込ませて推定させる
model=YOLO('./runs/detect/train/weights/best.pt')
results = model.predict(source=real_path, conf=0.25)
for result in results:
    boxes = result.boxes
bbox = boxes.xyxy.tolist()[0]

input_box = np.array(bbox)
masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,

plt.figure(figsize=(10, 10))
plt.imshow(image, cmap='gray')
show_mask(masks[0], plt.gca())
show_box(input_box, plt.gca())
plt.axis('off')
plt.show()

if os.path.exists('../axial_t1wce_2_class'):
    shutil.rmtree('../axial_t1wce_2_class')
if os.path.exists('./fixed_path.yaml'):
   os.remove('./fixed_path.yaml')
if os.path.exists('./runs'):
    shutil.rmtree('./runs')
if os.path.exists('./test.yaml'):
   os.remove('./test.yaml')