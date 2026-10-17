import os
from collections import defaultdict

root_dir = r'f:\project images data\Data rough'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

disease_stats = defaultdict(lambda: {'count': 0, 'sources': set()})

# 1. AL9EE
al9ee_path = os.path.join(root_dir, 'AL9EE', 'AL9EE')
if os.path.exists(al9ee_path):
    for d in os.listdir(al9ee_path):
        dp = os.path.join(al9ee_path, d)
        if os.path.isdir(dp):
            cnt = len([f for f in os.listdir(dp) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS])
            disease_stats[d]['count'] += cnt
            disease_stats[d]['sources'].add('AL9EE')

# 2. archive
archive_all = os.path.join(root_dir, 'archive', 'AppleLeafDisease')
if os.path.exists(archive_all):
    for sub in ['All', 'Train', 'Test', 'Aug_Set']:
        sp = os.path.join(archive_all, sub)
        if os.path.exists(sp):
            for d in os.listdir(sp):
                dp = os.path.join(sp, d)
                if os.path.isdir(dp):
                    cnt = len([f for f in os.listdir(dp) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS])
                    # Standardize names
                    std_name = d.replace('_', ' ').title()
                    disease_stats[std_name]['count'] += cnt
                    disease_stats[std_name]['sources'].add('archive')

# 3. Apple Disease Dataset
add_path = os.path.join(root_dir, 'Apple Disease Dataset')
if os.path.exists(add_path):
    for sub in ['train', 'test', 'validation']:
        sp = os.path.join(add_path, sub)
        if os.path.exists(sp):
            for d in os.listdir(sp):
                dp = os.path.join(sp, d)
                if os.path.isdir(dp):
                    cnt = len([f for f in os.listdir(dp) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS])
                    std_name = d.replace('_', ' ').title()
                    disease_stats[std_name]['count'] += cnt
                    disease_stats[std_name]['sources'].add('Apple Disease Dataset')

# 4. Indigenous
indig_path = os.path.join(root_dir, 'Indigenous Dataset for Apple Leaf Disease Detection and Classification', 'Indigenous Dataset for Apple Leaf Disease Detection and Classification')
if os.path.exists(indig_path):
    for sub in ['Train_Set', 'Test_Set']:
        sp = os.path.join(indig_path, sub)
        if os.path.exists(sp):
            for d in os.listdir(sp):
                dp = os.path.join(sp, d)
                if os.path.isdir(dp):
                    cnt = len([f for f in os.listdir(dp) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS])
                    std_name = d.replace('_', ' ').title()
                    disease_stats[std_name]['count'] += cnt
                    disease_stats[std_name]['sources'].add('Indigenous')

# 5. alternaria
alt_path = os.path.join(root_dir, 'alternaria')
if os.path.exists(alt_path):
    for d in os.listdir(alt_path):
        dp = os.path.join(alt_path, d)
        if os.path.isdir(dp):
            cnt = len([f for f in os.listdir(dp) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS])
            std_name = d.replace('_', ' ').title()
            disease_stats[std_name]['count'] += cnt
            disease_stats[std_name]['sources'].add('alternaria')

# 6. apple_disease.coco
coco_path = os.path.join(root_dir, 'apple_disease.coco')
if os.path.exists(coco_path):
    total_coco = 0
    for sub in ['train', 'valid', 'test']:
        sp = os.path.join(coco_path, sub)
        if os.path.exists(sp):
            total_coco += len([f for f in os.listdir(sp) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS])
    disease_stats['COCO Annotated Images']['count'] += total_coco
    disease_stats['COCO Annotated Images']['sources'].add('apple_disease.coco')

print('=' * 80)
print(f'{"DISEASE / CONDITION CLASS":<35} | {"TOTAL IMAGES":<14} | SOURCES')
print('=' * 80)
for k, v in sorted(disease_stats.items(), key=lambda x: -x[1]['count']):
    src_str = ', '.join(sorted(v['sources']))
    print(f'{k:<35} | {v["count"]:>12} | {src_str}')
print('=' * 80)
