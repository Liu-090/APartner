import json
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import shutil
import logging
import sys
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import random
from collections import defaultdict
import math

# ==================== 日志设置 ====================
def setup_logging():
    """设置日志系统"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 文件处理器
    log_filename = f'processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # 移除所有现有处理器，添加新的
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 初始化日志
logger = setup_logging()

# ==================== 平衡特征提取器 ====================
class BalancedFeatureExtractor:
    """平衡的特征提取器"""
    
    def __init__(self):
        self.image_width = 1920
        self.image_height = 1080
        
        # 统一的映射定义
        self.class_mapping = {'car': 0, 'van': 1, 'truck': 2, 'bus': 3, 'other': 4}
        self.color_mapping = {
            'white': 0, 'black': 1, 'red': 2, 'silver_grey': 3, 'silver-gray': 3,
            'blue': 4, 'green': 5, 'yellow': 6, 'gray': 7, 'grey': 7, 'unknown': 8,
            'dark_brown': 9, 'yellow_orange': 10
        }
        
        # 固定维度
        self._text_feature_dim = 35
        self._object_feature_dim = 25
        
        logger.info(f"平衡特征维度 - 文本: {self._text_feature_dim}, 对象: {self._object_feature_dim}")
    
    def get_text_feature_dim(self):
        return self._text_feature_dim
    
    def get_object_feature_dim(self):
        return self._object_feature_dim
    
    def extract_text_features(self, descriptions: List[Dict]) -> np.ndarray:
        """提取文本特征"""
        if not descriptions:
            return np.zeros((0, self._text_feature_dim), dtype=np.float32)
        
        text_features = []
        
        for desc in descriptions:
            feature_vector = []
            
            # 1. 颜色特征 (11维)
            color_vec = [0.0] * 11
            for color in desc.get('color', []):
                if color in self.color_mapping:
                    idx = self.color_mapping[color]
                    color_vec[idx] = 1.0
            feature_vector.extend(color_vec)
            
            # 2. 类型特征 (5维)
            type_vec = [0.0] * 5
            for obj_type in desc.get('object_type', []):
                if obj_type in self.class_mapping:
                    idx = self.class_mapping[obj_type]
                    type_vec[idx] = 1.0
            feature_vector.extend(type_vec)
            
            # 3. 位置特征 (5维)
            position_keywords = ['left', 'right', 'top', 'bottom', 'center']
            position_vec = [0.0] * 5
            position_data = desc.get('position', [])
            for i, pos in enumerate(position_keywords):
                if i < 5 and pos in position_data:
                    position_vec[i] = 1.0
            feature_vector.extend(position_vec)
            
            # 4. 尺寸特征 (6维)
            size_vec = [0.0] * 6
            size_constraints = desc.get('size_constraints', [])
            if size_constraints:
                size_vec[0] = 1.0  # 是否有尺寸约束
                try:
                    min_vals = [c.get('min', 0) for c in size_constraints]
                    max_vals = [c.get('max', 0) for c in size_constraints]
                    if min_vals:
                        size_vec[1] = min(min_vals) / 10.0
                        size_vec[2] = max(min_vals) / 10.0
                    if max_vals:
                        size_vec[3] = min(max_vals) / 10.0
                        size_vec[4] = max(max_vals) / 10.0
                    size_vec[5] = len(size_constraints) / 3.0
                except:
                    pass
            feature_vector.extend(size_vec)
            
            # 5. 距离特征 (4维)
            distance_vec = [0.0] * 4
            distance_constraints = desc.get('distance_constraints', [])
            if distance_constraints:
                distance_vec[0] = 1.0
                try:
                    distance_vec[1] = min(distance_constraints) / 100.0
                    distance_vec[2] = max(distance_constraints) / 100.0
                    distance_vec[3] = len(distance_constraints) / 3.0
                except:
                    pass
            feature_vector.extend(distance_vec)
            
            # 6. 遮挡和方向特征 (4维)
            occlusion_vec = [0.0] * 2
            orientation_vec = [0.0] * 2
            
            occlusion_data = desc.get('occlusion', [])
            orientation_data = desc.get('orientation', [])
            
            if 'occluded' in occlusion_data or 'partially occluded' in occlusion_data:
                occlusion_vec[0] = 1.0
            if 'not occluded' in occlusion_data:
                occlusion_vec[1] = 1.0
            
            if 'facing me' in orientation_data or 'front' in orientation_data:
                orientation_vec[0] = 1.0
            if 'facing away' in orientation_data or 'back' in orientation_data:
                orientation_vec[1] = 1.0
            
            feature_vector.extend(occlusion_vec)
            feature_vector.extend(orientation_vec)
            
            # 维度强制匹配
            if len(feature_vector) != self._text_feature_dim:
                if len(feature_vector) > self._text_feature_dim:
                    feature_vector = feature_vector[:self._text_feature_dim]
                else:
                    feature_vector.extend([0.0] * (self._text_feature_dim - len(feature_vector)))
            
            text_features.append(feature_vector)
        
        return np.array(text_features, dtype=np.float32)
    
    def extract_object_features(self, object_data: List[Dict]) -> np.ndarray:
        """提取对象特征"""
        if not object_data:
            return np.zeros((0, self._object_feature_dim), dtype=np.float32)
        
        features = []
        
        for obj in object_data:
            feature_vector = []
            
            # 1. 几何特征 (16维)
            try:
                bbox = obj.get('bbox', [0, 0, 100, 100])
                dimensions = obj.get('dimensions', [1.5, 1.8, 4.0])
                location = obj.get('location', [0, 0, 0])
                
                distance = np.sqrt(location[0]**2 + location[1]**2 + location[2]**2)
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                bbox_area = bbox_width * bbox_height
                bbox_center_x = (bbox[0] + bbox[2]) / 2
                bbox_center_y = (bbox[1] + bbox[3]) / 2
                
                geom_features = [
                    bbox[0] / self.image_width,
                    bbox[1] / self.image_height,
                    bbox[2] / self.image_width,
                    bbox[3] / self.image_height,
                    bbox_center_x / self.image_width,
                    bbox_center_y / self.image_height,
                    bbox_width / self.image_width,
                    bbox_height / self.image_height,
                    bbox_area / (self.image_width * self.image_height),
                    dimensions[0] / 5.0,
                    dimensions[1] / 5.0,
                    dimensions[2] / 10.0,
                    location[0] / 50.0,
                    location[1] / 50.0,
                    location[2] / 50.0,
                    distance / 100.0,
                ]
            except:
                geom_features = [0.0] * 16
            
            feature_vector.extend(geom_features)
            
            # 2. 类别特征 (5维)
            class_features = [0.0] * 5
            class_idx = self.class_mapping.get(obj.get('class', 'car'), 4)
            if class_idx < 5:
                class_features[class_idx] = 1.0
            feature_vector.extend(class_features)
            
            # 3. 颜色特征 (4维)
            color_features = [0.0] * 4
            color = obj.get('color', 'unknown')
            color_categories = {
                'black': 0, 'white': 1, 'red': 2, 'blue': 2, 'green': 2,
                'yellow': 2, 'gray': 0, 'grey': 0, 'silver_grey': 0, 'silver-gray': 0,
                'dark_brown': 0, 'yellow_orange': 2, 'unknown': 3
            }
            color_idx = color_categories.get(color, 3)
            if color_idx < 4:
                color_features[color_idx] = 1.0
            feature_vector.extend(color_features)
            
            # 维度强制匹配
            if len(feature_vector) != self._object_feature_dim:
                if len(feature_vector) > self._object_feature_dim:
                    feature_vector = feature_vector[:self._object_feature_dim]
                else:
                    feature_vector.extend([0.0] * (self._object_feature_dim - len(feature_vector)))
            
            features.append(feature_vector)
        
        return np.array(features, dtype=np.float32)

# ==================== 平衡文本处理器 ====================
class BalancedTextProcessor:
    """平衡的文本处理器"""
    
    def __init__(self):
        self.color_keywords = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'gray', 'grey', 'silver', 'dark-brown', 'yellow-orange']
        self.position_keywords = ['left', 'right', 'top', 'bottom', 'center']
        self.type_keywords = ['car', 'van', 'truck', 'bus', 'vehicle']
        self.occlusion_keywords = ['occluded', 'not occluded', 'partially occluded']
        self.orientation_keywords = ['facing me', 'facing away', 'front', 'back']
    
    def extract_features(self, description: str) -> Dict:
        """从描述文本中提取特征"""
        description_lower = description.lower()
        features = {
            'color': [],
            'object_type': [],
            'position': [],
            'size_constraints': [],
            'distance_constraints': [],
            'occlusion': [],
            'orientation': [],
            'constraint_count': 0
        }
        
        # 提取颜色
        for color in self.color_keywords:
            if re.search(r'\b' + re.escape(color) + r'\b', description_lower):
                standardized_color = color.replace('-', '_')
                features['color'].append(standardized_color)
        
        # 提取类型
        for obj_type in self.type_keywords:
            if re.search(r'\b' + re.escape(obj_type) + r'\b', description_lower):
                features['object_type'].append(obj_type)
        
        # 提取位置
        for pos in self.position_keywords:
            if re.search(r'\b' + re.escape(pos) + r'\b', description_lower):
                features['position'].append(pos)
        
        # 提取遮挡状态
        for occ in self.occlusion_keywords:
            if re.search(r'\b' + re.escape(occ) + r'\b', description_lower):
                features['occlusion'].append(occ)
        
        # 提取朝向
        for orient in self.orientation_keywords:
            if re.search(r'\b' + re.escape(orient) + r'\b', description_lower):
                features['orientation'].append(orient)
        
        # 尺寸约束提取
        size_patterns = [
            r'(?:height|width|length).*?(\d+\.?\d*)\s*to\s*(\d+\.?\d*)\s*meters?',
            r'(?:height|width|length).*?(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*meters?',
            r'(\d+\.?\d*)\s*to\s*(\d+\.?\d*)\s*meters?',
            r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*meters?',
        ]
        
        for pattern in size_patterns:
            matches = re.findall(pattern, description_lower)
            for match in matches:
                try:
                    if len(match) == 2:
                        min_val = float(match[0])
                        max_val = float(match[1])
                        if 0.5 <= min_val <= 20.0 and 0.5 <= max_val <= 20.0 and min_val < max_val:
                            features['size_constraints'].append({
                                'min': min_val,
                                'max': max_val
                            })
                except ValueError:
                    continue
        
        # 距离约束
        distance_patterns = [
            r'(\d+\.?\d*)\s*meters?\s*(?:away|distance)',
        ]
        
        for pattern in distance_patterns:
            distance_matches = re.findall(pattern, description_lower)
            for match in distance_matches:
                try:
                    distance = float(match)
                    if 5.0 <= distance <= 200.0:
                        features['distance_constraints'].append(distance)
                except ValueError:
                    continue
        
        # 计算约束数量
        features['constraint_count'] = (
            len(features['color']) + len(features['object_type']) + 
            len(features['position']) + len(features['occlusion']) +
            len(features['orientation']) + len(features['size_constraints']) + 
            len(features['distance_constraints'])
        )
        
        return features
    
    def process_descriptions(self, descriptions: List[str]) -> List[Dict]:
        """处理描述列表"""
        processed = []
        for desc in descriptions:
            features = self.extract_features(desc)
            processed.append(features)
        return processed

# ==================== 平衡匹配模型 ====================
class BalancedMatchingModel(nn.Module):
    """平衡匹配模型 - 适中的复杂度"""
    
    def __init__(self, text_dim=35, object_dim=25, hidden_dim=64):
        super(BalancedMatchingModel, self).__init__()
        
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.object_encoder = nn.Sequential(
            nn.Linear(object_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 适中的匹配头
        self.matching_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"平衡匹配模型: 文本{text_dim}->{hidden_dim}, 对象{object_dim}->{hidden_dim}")
    
    def forward(self, text_features, object_features):
        # 编码特征
        text_encoded = self.text_encoder(text_features)
        object_encoded = self.object_encoder(object_features)
        
        # 计算相似度
        combined = torch.cat([text_encoded, object_encoded], dim=1)
        similarity_scores = self.matching_head(combined).squeeze()
        
        return similarity_scores

# ==================== 平衡数据处理器 ====================
class BalancedDataProcessor:
    """平衡的数据处理器：生成平衡的训练数据"""
    
    def __init__(self):
        self.image_width = 1920
        self.image_height = 1080
    
    def parse_object_string(self, obj_str):
        """解析对象字符串"""
        try:
            if isinstance(obj_str, str):
                if obj_str.startswith('[') and obj_str.endswith(']'):
                    # 训练集格式
                    clean_str = obj_str.strip()[1:-1]
                    parts = [part.strip().strip("'\"") for part in clean_str.split(',')]
                else:
                    # 测试集格式
                    parts = obj_str.strip().split()
                
                # 确保有足够的部分
                if len(parts) < 15:
                    parts.extend(['0'] * (15 - len(parts)))
                
                # 解析对象数据
                obj_data = {
                    'class': parts[0] if parts[0] else 'car',
                    'truncated': float(parts[1]) if len(parts) > 1 else 0,
                    'occluded': int(float(parts[2])) if len(parts) > 2 else 0,
                    'alpha': float(parts[3]) if len(parts) > 3 else 0.0,
                    'bbox': [
                        float(parts[4]) if len(parts) > 4 else 0,
                        float(parts[5]) if len(parts) > 5 else 0,
                        float(parts[6]) if len(parts) > 6 else 100,
                        float(parts[7]) if len(parts) > 7 else 100
                    ],
                    'dimensions': [
                        float(parts[8]) if len(parts) > 8 else 1.5,
                        float(parts[9]) if len(parts) > 9 else 1.8,
                        float(parts[10]) if len(parts) > 10 else 4.0
                    ],
                    'location': [
                        float(parts[11]) if len(parts) > 11 else 0,
                        float(parts[12]) if len(parts) > 12 else 0,
                        float(parts[13]) if len(parts) > 13 else 0
                    ],
                    'rotation_y': float(parts[14]) if len(parts) > 14 else 0.0,
                    'color': 'unknown'
                }
                
                # 提取颜色信息
                if len(parts) > 15:
                    color_candidate = parts[15].strip("'\"")
                    if color_candidate in ['white', 'black', 'red', 'blue', 'green', 'yellow', 'gray', 'grey', 'silver', 'dark-brown', 'yellow-orange']:
                        obj_data['color'] = color_candidate.replace('-', '_')
                
                return obj_data
            
        except Exception as e:
            logger.error(f"解析对象数据失败: {e}")
        
        # 默认返回
        return {
            'class': 'car', 'truncated': 0, 'occluded': 0, 'alpha': 0.0,
            'bbox': [0, 0, 100, 100], 'dimensions': [1.5, 1.8, 4.0],
            'location': [0, 0, 0], 'rotation_y': 0.0, 'color': 'unknown'
        }
    
    def process_test_data(self, test_data):
        """处理测试集数据"""
        return [self.parse_object_string(obj_str) for obj_str in test_data]
    
    def process_training_data(self, json_data):
        """处理训练集数据 - 生成平衡的训练数据"""
        all_text_features = []
        all_object_features = []
        all_labels = []
        
        if not isinstance(json_data, list) or len(json_data) == 0:
            return all_text_features, all_object_features, all_labels
        
        annotations = json_data[0] if isinstance(json_data[0], list) else json_data
        
        text_processor = BalancedTextProcessor()
        feature_extractor = BalancedFeatureExtractor()
        
        # 收集所有描述和对象
        all_descriptions = []
        all_objects = []
        annotation_objects = []  # 每个标注对应的对象列表
        
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
                
            description = ann.get('public_description', '')
            label_3_data = ann.get('label_3', [])
            
            if not description or not label_3_data:
                continue
            
            # 处理对象数据
            objects = [self.parse_object_string(obj_str) for obj_str in label_3_data]
            
            all_descriptions.append(description)
            all_objects.extend(objects)
            annotation_objects.append(objects)
        
        # 处理文本描述
        processed_descriptions = text_processor.process_descriptions(all_descriptions)
        text_features = feature_extractor.extract_text_features(processed_descriptions)
        object_features = feature_extractor.extract_object_features(all_objects)
        
        if len(text_features) > 0 and len(object_features) > 0:
            # 创建正样本 - 每个描述匹配其对应的对象
            obj_start_idx = 0
            for i, desc in enumerate(processed_descriptions):
                num_objects = len(annotation_objects[i]) if i < len(annotation_objects) else 0
                
                for j in range(num_objects):
                    if obj_start_idx + j < len(object_features):
                        all_text_features.append(text_features[i])
                        all_object_features.append(object_features[obj_start_idx + j])
                        all_labels.append(1.0)  # 正样本
                
                obj_start_idx += num_objects
            
            # 创建负样本 - 平衡正负样本比例
            num_positive = len(all_labels)
            logger.info(f"正样本数量: {num_positive}")
            
            # 生成负样本 - 每个正样本对应1个负样本（1:1比例）
            num_negative = num_positive
            
            for _ in range(num_negative):
                # 随机选择描述和对象
                desc_idx = random.randint(0, len(text_features) - 1)
                obj_idx = random.randint(0, len(object_features) - 1)
                
                # 检查是否为正样本
                is_positive = False
                obj_start_idx = 0
                for i in range(len(annotation_objects)):
                    num_objs = len(annotation_objects[i])
                    if i == desc_idx and obj_start_idx <= obj_idx < obj_start_idx + num_objs:
                        is_positive = True
                        break
                    obj_start_idx += num_objs
                
                # 如果不是正样本，则添加为负样本
                if not is_positive:
                    all_text_features.append(text_features[desc_idx])
                    all_object_features.append(object_features[obj_idx])
                    all_labels.append(0.0)  # 负样本
            
            logger.info(f"总样本数量: {len(all_labels)}, 正样本: {num_positive}, 负样本: {len(all_labels) - num_positive}")
        
        return all_text_features, all_object_features, all_labels

# ==================== 平衡ML训练器 ====================
class BalancedMLTrainer:
    """平衡的ML训练器 - 使用适中的训练策略"""
    
    def __init__(self):
        self.feature_extractor = BalancedFeatureExtractor()
        self.text_processor = BalancedTextProcessor()
        self.data_processor = BalancedDataProcessor()
        
        text_dim = self.feature_extractor.get_text_feature_dim()
        obj_dim = self.feature_extractor.get_object_feature_dim()
        self.model = BalancedMatchingModel(text_dim, obj_dim)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.criterion = nn.BCELoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.8)
    
    def train_from_directory(self, train_data_dir, epochs=15, max_files=500):
        """从目录训练模型 - 使用适中的训练策略"""
        logger.info("开始平衡ML模型训练...")
        
        json_files = [f for f in os.listdir(train_data_dir) if f.endswith('.json')]
        logger.info(f"找到 {len(json_files)} 个训练文件，使用前 {max_files} 个")
        
        if not json_files:
            logger.warning("没有找到训练文件")
            return False
        
        # 收集训练数据
        all_text_features = []
        all_object_features = []
        all_labels = []
        
        processed_files = 0
        for json_file in json_files[:max_files]:
            try:
                json_path = os.path.join(train_data_dir, json_file)
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                text_features, object_features, labels = self.data_processor.process_training_data(json_data)
                
                if text_features and object_features and labels:
                    all_text_features.extend(text_features)
                    all_object_features.extend(object_features)
                    all_labels.extend(labels)
                    processed_files += 1
                    
                    if processed_files % 50 == 0:
                        logger.info(f"已处理 {processed_files}/{max_files} 个训练文件")
                    
            except Exception as e:
                logger.error(f"处理训练文件 {json_file} 失败: {e}")
                continue
        
        if not all_text_features or not all_object_features:
            logger.warning("没有有效的训练数据")
            return False
        
        # 统计正负样本比例
        positive_count = sum(all_labels)
        negative_count = len(all_labels) - positive_count
        logger.info(f"训练数据统计: 正样本 {positive_count}, 负样本 {negative_count}, 总计 {len(all_labels)}")
        
        try:
            # 转换为numpy数组
            all_text_features_np = np.array(all_text_features, dtype=np.float32)
            all_object_features_np = np.array(all_object_features, dtype=np.float32)
            all_labels_np = np.array(all_labels, dtype=np.float32)
            
            logger.info(f"数据形状: 文本{all_text_features_np.shape}, 对象{all_object_features_np.shape}, 标签{all_labels_np.shape}")
            
            # 转换为tensor
            text_tensor = torch.from_numpy(all_text_features_np)
            object_tensor = torch.from_numpy(all_object_features_np)
            label_tensor = torch.from_numpy(all_labels_np)
            
            # 强制训练 - 确保维度匹配
            self.model.train()
            dataset_size = len(text_tensor)
            
            # 强制检查维度
            if len(text_tensor) != len(object_tensor):
                min_size = min(len(text_tensor), len(object_tensor))
                text_tensor = text_tensor[:min_size]
                object_tensor = object_tensor[:min_size]
                label_tensor = label_tensor[:min_size]
                dataset_size = min_size
                logger.info(f"强制调整数据维度到: {dataset_size}")
            
            # 创建数据集
            dataset = torch.utils.data.TensorDataset(text_tensor, object_tensor, label_tensor)
            batch_size = min(32, dataset_size)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            for epoch in range(epochs):
                total_loss = 0.0
                num_batches = 0
                
                for batch_text, batch_object, batch_labels in dataloader:
                    # 强制维度检查
                    if len(batch_text) != len(batch_object):
                        min_batch_size = min(len(batch_text), len(batch_object))
                        batch_text = batch_text[:min_batch_size]
                        batch_object = batch_object[:min_batch_size]
                        batch_labels = batch_labels[:min_batch_size]
                    
                    self.optimizer.zero_grad()
                    
                    # 前向传播
                    predictions = self.model(batch_text, batch_object)
                    
                    # 强制维度匹配
                    if predictions.shape != batch_labels.shape:
                        min_pred_size = min(predictions.size(0), batch_labels.size(0))
                        predictions = predictions[:min_pred_size]
                        batch_labels = batch_labels[:min_pred_size]
                    
                    loss = self.criterion(predictions, batch_labels)
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                self.scheduler.step()
                
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # 计算准确率
                with torch.no_grad():
                    self.model.eval()
                    val_predictions = self.model(text_tensor, object_tensor)
                    val_predictions_binary = (val_predictions > 0.5).float()
                    accuracy = (val_predictions_binary == label_tensor).float().mean()
                    
                    # 计算正负样本的准确率
                    pos_mask = label_tensor == 1.0
                    neg_mask = label_tensor == 0.0
                    
                    pos_accuracy = (val_predictions_binary[pos_mask] == label_tensor[pos_mask]).float().mean() if pos_mask.sum() > 0 else torch.tensor(0.0)
                    neg_accuracy = (val_predictions_binary[neg_mask] == label_tensor[neg_mask]).float().mean() if neg_mask.sum() > 0 else torch.tensor(0.0)
                    
                    self.model.train()
                
                logger.info(f"Epoch {epoch+1}/{epochs}, 平均Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
                logger.info(f"准确率: 总体 {accuracy:.4f}, 正样本 {pos_accuracy:.4f}, 负样本 {neg_accuracy:.4f}")
            
            # 保存模型
            torch.save(self.model.state_dict(), "balanced_matching_model.pth")
            logger.info("平衡ML模型训练完成并保存")
            return True
            
        except Exception as e:
            logger.error(f"训练过程失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

# ==================== 平衡匹配器 ====================
class BalancedMatcher:
    """平衡匹配器：使用适中的阈值"""
    
    def __init__(self, use_ml=True):
        self.text_processor = BalancedTextProcessor()
        self.feature_extractor = BalancedFeatureExtractor()
        self.data_processor = BalancedDataProcessor()
        self.use_ml = use_ml
        
        if use_ml:
            try:
                text_dim = self.feature_extractor.get_text_feature_dim()
                obj_dim = self.feature_extractor.get_object_feature_dim()
                self.ml_model = BalancedMatchingModel(text_dim, obj_dim)
                self.load_ml_model()
                logger.info("平衡ML模型加载成功")
            except Exception as e:
                logger.warning(f"平衡ML模型加载失败: {e}")
                self.use_ml = False
        else:
            logger.info("使用规则匹配")
    
    def load_ml_model(self, model_path="balanced_matching_model.pth"):
        """加载ML模型"""
        if os.path.exists(model_path):
            self.ml_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.ml_model.eval()
            logger.info("平衡ML模型加载成功")
        else:
            logger.warning("平衡ML模型文件不存在")
            self.use_ml = False
    
    def balanced_rule_match(self, objects, descriptions_text):
        """平衡的规则匹配"""
        if not objects or not descriptions_text:
            return []
        
        processed_descriptions = self.text_processor.process_descriptions(descriptions_text)
        results = []
        
        for obj in objects:
            obj_matches = []
            for desc_features in processed_descriptions:
                similarity = self.calculate_balanced_similarity(obj, desc_features)
                # 使用适中的阈值
                threshold = 0.5
                obj_matches.append(1 if similarity >= threshold else 0)
            results.append(obj_matches)
        
        return results
    
    def calculate_balanced_similarity(self, obj, desc_features):
        """计算平衡的相似度"""
        score = 0.0
        total_weights = 0.0
        
        # 类型匹配 (权重: 0.3)
        if desc_features.get('object_type'):
            total_weights += 0.3
            obj_class = obj.get('class', 'car')
            if obj_class in desc_features['object_type']:
                score += 0.3
        
        # 颜色匹配 (权重: 0.25)
        if desc_features.get('color'):
            total_weights += 0.25
            obj_color = obj.get('color', 'unknown')
            if obj_color in desc_features['color']:
                score += 0.25
        
        # 位置匹配 (权重: 0.2)
        if desc_features.get('position'):
            total_weights += 0.2
            bbox = obj.get('bbox', [0, 0, 100, 100])
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            position_matched = False
            for pos in desc_features['position']:
                if pos == 'left' and center_x < 600:
                    position_matched = True
                    break
                elif pos == 'right' and center_x > 1320:
                    position_matched = True
                    break
                elif pos == 'top' and center_y < 400:
                    position_matched = True
                    break
                elif pos == 'bottom' and center_y > 680:
                    position_matched = True
                    break
                elif pos == 'center' and 600 <= center_x <= 1320 and 400 <= center_y <= 680:
                    position_matched = True
                    break
            
            if position_matched:
                score += 0.2
        
        # 尺寸匹配 (权重: 0.15)
        if desc_features.get('size_constraints'):
            total_weights += 0.15
            dimensions = obj.get('dimensions', [1.5, 1.8, 4.0])
            size_matched = False
            
            for constraint in desc_features['size_constraints']:
                min_val = constraint.get('min', 0)
                max_val = constraint.get('max', float('inf'))
                
                # 检查所有维度
                height_match = min_val <= dimensions[0] <= max_val
                width_match = min_val <= dimensions[1] <= max_val  
                length_match = min_val <= dimensions[2] <= max_val
                
                if height_match or width_match or length_match:
                    size_matched = True
                    break
            
            if size_matched:
                score += 0.15
        
        # 距离匹配 (权重: 0.1)
        if desc_features.get('distance_constraints'):
            total_weights += 0.1
            location = obj.get('location', [0, 0, 0])
            distance = np.sqrt(location[0]**2 + location[1]**2 + location[2]**2)
            distance_matched = False
            
            for dist_constraint in desc_features['distance_constraints']:
                if abs(distance - dist_constraint) < 15.0:
                    distance_matched = True
                    break
            
            if distance_matched:
                score += 0.1
        
        return score / total_weights if total_weights > 0 else 0.0
    
    def ml_based_match(self, objects, descriptions_text):
        """基于ML的匹配 - 使用适中的阈值"""
        if not self.use_ml or not objects or not descriptions_text:
            return self.balanced_rule_match(objects, descriptions_text)
        
        try:
            processed_descriptions = self.text_processor.process_descriptions(descriptions_text)
            text_features = self.feature_extractor.extract_text_features(processed_descriptions)
            object_features = self.feature_extractor.extract_object_features(objects)
            
            results = []
            
            # 对每个对象和每个描述进行匹配
            for obj_idx, obj_feat in enumerate(object_features):
                obj_matches = []
                for desc_idx, desc_feat in enumerate(text_features):
                    # 准备输入
                    text_tensor = torch.FloatTensor(desc_feat).unsqueeze(0)
                    obj_tensor = torch.FloatTensor(obj_feat).unsqueeze(0)
                    
                    # ML预测
                    with torch.no_grad():
                        ml_score = self.ml_model(text_tensor, obj_tensor).item()
                    
                    # 使用适中的阈值
                    obj_matches.append(1 if ml_score > 0.5 else 0)
                
                results.append(obj_matches)
            
            return results
            
        except Exception as e:
            logger.error(f"ML匹配失败: {e}, 回退到规则匹配")
            return self.balanced_rule_match(objects, descriptions_text)
    
    def hybrid_match(self, objects, descriptions_text):
        """混合匹配：结合规则和ML"""
        if not self.use_ml:
            return self.balanced_rule_match(objects, descriptions_text)
        
        try:
            processed_descriptions = self.text_processor.process_descriptions(descriptions_text)
            text_features = self.feature_extractor.extract_text_features(processed_descriptions)
            object_features = self.feature_extractor.extract_object_features(objects)
            
            hybrid_results = []
            
            # 对每个对象和每个描述进行混合匹配
            for obj_idx, obj_feat in enumerate(object_features):
                obj_matches = []
                for desc_idx, desc_feat in enumerate(text_features):
                    # 准备输入
                    text_tensor = torch.FloatTensor(desc_feat).unsqueeze(0)
                    obj_tensor = torch.FloatTensor(obj_feat).unsqueeze(0)
                    
                    # ML预测
                    with torch.no_grad():
                        ml_score = self.ml_model(text_tensor, obj_tensor).item()
                    
                    # 规则分数
                    rule_score = self.calculate_balanced_similarity(
                        objects[obj_idx], 
                        processed_descriptions[desc_idx]
                    )
                    
                    # 加权组合 (ML权重0.6，规则权重0.4)
                    combined_score = 0.6 * ml_score + 0.4 * rule_score
                    
                    # 使用适中的阈值
                    obj_matches.append(1 if combined_score > 0.5 else 0)
                
                hybrid_results.append(obj_matches)
            
            return hybrid_results
            
        except Exception as e:
            logger.error(f"混合匹配失败: {e}, 回退到规则匹配")
            return self.balanced_rule_match(objects, descriptions_text)
    
    def match(self, objects, descriptions_text):
        """主匹配函数 - 使用混合匹配"""
        return self.hybrid_match(objects, descriptions_text)

# ==================== 主程序 ====================
def main():
    """主程序"""
    logger.info("=== 平衡的3D视觉定位与文本匹配系统启动 ===")
    
    # 路径设置
    train_data_dir = "C:/Users/38543/Desktop/人工智能专项赛文献/VLMOD(train+test)/MonoMulti3D/train"
    test_json_folder = "C:/Users/38543/Desktop/人工智能专项赛文献/VLMOD(train+test)/MonoMulti3D/test"
    output_folder = "balanced_result"
    
    # 训练平衡ML模型
    ml_trained = False
    if os.path.exists(train_data_dir):
        trainer = BalancedMLTrainer()
        ml_trained = trainer.train_from_directory(train_data_dir, epochs=15, max_files=500)
        if ml_trained:
            logger.info("平衡ML模型训练成功，使用混合匹配")
        else:
            logger.info("平衡ML模型训练失败，使用纯规则匹配")
    else:
        logger.info("训练数据目录不存在，使用纯规则匹配")
    
    # 处理测试集
    matcher = BalancedMatcher(use_ml=ml_trained)
    data_processor = BalancedDataProcessor()
    
    if not os.path.exists(test_json_folder):
        logger.error(f"测试集文件夹不存在: {test_json_folder}")
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    json_files = [f for f in os.listdir(test_json_folder) if f.endswith('.json')]
    logger.info(f"找到 {len(json_files)} 个测试文件")
    
    processed_count = 0
    
    for json_file in json_files:
        json_path = os.path.join(test_json_folder, json_file)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # 测试集格式处理
            descriptions_text = json_data.get('public_description', [])
            test_data = json_data.get('test_data', [])
            objects = data_processor.process_test_data(test_data)
            
            logger.info(f"处理文件 {json_file}: {len(descriptions_text)} 描述, {len(objects)} 对象")
            
            # 进行匹配
            matches = matcher.match(objects, descriptions_text)
            
            # 检查匹配结果
            total_matches = sum(sum(match) for match in matches)
            total_possible = len(matches) * len(descriptions_text)
            match_percentage = total_matches / total_possible * 100 if total_possible > 0 else 0
            logger.info(f"匹配统计: 总匹配数 {total_matches}/{total_possible} ({match_percentage:.1f}%)")
            
            # 如果匹配率过低，使用更宽松的规则匹配
            if match_percentage < 10:
                logger.warning(f"匹配率过低 ({match_percentage:.1f}%)，使用更宽松的规则匹配")
                # 临时降低阈值
                original_threshold = 0.5
                threshold = 0.3
                
                processed_descriptions = matcher.text_processor.process_descriptions(descriptions_text)
                matches = []
                
                for obj in objects:
                    obj_matches = []
                    for desc_features in processed_descriptions:
                        similarity = matcher.calculate_balanced_similarity(obj, desc_features)
                        obj_matches.append(1 if similarity >= threshold else 0)
                    matches.append(obj_matches)
                
                total_matches = sum(sum(match) for match in matches)
                logger.info(f"宽松匹配后: 总匹配数 {total_matches}/{total_possible}")
            
            # 保存结果
            output_file = json_file.replace('.json', '.txt')
            output_path = os.path.join(output_folder, output_file)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for match in matches:
                    f.write(' '.join(str(x) for x in match) + '\n')
            
            processed_count += 1
            
            if processed_count % 10 == 0:
                logger.info(f"已处理: {processed_count}/{len(json_files)} 个文件")
                
        except Exception as e:
            logger.error(f"处理文件 {json_file} 失败: {e}")
            # 生成适中的默认输出
            objects = data_processor.process_test_data(test_data) if 'test_data' in locals() else []
            num_descriptions = len(descriptions_text) if 'descriptions_text' in locals() else 3
            # 默认部分匹配
            default_matches = []
            for i in range(len(objects)):
                # 每个对象随机匹配0-1个描述
                match_vector = [0, 0, 0]
                if i < len(objects) // 2:  # 一半的对象匹配第一个描述
                    match_vector[0] = 1
                default_matches.append(match_vector)
            
            output_file = json_file.replace('.json', '.txt')
            output_path = os.path.join(output_folder, output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                for match in default_matches:
                    f.write(' '.join(str(x) for x in match) + '\n')
    
    # 创建提交文件
    if processed_count > 0:
        try:
            shutil.make_archive("balanced_final_submission", 'zip', output_folder)
            logger.info(f"ZIP文件已创建: balanced_final_submission.zip")
        except Exception as e:
            logger.error(f"创建ZIP文件失败: {e}")
    
    logger.info(f"=== 处理完成: {processed_count} 个文件 ===")

if __name__ == "__main__":
    main()
