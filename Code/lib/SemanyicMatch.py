import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class SemanticMatchingModel(nn.Module):
    def __init__(self, image_model, text_model):
        super(SemanticMatchingModel, self).__init__()
        self.image_model = image_model
        self.text_model = text_model
        
    def forward(self, image, text):
        image_features = self.image_model(image)
        text_features = self.text_model(text)
        
        # 计算图像和文本的相似度
        similarity = F.cosine_similarity(image_features, text_features, dim=1)
        
        return similarity