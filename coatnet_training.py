import torch
import torch.nn as nn
import timm

class PureCoAtNetModel(nn.Module):
    def __init__(self, num_vascular=2, num_bleeding=2, num_ulceration=2):
        super(PureCoAtNetModel, self).__init__()
        
        self.backbone = timm.create_model('coatnet_3_rw_224', pretrained=False, num_classes=0)
        
        backbone_features = 1536  # Feature dimension of coatnet_3_rw_224
        
        # Classification heads for three tasks
        self.vascular_head = nn.Linear(backbone_features, num_vascular)
        self.bleeding_head = nn.Linear(backbone_features, num_bleeding)
        self.ulceration_head = nn.Linear(backbone_features, num_ulceration)
        
    def forward(self, x):
        # Extract features through backbone
        features = self.backbone(x)
        
        # Pass through each task head
        vascular_out = self.vascular_head(features)
        bleeding_out = self.bleeding_head(features)
        ulceration_out = self.ulceration_head(features)
        
        return [vascular_out, bleeding_out, ulceration_out]
    
    def get_model_info(self):
        """Get model information"""
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'backbone': 'coatnet_3_rw_224',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'feature_dim': 1536,  # Feature dimension of coatnet_3_rw_224
            'num_tasks': 3,
            'task_names': ['vascular', 'bleeding', 'ulceration']
        }


def create_pure_coatnet_model(backbone_name='coatnet_3_rw_224', num_classes_per_task=(3, 4, 4), pretrained=True, dropout_rate=0.5):
    """Create pure CoAtNet model (without additional attention mechanism)
    
    Args:
        backbone_name: Backbone network name
        num_classes_per_task: Number of classes for each task (vascular, bleeding, ulceration)
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout ratio
    """
    # Dynamically adjust feature dimension according to backbone name
    if 'coatnet_3' in backbone_name:
        backbone_features = 1536  # Feature dimension of coatnet_3_rw_224
    elif 'coatnet_2' in backbone_name:
        backbone_features = 1024  # Feature dimension of coatnet_2
    elif 'coatnet_1' in backbone_name:
        backbone_features = 640   # Feature dimension of coatnet_1
    elif 'coatnet_0' in backbone_name:
        backbone_features = 512   # Feature dimension of coatnet_0
    else:
        backbone_features = 1536  # Default value
    
    # Create model instance
    model = PureCoAtNetModel(
        num_vascular=num_classes_per_task[0],
        num_bleeding=num_classes_per_task[1],
        num_ulceration=num_classes_per_task[2]
    )
    
    # Load pretrained weights if needed
    if pretrained:
        try:
            # Create a temporary pretrained model to load weights
            temp_model = timm.create_model(backbone_name, pretrained=True)
            # Extract pretrained backbone weights
            backbone_state_dict = {}
            for name, param in temp_model.state_dict().items():
                if name.startswith('stem') or name.startswith('stages'):
                    backbone_state_dict[name] = param
            
            # Try to load backbone weights
            model.backbone.load_state_dict(backbone_state_dict, strict=False)
            print(f"Successfully loaded pretrained weights for {backbone_name}")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}, using random initialization")
    
    return model