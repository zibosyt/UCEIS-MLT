import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
from SimAM import  insert_attention_to_coatnet


class ImprovedMultiTaskModel(nn.Module):
    """Improved multi-task model supporting multiple powerful backbone networks"""
    
    def __init__(self, num_classes_per_task=(3, 4, 4), backbone_name='coatnet_3_rw_224', 
                 pretrained=True, dropout_rate=0.5, use_attention=True):
        super(ImprovedMultiTaskModel, self).__init__()
        
        self.backbone_name = backbone_name
        self.use_attention = use_attention
        
        # Supported backbone network configurations
        self.backbone_configs = {
            # CoAtNet series - now fully supported!
            'coatnet_1_rw_224': {
                'pretrained_file': None,
                'input_size': 224
            },
            'coatnet_2_rw_224': {
                'pretrained_file': 'coatnet_2_rw_224.bin',  # Optional local weights
                'input_size': 224
            },
            'coatnet_3_rw_224': {
                'pretrained_file': None,
                'input_size': 224
            },
            # EfficientNet series - as alternatives
            'efficientnet_b2': {
                'pretrained_file': None,
                'input_size': 224
            },
            'efficientnet_b3': {
                'pretrained_file': None,
                'input_size': 224
            },
            'efficientnet_b4': {
                'pretrained_file': None,
                'input_size': 224
            },
            # ResNet series
            'resnet50': {
                'pretrained_file': None,
                'input_size': 224
            },
            'resnet101': {
                'pretrained_file': None,
                'input_size': 224
            }
        }
        
        # Create backbone network
        self.backbone = self._create_backbone(backbone_name, pretrained)
        
        # Get feature dimension
        self.feature_dim = self._get_feature_dim()
        
        # Feature fusion layer - enhance feature representation
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),  # Smaller dropout
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8)   # Medium dropout
        )
        
        # Task-specific classification heads - deeper networks
        self.vascular_head = self._make_classifier(512, num_classes_per_task[0], dropout_rate)
        self.bleeding_head = self._make_classifier(512, num_classes_per_task[1], dropout_rate)
        self.ulceration_head = self._make_classifier(512, num_classes_per_task[2], dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"Model created:")
        print(f"  - Backbone: {backbone_name}")
        print(f"  - Feature dimension: {self.feature_dim}")
        print(f"  - Using attention mechanism: {use_attention}")
        print(f"  - Dropout rate: {dropout_rate}")
    
    def _create_backbone(self, backbone_name, pretrained):
        """Create backbone network"""
        if backbone_name not in self.backbone_configs:
            raise ValueError(f"Unsupported backbone network: {backbone_name}")
        
        config = self.backbone_configs[backbone_name]
        
        try:
            # Create model without classification head
            backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
            print(f"Successfully created {backbone_name} backbone")
            
            # If local pretrained weights file exists, load it
            if config['pretrained_file'] and os.path.exists(config['pretrained_file']):
                print(f"Loading local pretrained weights: {config['pretrained_file']}")
                state_dict = torch.load(config['pretrained_file'], map_location='cpu')
                # Filter out classification head weights
                filtered_state_dict = {k: v for k, v in state_dict.items() 
                                     if not k.startswith('head') and not k.startswith('classifier')}
                backbone.load_state_dict(filtered_state_dict, strict=False)
            
            # Add attention mechanism to CoAtNet (if supported)
            if self.use_attention and 'coatnet' in backbone_name:
                try:
                    insert_attention_to_coatnet(backbone, attention_type='both')
                    print("Added attention mechanism to CoAtNet")
                except Exception as e:
                    print(f"Failed to add attention mechanism: {e}")
            
            return backbone
            
        except Exception as e:
            print(f"Failed to create {backbone_name}: {e}")
            print("Falling back to coatnet_2_rw_224")
            # Fall back to CoAtNet-2
            try:
                backbone = timm.create_model('coatnet_2_rw_224', pretrained=pretrained, num_classes=0)
                return backbone
            except:
                print("CoAtNet-2 also failed, falling back to efficientnet_b2")
                backbone = timm.create_model('efficientnet_b2', pretrained=pretrained, num_classes=0)
                return backbone
    
    def _get_feature_dim(self):
        """Get feature dimension of backbone network"""
        try:
            return self.backbone.num_features
        except AttributeError:
            # If no num_features attribute, get through forward pass
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = self.backbone(dummy_input)
                return features.shape[1]
    
    def _make_classifier(self, in_features, num_classes, dropout_rate):
        """Create classification head"""
        return nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),  # Smaller dropout for final layer
            nn.Linear(128, num_classes)
        )
    
    def _initialize_weights(self):
        """Initialize weights"""
        for module in [self.feature_fusion, self.vascular_head, self.bleeding_head, self.ulceration_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)
        
        # Feature fusion
        fused_features = self.feature_fusion(features)
        
        # Outputs for three tasks
        vascular_output = self.vascular_head(fused_features)
        bleeding_output = self.bleeding_head(fused_features)
        ulceration_output = self.ulceration_head(fused_features)
        
        return vascular_output, bleeding_output, ulceration_output
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'backbone': self.backbone_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'feature_dim': self.feature_dim
        }


class EnsembleModel(nn.Module):
    """Ensemble model integrating multiple different backbone networks"""
    
    def __init__(self, backbone_names, num_classes_per_task=(3, 4, 4), 
                 pretrained=True, dropout_rate=0.5):
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList()
        self.backbone_names = backbone_names
        
        # Create multiple different models
        for backbone_name in backbone_names:
            model = ImprovedMultiTaskModel(
                num_classes_per_task=num_classes_per_task,
                backbone_name=backbone_name,
                pretrained=pretrained,
                dropout_rate=dropout_rate
            )
            self.models.append(model)
        
        print(f"Ensemble model created with {len(self.models)} sub-models:")
        for name in backbone_names:
            print(f"  - {name}")
    
    def forward(self, x):
        # Get outputs from all models
        all_outputs = []
        for model in self.models:
            outputs = model(x)
            all_outputs.append(outputs)
        
        # Average ensemble for each task
        ensemble_outputs = []
        for task_idx in range(3):  # Three tasks
            task_outputs = [outputs[task_idx] for outputs in all_outputs]
            # Average ensemble
            ensemble_output = torch.stack(task_outputs).mean(dim=0)
            ensemble_outputs.append(ensemble_output)
        
        return tuple(ensemble_outputs)
    
    def get_ensemble_info(self):
        """Get ensemble model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'num_models': len(self.models),
            'backbone_names': self.backbone_names,
            'total_params': total_params,
            'trainable_params': trainable_params
        }


def create_model(backbone_name='coatnet_3_rw_224', num_classes_per_task=(3, 4, 4), 
                pretrained=True, dropout_rate=0.5, use_attention=True):
    """Convenience function to create improved multi-task model"""
    return ImprovedMultiTaskModel(
        num_classes_per_task=num_classes_per_task,
        backbone_name=backbone_name,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        use_attention=use_attention
    )


def create_ensemble_model(backbone_names=['coatnet_3_rw_224', 'coatnet_2_rw_224'], 
                         num_classes_per_task=(3, 4, 4), pretrained=True, dropout_rate=0.5):
    """Convenience function to create ensemble model"""
    return EnsembleModel(
        backbone_names=backbone_names,
        num_classes_per_task=num_classes_per_task,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )


if __name__ == "__main__":
    # Test different backbone networks
    backbone_names = [
        'coatnet_3_rw_224',
        'coatnet_2_rw_224', 
        'coatnet_1_rw_224',
        'efficientnet_b3'
    ]
    
    print("Testing different backbone networks:")
    for backbone_name in backbone_names:
        try:
            model = create_model(backbone_name=backbone_name, pretrained=False)
            info = model.get_model_info()
            print(f"\n{backbone_name}:")
            print(f"  Total parameters: {info['total_params']:,}")
            print(f"  Trainable parameters: {info['trainable_params']:,}")
            print(f"  Feature dimension: {info['feature_dim']}")
            
            # Test forward pass
            dummy_input = torch.randn(2, 3, 224, 224)
            outputs = model(dummy_input)
            print(f"  Output shapes: {[out.shape for out in outputs]}")
            
        except Exception as e:
            print(f"\n{backbone_name}: Creation failed - {e}")
    
    print("\n" + "="*50)
    print("Testing ensemble model:")
    try:
        ensemble = create_ensemble_model(
            backbone_names=['coatnet_2_rw_224', 'efficientnet_b3'], 
            pretrained=False
        )
        info = ensemble.get_ensemble_info()
        print(f"Ensemble model information:")
        print(f"  Number of sub-models: {info['num_models']}")
        print(f"  Total parameters: {info['total_params']:,}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        outputs = ensemble(dummy_input)
        print(f"  Output shapes: {[out.shape for out in outputs]}")
        
    except Exception as e:
        print(f"Ensemble model creation failed: {e}")