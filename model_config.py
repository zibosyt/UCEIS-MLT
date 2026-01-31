"""
Model configuration file
Define configurations and recommended settings for different backbone networks
"""

# Backbone network configurations
BACKBONE_CONFIGS = {
    # CoAtNet series - now supported!
    'coatnet_2_rw_224': {
        'description': 'CoAtNet-2 (Original model)',
        'params': '73M',
        'input_size': 224,
        'batch_size': 32,
        'lr': 1e-4,
        'pretrained_file': 'coatnet_2_rw_224.bin',  # Optional local weights
        'memory_usage': 'Medium',
        'training_time': 'Medium',
        'recommended': True
    },
    'coatnet_3_rw_224': {
        'description': 'CoAtNet-3 (Larger and stronger)',
        'params': '168M', 
        'input_size': 224,
        'batch_size': 24,  # Smaller batch size
        'lr': 8e-5,        # Smaller learning rate
        'pretrained_file': None,
        'memory_usage': 'High',
        'training_time': 'High',
        'recommended': True
    },
    'coatnet_1_rw_224': {
        'description': 'CoAtNet-1 (Lightweight CoAtNet)',
        'params': '42M',
        'input_size': 224,
        'batch_size': 40,
        'lr': 1.2e-4,
        'pretrained_file': None,
        'memory_usage': 'Medium',
        'training_time': 'Medium',
        'recommended': True
    },
    
    # EfficientNet series - as alternatives
    'efficientnet_b2': {
        'description': 'EfficientNet-B2 (Efficient alternative)',
        'params': '9M',
        'input_size': 224,
        'batch_size': 40,
        'lr': 6e-4,
        'pretrained_file': None,
        'memory_usage': 'Low-Medium',
        'training_time': 'Medium',
        'recommended': True
    },
    'efficientnet_b3': {
        'description': 'EfficientNet-B3 (Larger alternative)',
        'params': '12M',
        'input_size': 224,
        'batch_size': 32,
        'lr': 5e-4,
        'pretrained_file': None,
        'memory_usage': 'Medium',
        'training_time': 'Medium',
        'recommended': True
    },
    'efficientnet_b4': {
        'description': 'EfficientNet-B4 (Large alternative)',
        'params': '19M',
        'input_size': 224,
        'batch_size': 24,
        'lr': 4e-4,
        'pretrained_file': None,
        'memory_usage': 'Medium',
        'training_time': 'Medium-High',
        'recommended': True
    },
    
    # ResNet series - classic choices
    'resnet50': {
        'description': 'ResNet-50 (Classic CNN)',
        'params': '26M',
        'input_size': 224,
        'batch_size': 32,
        'lr': 1e-3,
        'pretrained_file': None,
        'memory_usage': 'Medium',
        'training_time': 'Medium',
        'recommended': True
    },
    'resnet101': {
        'description': 'ResNet-101 (Deeper ResNet)',
        'params': '45M',
        'input_size': 224,
        'batch_size': 24,
        'lr': 8e-4,
        'pretrained_file': None,
        'memory_usage': 'Medium-High',
        'training_time': 'High',
        'recommended': True
    }
}

# Recommended backbone network combinations (for ensemble learning)
ENSEMBLE_CONFIGS = {
    'lightweight': {
        'backbones': ['coatnet_1_rw_224', 'efficientnet_b2'],
        'description': 'Lightweight ensemble (CoAtNet + EfficientNet)',
        'memory_usage': 'Medium',
        'training_time': 'Medium'
    },
    'balanced': {
        'backbones': ['coatnet_2_rw_224', 'efficientnet_b3'],
        'description': 'Balanced ensemble (Balance between performance and efficiency)',
        'memory_usage': 'Medium-High', 
        'training_time': 'Medium-High'
    },
    'powerful': {
        'backbones': ['coatnet_3_rw_224', 'coatnet_2_rw_224', 'efficientnet_b4'],
        'description': 'Powerful ensemble (Best performance)',
        'memory_usage': 'High',
        'training_time': 'High'
    },
    'diverse': {
        'backbones': ['coatnet_3_rw_224', 'resnet101', 'efficientnet_b4'],
        'description': 'Diverse ensemble (Different architecture combination)',
        'memory_usage': 'High',
        'training_time': 'High'
    }
}

# Training configurations
TRAINING_CONFIGS = {
    'fast': {
        'num_epochs': 50,
        'patience': 8,
        'warmup_epochs': 2,
        'description': 'Fast training (Suitable for testing)'
    },
    'standard': {
        'num_epochs': 100,
        'patience': 10,
        'warmup_epochs': 3,
        'description': 'Standard training (Recommended)'
    },
    'thorough': {
        'num_epochs': 150,
        'patience': 15,
        'warmup_epochs': 5,
        'description': 'Thorough training (Best performance)'
    }
}


def get_recommended_backbones():
    """Get recommended backbone networks"""
    return [name for name, config in BACKBONE_CONFIGS.items() if config['recommended']]


def get_backbone_config(backbone_name):
    """Get configuration for specified backbone network"""
    return BACKBONE_CONFIGS.get(backbone_name, None)


def print_backbone_info():
    """Print information for all backbone networks"""
    print("=" * 80)
    print("Available Backbone Networks:")
    print("=" * 80)
    
    for name, config in BACKBONE_CONFIGS.items():
        status = "Recommended" if config['recommended'] else "Optional"
        print(f"{status} {name}")
        print(f"    Description: {config['description']}")
        print(f"    Parameters: {config['params']}")
        print(f"    Recommended batch_size: {config['batch_size']}")
        print(f"    Recommended learning rate: {config['lr']}")
        print(f"    Memory usage: {config['memory_usage']}")
        print(f"    Training time: {config['training_time']}")
        print()


def print_ensemble_info():
    """Print ensemble configuration information"""
    print("=" * 80)
    print("Ensemble Learning Configurations:")
    print("=" * 80)
    
    for name, config in ENSEMBLE_CONFIGS.items():
        print(f"{name}")
        print(f"    Description: {config['description']}")
        print(f"    Backbones: {', '.join(config['backbones'])}")
        print(f"    Memory usage: {config['memory_usage']}")
        print(f"    Training time: {config['training_time']}")
        print()


def get_optimal_config_for_gpu(gpu_memory_gb):
    """Recommend optimal configuration based on GPU memory"""
    if gpu_memory_gb >= 24:  # RTX 4090, A100, etc.
        return {
            'backbone': 'coatnet_3_rw_224',
            'batch_size': 24,
            'ensemble': 'powerful'
        }
    elif gpu_memory_gb >= 16:  # RTX 4080, RTX 3090, etc.
        return {
            'backbone': 'coatnet_2_rw_224',
            'batch_size': 32,
            'ensemble': 'balanced'
        }
    elif gpu_memory_gb >= 12:  # RTX 4070 Ti, RTX 3080, etc.
        return {
            'backbone': 'coatnet_1_rw_224',
            'batch_size': 40,
            'ensemble': 'lightweight'
        }
    elif gpu_memory_gb >= 8:   # RTX 4060 Ti, RTX 3070, etc.
        return {
            'backbone': 'efficientnet_b3',
            'batch_size': 32,
            'ensemble': 'lightweight'
        }
    else:  # Below 8GB
        return {
            'backbone': 'efficientnet_b2',
            'batch_size': 40,
            'ensemble': None
        }


if __name__ == "__main__":
    print_backbone_info()
    print_ensemble_info()
    
    # Example: Recommend configuration based on GPU memory
    print("=" * 80)
    print("GPU Memory Recommended Configurations:")
    print("=" * 80)
    
    gpu_memories = [8, 12, 16, 24]
    for mem in gpu_memories:
        config = get_optimal_config_for_gpu(mem)
        print(f"{mem}GB GPU:")
        print(f"    Recommended backbone: {config['backbone']}")
        print(f"    Recommended batch_size: {config['batch_size']}")
        if config['ensemble']:
            print(f"    Recommended ensemble: {config['ensemble']}")
        print()