# UCEIS Multi-task Learning Project

This project implements a multi-task learning approach for UCEIS (Ulcerative Colitis Endoscopic Index Severity) classification using deep learning models.

## Environment Configuration

### Requirements

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### GPU Requirements

- **Recommended GPU**: NVIDIA GPU with 12GB+ VRAM
- **Minimum GPU**: NVIDIA GPU with 8GB VRAM
- **CUDA Version**: CUDA 11.8 (compatible with PyTorch 2.0.1+cu118)

### Project Structure

```
UCEIS-MLT-Project/
├── run_training.py          # Training script
├── run_evaluate.py         # Evaluation script
├── multi_task_model.py      # Multi-task model definitions
├── model_config.py          # Model configurations
├── coatnet_training.py     # Pure CoAtNet model
├── SimAM.py               # Attention mechanism
├── requirements.txt         # Python dependencies
├── kfold/                 # Cross-validation data
├── models/                # Saved model weights
└── results/               # Evaluation results
```

## Training

### Data Preparation

Ensure your data is organized as follows:

1. **Cross-validation data**: Place training and validation CSV files in the `kfold/` directory
   - `train_fold0.csv`, `val_fold0.csv` to `train_fold4.csv`, `val_fold4.csv`
   
2. **Images**: Store images in your dataset directory
   - Default path: `./images`
   - Modify the `images_dir` parameter in scripts if using a different path

### Running Training

Start the training process:

```bash
python run_training.py
```

### Training Options

The training script provides the following options:

1. **Full 5-fold cross-validation training**
   - Trains all 5 folds sequentially
   - Automatically saves best models for each fold

2. **Selective fold training**
   - Choose specific folds to train (1-5)
   - Useful for resuming interrupted training or testing specific folds

### Training Configuration

- **Backbone Network**: CoAtNet-3 (coatnet_3_rw_224)
- **Batch Size**: 24 (adjustable based on GPU memory)
- **Learning Rate**: 8e-5
- **Epochs**: 100 with early stopping (patience=10)
- **Loss Function**: Focal Loss with dynamic weighting
- **Optimizer**: AdamW with cosine annealing scheduler

### Model Saving

Trained models are saved in the `models/` directory with the following naming convention:
- `best_coatnet3_fold{fold}.pth` - Best model for each fold
- `checkpoint_coatnet3_fold{fold}.pth` - Training checkpoints for resuming

## Evaluation

### Running Evaluation

Evaluate trained models using the evaluation script:

```bash
python run_evaluate.py
```

### Evaluation Options

1. **5-fold cross-validation evaluation**
   - Evaluates all available trained folds
   - Aggregates results across folds

2. **Single model evaluation**
   - Evaluate a specific trained model
   - Specify model file and test data

### Evaluation Metrics

The evaluation script computes the following metrics for each task (vascular, bleeding, ulceration):

- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted precision score
- **Recall**: Weighted recall score
- **F1 Score**: Weighted F1 score
- **Kappa Coefficient**: Cohen's kappa for inter-rater agreement
- **Quadratic Weighted Kappa (QWK)**: Weighted kappa for ordinal classification

### Results Output

Evaluation results are saved in the `results/` directory:

1. **CSV Files**: Detailed predictions and confidence scores
   - `test_fold{fold}_result.csv` - Results for each fold
   - Includes patient_id, image_name, actual/predicted labels, and confidence scores

2. **Confusion Matrices**: Visual confusion matrices for each task
   - `confusion_matrices_fold{fold}.png` - Confusion matrix visualization
   - Shows classification performance across all classes

### Task Labels

The model predicts three tasks with the following labels:

- **Vascular**: Normal, Mild, Severe (3 classes)
- **Bleeding**: None, Mild, Moderate, Severe (4 classes)
- **Ulceration**: None, Mild, Moderate, Severe (4 classes)
