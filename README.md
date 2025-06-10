#PyMino-RL

A deep reinforcement learning and supervised learning framework for training an AI agent to play **PyMino**, a dynamic block-placement puzzle game inspired by classic Tetris and polyomino challenges.

---

##Overview

**PyMino-RL** leverages modern deep learning techniques—such as convolutional neural networks (CNNs), batch normalization, dropout, data augmentation, weighted loss sampling, and advanced learning rate scheduling—to train models that excel in complex block-placement puzzles. It supports both supervised learning and reinforcement learning paradigms and integrates features like curriculum learning and warmup schedules to improve training stability and performance.

---

## Features

**Convolutional Neural Networks** with batch normalization and dropout  
**Data Augmentation** with rotations, flips, and noise injection  
**Weighted Loss Sampling** based on game scores  
**Early Stopping** and **Learning Rate Scheduling** with warmup  
**Curriculum Learning** to train models progressively  
**Robust Evaluation** with scoring and variance metrics  
**GPU-accelerated** training with mixed precision (AMP)  
**Customizable Training Pipelines** via command-line arguments  

---

##Repository Structure

```
PyMino-RL/
│
├── pymino/
│   ├── core_game.py         # Game logic and mechanics
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── model.py             # PyMinoNet CNN architecture
│   └── utils.py             # Utility functions (augmentation, metrics, etc.)
│
├── train_pymino_bot_weighted_amp_v5_warmup_v3.py  # Main training script
├── pymino_train_eval_amp_v3.py                   # Evaluation script
│
├── requirements.txt                              # Python package dependencies
├── README.md                                     # This file!
└── LICENSE                                       # License information
```

---

##Installation

```bash
git clone https://github.com/ConnorOswalt/PyMino-RL.git
cd PyMino-RL
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

##Usage

###Train a Model

```bash
python train_pymino_bot_weighted_amp_v5_warmup_v3.py     --logs_folder="/path/to/logs"     --epochs1=80 --lr1=1e-4 --warmup_epochs1=3     --epochs2=80 --lr2=1.5e-4 --warmup_epochs2=3     --epochs3=80 --lr3=1e-4 --warmup_epochs3=3     --batch_size=512     --device="cuda"     --minimum_score=500     --max_score_cutoff=0
```

###Evaluate a Model

```bash
python pymino_train_eval_amp_v3.py     --model="path/to/your/model.pth"
```

---

##Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--logs_folder` | Path to folder containing PyMino game logs (.json) |
| `--epochs1/2/3` | Number of epochs for each training step |
| `--lr1/2/3` | Learning rate for each training step |
| `--warmup_epochs1/2/3` | Warmup epochs for each training step |
| `--batch_size` | Training batch size |
| `--device` | `cuda` or `cpu` |
| `--minimum_score` | Minimum score threshold to include logs |
| `--max_score_cutoff` | Upper score cutoff to filter out extremely high scores |

---

##Requirements

- Python 3.10+
- PyTorch 2.0+
- NVIDIA GPU + CUDA 12.2 (recommended)
- numpy, tqdm, json, argparse

Install them via:

```bash
pip install -r requirements.txt
```

---

##Results

After training on 80 epochs with warmup, models achieved:
- **Step 1 Validation Accuracy**: ~53%
- **Step 2 Validation Accuracy**: ~43%
- **Step 3 Validation Accuracy**: ~86%

Evaluation on 50 episodes yielded:
- **Average Score**: ~270-400
- **Best Single-Run Score**: >900

---

##Contributing

Contributions are welcome! If you’d like to submit a pull request or report an issue, please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) guide.

---

##License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

##Acknowledgments

- Inspired by the Blockudoku community and deep learning enthusiasts
- Special thanks to OpenAI’s GPT-4 for assisting with scripting and documentation

---
