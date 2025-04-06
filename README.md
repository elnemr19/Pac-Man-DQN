
# 🕹️ Deep Q-Network (DQN) Agent for Ms. Pac-Man

A Deep Reinforcement Learning agent that learns to play **Ms. Pac-Man** using the **Deep Q-Network (DQN)** algorithm with a convolutional neural network, implemented in **TensorFlow** and trained using **ALE (Arcade Learning Environment)** via **Gymnasium**.




![training_results](https://github.com/user-attachments/assets/731ebefe-b392-4812-9d1e-1178c87205f6)

---

## 🚀 Project Overview

This project demonstrates the power of Deep Q-Learning in learning to play the Atari game *Ms. Pac-Man*. The agent uses visual input frames from the game, applies preprocessing (grayscale + downsampling), and stacks them to understand motion and spatial features. It learns through experience replay and target network updates.

---

## 🧠 Features

- ✅ **Convolutional Neural Network** for visual state processing  
- 🔁 **Experience Replay** for stable training  
- 🎯 **Target Network** for improved Q-learning stability  
- 📉 **Epsilon-Greedy** exploration strategy with linear decay  
- 📦 **Checkpointing** & training logs  
- 📊 Training visualizations with Matplotlib  

---

## 🛠️ Installation

> Make sure you're using a GPU-enabled environment (like [Kaggle Kernels](https://www.kaggle.com/kernels) or Colab) for best performance.

```bash
pip install memory_profiler psutil gymnasium ale-py tensorflow matplotlib
```

## 🧩 Environment Setup
This project uses the ALE/MsPacman-v5 environment from Gymnasium:

```python
import gymnasium as gym
env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
```


## 🏗️ Model Architecture
```text
Input: (88, 80, N_FRAMES)
→ Conv2D(32, 8x8, stride=4)
→ Conv2D(64, 4x4, stride=2)
→ Conv2D(64, 3x3, stride=1)
→ Flatten
→ Dense(512)
→ Output: Q-values for 9 discrete actions
```


## 📈 Training Details

- Episodes: 1000

- Replay Buffer Size: Up to 80,000

- Batch Size: 32–64 (based on available RAM)

- Frame Stack: 3–4

- Learning Rate: 0.0001–0.00025

- Target Network Update: Every 1000 steps

- Checkpoint Frequency: Every 100 episodes

- Min Replay Start Size: 10,000

> Training results are saved under `./pacman_models/` in my kaggle

## 📊 Sample Training Output

```yaml
Ep  900 | R: 1470.0 | Avg R: 1085.0 | ε: 0.010 | Steps: 568314
Ep  910 | R: 1010.0 | Avg R: 1102.0 | ε: 0.010 | Steps: 574856
...
Training completed in 307.28 minutes
Average reward: 759.9
Best reward: 3580.0

```

## 📚 References

[OpenAI Gym Documentation](https://www.gymlibrary.dev/)

[Atari Learning Environment (ALE)](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
