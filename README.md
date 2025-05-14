


https://github.com/user-attachments/assets/903adce1-8938-4869-a22a-92da3f328f28



# 🐍 Snake RL - Q-Learning

This project implements the classic Snake game trained using the **Q-Learning** algorithm. It uses `pygame` for the visual environment and a `QLearningAgent` class that learns through trial and error.

---

## 🎮 How It Works

- **Available modes**:
  - **Training**: The agent learns from scratch using Q-Learning.
  - **Testing**: The agent plays using a saved Q-Table without further learning.

- **States**: Represented as tuples encoding the snake’s environment.
- **Actions**:
  - 0 = keep straight
  - 1 = turn left
  - 2 = turn right
- **Rewards**:
  - Eating food: +1
  - Crashing: -1
  - Regular movement: small penalty to encourage shorter paths

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install pygame numpy


## ⚙️  Run

```bash
python main.py
