# Reinforcement Learning Project

This repository contains implementations and experiments in **Reinforcement Learning (RL)**.  
The aim of this project is to explore and compare different RL algorithms across simulation environments, with a focus on understanding their performance, stability, and adaptability.

---

## 📌 Features
- Modular RL framework for training and evaluation
- Implemented algorithms:
  - Deep Q-Network (DQN)
  - Proximal Policy Optimization (PPO)
  - Advantage Actor-Critic (A2C)
- Configurable training pipeline (via YAML/JSON)
- Support for logging, plotting, and checkpointing
- Integration with **OpenAI Gym** and custom environments

---

## 📂 Repository Structure

RL_Code/
│── README.md # Project documentation
│── requirements.txt # Python dependencies
│── config/ # Training configurations
│── environments/ # Custom environments
│── models/ # Saved trained models
│── notebooks/ # Jupyter notebooks for experiments
│── src/ # Core RL implementation
│ ├── agents/ # RL agents (DQN, PPO, A2C, etc.)
│ ├── utils/ # Helper functions (logging, plotting, etc.)
│ └── train.py # Training script
│── results/ # Logs, plots, evaluation metrics



🔗 References

Sutton, R. S., & Barto, A. G. Reinforcement Learning: An Introduction

OpenAI Gym

Stable-Baselines3
