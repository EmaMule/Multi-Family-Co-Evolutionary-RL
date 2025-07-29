# A Multi-Family Co-Evolutionary Approach for Multi-Agent Reinforcement Learning

Project developed for the course of "Reiforcement Learning" during the MSc in Artificial Intelligence and Robotics at Sapienza University of Rome, A.Y. 2024-2025.

## Evolutive Multi-Family Strategies (EMS)
<p align="center">
  <img src="./figures/ems.png" alt="First Approach" width="800"/>
</p>

## Genetic Multi-Family Algorithm (GMA)
<p align="center">
  <img src="./figures/gma.png" alt="First Approach" width="800"/>
</p>

## How to run

1. Clone this repository 
    ```bash
    git clone https://github.com/EmaMule/RLProject.git
    ```
2. Install the requirements
   ```bash
   pip install -r requirements.txt
   ```
3. For executing the first algorithm (EMS) use the following command:
   ```bash
   python ./main.py EMS
   ```
4. For executing the second algorithm (GMA) use the following command:
   ```bash
   python ./main.py GMA
   ```

Alternatively you can use the provided notebook.ipynb by loading it on google colab or kaggle environments.

# Acknowledgement

The games considered are multi-agent environments from [PettingZoo's classic environments](https://pettingzoo.farama.org/environments/classic/)

The inspiration for the work comes from [Daan Klijn and A.E. Eiben. 2021. A coevolutionary approach to deep multi-agent reinforcement learning](https://arxiv.org/pdf/2104.05610). 

