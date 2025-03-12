<img width="719" alt="image" src="https://github.com/user-attachments/assets/d5bace9a-8cc2-4684-a1cb-6934b9101933" />

# Reinforcement Learning Project

## Description
This project implements reinforcement learning algorithms using the OpenAI Gym environment, specifically the Cliff Walking problem. It includes implementations of Q-learning and a random agent for comparison.

## Installation
To run this project, you need to have Python installed along with the following packages:
- gym
- numpy
- opencv-python

You can install the required packages using pip:
```bash
pip install gym numpy opencv-python
```

## Usage
### Q-learning
To run the Q-learning algorithm, execute the following command:
```bash
python q_learning.py
```
This will train the agent using the Q-learning algorithm and save the Q-table to a file.

### Evaluator
To evaluate the performance of the trained agent, run:
```bash
python evaluator.py
```
This will visualize the agent's actions in the Cliff Walking environment.

### Random Agent
To run a random agent in the environment, execute:
```bash
python random_agent.py
```
This will demonstrate the agent taking random actions until it reaches a terminal state.

## Environment Representation
The Cliff Walking environment is represented as a grid where:
- **S**: Start state
- **G**: Goal state
- **C**: Cliff (falling off results in a penalty)

```
| S |   |   |   |   |   |   |   |   |   |   | G |
|---|---|---|---|---|---|---|---|---|---|---|---|
|   | C | C | C | C | C | C | C | C | C | C |   |
```


## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License.

