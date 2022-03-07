# Atari - Solving Games with AI 🤖

The purpose of this repository is to demonstrate how ZenML enables you to build complex applications in an easy way and Use Zenml for various classes of tasks.

ZenML is an extensible, open-source MLOps framework to create production-ready machine learning pipelines. Built for data scientists, it has a simple, flexible syntax, is cloud- and tool-agnostic, and has interfaces/abstractions that are catered towards ML workflows.

At its core, ZenML pipelines execute ML-specific workflows from sourcing data to splitting, preprocessing, training, all the way to the evaluation of results and even serving. There are many built-in batteries to support common ML development tasks. ZenML is not here to replace the great tools that solve these individual problems. Rather, it integrates natively with popular ML tooling and gives standard abstraction to write your workflows.

Within this repo, we will use ZenML to build pipelines for developing One of the most powerful reinforcement learning algorithms which is DQN, which leans to solve Atari Games using AI.

The demo for a trained model which solves Atari is following:-
![](_assets/demo.gif)

## 🐍 Python Requirements

Before running this project, you have to install some python packages in your environment which you can do by following steps:

```
git clone https://github.com/zenml-io/zenfiles.git
cd Atari-Game-Player
pip install -r requirements.txt
```

## 📓 Diving into the code

We're ready to go now. You can run the code, using the `run_pipeline.py` script.

```
python run_pipeline.py train
```

## 📓 Explanation of Code

- `DQN/model.py` This file consists all the utility functions and classes which we need for developing our steps, all the classes and functions are explained in detail in their respective docstring.

- `pipelines/training_pipeline` This file makes the full pipeline for training the model, we had made use of Zenml pipeline module to make it.

- `steps/game_wrap.py` It is a wrapper for the Gym environment. It will manage the state fed to the DQN.
- `steps/build_dqn.py` It builds the keras model
- `steps/replay_buffer.py` It takes care of managing the stored experiences and sampling from them on demand.
- `steps/agent.py` It will put together the Keras DQN model (including the target network) and the ReplayBuffer. It will take care of things like choosing the action and performing gradient descent.
- `steps/get_information_meta.py` It returns a set of lists like frame number, rewards, and loss lists.
- `steps/train.py` It trains your model

Above are a short summary of what every step does!!!
