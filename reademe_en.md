# Progress
In October 2024, [AutoPlan2](https://github.com/LDLINGLINGLING/AutoPlan) was launched, significantly reducing the data construction cost for AutoPlan and focusing on constructing complex agent data for professional fields during cold start.

In September 2024, high-quality [cold start function call data construction](https://github.com/OpenBMB/MiniCPM-CookBook/tree/main/agent_demo) was achieved.

# AutoPlan
This project mainly accomplished complex task planning and execution in the military domain based on large models, utilizing an improved ReAct technology for long-chain agent execution. This repository is grateful for the strong support of Mr. Yin Junxi, who made a significant contribution to this project.
## Basic Principles:
The first version of the project's principle is as follows,
![Principle Diagram](https://github.com/LDLINGLINGLING/AutoPlan/assets/47373076/e6087bbd-b1cf-49de-a3d2-b84eb24da9fa)
The second version distilled the data from the first version, merging the two models into a smaller one, addressing the shortcomings of the first version such as `lack of multi-turn dialogue capability, excessive GPU memory usage, low inference efficiency, overcomplication of simple tasks, and lack of daily conversation functionality`.

## Usage
There are two ways to use this, either by training a task planning dataset (for the first version) or a fully distilled dataset (for the second version).
### Usage 1: Training Task Planning Dataset
`train_plan.json` and `test_plan.json` are the training and testing datasets for `task planning`. They can be used to train within qwen1/qwen1.5, after which the qwen model will acquire the ability to plan tasks.
![Training Data Example](https://github.com/LDLINGLINGLING/AutoPlan/assets/47373076/5b01b9d4-bf52-4502-b910-c3f8a8851417)

Change the default value of `allparams_split_task_chain` in `main.py` to the trained task planning qwen model. Set `execute_model_path` to the address of the qwen72b model and set `execute_reflexion` to `false`. Keep other parameters unchanged and run to obtain both task planning and execution capabilities.

### Usage 2: Training Distilled Task Execution Dataset
`train_react.json` and `test_react.json` are the datasets distilled from the task planning and task execution models, including manual annotations, which cover both `task planning and execution steps`.
Place `train_react.json` in qwen1/qwen1.5 for training, to integrate task planning and execution abilities into a single model; it is recommended to use qwen1.5 14b for training.
After training, set the `allparams_split_task_chain` default value in `main.py` to `false`. Change `execute_model_path` to the address of the trained model, and set `execute_reflexion` to `false`.
Inference will yield a model that has both task planning and execution capabilities.

## Demonstration of Results
### Task Planning Phase Results:
![Task Planning Result](https://github.com/LDLINGLINGLING/AutoPlan/assets/47373076/d4faf07c-2979-4cec-a21a-8cbe3442386c)

### Task Execution Phase Results:
![Task Execution Result 1](https://github.com/LDLINGLINGLING/AutoPlan/assets/47373076/389d22fe-e1e5-4595-8de9-d0683524bd93)
![Task Execution Result 2](https://github.com/LDLINGLINGLING/AutoPlan/assets/47373076/6f2b1dcc-4572-425a-8e7b-04a8a73e363e)
