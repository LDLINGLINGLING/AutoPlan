# AutoPlan
本项目主要完成了在军事领域下基于大模型的复杂任务规划和执行，使用到了改进的react技术进行长链条的agent执行。本仓库要感谢尹俊希老帅哥的大力支持，为本项目做出了非常大的帮助<br>
##  基本原理：<br>
本项目第一版原理如下图，<br>
![image](https://github.com/LDLINGLINGLING/AutoPlan/assets/47373076/e6087bbd-b1cf-49de-a3d2-b84eb24da9fa)
第二版将第一版进行了数据蒸馏，将上图中两个模型合并到一个较小模型中，解决了第一版存在的`多轮对话能力缺失，显存占用过高、推理效率低，简单任务复杂化和日常对话功能缺失`等不够完善的地方。
## usage
分为训练任务规划数据集（用于第一版）或者训练蒸馏后的完整数据集（用于第二版）两种
###  用法1、训练任务规划数据集：<br>
train_plan.json和test_plan.json分别为`进行任务规划`的训练数据集和测试数据集。可以放入qwen1/qwen1.5中训练，训练后qwen模型可获得任务规划能力。<br>
![image](https://github.com/LDLINGLINGLING/AutoPlan/assets/47373076/5b01b9d4-bf52-4502-b910-c3f8a8851417)

将main.py文件中allparams_split_task_chain的default值改为训练后的任务规划qwen模型。将execute_model_path改为qwen72b的的模型地址，execute_reflexion改为false。其他不变，运行即可获得任务规划和执行能力。<br>

###  用法2、训练蒸馏后的带有任务执行数据集：<br>
train_react.josn和test_react.json分别为对任务规划和任务执行两个模型蒸馏出来的数据，并且进行人工标注的数据，其中同时包括了`任务规划和任务执行步骤`。<br>
将train_react.json放到qwen1/qwen1.5内进行训练，可将任务规划和任务执行能力导入同一个模型，建议使用qwen1.5 14b进行训练.<br>
训练完成后将main.py文件中allparams_split_task_chain的default值改为false。将execute_model_path改为以上模型训练的模型地址，execute_reflexion改为false。<br>
推理可得一个模型同时获得任务规划和任务执行两个效果。
## 效果展示
###   任务规划阶段效果如下：
![oy060f7h](https://github.com/LDLINGLINGLING/AutoPlan/assets/47373076/d4faf07c-2979-4cec-a21a-8cbe3442386c)

###   任务执行阶段效果如下:
<img width="981" alt="294754670-46c7ed17-197f-487a-b9bc-893c49eaba36" src="https://github.com/LDLINGLINGLING/AutoPlan/assets/47373076/389d22fe-e1e5-4595-8de9-d0683524bd93">


续上
![image](https://github.com/LDLINGLINGLING/AutoPlan/assets/47373076/6f2b1dcc-4572-425a-8e7b-04a8a73e363e)



