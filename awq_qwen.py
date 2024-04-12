"""这个是讲任务分解模型和任务执行模型蒸馏到一个模型后进行awq的量化"""

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import json
import re
import logging
import torch

# Specify paths and hyperparameters for quantization
model_path = "/ai/ld/remote/Qwen1.5-main/examples/sft/output_qwen/tmp-checkpoint-200"#这个是被量化之前的模型地址
quant_path = "/ai/ld/remote/Qwen-main/output_qwen/quant_model/qwen_1.5_14b_awq_int4"#这个是被量化之后的模型保存地址
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
max_len = 1024



with open('/ai/ld/remote/Qwen-main/get_subtask/data_process/train_react.json','r',encoding='utf-8') as f:
    datas = json.load(f)
quant_data=[]
for i in datas:
    new_line=[]
    assert len(i['conversations'])==2
    for j in i['conversations']:
        new_data={"role": "system", "content": "You are a helpful assistant."}
        new_data['role']=j['from']
        content=re.sub('Thought:Thought:','Thought:',j['value'])
        content=re.sub('\nuser\nAnswer','\nAnswer',content)
        new_data['content']=content
        new_line.append(new_data)
    quant_data.append(new_line)

tokenizer = AutoTokenizer.from_pretrained(model_path)

data = []
for msg in quant_data:
    text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
    text_id=tokenizer.encode(text)
    text_id=text_id[:max_len]+[151645,198]
    text=tokenizer.decode(text_id)
    data.append(text.strip())

# Load your tokenizer and model with AutoGPTQ
model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto", safetensors=True)

import logging

# 设置基本的日志格式和级别
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# 创建一个FileHandler，用于将日志保存到指定文件
file_handler = logging.FileHandler("/ai/ld/remote/Qwen-main/output_qwen/quant_model/qwen_1.5_14b_int4/quantize.log")
file_handler.setLevel(logging.INFO)  # 可以设置不同的日志级别

# 使用与console相同的formatter
formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
file_handler.setFormatter(formatter)

# 将FileHandler添加到root logger
logging.getLogger().addHandler(file_handler)

# 现在，日志既会在控制台显示，也会被写入到'app.log'文件中
logging.info("This message will be saved to the log file and displayed in console.")

model.quantize(tokenizer, quant_config=quant_config, calib_data=data)
model.save_quantized(quant_path,safetensors=True, shard_size="4GB")
tokenizer.save_pretrained(quant_path)
