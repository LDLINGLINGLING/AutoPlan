"""这个是讲任务分解模型和任务执行模型蒸馏到一个模型后进行gptq的量化"""
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
import json
import re
import logging
import torch

# Specify paths and hyperparameters for quantization
model_path = "/ai/ld/remote/Qwen1.5-main/examples/sft/output_qwen/tmp-checkpoint-200"#这个是被量化之前的模型地址
quant_path = "/ai/ld/remote/Qwen-main/output_qwen/quant_model/qwen_1.5_14b_int4"#这个是被量化之后的模型保存地址
quantize_config = BaseQuantizeConfig(
    bits=8, # 4 or 8#使用的量化位数
    group_size=128,#每个量化组的大小，gptq是以组为最小单元进行量化，
    damp_percent=0.01,#阻尼系数，用于量化过程中减少量化带来的数智震荡
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    static_groups=False,#是否使用静态组，静态组简化计算，但是精度下降
    sym=True,#是否对称量化
    true_sequential=True,# 是否使用真正的序列量化，设置为True可以提高量化精度，但可能增加计算量
    model_name_or_path=None,
    model_file_base_name="model",# 模型文件基础名称，用于保存量化后的模型,
)
max_len = 4096

# Load your tokenizer and model with AutoGPTQ
tokenizer = AutoTokenizer.from_pretrained(model_path)

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


data = []
for msg in quant_data:
    text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
    model_inputs = tokenizer([text])
    input_ids = torch.tensor(model_inputs.input_ids[:max_len], dtype=torch.int)
    data.append(dict(input_ids=input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id)))

model = AutoGPTQForCausalLM.from_pretrained(model_path, device_map='auto',quantize_config=quantize_config,trust_remote_code=True)



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


model.quantize(data, cache_examples_on_gpu=False,batch_size=1,use_triton=True)

model.save_quantized(quant_path, use_safetensors=True)
tokenizer.save_pretrained(quant_path)
