from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch
from awq import AutoAWQForCausalLM
device = "cuda" # the device to load the model onto
model_path='/ai/ld/pretrain/qwen1.5_awq_int4_72b/qwen/Qwen1___5-72B-Chat-AWQ/'
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path, # the quantized model
    device_map="auto",
    trust_remote_code=True,
)
#model = AutoAWQForCausalLM.from_quantized(model_path)
print('模型加载完成')

prompt = "\nAnswer the following questions as best you can. You have access to the following tools:\n\ngoogle_search: Call this tool to interact with the 谷歌搜索 API. What is the 谷歌搜索 API useful for? 谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。 Parameters: [{\"name\": \"search_query\", \"description\": \"搜索关键词或短语\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\nmilitary_information_search: Call this tool to interact with the 军事情报搜索 API. What is the 军事情报搜索 API useful for? 军事情报搜索是一个通用搜索引擎，可用于访问军事情报网、查询军网、了解军事新闻等。 Parameters: [{\"name\": \"search_query\", \"description\": \"搜索关键词或短语\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\naddress_book: Call this tool to interact with the 通讯录 API. What is the 通讯录 API useful for? 通讯录是用来获取个人信息如电话、邮箱地址、公司地址的软件。 Parameters: [{\"name\": \"person_name\", \"description\": \"被查询者的姓名\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\nQQ_Email: Call this tool to interact with the qq邮箱 API. What is the qq邮箱 API useful for? qq邮箱是一个可以用来发送合接受邮件的工具 Parameters: [{\"E-mail address\": \"E-mail address\", \"description\": \"对方邮箱的地址 发给对方的内容\", \"required\": true, \"schema\": {\"type\": \"string\"}}, {\"E-mail content\": \"E-mail_content\", \"description\": \"发给对方的内容\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\nimage_gen: Call this tool to interact with the 文生图 API. What is the 文生图 API useful for? 文生图是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL Parameters: [{\"name\": \"prompt\", \"description\": \"英文关键词，描述了希望图像具有什么内容\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\nSituation_display: Call this tool to interact with the 态势显示 API. What is the 态势显示 API useful for? :态势显示是通过输入目标位置坐标和显示范围，从而显示当前敌我双方的战场态势图像，并生成图片 Parameters: [{\"coordinate\": \"[coordinate_x,coordinate_y]\", \"description\": \"目标位置的x和y坐标\", \"required\": true, \"schema\": {\"type\": \"string\"}}, {\"radio\": \"radio\", \"description\": \"态势图像显示的范围半径,单位是km,默认值为300km\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\ncalendar: Call this tool to interact with the 万年历 API. What is the 万年历 API useful for? 万年历获取当前时间的工具 Parameters: [{\"time\": \"time_query\", \"description\": \"目标的地点\", \"location\": \"location_query\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\nmap_search: Call this tool to interact with the 地图 API. What is the 地图 API useful for? 地图是一个可以查询地图上所有单位位置信息的工具，返回所有敌军的位置信息。 Parameters: [{\"lauch\": \"yes\", \"description\": \"yes代表启用地图搜索\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\nknowledge_graph: Call this tool to interact with the 知识图谱 API. What is the 知识图谱 API useful for? 知识图谱是输入武器种类获取该武器的属性，也可以输入某种属性获得所有武器的该属性 Parameters: [{\"weapon\": \"weapon_query\", \"description\": \"武器名称,比如飞机、坦克,所有武器\", \"required\": true, \"schema\": {\"type\": \"string\"}}, {\"attribute\": \"attribute\", \"description\": \"输出武器的该属性：射程/速度/重量/适应场景/克制武器/所有属性\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\nweapon_launch: Call this tool to interact with the 武器发射按钮 API. What is the 武器发射按钮 API useful for? 武器发射按钮是可以启动指定武器打击指定目标位置工具。 Parameters: [{\"weapon_and_coordinate\": [\"weapon_query\", \"target_name\", [\"x\", \"y\"]], \"description\": \"被启动的武器名称 被打击的目标名称 被打击目标的坐标地点\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\ndistance_calculation: Call this tool to interact with the 距离计算器 API. What is the 距离计算器 API useful for? 可以根据目标单位和地图api查询的位置信息，计算出地图上所有其他单位与目标单位的距离 Parameters: [{\"target_and_mapdict\": {\"weapon_query\": [\"x1\", \"y1\"], \"unit2\": [\"x2\", \"y2\"], \"unit3\": [\"x3\", \"y3\"], \"unit4\": [\"x4\", \"y4\"]}, \"description\": \"包括目标单位在内的所有地图上单位的名称和位置参数:{被计算的单位名称:[该单位的x坐标,该单位的y坐标],被计算的另外一个单位名称:[该单位的x坐标,该单位的y坐标],地图上的其他单位名称(可省略):[该单位的x坐标,该单位的y坐标](可省略)}\", \"required\": true, \"schema\": {\"type\": \"string\"}}] Format the arguments as a JSON object.\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [google_search, military_information_search, address_book, QQ_Email, image_gen, Situation_display, calendar, map_search, knowledge_graph, weapon_launch, distance_calculation]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can be repeated zero or more times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: 敌方坦克出现在坐标[20,30],请使用合适的武器进行打击,结果发送给张三指挥官。\n"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content":prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    max_new_tokens=512,
    max_length=4096
)

model_inputs = tokenizer([text], return_tensors="pt").to(device)
start=time.time()
generated_ids = model.generate(
    model_inputs.input_ids,
    top_k=1,
    do_sample=False,
    max_length=4096,
    eos_token_id=tokenizer.encode('<|im_end|>'),  
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
end=time.time()
print('共花费时间')
print(end-start)

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
print(response)