import pandas as pd
import json
from datasets import Dataset

df = pd.read_json("/root/autodl-tmp/workspace/train/data/data/gdpo_prompts.json", lines=True) 

simpo_data_list = {"chosen": [], "rejected": []}

for index, row in df.iterrows():
    context = row["context"]
    # 解析 context 字符串为 Python 列表
    if isinstance(context, str):
        try:
            context = json.loads(context) # 【修改点】使用 json.loads 替代 ast
        except Exception as e:
            print(f"第 {index} 行解析出错: {e}")
            continue
            
    # 构建 chosen 的完整对话流
    chosen_messages = context.copy()
    chosen_messages.append({"role": "assistant", "content": str(row["chosen"])})
    
    # 构建 rejected 的完整对话流
    rejected_messages = context.copy()
    rejected_messages.append({"role": "assistant", "content": str(row["rejected"])})
    
    simpo_data_list["chosen"].append(chosen_messages)
    simpo_data_list["rejected"].append(rejected_messages)

# 转换为 Hugging Face Dataset
hf_dataset = Dataset.from_dict(simpo_data_list)

# 划分训练集和测试集
dataset_dict = hf_dataset.train_test_split(test_size=0.05, seed=42)

# 保存为 Hugging Face 磁盘格式
dataset_dict.save_to_disk("./local_simpo_dataset")

print(f"数据已成功转换！")
print(f"训练集大小: {len(dataset_dict['train'])} 条")
print(f"测试集大小: {len(dataset_dict['test'])} 条")