import pandas as pd
import ast
from datasets import Dataset, DatasetDict

df = pd.read_json("/root/autodl-tmp/workspace/train/data/data/gdpo_prompts.json", lines=True) 

simpo_data_list = {"chosen": [], "rejected": []}

for _, row in df.iterrows():
    # 解析 context 字符串为 Python 列表。如果已经是 list 则无需解析
    context = row["context"]
    if isinstance(context, str):
        try:
            context = ast.literal_eval(context)
        except:
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

# 划分训练集和测试集（SimPO trainer 默认期望有 test split 用于 eval）
dataset_dict = hf_dataset.train_test_split(test_size=0.05, seed=42)

# 保存为 Hugging Face 磁盘格式
dataset_dict.save_to_disk("./local_simpo_dataset")
print("数据已成功转换为 SimPO 格式并保存至 ./local_simpo_dataset")