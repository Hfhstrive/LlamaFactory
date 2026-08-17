"""
Convert alpaca dataset into sharegpt format.

Usage: python convert_dataset.py --in_file alpaca_data.json --out_file alpaca_data_sharegpt.jsonl
"""

import argparse

from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True, help="input file name, csv or jsonl file.")
    parser.add_argument("--out_file", type=str, required=True, help="output file name, for example: out.jsonl")
    parser.add_argument("--data_type", type=str, default='alpaca',
                        choices=['alpaca', 'qa', 'sharegpt', 'messages'],
                        help="Input data type: alpaca, qa, sharegpt, or messages")
    parser.add_argument("--file_type", type=str, default='json',
                        choices=['json', 'jsonl', 'csv'],
                        help='input file type, json, jsonl or csv')
    args = parser.parse_args()
    print(args)

    data_files = {"train": args.in_file}

    # 根据文件类型加载数据
    if args.file_type == 'csv':
        if args.data_type in ['qa']:
            column_names = ['input', 'output']
        elif args.data_type in ['alpaca']:
            column_names = ['instruction', 'input', 'output']
        else:
            # 对于 messages 或 sharegpt 格式，不指定列名，让 load_dataset 自动检测
            raw_datasets = load_dataset('csv', data_files=data_files)
        if args.data_type in ['qa', 'alpaca']:
            raw_datasets = load_dataset('csv', data_files=data_files, column_names=column_names, delimiter='\t')
    elif args.file_type in ['json', 'jsonl']:
        raw_datasets = load_dataset('json', data_files=data_files)
    else:
        raise ValueError("File type not supported")

    ds = raw_datasets['train']


    def process_qa(examples):
        """处理 QA 格式：input 和 output 字段"""
        convs = []
        for q, a in zip(examples['input'], examples['output']):
            convs.append([
                {"from": "human", "value": q},
                {"from": "gpt", "value": a}
            ])
        return {"conversations": convs}


    def process_alpaca(examples):
        """处理 Alpaca 格式：instruction, input, output 字段"""
        convs = []
        for instruction, inp, output in zip(examples['instruction'], examples['input'], examples['output']):
            if inp and len(inp.strip()) > 0:
                instruction = instruction + '\n\n' + inp
            q = instruction
            a = output
            convs.append([
                {"from": "human", "value": q},
                {"from": "gpt", "value": a}
            ])
        return {"conversations": convs}


    def process_messages(examples):
        """
        处理 messages 格式：
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        转换为:
        {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
        """
        convs = []

        # role 映射表
        role_map = {
            'user': 'human',
            'assistant': 'gpt',
            'system': 'system',
            'human': 'human',
            'gpt': 'gpt'
        }

        # 遍历每条数据
        for i in range(len(examples['messages'])):
            messages = examples['messages'][i]
            conversation = []

            for msg in messages:
                # 获取角色，如果没有 role 字段，尝试从 from 字段获取
                role = msg.get('role', msg.get('from', 'human'))
                content = msg.get('content', msg.get('value', ''))

                # 映射角色
                mapped_role = role_map.get(role, role)

                conversation.append({
                    "from": mapped_role,
                    "value": content
                })

            convs.append(conversation)

        return {"conversations": convs}


    # 根据数据类型选择处理函数
    if args.data_type == 'alpaca':
        print("Processing Alpaca format...")
        ds = ds.map(process_alpaca, batched=True, remove_columns=ds.column_names, desc="Running process")
    elif args.data_type == 'qa':
        print("Processing QA format...")
        ds = ds.map(process_qa, batched=True, remove_columns=ds.column_names, desc="Running process")
    elif args.data_type == 'messages':
        print("Processing messages format...")
        # 确保 messages 字段存在
        if 'messages' not in ds.column_names:
            raise KeyError(f"Data type is 'messages' but 'messages' column not found. Available columns: {ds.column_names}")
        ds = ds.map(process_messages, batched=True, remove_columns=ds.column_names, desc="Running process")
    elif args.data_type == 'sharegpt':
        print("Processing ShareGPT format...")
        # 如果是 sharegpt 格式，需要重命名 items 为 conversations
        if "items" in ds.column_names:
            ds = ds.rename_column("items", "conversations")
        # 移除除了 conversations 以外的所有列
        columns_to_remove = [col for col in ds.column_names if col != 'conversations']
        if columns_to_remove:
            ds = ds.remove_columns(columns_to_remove)
    else:
        raise ValueError(f"Unsupported data type: {args.data_type}")

    # 保存转换后的数据集
    ds.to_json(f"{args.out_file}", lines=True, force_ascii=False)
    print(f"转换完成！输出文件: {args.out_file}")
    print(f"数据条数: {len(ds)}")
    print(f"列名: {ds.column_names}")
    print(f"示例数据: {ds[0]}")