import json
import re

def process_jsonl_file(input_file, output_file):
    all_data = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            converted_data = convert_to_tokenized_format(data)
            all_data.append(converted_data)
    
  
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(all_data, outfile, ensure_ascii=False, indent=4)

def convert_to_tokenized_format(data):
    tokenized_text = re.findall(r'\w+|[^\w\s]', data["text"], re.UNICODE)
    ner = []
    
    current_char_index = 0
    token_spans = []
    for token in tokenized_text:
        start_index = data["text"].find(token, current_char_index)
        end_index = start_index + len(token) - 1
        token_spans.append((start_index, end_index))
        current_char_index = end_index + 1

    for entity in data["entities"]:
        entity_start = entity["start_offset"]
        entity_end = entity["end_offset"] - 1


        token_start = None
        token_end = None
        for i, (start, end) in enumerate(token_spans):
            if token_start is None and start <= entity_start <= end:
                token_start = i
            if token_end is None and start <= entity_end <= end:
                token_end = i
                break
        
        if token_start is not None and token_end is not None:
            ner.append([token_start, token_end, entity["label"]])

    return {
        "tokenized_text": tokenized_text,
        "ner": ner
    }


input_file = 'nonacp.jsonl'  
output_file = 'nonacp.json'  

process_jsonl_file(input_file, output_file)


import json


with open('acp_ext.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


allowed_tags = {"Resource", "Subject", "Action"}


for item in data:
    item['ner'] = [ner for ner in item['ner'] if ner[2] in allowed_tags]


with open('filtered_acp.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=None)  # 设置 indent 为 None

print("过滤完成，结果已保存为 'filtered_acp.json'")
