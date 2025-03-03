from flask import Flask, request, jsonify
from paddlenlp import Taskflow


app = Flask(__name__)

# 设定为访问控制组件
schema = ['Subject', 'Action', 'Resource', 'Condition','Purpose']

my_ie = Taskflow("information_extraction", schema=schema, task_path='./checkpoint/model_best')


def convert(result):
    result = result[0]
    formatted_result = []
    for label, ents in result.items():
        for ent in ents:
            formatted_result.append(
                {
                    "label": label,
                    "start_offset": ent['start'],
                    "end_offset": ent['end']
                })

    return formatted_result


@app.route('/', methods=['POST'])
def get_result():
    text = request.json['text']
    print(text)
    result = my_ie(text)
    formatted_result = convert(result)

    return jsonify(formatted_result)


if __name__ == '__main__':
	# 设定ip跟端口
    app.run(host='0.0.0.0', port=5739)  
