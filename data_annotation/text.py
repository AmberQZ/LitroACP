from pprint import pprint
from paddlenlp import Taskflow
schema = ['SUBJECT', 'ACTION', 'RESOURACE', 'CONDITION']
my_ie = Taskflow("information_extraction", schema=schema, task_path='./checkpoint/model_best')
pprint(my_ie("PC can insert numbers to the table in the night."))