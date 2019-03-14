import subprocess
import requests
import json
import time

args = json.load(open('data.json','r'))

proc = subprocess.Popen([args['solution_python_executor'],
                         '%s/flask_prep.py' % args['solution_folder']])

while True:
    try:
        requests.get('http://127.0.0.1:5112/started')
        time.sleep(5)
        break
    except:
        pass

print(proc.pid)
print('DONE')