import requests

for i in range(5):
    try:
        requests.get('http://127.0.0.1:5112/shutdown')
        time.sleep(5)
        break
    except:
        pass

