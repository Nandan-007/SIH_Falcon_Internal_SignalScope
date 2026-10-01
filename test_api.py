import urllib.request
import json

url_health = "http://127.0.0.1:8000/health"
url_info = "http://127.0.0.1:8000/model-info"
url_predict = "http://127.0.0.1:8000/predict"
img_path = r"S:\gpu_training\test_images\real\-2X3vAmt.jpg"

try:
    print("Testing /health")
    req = urllib.request.Request(url_health)
    with urllib.request.urlopen(req) as res:
        print(json.dumps(json.loads(res.read()), indent=2))
    
    print("\nTesting /model-info")
    req = urllib.request.Request(url_info)
    with urllib.request.urlopen(req) as res:
        print(json.dumps(json.loads(res.read()), indent=2))
        
    print("\nTesting /predict")
    with open(img_path, 'rb') as f:
        file_data = f.read()
    
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    
    body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; filename=\"-2X3vAmt.jpg\"\r\n"
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode('utf-8') + file_data + f"\r\n--{boundary}--\r\n".encode('utf-8')
    
    req = urllib.request.Request(url_predict, data=body, method='POST')
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    req.add_header('Content-Length', str(len(body)))
    
    with urllib.request.urlopen(req) as res:
        print(json.dumps(json.loads(res.read()), indent=2))
        
except Exception as e:
    import traceback
    traceback.print_exc()

