# windows curl发送post请求
# curl -X POST http://127.0.0.1:8001/recipe/ -H "Content-Type: application/json" -d "{\"label\":\"haha099\",\"source\":\"ii\",\"url\":\"http://12iopfad.com\",\"submitter_id\": 9}"


# python 发送post请求
import httpx
import json
from pydantic import BaseModel, HttpUrl


# 创建客户端
client = httpx.Client()
 
# 要发送的数据
data = {
    "label": "vaop",
    "source": "pp",
    "url": "https://127.0.0.1:8001/recipe",
    "submitter_id": 5
}
 
# 将数据转换为JSON格式
json_data = json.dumps(data).encode('utf-8')
 
# 发送POST请求
response = client.post('http://127.0.0.1:8001/recipe/', content=json_data)
 
# 打印响应内容
print(response.status_code)
print(response.text)
 
# 关闭客户端
client.close()
