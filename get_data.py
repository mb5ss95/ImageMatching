import requests
import json
import sift_bfmatcher as sb

reportnum = '20060020008708'
ansick = 'http://scm.ansick.com'

url1 = requests.get("http://scm.ansick.com/product/ansick_product_same.php?c=target&reportnum="+reportnum)
data1 = json.loads(url1.text)['data']


for i in data1:
    img_name1 = ansick + i['img']
    item_name1 = i['name']
    requests.post('http://scm.ansick.com/product/ansick_naver_search.php?KEYWORD='+item_name1)
    url2 = requests.get('http://scm.ansick.com/product/ansick_product_same.php?c=compare&keyword='+item_name1)
    data2 = json.loads(url2.text)['data']
    for j in data2:
        img_name2 = ansick + j['img']

        sb.start(img_name1, img_name2)
        
