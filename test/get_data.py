import requests
import json
import sift_bf_ssim_polygon as sp

# reportnum = '20040020028607'
# # 20040020028607 200600200032064 200400170061194 201600080352 20040020028606
# ansick = 'http://scm.ansick.com'

# url1 = requests.get("https://scm.ansick.com/product/ansick_product_same.php?c=target&reportnum="+reportnum)
# data1 = json.loads(url1.text)['data']


# for i in data1:
#     img_name1 = ansick + i['img']
#     item_name1 = i['name']
#     requests.post('https://scm.ansick.com/product/ansick_naver_search.php?KEYWORD='+item_name1)
#     url2 = requests.get('https://scm.ansick.com/product/ansick_product_same.php?c=compare&keyword='+item_name1)
#     data2 = json.loads(url2.text)['data']
#     for j in data2:
#         img_name2 = ansick + j['img']
url1 = "https://www.shinailbo.co.kr/news/photo/202110/1466366_661801_2145.jpg"
url2 = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F36W98%2FbtqDVV2stjO%2FOxObRb2T25g0oiZ9njZPHK%2Fimg.jpg"

sp.start(url1, url2)