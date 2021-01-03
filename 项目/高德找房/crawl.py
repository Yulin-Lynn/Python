#!/usr/bin/env python3
# 实验楼：https://www.lanqiao.cn/courses/599/learning/?id=1978
# 爬取链接：https://sh.58.com/pinpaigongyu/?PGTID=0d100000-0000-2616-b5e0-1220762fd578&ClickID=2
#!/usr/bin/env python3
from bs4 import BeautifulSoup # 网页解析模块
import requests # 网络请求模块
import csv # csv 文件模块
import time
import lxml
# 网址
url = "https://sh.58.com/pinpaigongyu/pn/{page}/?minprice=600_1000"

page = 0

csv_file = open("rent.csv","w",encoding="utf-8")
csv_writer = csv.writer(csv_file, delimiter=",")

while True:
    page += 1
    print("fetch: ", url.format(page=page))
    time.sleep(1)
    # 抓取目标页面
    response = requests.get(url.format(page=page))
    # 设置编码模式
    response.encoding = "utf-8"

    # 创建一个BeautifulSoup对象，获取页面正文
    html = BeautifulSoup(response.text,features="lxml")
    # 获取当前页面的房子信息
    house_list = html.select(".list > li")

    # 循环至读不到房源时结束
    if not house_list:
        break
    for house in house_list:
        house_title = house.select("h2")[0].string
        house_url = house.select("a")[0]["href"]
        # 对标题进行分隔
        house_info_list = house_title.split()
        # 如果第二列是公寓名则取第一列作为地址
        if "公寓" in house_info_list[1] or "青年社区" in house_info_list[1]:
            house_location = house_info_list[0]
        else:
            house_location = house_info_list[1]

        house_money = house.select(".money")[0].select("b")[0].string
        
        csv_writer.writerow([house_title,house_location,house_money,house_url])

csv_file.close()