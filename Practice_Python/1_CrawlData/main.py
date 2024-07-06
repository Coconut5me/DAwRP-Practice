#%% - Import Lib
import requests
from bs4 import BeautifulSoup as bs


#%% - Pull data from Url
url = "https://tuoitre.vn/tet-nay-ha-noi-se-khong-mo-cua-pho-di-bo-ho-hoan-kiem-20240125120522211.htm"

response = requests.get(url)

# print(response.status_code)
#print(response.json()[0])

#returned_data = response.text

if response.status_code == 200:
    soup = bs(response.content, "html.parser")
    # comment-content
    print(soup.find("div", class_="h_login").text)
    print(soup.find("div", {"class": "h_login"}).text)

