import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.common.exceptions import TimeoutException
import json

def make_request(url):
    # Hàm này thực hiện yêu cầu HTTP và trả về nội dung phản hồi nếu thành công
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return None

def scrape_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    products = soup.find_all('div', class_='product-item')
    data = []

    for product in products:
        product_name = product.find('div', class_='product__name').find('h3').text.strip()
        img = product.find('div', class_='product__image').find('img')['src']
        product_price_tag = product.find('div', class_='box-info__box-price').find('p', class_='product__price--show')
        promotion_tag = product.find('div', class_='product__promotions').find('p', class_='coupon-price')

        # Convert Tag objects to strings
        product_price = product_price_tag.text.strip() if product_price_tag else None
        promotion = promotion_tag.text.strip() if promotion_tag else None

        product_info = {
            "product_name": product_name,
            "img": img,
            "product_price": product_price,
            "promotion": promotion
        }

        data.append(product_info)

    return data

def save_to_json(data, filename):
    # Hàm này lưu dữ liệu vào một tệp JSON
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)
    print(f'Data saved to {filename}')
