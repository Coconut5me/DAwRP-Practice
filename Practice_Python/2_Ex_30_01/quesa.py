import requests
import json

# gửi yêu cầu đến API
api_url = "https://fakestoreapi.com/products"
response = requests.get(api_url)

# kiểm tra yêu cầu thành công không
if response.status_code == 200:
    products_data = response.json()

    # lưu dữ liệu vào file JSON
    with open("data/products.json", "w", encoding="utf-8") as json_file:
        json.dump(products_data, json_file, indent=2, ensure_ascii=False)
    print("Dữ liệu đã được lưu vào file 'products_data.json'")
else:
    print(f"Yêu cầu không thành công. Mã trạng thái: {response.status_code}")
