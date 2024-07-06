import requests
import xml.etree.ElementTree as ET

api_url = "https://fakestoreapi.com/products"
response = requests.get(api_url)

if response.status_code == 200:
    # chuyển đổi nội dung JSON thành danh sách Python
    products_data = response.json()

    # tạo phần tử gốc cho tài liệu XML
    root = ET.Element("products")

    # lặp qua danh sách sản phẩm và tạo các phần tử XML
    for product in products_data:
        product_element = ET.SubElement(root, "product")
        for key, value in product.items():
            if key == "rating":
                rating_element = ET.SubElement(product_element, "rating")
                for k, v in value.items():
                    ET.SubElement(rating_element, k).text = str(v)
            else:
                ET.SubElement(product_element, key).text = str(value)

    # tạo cây XML từ phần tử gốc
    xml_tree = ET.ElementTree(root)

    # ghi cây XML vào tệp
    xml_tree.write("data/products_ver1.xml", encoding="utf-8")
    print("Dữ liệu đã được lưu vào file 'products_data.xml'")
else:
    print(f"Yêu cầu không thành công. Mã trạng thái: {response.status_code}")
