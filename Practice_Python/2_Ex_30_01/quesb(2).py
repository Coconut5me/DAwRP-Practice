import json
import dicttoxml

# đc dữ liệu từ tệp JSON
with open("data/products.json", "r", encoding="utf-8") as json_file:
    products_data = json.load(json_file)

# chuyển đổi danh sách Python thành XML
xml_data = dicttoxml.dicttoxml(products_data, custom_root='products', attr_type=False)

# ghi dữ liệu XML vào tệp
with open("data/products_ver2.xml", "wb") as xml_file:
    xml_file.write(xml_data)
print("Dữ liệu đã được chuyển từ JSON sang XML và lưu vào file 'products_ver2.xml'")
