# from func import *
#
# content = '''Trường Đại học Kinh tế - Luật
# Khoa Hệ thống thông tin
# '''
#
# #saveTxt(content, "data/test.txt", "a")
# #saveTxt(mode="a", content=content, path="data/test2.txt")
#
# s = readTxt("data/test.txt")
# print(s)

# from xml.etree import ElementTree as ET
import xml.etree.ElementTree as ET
tree = ET.parse(("data/products.xml"))
root = tree.getroot()
print(f"Root tag: {root.tag}")
print(f"Root attr: {root.attrib}")
print("------"*4)

for child in root:
    print(f"Child tag: {child.tag}, Child Attr: {child.attrib}")

print("------"*4)
print(root[0][2].text)
print(root[2][1].text)

# print("------"*4)
# for product_elements in root.iter("product"):
#     print(product_elements[1].text)

# print("------"*4)
# for product_elements in root.iter("product"):
#     print(product_elements.find("price").text)

# print("------"*4)
# for product_elements in root.findall("product"):
#     print(product_elements.tag)

# print("------"*4)
# for product_elements in root.findall("product"):
#     print(f"{product_elements.find('id').text} - {product_elements.get('name')}")

# for p_element in root.iter("product"):
#     slogan_e = p_element.find("slogan")
#     text_formatted = slogan_e.text.title()
#     slogan_e.text = text_formatted
#     slogan_e.set("formatted", "yes")
#
# tree.write("data/updated_products.xml", encoding="utf-8")


# ver_1
for p_element in root.iter("product"):
    price_e = p_element.find("price")
    price_value = float(price_e.text.replace("$", ""))
    price_formatted = f"{price_value * 23000:.0f}VND"
    price_e.text = price_formatted
    price_e.set("formatted", "yes")

tree.write("data/updated_prices.xml", encoding="utf-8")


price = "$1.5"
new_price = price.lstrip("$")
print(new_price)