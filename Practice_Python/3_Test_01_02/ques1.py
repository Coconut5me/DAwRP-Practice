from func import *

url = 'https://cellphones.com.vn/laptop/mac.html'
html_content = make_request(url)

if html_content:
    scraped_data = scrape_data(html_content)
    save_to_json(scraped_data, 'data/cellphones_products_mac')
