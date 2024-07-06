import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep

# Set up the webdriver
driver = webdriver.Chrome()
url = "https://s.cafef.vn/du-lieu.chn"
driver.get(url)

# Navigate to the historical data page
btn_lichsudulieu = driver.find_element(By.XPATH, '//*[@id="pagewrap"]/div[1]/div[1]/div[2]/a[3]')
btn_lichsudulieu.click()

# Enter the search criteria
input_search = driver.find_element(By.ID, "ContentPlaceHolder1_ctl00_acp_inp_disclosure")
input_search.send_keys("Techcombank")
input_search.send_keys(Keys.ENTER)

# Set the date range
input_time = driver.find_element(By.ID, "date-inp-disclosure")
input_time.send_keys("01/01/2023 - 31/12/2023")

# Click the search button
choose=driver.find_element(By.XPATH,'/html/body/div[3]/div[4]/button[2]')
choose.click()

# Create a CSV file to store the data using 'with open'
csv_file_path = 'data/TCB_stock_data1.csv'
csv_header = ['Date', 'High', 'Low', 'Open', 'Close']

with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_header)

    # Loop through the pages
    for page in range(1, 14):  # 13 pages in total
        # Assuming driver is already initialized and navigated to the page
        tbl_data = driver.find_element(By.ID, "owner-contents-table")
        all_rows = tbl_data.find_elements(By.XPATH, '//*[@id="render-table-owner"]')

        # Lặp qua từng hàng để trích xuất thông tin
        for row in all_rows:
            date = row.find_element(By.CLASS_NAME, "owner_time").text
            close = row.find_elements(By.CLASS_NAME, "owner_priceClose")[0].text
            open = row.find_elements(By.CLASS_NAME, "owner_price_td")[0].text
            low = row.find_elements(By.CLASS_NAME, "owner_price_td")[1].text
            high = row.find_elements(By.CLASS_NAME, "owner_price_td")[2].text

    csv_writer.writerow([date, high, low, open, close])

# Close the webdriver
driver.quit()
