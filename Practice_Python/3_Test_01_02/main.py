from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from time import sleep

from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

#%%
driver = webdriver.Chrome()
url = "https://s.cafef.vn/du-lieu.chn"
driver.get(url)

#%%
# //*[@id="pagewrap"]/div[1]/div[1]/div[2]/a[3]
btn_lichsudulieu = driver.find_element(By.XPATH, '//*[@id="pagewrap"]/div[1]/div[1]/div[2]/a[3]')
btn_lichsudulieu.click()

#%%
input_search = driver.find_element(By.ID,"ContentPlaceHolder1_ctl00_acp_inp_disclosure")
input_search.send_keys("Techcombank")

#%%
input_search.send_keys(Keys.ENTER)

#%%
input_time = driver.find_element(By.ID, "date-inp-disclosure")
input_time.send_keys("01/01/2023 - 31/12/2023")

#%%
btn_xem = driver.find_element(By.ID, "owner-find")
btn_xem = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "owner-find")))
btn_xem.click()

#%%
tbl_data = driver.find_element(By.Id, "owner-contents-table")
all_rows = tbl_data.find_elements(By.TAG_NAME, "tr")
for row in all_rows:
    print(row.text)
    sleep(1)

#%%
btn_next = driver.find_element(By.XPATH, '//*[@id="divStart"]/div/div[3]/div[3]')
btn_next.click()
