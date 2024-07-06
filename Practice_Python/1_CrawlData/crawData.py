#%% - Import Lib
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.common.exceptions import TimeoutException

#&& - Crawl data
driver = webdriver.Chrome()

url = "https://www.lazada.vn/products/duohanzi-quan-dai-phu-nu-long-cuu-mua-thu-dong-quan-co-dot-mut-cua-phu-nu-dan-hoi-eo-cao-gian-di-quan-cang-mau-xam-mong-phu-nu-trung-nien-quan-thang-i2384218740-s11592351451.html?spm=a2o4n.home-vn.6598063730.20.19053bdchMdUO2&search=1&mp=1&c=fs&clickTrackInfo=rs%3A0.03415354713797569%3Bfs_item_discount_price%3A55.000%3Bitem_id%3A2384218740%3Bpctr%3A0.03415354713797569%3Bcalib_pctr%3A0.0%3Bvoucher_price%3A55000.00%3Bmt%3Ahot%3Bpromo_price%3A55000%3Bfs_utdid%3A-1%3Bfs_item_sold_cnt%3A18%3Babid%3A287818%3Bfs_item_price%3A144.000%3Bpvid%3Ac2a9199b-6c2c-40c3-a102-81c45776c04d%3Bfs_min_price_l30d%3A0%3Bdata_type%3Aflashsale%3Bfs_pvid%3Ac2a9199b-6c2c-40c3-a102-81c45776c04d%3Btime%3A1706175972%3Bfs_biz_type%3Afs%3Bscm%3A1007.17760.287818.%3Bchannel_id%3A0000%3Bfs_item_discount%3A62%25%3Bcampaign_id%3A267097&scm=1007.17760.287818.0"
driver.get(url)

timeout = 20
try:
    WebDriverWait(driver, timeout).until(ec.visibility_of_element_located((By.CLASS_NAME, "content")))
except TimeoutException:
    print("Time out waiting for page to load!!!")
    driver.quit()

#%% - Get Data
comments = driver.find_elements(By.CLASS_NAME, "content")
for comment in comments:
    print(comment.text)
#%%
btn_element = driver.find_elements(By.CLASS_NAME, "next-btn")[2]
webdriver.ActionChains(driver).move_to_element(btn_element).click(btn_element).perform()

#%%
comments2 = driver.find_elements((By.CLASS_NAME, "content"))
for comment in comments2:
    print(comment.text)