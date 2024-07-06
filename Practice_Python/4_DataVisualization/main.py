#%% - Import Lib
# import pandas as pd
import matplotlib.pyplot as plt

#%% - Some configs
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 14

#%%
import pandas as pd
df = pd.read_csv('data/NetProfit.csv')
dat = df[['Year', 'VIC']]
plt.plot('Year', 'VIC', data=df, color='#FF0000', linestyle='--', marker='s')
plt.plot('Year', 'VNM', data=df, color='green', linestyle='-', marker='o')
plt.plot('Year', 'VCB', data=df, color='b', linestyle='--', marker='*')
plt.plot('Year', 'PNJ', data=df, color='orange', linestyle=':', marker='+')
plt.title("Lợi nhuận của VIC, VNM, VCB, PNJ (2010-2020)", fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Net Profit")
plt.legend()
plt.show()


