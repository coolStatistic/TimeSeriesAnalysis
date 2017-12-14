import pandas as pd

# 数据预处理
dataframe = pd.read_excel("airport.xls",
        headder=None,index_col=None)

dataframe = dataframe.transpose()
dataframe = dataframe.iloc[2:, ]
dataframe[1] = dataframe[0]
dataframe[0] = dataframe.index
dataframe.columns = ['Date', 'Num']
dataframe = dataframe.iloc[::-1]

# 日期转换
# 日期格式转换函数
def convert_to_date(s):
    d = pd.to_datetime(s, format='%Y年%m月')
    return d

dataframe['Date'] = dataframe['Date'].apply(convert_to_date)

dataframe.to_csv('clean_train_data.csv', index=None)
