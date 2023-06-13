# 数据异常处理
# 房价的最大值远高于均值，我们需要去除掉那些房价为50的数据
# boston_df = boston_df.loc[boston_df['PRICE'] != 50]