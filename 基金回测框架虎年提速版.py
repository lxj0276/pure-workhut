import numpy as np
import pandas as pd
from functools import lru_cache,reduce
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

'''
框架说明
1。虎年快乐

2。使用示例如下：
#加载因子部分，并调整因子格式
fac=pd.read_csv('12月夏普比率因子.csv',encoding='gbk')
fac=fac.iloc[:,1:]
fac=fac.set_index(['date','F_INFO_WINDCODE']).unstack()
fac.index=pd.to_datetime(fac.index)
fac.columns=[i[1] for i in list(fac.columns)]

#回测部分
lu=HappyLujie(fac,only_buy=True,filename='夏普比')

3。本代码中，包含两个类TigerProsperous和HappyLujie，其中HappyLujie是回测时需要调用的。
而TigerProsperous是HappyLujie的内核，使用者无需调用。

4。进行回测需要两步。第一步，算出因子df，因子的格式应为矩阵格式，index为时间戳格式的时间（需为每月月底或每季度最后一天），columns为带.OF后缀的基金代码。
第二步，把因子df带入回测框架，进行回测。语句为lu=HappyLujie(fac,freq='Q',only_buy=False,group_num=10,buy_num=30,risk_free_rate=0,filename='回测结果')
回测结果展示：一张净值走势图，和一个绩效评价与各期净值excel表。

5。回测速度说明：此框架速度非常快，尤其是你需要测试多于一个因子时，第一次测试时，需要花几秒钟加载净值数据。此后，从第二次回测开始，只需要一眨眼的时间，就能完成回测。

6。2种功能说明，第一种为分组式回测，依据因子值，把基金池分为若干组，分别测每组的净值（以及多空对冲的净值）。
第二种为绝对数值型回测，依据因子值，每期选出因子值最大的若干只基金，和因子值最小的若干只基金，分别测试最大若干只和最小若干只的净值（以及大小对冲的净值）。

7。参数说明：
①fac：因子矩阵。为一个dataframe格式。因子的格式应为矩阵格式，index为时间戳格式的时间（需为每月月底或每季度最后一天），columns为带.OF后缀的基金代码。
②freq：回测频率。可选值为'M'或'Q'，默认为'Q'。'M'为月频调仓，每月第一个交易日，以当天净值买入，每月最后一个交易日，以当天净值卖出。'Q'为季度，季度第一天买入，最后一天卖出。
③only_buy：是否是分组式回测。可选值为True或False，默认为False。为False时表示采用分组式回测，为True时表示采用选出因子值最大若干只和最小若干只的方式回测。
④group_num：分组回测时，分为多少组。可选值为大于1的正整数，默认为10。此时多空对冲即第一组和第十组对冲。此参数只有在only_buy为False时才会生效。
⑤buy_num：分别买入最大最小若干只时，买入最大最小的各多少只。可选值为大于1的正整数，默认值为30。此时大小对冲即，最大的若干只和最小的若干只对冲。此参数只有在only_buy为True时才会生效。
⑥risk_free_rate：无风险利率。可选值为任意浮点数，默认值为0。此参数在计算夏普比率的时候有用。
⑦filename：文件名字。可选值为任意字符串，默认值为'回测结果'。回测结束后，净值走势图，以及绩效评价与各期净值excel表，这两个文件的名称，将为filename+'.png'和filename+'.xlsx'。

8。备注：代码第51行处，有一个路径，这是基金净值数据在我电脑上的路径，你把它换成你自己电脑上的路径就好了。
'''



class TigerProsperous(object):
    def __new__(cls,freq='Q'):
        cls.navs_path='/Users/chenzongwei/pythoncode/数据库/基金数据库/nav_all_20220119.txt'
        cls.freq=freq
        return super(TigerProsperous,cls).__new__(cls)

    @classmethod
    @lru_cache(maxsize=None)
    def read_in_nav(cls):
        '''读入净值数据，并处理成收益率数据'''
        navs=pd.read_csv(cls.navs_path,sep='\t')
        navs.columns=['code','date','x','y','nav']
        navs=navs.drop_duplicates(subset=['date','code'])
        navs=navs[['date','code','nav']].set_index(['date','code']).unstack()
        navs.columns=[i[1] for i in list(navs.columns)]
        navs.index=pd.to_datetime(navs.index,format='%Y%m%d')
        navs_count=navs.count(axis=1).to_frame('num')
        navs=pd.concat([navs,navs_count],axis=1)
        navs=navs[navs.num>=500]
        cls.navs=navs
        freq_navs_last=navs.resample(cls.freq).last()
        freq_navs_first=navs.resample(cls.freq).first()
        rets=(freq_navs_last-freq_navs_first)/freq_navs_first
        rets=rets.stack().reset_index()
        rets.columns=['date','code','ret']
        cls.rets=rets

    @classmethod
    def get_groups(cls,df,groups_num):
        '''依据因子值，判断是在第几组'''
        if 'group' in list(df.columns):
            df=df.drop(columns=['group'])
        df=df.sort_values(['fac'],ascending=False)
        each_group=round(df.shape[0]/groups_num)
        l=list(map(lambda x,y:[x]*y,list(range(1,groups_num+1)),[each_group]*groups_num))
        l=reduce(lambda x,y:x+y,l)
        if len(l)<df.shape[0]:
            l=l+[groups_num]*(df.shape[0]-len(l))
        l=l[:df.shape[0]]
        df.insert(0,'group',l)
        return df

    @classmethod
    def get_only_buy(cls,df,buy_num):
        '''依据因子值，判断是在第几组'''
        df=df.sort_values('fac')
        df_head=df.head(buy_num)
        df_tail=df.tail(buy_num)
        df_head=df_head.assign(group=1)
        df_tail=df_tail.assign(group=-1)
        df=pd.concat([df_head,df_tail])
        return df

    @classmethod
    def get_ic_rankic(cls, df):
        '''计算IC和RankIC'''
        df1 = df[['ret', 'fac']]
        ic = df1.corr(method='pearson').iloc[0, 1]
        rankic = df1.corr(method='spearman').iloc[0, 1]
        df2 = pd.DataFrame({'ic': [ic], 'rankic': [rankic]})
        return df2

    @classmethod
    def get_icir_rankicir(cls, df):
        '''计算ICIR和RankICIR'''
        ic = df.ic.mean()
        rankic = df.rankic.mean()
        icir = ic / np.std(df.ic) * (12 ** (0.5))
        rankicir = rankic / np.std(df.rankic) * (12 ** (0.5))
        return pd.DataFrame({'IC': [ic], 'ICIR': [icir], 'RankIC': [rankic], 'RankICIR': [rankicir]}, index=['评价指标'])

    @classmethod
    def get_ic_icir_and_rank(cls, df):
        '''计算IC、ICIR、RankIC、RankICIR'''
        df1 = df.groupby('date').apply(cls.get_ic_rankic)
        df2 = cls.get_icir_rankicir(df1)
        df2 = df2.T
        return df2

    @classmethod
    def comments_on_twins(cls,series,risk_free_rate=0):
        '''对twins中的结果给出评价
        评价指标包括年化收益率、总收益率、年化波动率、年化夏普比率、最大回撤率、胜率'''
        ret=(series.iloc[-1]-series.iloc[0])/series.iloc[0]
        duration=(series.index[-1]-series.index[0]).days
        year=duration/365
        ret_yearly=(series.iloc[-1]/series.iloc[0])**(1/year)-1
        max_draw=-(series/series.expanding(1).max()-1).min()
        series1=series.pct_change()
        if cls.freq=='Q':
            vol=np.std(series1)*(4**0.5)
        elif cls.freq=='M':
            vol=np.std(series1)*(12**0.5)
        sharpe=(ret_yearly-risk_free_rate)/vol
        wins=series[series1>0]
        loses=series[series1<0]
        win_rate=len(wins)/(len(wins)+len(loses))
        return pd.Series([ret,ret_yearly,vol,sharpe,max_draw,win_rate],index=['总收益率','年化收益率','年化波动率','年化夏普比率','最大回撤率','胜率'])

    def deal_with_factor(self,fac,only_buy=False,group_num=10,buy_num=30,risk_free_rate=0,filename='回测结果'):
        '''输入的fac，应为矩阵形式，index是时间戳格式的时间，columns是带后缀名的基金代码
        处理因子数据，将其日期推后一期，并与收益率匹配到一起'''
        if self.freq=='Q':
            fac.index=fac.index+pd.DateOffset(months=3)
        elif self.freq=='M':
            fac.index=fac.index+pd.DateOffset(months=1)
        fac=fac.resample(self.freq).last()
        fac=fac.stack().reset_index()
        fac.columns=['date','code','fac']
        twins=pd.merge(self.rets,fac,on=['date','code'])
        ics=self.get_ic_icir_and_rank(twins)
        if not only_buy:
            twins=twins.groupby('date',as_index=False).apply(lambda x:self.get_groups(x,group_num))
        else:
            twins=twins.groupby('date',as_index=False).apply(lambda x:self.get_only_buy(x,buy_num))
        twins=twins[['date','group','ret']]
        twins=twins.groupby(['date','group']).mean().unstack()
        twins.columns=[i[1] for i in list(twins.columns)]
        if not only_buy:
            twins=twins.assign(long_short_pos=twins[group_num]-twins[1])
            twins=twins.assign(long_short_neg=twins[1]-twins[group_num])
        else:
            twins=twins.assign(long_short_pos=twins[1]-twins[-1])
            twins=twins.assign(long_short_neg=twins[-1]-twins[1])
        twins=twins+1
        twins=twins.cumprod()
        twins=twins.apply(lambda x:x/x.iloc[0])
        if twins.long_short_pos.iloc[-1]>twins.long_short_neg.iloc[-1]:
            twins=twins.assign(long_short=twins.long_short_pos)
        else:
            twins=twins.assign(long_short=twins.long_short_neg)
        twins=twins.drop(columns=['long_short_pos','long_short_neg'])
        self.twins=twins
        # self.comments=twins.apply(lambda x:self.comments_on_twins(x,risk_free_rate))
        # ics=ics.reset_index()
        # self.comments=self.comments.reset_index()
        # self.comments=pd.concat([ics,self.comments],axis=1)
        # pic_name=filename+'.png'
        # excel_name=filename+'.xlsx'
        # self.twins.plot()
        # plt.savefig(pic_name)
        # print(self.comments)
        # w=pd.ExcelWriter(excel_name)
        # self.comments.to_excel(w,sheet_name='评价指标')
        # self.twins.to_excel(w,sheet_name='各组净值序列')
        # w.save()
        # w.close()



class HappyLujie(object):
    def __init__(self,fac,freq='Q',only_buy=False,group_num=10,buy_num=30,risk_free_rate=0,filename='回测结果'):
        self.lu=TigerProsperous(freq=freq)
        self.lu.read_in_nav()
        self.lu.deal_with_factor(fac=fac,only_buy=only_buy,group_num=group_num,buy_num=buy_num,risk_free_rate=risk_free_rate,filename=filename)

    def __call__(self):
        return self.lu.comments


