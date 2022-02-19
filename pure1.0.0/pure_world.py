import os
import tqdm
import numpy as np
import pandas as pd
import scipy.io as scio
import statsmodels.formula.api as smf
from functools import partial, reduce
import warnings
warnings.filterwarnings('ignore')
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import plotly.express as pe
import plotly.io as pio
import scipy.stats as ss
from loguru import logger
import time

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import h5py  # 导入工具包


class pure_moon():
    def __init__(self):
        now=datetime.datetime.now()
        now=datetime.datetime.strftime(now,format='%Y-%m-%d %H:%M:%S')
        # logger.add('pure_moon'+now+'.log')
        # 绝对路径前缀
        self.path_prefix = '/Users/chenzongwei/pythoncode/数据库/'
        # 股票代码文件
        self.codes_path = '日频数据/AllStockCode.mat'
        # 交易日期文件
        self.tradedays_path = '日频数据/TradingDate_Daily.mat'
        # 上市天数文件
        self.ages_path = '日频数据/AllStock_DailyListedDate.mat'
        # st日子标志文件
        self.sts_path = '日频数据/AllStock_DailyST.mat'
        # 交易状态文件
        self.states_path = '日频数据/AllStock_DailyStatus.mat'
        # 复权开盘价数据文件
        self.opens_path = '日频数据/AllStock_DailyOpen_dividend.mat'
        # 复权收盘价数据文件
        self.closes_path = '日频数据/AllStock_DailyClose_dividend.mat'
        #复权最高价数据文件
        self.highs_path = '日频数据/Allstock_DailyHigh_dividend.mat'
        #复权最低价数据文件
        self.lows_path = '日频数据/Allstock_DailyLow_dividend.mat'
        # 不复权收盘价数据文件
        self.pricloses_path = '日频数据/AllStock_DailyClose.mat'
        # 流通股本数据文件
        self.flowshares_path = '日频数据/AllStock_DailyAShareNum.mat'
        # 成交量数据文件
        self.amounts_path = '日频数据/AllStock_DailyVolume.mat'
        # 换手率数据文件
        self.turnovers_path = '日频数据/AllStock_DailyTR.mat'
        # 因子数据文件
        self.factors_file = ''
        # 已经算好的月度st状态文件
        self.sts_monthly_file = '日频数据/sts_monthly.feather'
        # 已经算好的月度交易状态文件
        self.states_monthly_file = '日频数据/states_monthly.feather'
        # 已经算好的月度st_by10状态文件
        self.sts_monthly_by10_file = '日频数据/sts_monthly_by10.feather'
        # 已经算好的月度交易状态文件
        self.states_monthly_by10_file = '日频数据/states_monthly_by10.feather'
        # 拼接绝对路径前缀和相对路径
        dirs = dir(self)
        dirs.remove('new_path')
        dirs.remove('set_factor_file')
        dirs = [i for i in dirs if i.endswith('path')] + [i for i in dirs if i.endswith('file')]
        dirs_values = list(map(lambda x, y: getattr(x, y), [self] * len(dirs), dirs))
        dirs_values = list(map(lambda x, y: x + y, [self.path_prefix] * len(dirs), dirs_values))
        for attr, value in zip(dirs, dirs_values):
            setattr(self, attr, value)

    def __call__(self, fallmount=0):
        '''调用对象则返回因子值'''
        df=self.factors.copy()
        df=df.set_index(['date', 'code']).unstack()
        df.columns=list(map(lambda x:x[1],list(df.columns)))
        if fallmount == 0:
            return df
        else:
            return pure_fallmount(df)

    def set_factor_file(self, factors_file):
        '''设置因子文件的路径，因子文件列名应为股票代码，索引为时间'''
        self.factors_file = factors_file
        self.factors = pd.read_feather(self.factors_file)
        self.factors = self.factors.set_index('date')
        self.factors = self.factors.resample('M').last()
        self.factors = self.factors.reset_index()

    def set_factor_df_date_as_index(self, df):
        '''设置因子数据的dataframe，因子表列名应为股票代码，索引应为时间'''
        df = df.reset_index()
        df.columns = ['date'] + list(df.columns)[1:]
        # df.date=df.date.apply(self.next_month_end)
        # df=df.set_index('date')
        self.factors = df
        self.factors = self.factors.set_index('date')
        self.factors = self.factors.resample('M').last()
        self.factors = self.factors.reset_index()

    def set_factor_df_wide(self, df):
        '''从dataframe读入因子宽数据'''
        self.factors = df
        # self.factors.date=self.factors.date.apply(self.next_month_end)
        # self.factors=self.factors.set_index('date')
        self.factors = self.factors.set_index('date')
        self.factors = self.factors.resample('M').last()
        self.factors = self.factors.reset_index()

    # def set_factor_df_long(self,df):
    #     '''从dataframe读入因子长数据'''
    #     self.factors=df
    #     self.factors.columns=['date','code','fac']

    def new_path(self, **kwargs):
        '''修改日频数据文件的路径，便于更新数据
        要修改的路径以字典形式传入，键为属性名，值为要设置的新路径'''
        for key, value in kwargs.items():
            setattr(self, key, value)

    def col_and_index(self):
        '''读取股票代码，作为未来表格的行名
        读取交易日历，作为未来表格的索引'''
        self.codes = list(scio.loadmat(self.codes_path).values())[3]
        self.tradedays = list(scio.loadmat(self.tradedays_path).values())[3].astype(str)
        self.codes = self.codes.flatten().tolist()
        self.tradedays = self.tradedays.flatten().tolist()
        self.codes = list(map(lambda x: x[0], self.codes))

    def loadmat(self, path):
        '''重写一个加载mat文件的函数，以使代码更简洁'''
        return list(scio.loadmat(path).values())[3]

    def make_df(self, data):
        '''将读入的数据，和股票代码与时间拼接，做成dataframe'''
        data = pd.DataFrame(data, columns=self.codes, index=self.tradedays)
        data.index = pd.to_datetime(data.index, format='%Y%m%d')
        return data

    def load_all_files(self):
        '''加全部的mat文件'''
        attrs = dir(self)
        attrs = [i for i in attrs if i.endswith('path')]
        attrs.remove('codes_path')
        attrs.remove('tradedays_path')
        attrs.remove('new_path')
        for attr in attrs:
            new_attr = attr[:-5]
            setattr(self, new_attr, self.make_df(self.loadmat(getattr(self, attr))))
        self.opens = self.opens.replace(0, np.nan)
        self.closes = self.closes.replace(0, np.nan)

    def judge_month_st(self, df):
        '''比较一个月内st的天数，如果st天数多，就删除本月，如果正常多，就保留本月'''
        st_count = len(df[df == 1])
        normal_count = len(df[df != 1])
        if st_count >= normal_count:
            return 0
        else:
            return 1

    def judge_month_st_by10(self, df):
        '''比较一个月内正常交易的天数，如果少于10天，就删除本月'''
        normal_count = len(df[df != 1])
        if normal_count < 10:
            return 0
        else:
            return 1

    def judge_month_state(self, df):
        '''比较一个月内非正常交易的天数，如果非正常交易天数多，就删除本月，否则保留本月'''
        abnormal_count = len(df[df == 0])
        normal_count = len(df[df == 1])
        if abnormal_count >= normal_count:
            return 0
        else:
            return 1

    def judge_month_state_by10(self, df):
        '''比较一个月内正常交易天数，如果少于10天，就删除本月'''
        normal_count = len(df[df == 1])
        if normal_count < 10:
            return 0
        else:
            return 1

    def read_add(self, pridf, df, func):
        '''由于数据更新，过去计算的月度状态可能需要追加'''
        logger.info(f'this is max_index of pridf{pridf.index.max()}')
        logger.info(f'this is max_index of df{df.index.max()}')
        if pridf.index.max() > df.index.max():
            df_add = pridf[pridf.index > df.index.max()]
            df_add = df_add.resample('M').apply(func)
            df = pd.concat([df, df_add])
            return df
        else:
            return df

    def write_feather(self, df, path):
        '''将算出来的数据存入本地，以免造成重复运算'''
        df1 = df.copy()
        df1 = df1.reset_index()
        df1.to_feather(path)

    def daily_to_monthly(self, pridf, path, func):
        '''把日度的交易状态、st、上市天数，转化为月度的，并生成能否交易的判断
        读取本地已经算好的文件，并追加新的时间段部分，如果本地没有就直接全部重新算'''
        try:
            logger.info('try to read the prepared state file')
            month_df = pd.read_feather(path).set_index('index')
            logger.info('state file load success')
            month_df = self.read_add(pridf, month_df, func)
            logger.info('adding after state file has finish')
            self.write_feather(month_df, path)
            logger.info('the feather is new now')
        except Exception as e:
            logger.error('error occurs when read state files')
            logger.error(e)
            print('state file rewriting……')
            month_df = pridf.resample('M').apply(func)
            self.write_feather(month_df, path)
        return month_df

    def daily_to_monthly_by10(self, pridf, path, func):
        '''把日度的交易状态、st、上市天数，转化为月度的，并生成能否交易的判断
        读取本地已经算好的文件，并追加新的时间段部分，如果本地没有就直接全部重新算'''
        try:
            month_df = pd.read_feather(path).set_index('date')
            month_df = self.read_add(pridf, month_df, func)
            self.write_feather(month_df, path)
        except Exception:
            print('rewriting')
            month_df = pridf.resample('M').apply(func)
            self.write_feather(month_df, path)
        return month_df

    def judge_month(self):
        '''生成一个月综合判断的表格'''
        self.sts_monthly = self.daily_to_monthly(self.sts, self.sts_monthly_file, self.judge_month_st)
        self.states_monthly = self.daily_to_monthly(self.states, self.states_monthly_file, self.judge_month_state)
        self.ages_monthly = self.ages.resample('M').last()
        self.ages_monthly = np.sign(self.ages_monthly.applymap(lambda x: x - 60)).replace(-1, 0)
        self.tris_monthly = self.sts_monthly * self.states_monthly * self.ages_monthly
        self.tris_monthly = self.tris_monthly.replace(0, np.nan)

    def judge_month_by10(self):
        '''生成一个月综合判断的表格'''
        self.sts_monthly = self.daily_to_monthly(self.sts, self.sts_monthly_by10_file, self.judge_month_st_by10)
        self.states_monthly = self.daily_to_monthly(self.states, self.states_monthly_by10_file,
                                                    self.judge_month_state_by10)
        self.ages_monthly = self.ages.resample('M').last()
        self.ages_monthly = np.sign(self.ages_monthly.applymap(lambda x: x - 60)).replace(-1, 0)
        self.tris_monthly = self.sts_monthly * self.states_monthly * self.ages_monthly
        self.tris_monthly = self.tris_monthly.replace(0, np.nan)

    def get_rets_month(self):
        '''计算每月的收益率，并根据每月做出交易状态，做出删减'''
        self.opens_monthly = self.opens.resample('M').first()
        self.closes_monthly = self.closes.resample('M').last()
        self.rets_monthly = (self.closes_monthly - self.opens_monthly) / self.opens_monthly
        self.rets_monthly = self.rets_monthly * self.tris_monthly
        self.rets_monthly = self.rets_monthly.stack().reset_index()
        self.rets_monthly.columns = ['date', 'code', 'ret']

    def neutralize_factors(self, df):
        '''组内对因子进行市值中性化'''
        ols_result = smf.ols('fac~cap_size', data=df).fit()
        ols_w = ols_result.params['cap_size']
        ols_b = ols_result.params['Intercept']
        df.fac = df.fac - ols_w * df.cap_size - ols_b
        df = df[['fac']]
        return df

    def get_log_cap(self,boxcox=False):
        '''获得对数市值'''
        self.pricloses = self.pricloses.replace(0, np.nan)
        self.flowshares = self.flowshares.replace(0, np.nan)
        self.pricloses = self.pricloses.resample('M').last()
        self.pricloses = self.pricloses.stack().reset_index()
        self.pricloses.columns = ['date', 'code', 'priclose']
        self.flowshares = self.flowshares.resample('M').last()
        self.flowshares = self.flowshares.stack().reset_index()
        self.flowshares.columns = ['date', 'code', 'flowshare']
        self.flowshares = pd.merge(self.flowshares, self.pricloses, on=['date', 'code'])
        self.cap = self.flowshares.assign(cap_size=self.flowshares.flowshare * self.flowshares.priclose)
        if boxcox:
            self.cap.cap_size=ss.boxcox(self.cap.cap_size)[0]
        else:
            self.cap.cap_size = np.log(self.cap.cap_size)

    def get_neutral_factors(self):
        '''对因子进行市值中性化'''
        self.factors.date = self.factors.date.apply(self.next_month_end)
        self.factors = self.factors.set_index('date')
        self.factors = self.factors * self.tris_monthly
        self.factors = self.factors.reset_index()
        self.factors=self.factors.rename(columns={'index':'date'})
        vanish_top_time = self.factors.date.min()
        self.factors.date = self.factors.date.shift(1)
        vanish_top_time = vanish_top_time - pd.Timedelta(days=vanish_top_time.day)
        self.factors.date = self.factors.date.fillna(vanish_top_time)
        self.factors = self.factors.set_index('date')
        self.factors = self.factors.stack().reset_index()
        self.factors.columns = ['date', 'code', 'fac']
        self.factors = pd.merge(self.factors, self.cap, how='inner', on=['date', 'code'])
        self.factors = self.factors.set_index(['date', 'code'])
        self.factors = self.factors.groupby(['date']).apply(self.neutralize_factors)
        self.factors = self.factors.reset_index()

    def deal_with_factors(self):
        '''删除不符合交易条件的因子数据'''
        self.factors.date = self.factors.date.apply(self.next_month_end)
        self.factors = self.factors.set_index('date')
        self.factors = self.factors * self.tris_monthly
        self.factors = self.factors.stack().reset_index()
        self.factors.columns = ['date', 'code', 'fac']

    def deal_with_factors_after_neutralize(self):
        '''中性化之后的因子处理方法'''
        self.factors = self.factors.set_index(['date', 'code'])
        self.factors = self.factors.unstack()
        self.factors = self.factors.reset_index()
        self.factors.date = self.factors.date.apply(self.next_month_end)
        self.factors = self.factors.set_index('date')
        self.factors.columns = list(map(lambda x: x[1], list(self.factors.columns)))
        self.factors = self.factors.stack().reset_index()
        self.factors.columns = ['date', 'code', 'fac']

    def find_limit(self, df, up=1):
        '''计算涨跌幅超过9.8%的股票，并将其存储进一个长列表里
        其中时间列，为某月的最后一天；涨停日虽然为下月初第一天，但这里标注的时间统一为上月最后一天'''
        limit_df = np.sign(df.applymap(lambda x: x - up * 0.098)).replace(-1 * up, np.nan)
        limit_df = limit_df.stack().reset_index()
        limit_df.columns = ['date', 'code', 'limit_up_signal']
        limit_df = limit_df[['date', 'code']]
        return limit_df

    def get_limit_ups_downs(self):
        '''找月初第一天就涨停或者跌停的股票'''
        self.opens_monthly_shift = self.opens_monthly.copy()
        self.opens_monthly_shift.index = self.opens_monthly_shift.index.shift(-1)
        self.rets_monthly_begin = (self.opens_monthly_shift - self.closes_monthly) / self.closes_monthly
        self.limit_ups = self.find_limit(self.rets_monthly_begin, up=1)
        self.limit_downs = self.find_limit(self.rets_monthly_begin, up=-1)

    def get_ic_rankic(self, df):
        '''计算IC和RankIC'''
        df1 = df[['ret', 'fac']]
        ic = df1.corr(method='pearson').iloc[0, 1]
        rankic = df1.corr(method='spearman').iloc[0, 1]
        df2 = pd.DataFrame({'ic': [ic], 'rankic': [rankic]})
        return df2

    def get_icir_rankicir(self, df):
        '''计算ICIR和RankICIR'''
        ic = df.ic.mean()
        rankic = df.rankic.mean()
        icir = ic / np.std(df.ic) * (12 ** (0.5))
        rankicir = rankic / np.std(df.rankic) * (12 ** (0.5))
        return pd.DataFrame({'IC': [ic], 'ICIR': [icir], 'RankIC': [rankic], 'RankICIR': [rankicir]}, index=['评价指标'])

    def get_ic_icir_and_rank(self, df):
        '''计算IC、ICIR、RankIC、RankICIR'''
        df1 = df.groupby('date').apply(self.get_ic_rankic)
        df2 = self.get_icir_rankicir(df1)
        df2 = df2.T
        return df2

    def get_groups(self, df, groups_num):
        '''依据因子值，判断是在第几组'''
        if 'group' in list(df.columns):
            df = df.drop(columns=['group'])
        df = df.sort_values(['fac'], ascending=False)
        each_group = round(df.shape[0] / groups_num)
        l = list(map(lambda x, y: [x] * y, list(range(1, groups_num + 1)), [each_group] * groups_num))
        l = reduce(lambda x, y: x + y, l)
        if len(l) < df.shape[0]:
            l = l + [groups_num] * (df.shape[0] - len(l))
        l = l[:df.shape[0]]
        df.insert(0, 'group', l)
        return df

    def next_month_end(self, x):
        '''找到下个月最后一天'''
        x1 = x = x + relativedelta(months=1)
        while x1.month == x.month:
            x1 = x1 + relativedelta(days=1)
        return x1 - relativedelta(days=1)

    def limit_old_to_new(self, limit, data):
        '''获取跌停股在旧月的组号，然后将日期调整到新月里
        涨停股则获得新月里涨停股的代码和时间，然后直接删去'''
        data1 = data.copy()
        data1 = data1.reset_index()
        data1.columns = ['data_index'] + list(data1.columns)[1:]
        old = pd.merge(limit, data1, how='inner', on=['date', 'code'])
        old = old.set_index('data_index')
        old = old[['group', 'date', 'code']]
        old.date = list(map(self.next_month_end, list(old.date)))
        return old

    def get_data(self, groups_num):
        '''拼接因子数据和每月收益率数据，并对涨停和跌停股加以处理'''
        self.data = pd.merge(self.rets_monthly, self.factors, how='inner', on=['date', 'code'])
        self.ic_icir_and_rank = self.get_ic_icir_and_rank(self.data)
        self.data = self.data.groupby('date').apply(lambda x: self.get_groups(x, groups_num))
        self.data = self.data.reset_index(drop=True)
        self.limit_ups = self.limit_old_to_new(self.limit_ups, self.data)
        self.limit_downs = self.limit_old_to_new(self.limit_downs, self.data)
        self.data = self.data.drop(self.limit_ups.index)
        self.rets_monthly_limit_downs = pd.merge(self.rets_monthly, self.limit_downs, how='inner', on=['date', 'code'])
        self.data = pd.concat([self.data, self.rets_monthly_limit_downs])

    def select_data_time(self, time_start, time_end):
        '''筛选特定的时间段'''
        if time_start:
            self.data = self.data[self.data.date >= time_start]
        if time_end:
            self.data = self.data[self.data.date <= time_end]

    def make_start_to_one(self, l):
        '''让净值序列的第一个数变成1'''
        min_date = self.factors.date.min()
        add_date = min_date - relativedelta(days=min_date.day)
        add_l = pd.Series([1], index=[add_date])
        l = pd.concat([add_l, l])
        return l

    def to_group_ret(self,l):
        '''每一组的年化收益率'''
        ret=l[-1]**(12/len(l))-1
        return ret

    def get_group_rets_net_values(self, groups_num):
        '''计算组内每一期的平均收益，生成每日收益率序列和净值序列'''
        self.group_rets = self.data.groupby(['date', 'group']).apply(lambda x: x.ret.mean())
        # dropna是因为如果股票行情数据比因子数据的截止日期晚，而最后一个月发生月初跌停时，会造成最后某组多出一个月的数据
        self.group_rets = self.group_rets.unstack()
        self.group_rets = self.group_rets[self.group_rets.index <= self.factors.date.max()]
        self.group_rets.columns = list(map(str, list(self.group_rets.columns)))
        self.group_rets = self.group_rets.add_prefix('group')
        self.long_short_rets = self.group_rets['group1'] - self.group_rets['group' + str(groups_num)]
        self.long_short_net_values = (self.long_short_rets + 1).cumprod()
        if self.long_short_net_values[-1] <= self.long_short_net_values[0]:
            self.long_short_rets = self.group_rets['group' + str(groups_num)] - self.group_rets['group1']
            self.long_short_net_values = (self.long_short_rets + 1).cumprod()
        self.long_short_net_values = self.make_start_to_one(self.long_short_net_values)
        self.group_rets = self.group_rets.assign(long_short=self.long_short_rets)
        self.group_net_values = self.group_rets.applymap(lambda x: x + 1)
        self.group_net_values = self.group_net_values.cumprod()
        self.group_net_values = self.group_net_values.apply(self.make_start_to_one)
        a=groups_num**(0.5)
        #判断是否要两个因子画表格
        if a==int(a):
            self.square_rets=self.group_net_values.iloc[:,:-1].apply(self.to_group_ret).to_numpy()
            self.square_rets=self.square_rets.reshape((int(a),int(a)))
            self.square_rets=pd.DataFrame(self.square_rets,columns=list(range(1,int(a)+1)),index=list(range(1,int(a)+1)))
            print('这是self.square_rets',self.square_rets)

    def get_long_short_comments(self):
        '''计算多空对冲的相关评价指标
        包括年化收益率、年化波动率、信息比率、月度胜率、最大回撤率'''
        self.long_short_ret_yearly = self.long_short_net_values[-1] ** (12 / len(self.long_short_net_values)) - 1
        self.long_short_vol_yearly = np.std(self.long_short_rets) * (12 ** 0.5)
        self.long_short_info_ratio = self.long_short_ret_yearly / self.long_short_vol_yearly
        self.long_short_win_times = len(self.long_short_rets[self.long_short_rets > 0])
        self.long_short_win_ratio = self.long_short_win_times / len(self.long_short_rets)
        self.retreats = []
        for i, net_value in enumerate(list(self.long_short_net_values)[1:]):
            before_max = max(list(self.long_short_net_values[:i + 1]))
            self.retreats.append((before_max - net_value) / before_max)
        self.max_retreat = max(self.retreats)
        self.long_short_comments = pd.DataFrame({
            '评价指标': [
                self.long_short_ret_yearly,
                self.long_short_vol_yearly,
                self.long_short_info_ratio,
                self.long_short_win_ratio,
                self.max_retreat
            ]
        }, index=['年化收益率', '年化波动率', '信息比率', '月度胜率', '最大回撤率'])

    def get_total_comments(self):
        '''综合IC、ICIR、RankIC、RankICIR,年化收益率、年化波动率、信息比率、月度胜率、最大回撤率'''
        self.total_comments = pd.concat([self.ic_icir_and_rank, self.long_short_comments])

    def plot_net_values(self, y2, filename):
        '''使用matplotlib来画图，y2为是否对多空组合采用双y轴'''
        self.group_net_values.plot(secondary_y=y2)
        filename_path = filename + '.png'
        plt.savefig(filename_path)

    def plotly_net_values(self, filename):
        '''使用plotly.express画图'''
        fig = pe.line(self.group_net_values)
        filename_path = filename + '.html'
        pio.write_html(fig, filename_path, auto_open=True)

    def run(self, groups_num, neutralize=False, boxcox=False, by10=False, y2=False, plt=True, plotly=False, filename='分组净值图',
            time_start=None, time_end=None, print_comments=True):
        '''运行回测框架'''
        self.col_and_index()
        logger.success('already get columns and index')
        self.load_all_files()
        logger.success('successfully load all files')
        if by10:
            self.judge_month_by10()
        else:
            logger.success('start judge month')
            self.judge_month()
            logger.success('judge month finished')
        self.get_rets_month()
        if neutralize:
            self.get_log_cap()
            self.get_neutral_factors()
            self.deal_with_factors_after_neutralize()
        elif boxcox:
            self.get_log_cap(boxcox=True)
            self.get_neutral_factors()
            self.deal_with_factors_after_neutralize()
        else:
            self.deal_with_factors()
        self.get_limit_ups_downs()
        self.get_data(groups_num)
        self.select_data_time(time_start, time_end)
        self.get_group_rets_net_values(groups_num)
        self.get_long_short_comments()
        self.get_total_comments()
        if plt:
            if filename:
                self.plot_net_values(y2=y2, filename=filename)
            else:
                self.plot_net_values(y2=y2,
                                     filename=self.factors_file.split('.')[-2].split('/')[-1] + str(groups_num) + '分组')
        if plotly:
            if filename:
                self.plotly_net_values(filename=filename)
            else:
                self.plotly_net_values(
                    filename=self.factors_file.split('.')[-2].split('/')[-1] + str(groups_num) + '分组')
        if print_comments:
            print(self.total_comments)


class pure_fall():
    def __init__(
            self,
            minute_files_path='/Users/chenzongwei/pythoncode/数据库/分钟数据',
            minute_columns=['date', 'open', 'high', 'low', 'close', 'amount', 'money'],
            daily_factors_path='/Users/chenzongwei/pythoncode/数据库/因子数据/日频_',
            monthly_factors_path='/Users/chenzongwei/pythoncode/数据库/因子数据/月频_'
    ):
        # 分钟数据文件夹路径
        self.minute_files_path = minute_files_path
        # 分钟数据文件夹
        self.minute_files = os.listdir(self.minute_files_path)
        self.minute_files = [i for i in self.minute_files if i.endswith('.mat')]
        # 分钟数据的表头
        self.minute_columns = minute_columns
        # 分钟数据日频化之后的数据表
        self.daily_factors_list = []
        # 将分钟数据拼成一张日频因子表
        self.daily_factors = None
        # 最终月度因子表格
        self.monthly_factors = None
        # 日频因子文件保存路径
        self.daily_factors_path = daily_factors_path
        # 月频因子文件保存路径
        self.monthly_factors_path = monthly_factors_path
        #沪深300和中证500成分股
        self.hs300=pd.read_excel('/Users/chenzongwei/pythoncode/数据库/日频数据/300和500成分股.xlsx',sheet_name='沪深300').stack().reset_index(level=1)
        self.zz500=pd.read_excel('/Users/chenzongwei/pythoncode/数据库/日频数据/300和500成分股.xlsx',sheet_name='中证500').stack().reset_index(level=1)
        self.hs300.columns=['date','code']
        self.zz500.columns=['date','code']
        self.hs300.date=pd.to_datetime(self.hs300.date)
        self.zz500.date=pd.to_datetime(self.zz500.date)
        self.hs300=self.hs300.sort_values(['date'])
        self.zz500=self.zz500.sort_values(['date'])
        add_top_hs300=self.hs300.date.min()
        add_top_zz500=self.zz500.date.min()
        add_top_hs300=add_top_hs300-pd.Timedelta(days=add_top_hs300.day)
        add_top_zz500=add_top_zz500-pd.Timedelta(days=add_top_zz500.day)
        self.hs300.date=self.hs300.date.shift(1)
        self.zz500.date=self.zz500.date.shift(1)
        self.hs300.date=self.hs300.date.fillna(add_top_hs300)
        self.zz500.date=self.zz500.date.fillna(add_top_zz500)
        self.hs300=self.hs300.set_index(['date'])
        self.zz500=self.zz500.set_index(['date'])
        self.hs300=self.hs300.groupby(['code']).resample('M').last()
        self.zz500=self.zz500.groupby(['code']).resample('M').last()
        self.hs300=self.hs300.drop(columns=['code']).reset_index()
        self.zz500=self.zz500.drop(columns=['code']).reset_index()


    def __call__(self):
        '''为了防止属性名太多，忘记了要调用哪个才是结果，因此可以直接输出月度数据表'''
        return self.monthly_factors

    def __add__(self, selfas):
        '''将几个因子截面标准化之后，因子值相加'''
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2s=[]
        for selfa in selfas:
            fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2s.append(fac2)
        for i in fac2s:
            fac1=fac1+i
        new_pure=pure_fall()
        new_pure.monthly_factors=fac1
        return new_pure

    def __mul__(self,selfas):
        '''将几个因子横截面标准化之后，使其都为正数，然后因子值相乘'''
        fac1=self.standardlize_in_cross_section(self.monthly_factors)
        fac1=fac1-fac1.min()
        fac2s=[]
        for selfa in selfas:
            fac2=self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2=fac2-fac2.min()
            fac2s.append(fac2)
        for i in fac2s:
            fac1=fac1*i
        new_pure=pure_fall()
        new_pure.monthly_factors=fac1
        return new_pure

    def __truediv__(self, selfa):
        '''两个一正一副的因子，可以用此方法相减'''
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
        fac=fac1-fac2
        new_pure=pure_fall()
        new_pure.monthly_factors=fac
        return new_pure

    def __floordiv__(self, selfa):
        '''两个因子一正一负，可以用此方法相除'''
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
        fac1=fac1-fac1.min()
        fac2=fac2-fac2.min()
        fac=fac1/fac2
        fac=fac.replace(np.inf,np.nan)
        new_pure=pure_fall()
        new_pure.monthly_factors=fac
        return new_pure

    def __sub__(self, selfa):
        '''用主因子剔除其他相关因子、传统因子等
        selfa可以为多个因子对象组成的元组或列表，每个辅助因子只需要有月度因子文件路径即可'''
        fac_main = self.wide_to_long(self.monthly_factors, 'fac')
        fac_helps = [i.monthly_factors for i in selfa]
        help_names = ['help' + str(i) for i in range(1, (len(fac_helps) + 1))]
        fac_helps = list(map(self.wide_to_long, fac_helps, help_names))
        fac_helps = pd.concat(fac_helps, axis=1)
        facs = pd.concat([fac_main, fac_helps], axis=1).dropna()
        facs = facs.groupby('date').apply(lambda x: self.de_in_group(x, help_names))
        facs = facs.unstack()
        facs.columns = list(map(lambda x: x[1], list(facs.columns)))
        return facs

    def __gt__(self,selfa):
        '''用于输出25分组表格，使用时，以x>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子'''
        x=self.monthly_factors.copy()
        y=selfa.monthly_factors.copy()
        x=x.stack().reset_index()
        y=y.stack().reset_index()
        x.columns=['date','code','fac']
        y.columns=['date','code','fac']
        shen=pure_moon()
        x=x.groupby('date').apply(lambda df:shen.get_groups(df,5))
        x=x.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupx'})
        xy=pd.merge(x,y,on=['date','code'])
        xy=xy.groupby(['date','groupx']).apply(lambda df:shen.get_groups(df,5))
        xy=xy.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupy'})
        xy=xy.assign(fac=xy.groupx*5+xy.groupy)
        xy=xy[['date','code','fac']]
        xy=xy.set_index(['date','code']).unstack()
        xy.columns=[i[1] for i in list(xy.columns)]
        new_pure=pure_fall()
        new_pure.monthly_factors=xy
        return new_pure

    def __rshift__(self, selfa):
        '''用于输出100分组表格，使用时，以x>>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子'''
        x=self.monthly_factors.copy()
        y=selfa.monthly_factors.copy()
        x=x.stack().reset_index()
        y=y.stack().reset_index()
        x.columns=['date','code','fac']
        y.columns=['date','code','fac']
        shen=pure_moon()
        x=x.groupby('date').apply(lambda df:shen.get_groups(df,10))
        x=x.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupx'})
        xy=pd.merge(x,y,on=['date','code'])
        xy=xy.groupby(['date','groupx']).apply(lambda df:shen.get_groups(df,10))
        xy=xy.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupy'})
        xy=xy.assign(fac=xy.groupx*10+xy.groupy)
        xy=xy[['date','code','fac']]
        xy=xy.set_index(['date','code']).unstack()
        xy.columns=[i[1] for i in list(xy.columns)]
        new_pure=pure_fall()
        new_pure.monthly_factors=xy
        return new_pure

    def wide_to_long(self, df, i):
        '''将宽数据转化为长数据，用于因子表转化和拼接'''
        df = df.stack().reset_index()
        df.columns = ['date', 'code', i]
        df = df.set_index(['date', 'code'])
        return df

    def de_in_group(self, df, help_names):
        '''对每个时间，分别做回归，剔除相关因子'''
        ols_order = 'fac~' + '+'.join(help_names)
        ols_result = smf.ols(ols_order, data=df).fit()
        params = {i: ols_result.params[i] for i in help_names}
        predict = [params[i] * df[i] for i in help_names]
        predict = reduce(lambda x, y: x + y, predict)
        df.fac = df.fac - predict - ols_result.params['Intercept']
        df = df[['fac']]
        return df

    def mat_to_df(self, mat):
        '''将mat文件变成'''
        mat_path = '/'.join([self.minute_files_path, mat])
        df = list(scio.loadmat(mat_path).values())[3]
        df = pd.DataFrame(df, columns=self.minute_columns)
        df.date = pd.to_datetime(df.date.apply(str), format='%Y%m%d')
        df = df.set_index('date')
        return df

    def add_suffix(self, code):
        '''给股票代码加上后缀'''
        if not isinstance(code, str):
            code = str(code)
        if len(code) < 6:
            code = '0' * (6 - len(code)) + code
        if code.startswith('0') or code.startswith('3'):
            code = '.'.join([code, 'SZ'])
        elif code.startswith('6'):
            code = '.'.join([code, 'SH'])
        elif code.startswith('8'):
            code = '.'.join([code, 'BJ'])
        return code

    def minute_to_daily(self, func):
        '''
        将分钟数据变成日频因子，并且添加到日频因子表里
        通常应该每天生成一个指标，最后一只股票会生成一个series
        '''
        for mat in tqdm.tqdm(self.minute_files):
            df = self.mat_to_df(mat)
            the_func = partial(func)
            df = df.groupby('date').apply(the_func)
            code = self.add_suffix(mat[-10:-4])
            df = df.to_frame(name=code)
            self.daily_factors_list.append(df)
        self.daily_factors = pd.concat(self.daily_factors_list, axis=1)
        self.daily_factors.reset_index().to_feather(self.daily_factors_path)

    def minute_to_daily_whole(self, func):
        '''
        将分钟数据变成日频因子，并且添加到日频因子表里
        通常应该每天生成一个指标，最后一只股票会生成一个series
        '''
        for mat in tqdm.tqdm(self.minute_files):
            self.code=self.add_suffix(mat[-10:-4])
            df = self.mat_to_df(mat)
            the_func = partial(func)
            df = func(df)
            if isinstance(df,pd.DataFrame):
                df.columns = [self.code]
                self.daily_factors_list.append(df)
        self.daily_factors = pd.concat(self.daily_factors_list, axis=1)
        self.daily_factors.reset_index().to_feather(self.daily_factors_path)

    def standardlize_in_cross_section(self, df):
        '''
        在横截面上做标准化
        输入的df应为，列名是股票代码，索引是时间
        '''
        df = df.T
        df = (df - df.mean()) / df.std()
        df = df.T
        return df

    def get_daily_factors(self, func, whole):
        '''调用分钟到日度方法，算出日频数据'''
        try:
            self.daily_factors = pd.read_feather(self.daily_factors_path)
            self.daily_factors = self.daily_factors.set_index('date')
        except Exception:
            if whole:
                self.minute_to_daily_whole(func)
            else:
                self.minute_to_daily(func)

    def get_neutral_monthly_factors(self, df,boxcox=False):
        '''对月度因子做市值中性化处理'''
        shen = pure_moon()
        shen.set_factor_df_date_as_index(df)
        if boxcox:
            shen.run(5, boxcox=True, plt=False, print_comments=False)
        else:
            shen.run(5,neutralize=True,plt=False,print_comments=False)
        new_factors = shen.factors.copy()
        new_factors = new_factors.set_index(['date', 'code']).unstack()
        new_factors.columns = list(map(lambda x: x[1], list(new_factors.columns)))
        new_factors = new_factors.reset_index()
        add_start_point = new_factors.date.min()
        add_start_point = add_start_point - pd.Timedelta(days=add_start_point.day)
        new_factors.date = new_factors.date.shift(1)
        new_factors.date = new_factors.date.fillna(add_start_point)
        new_factors = new_factors.set_index('date')
        return new_factors

    def get_monthly_factors(self, func, neutralize,boxcox,hs300,zz500):
        '''将日频的因子转化为月频因子'''
        two_parts=self.monthly_factors_path.split('.')
        self.monthly_factors_path_hs300=two_parts[0]+'沪深300.'+two_parts[1]
        self.monthly_factors_path_zz500=two_parts[0]+'中证500.'+two_parts[1]
        try:
            if hs300:
                self.monthly_factors=pd.read_feather(self.monthly_factors_path_hs300)
            elif zz500:
                self.monthly_factors=pd.read_feather(self.monthly_factors_path_zz500)
            else:
                self.monthly_factors = pd.read_feather(self.monthly_factors_path)
            self.monthly_factors = self.monthly_factors.set_index('date')
        except Exception:
            the_func = partial(func)
            self.monthly_factors = the_func(self.daily_factors)
            if hs300:
                self.monthly_factors = self.monthly_factors.stack().reset_index()
                self.monthly_factors.columns = ['date', 'code', 'fac']
                self.monthly_factors=pd.merge(self.monthly_factors,self.hs300,on=['date', 'code'])
                self.monthly_factors=self.monthly_factors.drop_duplicates(subset=['date','code'])
                self.monthly_factors=self.monthly_factors.set_index(['date','code'])
                self.monthly_factors=self.monthly_factors.unstack()
                self.monthly_factors.columns=[i[1] for i in list(self.monthly_factors.columns)]
            if zz500:
                self.monthly_factors = self.monthly_factors.stack().reset_index()
                self.monthly_factors.columns = ['date', 'code', 'fac']
                self.monthly_factors = pd.merge(self.monthly_factors, self.zz500,on=['date', 'code'])
                self.monthly_factors=self.monthly_factors.set_index(['date','code'])
                self.monthly_factors=self.monthly_factors.unstack()
                self.monthly_factors.columns=[i[1] for i in list(self.monthly_factors.columns)]
            if neutralize:
                self.monthly_factors = self.get_neutral_monthly_factors(self.monthly_factors)
            elif boxcox:
                self.monthly_factors=self.get_neutral_monthly_factors(self.monthly_factors,boxcox=True)
            if hs300:
                self.monthly_factors.reset_index().to_feather(self.monthly_factors_path_hs300)
            elif zz500:
                self.monthly_factors.reset_index().to_feather(self.monthly_factors_path_zz500)
            else:
                self.monthly_factors.reset_index().to_feather(self.monthly_factors_path)

    def run(self, whole=False,daily_func=None, monthly_func=None, neutralize=False,boxcox=False,hs300=False,zz500=False):
        '''执必要的函数，将分钟数据变成月度因子'''
        self.get_daily_factors(daily_func,whole)
        self.get_monthly_factors(monthly_func, neutralize,boxcox,hs300,zz500)


class pure_fallmount(pure_fall):
    '''继承自父类，专为做因子截面标准化之后相加和因子剔除其他辅助因子的作用'''

    def __init__(self, monthly_factors):
        '''输入月度因子值，以设定新的对象'''
        super(pure_fall, self).__init__()
        self.monthly_factors = monthly_factors

    def __call__(self):
        '''完全与父类功能相同'''
        return self.monthly_factors

    def __add__(self, selfas):
        '''返回一个对象，而非一个表格，如需表格请调用对象'''
        fac1 = self.standardlize_in_cross_section(self.monthly_factors)
        fac2s = []
        for selfa in selfas:
            fac2 = self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2s.append(fac2)
        for i in fac2s:
            fac1 = fac1 + i
        new_pure = pure_fallmount(fac1)
        return new_pure

    def __mul__(self,selfas):
        '''将几个因子横截面标准化之后，使其都为正数，然后因子值相乘'''
        fac1=self.standardlize_in_cross_section(self.monthly_factors)
        fac1=fac1-fac1.min()
        fac2s=[]
        for selfa in selfas:
            fac2=self.standardlize_in_cross_section(selfa.monthly_factors)
            fac2=fac2-fac2.min()
            fac2s.append(fac2)
        for i in fac2s:
            fac1=fac1*i
        new_pure=pure_fall()
        new_pure.monthly_factors=fac1
        return new_pure

    def __sub__(self, selfa):
        '''返回对象，如需表格，请调用对象'''
        fac_main = self.wide_to_long(self.monthly_factors, 'fac')
        fac_helps = [i.monthly_factors for i in selfa]
        help_names = ['help' + str(i) for i in range(1, (len(fac_helps) + 1))]
        fac_helps = list(map(self.wide_to_long, fac_helps, help_names))
        fac_helps = pd.concat(fac_helps, axis=1)
        facs = pd.concat([fac_main, fac_helps], axis=1).dropna()
        facs = facs.groupby('date').apply(lambda x: self.de_in_group(x, help_names))
        facs = facs.unstack()
        facs.columns = list(map(lambda x: x[1], list(facs.columns)))
        new_pure = pure_fallmount(facs)
        return new_pure

    def __gt__(self,selfa):
        '''用于输出25分组表格，使用时，以x>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子'''
        x=self.monthly_factors.copy()
        y=selfa.monthly_factors.copy()
        x=x.stack().reset_index()
        y=y.stack().reset_index()
        x.columns=['date','code','fac']
        y.columns=['date','code','fac']
        shen=pure_moon()
        x=x.groupby('date').apply(lambda df:shen.get_groups(df,5))
        x=x.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupx'})
        xy=pd.merge(x,y,on=['date','code'])
        xy=xy.groupby(['date','groupx']).apply(lambda df:shen.get_groups(df,5))
        xy=xy.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupy'})
        xy=xy.assign(fac=xy.groupx*5+xy.groupy)
        xy=xy[['date','code','fac']]
        xy=xy.set_index(['date','code']).unstack()
        xy.columns=[i[1] for i in list(xy.columns)]
        new_pure=pure_fallmount(xy)
        return new_pure

    def __rshift__(self, selfa):
        '''用于输出100分组表格，使用时，以x>>y的形式使用，其中x,y均为pure_fall对象
        计算时使用的是他们的月度因子表，即self.monthly_factors属性，为宽数据形式的dataframe
        x应为首先用来的分组的主因子，y为在x分组后的组内继续分组的次因子'''
        x=self.monthly_factors.copy()
        y=selfa.monthly_factors.copy()
        x=x.stack().reset_index()
        y=y.stack().reset_index()
        x.columns=['date','code','fac']
        y.columns=['date','code','fac']
        shen=pure_moon()
        x=x.groupby('date').apply(lambda df:shen.get_groups(df,10))
        x=x.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupx'})
        xy=pd.merge(x,y,on=['date','code'])
        xy=xy.groupby(['date','groupx']).apply(lambda df:shen.get_groups(df,10))
        xy=xy.reset_index(drop=True).drop(columns=['fac']).rename(columns={'group':'groupy'})
        xy=xy.assign(fac=xy.groupx*10+xy.groupy)
        xy=xy[['date','code','fac']]
        xy=xy.set_index(['date','code']).unstack()
        xy.columns=[i[1] for i in list(xy.columns)]
        new_pure=pure_fallmount(xy)
        return new_pure

class pure_winter():
    def __init__(self):
        # barra因子数据
        self.barras = self.read_hdf5('/Users/chenzongwei/pythoncode/数据库/barra数据/FactorLoading_Style.h5')
        # 行业哑变量数据
        self.industrys = self.read_hdf5('/Users/chenzongwei/pythoncode/数据库/barra数据/FactorLoading_Industry.h5')

    def __call__(self,fallmount=0):
        '''返回纯净因子值'''
        if fallmount==0:
            return self.snow_fac
        else:
            return pure_fallmount(self.snow_fac)

    def read_hdf5(self, file_full_root, subgroup=None):
        '''读入h5文件'''
        if os.path.exists(file_full_root):
            file_h = pd.HDFStore(file_full_root, 'r')
            file_keys_raw = file_h.keys()
            file_keys = [x[1:] for x in file_keys_raw]
            if len(file_keys) == 1 and subgroup is None:
                result = file_h[file_keys[0]]
            elif len(file_keys) == 1 and subgroup is not None:
                if subgroup in file_keys:
                    result = file_h[subgroup]
                else:
                    result = pd.DataFrame()
            elif len(file_keys) > 1 and subgroup is not None:
                try:
                    result = file_h[subgroup]
                except:
                    result = pd.DataFrame()
            elif len(file_keys) > 1 and subgroup is None:
                result = {}
                for i in file_keys:
                    result[i] = file_h[i]
            else:
                # 原始文件是空的
                if subgroup is not None:
                    result = pd.DataFrame()
                else:
                    result = None
            file_h.close()
        else:
            result = None
        return result

    def last_month_end(self, x):
        '''找到下个月最后一天'''
        x1 = x = x - relativedelta(months=1)
        while x1.month == x.month:
            x1 = x1 + relativedelta(days=1)
        return x1 - relativedelta(days=1)

    def set_factors_df(self, df):
        '''传入因子dataframe，应为三列，第一列是时间，第二列是股票代码，第三列是因子值'''
        df.columns = ['date', 'code', 'fac']
        df = df.set_index(['date', 'code'])
        df = df.unstack().reset_index()
        df.date = df.date.apply(self.last_month_end)
        df = df.set_index(['date']).stack()
        self.factors = df

    def set_factors_df_wide(self, df):
        '''传入因子数据，时间为索引，代码为列名'''
        df=df.reset_index()
        df.date = df.date.apply(self.last_month_end)
        df=df.set_index(['date'])
        df=df.stack().reset_index()
        df.columns = ['date', 'code','fac']
        self.factors=df

    def daily_to_monthly(self, df):
        '''将日度的barra因子月度化'''
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
        df = df.resample('M').last()
        return df

    def get_monthly_barras_industrys(self):
        '''将barra因子和行业哑变量变成月度数据'''
        for key, value in self.barras.items():
            self.barras[key] = self.daily_to_monthly(value)
        for key, value in self.industrys.items():
            self.industrys[key] = self.daily_to_monthly(value)

    def wide_to_long(self, df, name):
        '''将宽数据变成长数据，便于后续拼接'''
        df = df.stack().reset_index()
        df.columns = ['date', 'code', name]
        df = df.set_index(['date', 'code'])
        return df

    def get_wide_barras_industrys(self):
        '''将barra因子和行业哑变量都变成长数据'''
        for key, value in self.barras.items():
            self.barras[key] = self.wide_to_long(value, key)
        for key, value in self.industrys.items():
            self.industrys[key] = self.wide_to_long(value, key)

    def get_corr_pri_ols_pri(self):
        '''拼接barra因子和行业哑变量，生成用于求相关系数和纯净因子的数据表'''
        if self.factors.shape[0]>1:
            self.factors=self.factors.set_index(['date', 'code'])
        self.corr_pri = pd.concat([self.factors] + list(self.barras.values()), axis=1).dropna()
        self.ols_pri = pd.concat([self.corr_pri] + list(self.industrys.values()), axis=1).dropna()

    def get_corr(self):
        '''计算每一期的相关系数，再求平均值'''
        self.corr_by_step = self.corr_pri.groupby(['date']).apply(lambda x: x.corr().head(1))
        self.corr = self.corr_by_step.mean()

    def ols_in_group(self, df):
        '''对每个时间段进行回归，并计算残差'''
        xs = list(df.columns)
        xs = [i for i in xs if i != 'fac']
        xs_join = '+'.join(xs)
        ols_formula = 'fac~' + xs_join
        ols_result = smf.ols(ols_formula, data=df).fit()
        ols_ws = {i: ols_result.params[i] for i in xs}
        ols_b = ols_result.params['Intercept']
        to_minus = [ols_ws[i] * df[i] for i in xs]
        to_minus = reduce(lambda x, y: x + y, to_minus)
        df = df.assign(snow_fac=df.fac - to_minus - ols_b)
        df = df[['snow_fac']]
        df = df.rename(columns={'snow_fac': 'fac'})
        return df

    def get_snow_fac(self):
        '''获得纯净因子'''
        self.snow_fac = self.ols_pri.groupby(['date']).apply(self.ols_in_group)
        self.snow_fac = self.snow_fac.unstack()
        self.snow_fac.columns = list(map(lambda x: x[1], list(self.snow_fac.columns)))

    def run(self):
        '''运行一些必要的函数'''
        self.get_monthly_barras_industrys()
        self.get_wide_barras_industrys()
        self.get_corr_pri_ols_pri()
        self.get_corr()
        self.get_snow_fac()


class pure_snowtrain(pure_winter):
    '''直接返回纯净因子'''

    def __init__(self, factors):
        '''直接输入原始因子数据'''
        super(pure_snowtrain, self).__init__()
        self.set_factors_df_wide(factors)
        self.run()

    def __call__(self, fallmount=0):
        '''可以直接返回pure_fallmount对象，或纯净因子矩阵'''
        if fallmount == 0:
            return self.snow_fac
        else:
            return pure_fallmount(self.snow_fac)


class pure_moonlight(pure_moon):
    '''继承自pure_moon回测框架，使用其中的日频复权收盘价数据，以及换手率数据'''

    def __init__(self):
        '''加载全部数据'''
        super(pure_moonlight, self).__init__()
        self.col_and_index()
        self.load_all_files()
        self.judge_month()
        self.get_log_cap()
        # 对数市值
        self.cap_as_factor = self.cap[['date', 'code', 'cap_size']].set_index(['date', 'code']).unstack()
        self.cap_as_factor.columns = list(map(lambda x: x[1], list(self.cap_as_factor.columns)))
        # 传统反转因子ret20
        self.ret20_database = '/Users/chenzongwei/pythoncode/数据库/因子数据/月频_反转因子ret20.feather'
        # 传统换手率因子turn20
        self.turn20_database = '/Users/chenzongwei/pythoncode/数据库/因子数据/月频_换手率因子turn20.feather'
        # 传统波动率因子vol20
        self.vol20_database = '/Users/chenzongwei/pythoncode/数据库/因子数据/月频_波动率因子vol20.feather'
        # #自动更新
        self.get_updated_factors()

    def __call__(self, name):
        '''可以通过call方式，直接获取对应因子数据'''
        value = getattr(self, name)
        return value

    def get_ret20(self, pri):
        '''计算20日涨跌幅因子'''
        past = pri.iloc[:-20, :]
        future = pri.iloc[20:, :]
        ret20 = (future.to_numpy() - past.to_numpy()) / past.to_numpy()
        df = pd.DataFrame(ret20, columns=pri.columns, index=future.index)
        df = df.resample('M').last()
        return df

    def get_turn20(self, pri):
        '''计算20换手率因子'''
        turns = pri.rolling(20).mean()
        turns = turns.resample('M').last().reset_index()
        turns.columns = ['date'] + list(turns.columns)[1:]
        self.factors = turns
        self.get_neutral_factors()
        df = self.factors.copy()
        df = df.set_index(['date', 'code'])
        df = df.unstack()
        df.columns = list(map(lambda x: x[1], list(df.columns)))
        return df

    def get_vol20(self, pri):
        '''计算20日波动率因子'''
        rets = pri.pct_change()
        vol = rets.rolling(20).apply(np.std)
        df = vol.resample('M').last()
        return df

    def update_single_factor_in_database(self, path, pri, func):
        '''
        用基础数据库更新因子数据库
        执行顺序为，先读取文件，如果没有，就直接全部计算，然后存储
        如果有，就读出来看看。数据是不是最新的
        如果不是最新的，就将原始数据在上次因子存储的日期处截断
        计算出新的一段时间的因子值，然后追加写入因子文件中
        '''
        the_func = partial(func)
        try:
            df = pd.read_feather(path)
            df.columns = ['date'] + list(df.columns)[1:]
            df = df.set_index('date')
            if df.index.max() < pri.index.max():
                to_add = pri[(pri.index > df.index.max()) & (pri.index <= pri.index.max())]
                to_add = the_func(to_add)
                df = pd.concat([df, to_add])
                print('之前数据有点旧了，已为您完成更新')
            else:
                print('数据很新，无需更新')
            df1 = df.reset_index()
            df1.columns = ['date'] + list(df1.columns)[1:]
            df1.to_feather(path)
            return df
        except Exception:
            df = the_func(pri)
            df1 = df.reset_index()
            df1.columns = ['date'] + list(df1.columns)[1:]
            df1.to_feather(path)
            print('新因子建库完成')
            return df

    def get_updated_factors(self):
        '''更新因子数据'''
        self.ret20 = self.update_single_factor_in_database(self.ret20_database, self.closes, self.get_ret20)
        self.turn20 = self.update_single_factor_in_database(self.turn20_database, self.turnovers, self.get_turn20)
        self.vol20 = self.update_single_factor_in_database(self.vol20_database, self.closes, self.get_vol20)


class pure_moonnight(pure_moon):
    '''封装选股框架'''

    def __init__(self, factors, groups_num, neutralize=False, boxcox=False,by10=False, y2=False, plt=True, plotly=False,
                 filename='分组净值图', time_start=None, time_end=None, print_comments=True):
        '''直接输入因子数据'''
        super(pure_moonnight, self).__init__()
        if isinstance(factors, str):
            self.set_factor_file(factors)
        else:
            self.set_factor_df_date_as_index(factors)
        self.run(groups_num, neutralize,boxcox, by10, y2, plt, plotly, filename, time_start, time_end, print_comments)

    def __call__(self, fallmount=0):
        '''调用则返回因子数据'''
        if fallmount == 0:
            return self.factors
        else:
            return pure_fallmount(self.factors)

class pure_newyear():
    '''转为生成25分组和百分组的收益矩阵而封装'''
    def __init__(self,facx,facy,group_num_single,namex='主',namey='次'):
        '''初始化时即进行回测'''
        homex=pure_fallmount(facx)
        homey=pure_fallmount(facy)
        if group_num_single==5:
            homexy=homex>homey
        elif group_num_single==10:
            homexy=homex>>homey
        shen=pure_moonnight(homexy(),group_num_single**2,plt=False,print_comments=False)
        sq=shen.square_rets.copy()
        sq.index=[namex+str(i) for i in list(sq.index)]
        sq.columns=[namey+str(i) for i in list(sq.columns)]
        self.square_rets=sq

    def __call__(self):
        '''调用对象时，返回最终结果，正方形的分组年化收益率表'''
        return self.square_rets

class pure_dawn():
    '''
    因子切割论的母框架，可以对两个因子进行类似于因子切割的操作
    可用于派生任何"以两个因子生成一个因子"的子类
    '''
    def __init__(self,fac1,fac2):
        self.fac1=fac1
        self.fac2=fac2

    def __call__(self):
        '''返回最终月度因子值'''
        return self.fac

    def get_fac_long_and_tradedays(self):
        '''将两个因子的矩阵转化为长列表'''
        self.fac1=self.fac1.stack().reset_index()
        self.fac1.columns=['date','code','fac1']
        self.fac2=self.fac2.stack().reset_index()
        self.fac2.columns=['date','code','fac2']
        fac=pd.merge(self.fac1,self.fac2,on=['date','code'])
        fac=fac.sort_values(['date','code'])
        self.fac=fac
        self.tradedays=sorted(list(set(self.fac.date)))

    def get_month_starts_and_ends(self):
        '''计算出每个月回看期间的起点日和终点日'''
        self.month_ends=[i for i,j in zip(self.tradedays[:-1],self.tradedays[1:]) if i.month!=j.month]
        self.month_ends.append(self.tradedays[-1])
        self.month_starts=[self.find_begin(self.tradedays,i) for i in self.month_ends]
        self.month_starts[0]=self.tradedays[0]

    def find_begin(self,tradedays,end_day,backsee=20):
        '''找出回看若干天的开始日，默认为20'''
        end_day_index=tradedays.index(end_day)
        start_day_index=end_day_index-backsee+1
        start_day=tradedays[start_day_index]
        return start_day

    def make_monthly_factors_single_code(self,df,func):
        '''
        对单一股票来计算月度因子
        func为单月执行的函数，返回值应为月度因子，如一个float或一个list
        df为一个股票的四列表，包含时间、代码、因子1和因子2
        '''
        res={}
        for start,end in zip(self.month_starts,self.month_ends):
            this_month=df[(df.date>=start)&(df.date<=end)]
            res[end]=func(this_month)
        dates=list(res.keys())
        corrs=list(res.values())
        part=pd.DataFrame({'date':dates,'corr':corrs})
        return part

    def get_monthly_factor(self,func):
        '''运行自己写的函数，获得月度因子'''
        tqdm.tqdm.pandas(desc='when the dawn comes, tonight will be a memory too.')
        self.fac=self.fac.groupby(['code']).progress_apply(lambda x:self.make_monthly_factors_single_code(x,func))
        self.fac=self.fac.reset_index(level=1,drop=True).reset_index().set_index(['date','code']).unstack()
        self.fac.columns=[i[1] for i in list(self.fac.columns)]
        self.fac=self.fac.resample('M').last()

    def run(self,func):
        '''运行必要的函数'''
        self.get_fac_long_and_tradedays()
        self.get_month_starts_and_ends()
        self.get_monthly_factor(func)
