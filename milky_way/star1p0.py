import os
import sys
import tqdm
import datetime
import numpy as np
import pandas as pd
import scipy.io as scio
from itertools import chain
from dateutil.relativedelta import relativedelta
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as pff
from plotly.offline import plot
import plotly.io as pio
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class star_data_core_time_single_point:
    '''单个时刻类，包含三个属性，日期、第几分钟和上个交易日日期，是交易日历和当前时刻的构成元素'''
    def __init__(self,daily,minute_order):
        #该时刻的日期
        self.daily=daily
        #该时刻是该日的第几个分钟(1-242之间)
        self.minute_order=minute_order

    def set_daily_last(self,daily_last):
        '''设置该时刻的上个交易日的日期'''
        self.daily_last=daily_last

    def is_end_of_day(self):
        '''判断该时刻是否为当天的收盘时刻'''
        if self.minute_order==242:
            return True
        else:
            return False

    def to_datetime(self):
        '''将single_point格式的对象转化为datetime'''
        if self.minute_order<=121:
            return self.daily+relativedelta(hours=9,minutes=(29+self.minute_order))
        else:
            return self.daily+relativedelta(hours=13,minutes=self.minute_order)

class star_data_params:
    '''设置参数相关'''
    def __init__(self):
        #交易手续费，默认千分之二
        self.trade_cost=0.002
        #交易滑点，默认千分之五
        self.slippage=0.005
        #初始资金，默认100万
        self.initial_cash=1000000

    def set_trade_cost(self,trade_cost):
        '''设置手续费'''
        self.trade_cost=trade_cost

    def set_slippage(self,slippage):
        '''设置滑点'''
        self.slippage=slippage

    def set_initial_cash(self,initial_cash):
        '''设置初始资金'''
        self.initial_cash=initial_cash

    def set_stock_list(self,stock_list):
        '''设置股票池'''
        self.stock_list=stock_list

class star_data_core_time_tradedays_series:
    '''交易日历列表'''
    def __init__(self):
        tradedays_path='/Users/chenzongwei/pythoncode/日频数据/TradingDate_Daily.mat'
        self.tradedays=list(scio.loadmat(tradedays_path).values())[3].flatten()
        self.tradedays=pd.to_datetime(pd.Series(self.tradedays),format='%Y%m%d')

    def timestamp_to_single_point(self,x):
        '''把单个时间日期，转化为242个single_point的列表'''
        xs=list(map(star_data_core_time_single_point,[x]*242,list(range(1,243))))
        return xs

    def date_convert(self):
        '''把交易日历，转化为single_point对象组成的迭代器'''
        self.tradedays=self.tradedays.apply(self.timestamp_to_single_point)
        self.tradedays=iter(list(chain.from_iterable(self.tradedays.to_numpy())))

    def set_backtest_period(self,start,end):
        '''设置起始时间'''
        self.backtest_start_day=pd.Timestamp(start)
        self.backtest_end_day=pd.Timestamp(end)
        self.tradedays=self.tradedays[
            (self.tradedays>=self.backtest_start_day)&(self.tradedays<=self.backtest_end_day)]
        #转化为迭代器形式
        self.date_convert()

class star_data_core_pri_data_and_iter_entry_dates(star_data_params,star_data_core_time_tradedays_series):
    '''原始数据迭代器和开始迭代时间表'''
    def __init__(self):
        #继承交易日历
        star_data_params.__init__(self)
        star_data_core_time_tradedays_series.__init__(self)
        #原始数据字典
        self.pri_data={}
        #开始进入迭代数据的时间
        self.iter_entry_dates={}

    def find_and_read_single_pri_data(self,code,files,start,end):
        '''根据股票代码，读入对应的数据，将数据转化为生成器，并返回进入迭代时间'''
        if '.' in code:
            code=code[:-3]
        file=[i for i in files if code in i][0]
        file='/'.join(['/Users/chenzongwei/pythoncode/分钟数据',file])
        data=list(scio.loadmat(file).values())[3]
        data=pd.DataFrame(data,columns=['date','open','high','low','close','amount','money'])
        data=data.fillna(method='ffill')
        data.date=pd.to_datetime(data.date,format='%Y%m%d')
        data=data[(data.date>=start)&(data.date<=end)]
        iter_entry_daily=data.date.min()
        data=data.itertuples()
        return iter_entry_daily,data

    def read_in_all_pri_data(self):
        '''根据股票池读入原始数据'''
        pri_data_files=os.listdir('/Users/chenzongwei/pythoncode/分钟数据')
        for stock in self.stock_list:
            entry_date,pri_data=self.find_and_read_single_pri_data(
                stock,pri_data_files,self.backtest_start_day,self.backtest_end_day)
            if entry_date in self.iter_entry_dates:
                self.iter_entry_dates[entry_date].append(stock)
            else:
                self.iter_entry_dates[entry_date]=[stock]
            self.pri_data[stock]=pri_data

class star_data_core_iter_data(star_data_core_pri_data_and_iter_entry_dates):
    '''迭代数据'''
    def __init__(self):
        star_data_core_pri_data_and_iter_entry_dates.__init__(self)
        self.iter_data={}

    def update_iter_data(self,now):
        '''更新正在迭代的数据，只在self.now.minute_order==1时更新'''
        if now.minute_order==1:
            if now.daily in self.iter_entry_dates:
                for stock in self.iter_entry_dates[now.daily]:
                    self.iter_data[stock]=self.pri_data[stock]

class star_data_now_basic_info(star_data_core_iter_data):
    '''当前分钟的基础数据'''
    def __init__(self):
        star_data_core_iter_data.__init__(self)
        self.now_data={}
        self.last_closes={}
        self.history_data={}
        #现金
        self.now_cash=self.initial_cash
        #当前持仓详情字典
        self.positions={}
        #当前存活的卖出订单
        self.breathing_orders_buy=None
        #当前存活的买入订单
        self.breathing_orders_sell=None
        #订单记录
        self.order_records=star_data_order_records()
        #持仓市值
        self.now_cap=0
        #账户净值
        self.now_net=self.now_cash+self.now_cap
        #净值记录
        self.net_records={}
        #持仓市值记录
        self.cap_records={}
        #现金记录
        self.cash_records={}

    def minute_to_daily(self,df):
        '''将分钟数据转化为日度数据'''
        date=df.date.iloc[0]
        open=df.open.iloc[0]
        high=df.high.max()
        low=df.low.min()
        close=df.close.iloc[-1]
        amount=df.amount.sum()
        money=df.money.sum()
        daily=pd.DataFrame({
            'date':[date],
            'minute_order':[0],
            'open':[open],
            'high':[high],
            'low':[low],
            'close':[close],
            'amount':[amount],
            'money':[money]
        })
        return daily

    def update_minute_basic(self,now):
        '''每次迭代时更新数据，每个时刻都更新'''
        to_del=[]
        for k,v in self.iter_data.items():
            try:
                here=next(v)
                self.now_data[k]=here
                history_add=pd.DataFrame({
                    'date':[now.daily],
                    'minute_order':[now.minute_order],
                    'open':[here.open],
                    'high':[here.high],
                    'low':[here.low],
                    'close':[here.close],
                    'amount':[here.amount],
                    'money':[here.money]
                })

                if k in self.history_data:
                    self.history_data[k]=pd.concat([self.history_data[k],history_add])
                else:
                    self.history_data[k]=history_add
            except Exception:
                to_del.append(k)
        for k in to_del:
            del self.iter_data[k]
        if now.minute_order==242:
            for k,v in self.now_data.items():
                self.last_closes[k]=v.close
            for k,v in self.history_data.items():
                month_ago=now.daily-relativedelta(months=1)
                old_history=v[(v.date<=month_ago)&(v.minute_order!=0)]
                if old_history.shape[0]>0:
                    old_history_ok=v[(v.date<=month_ago)&(v.minute_order==0)]
                    new_history=v[v.date>month_ago]
                    old_history=old_history.groupby('date').apply(self.minute_to_daily)
                    new_v=pd.concat([old_history_ok,old_history,new_history])
                    self.history_data[k]=new_v

class star_data_now_positions:
    '''当前持仓相关信息'''
    def __init__(self,code,hold_amount,buy_price,buy_value=None):
        '''股票代码，买入数量，买入价格'''
        self.code=code
        self.hold_amount=hold_amount
        self.buy_price=buy_price
        if buy_value is None:
            self.buy_value=self.buy_price*self.hold_amount
        else:
            self.buy_value=buy_value

    def __add__(self,other):
        '''在已有持仓的基础上追加持仓'''
        new=star_data_now_positions(
            code=self.code,
            hold_amount=self.hold_amount+other.hold_amount,
            buy_price=0,
            buy_value=self.buy_value+other.buy_value
        )
        new.buy_price=new.buy_value/new.hold_amount
        return new

    def get_last_amount(self):
        '''获得上一交易日最终持仓'''
        try:
            return self.last_amount
        except Exception:
            return 0

    def update_now_positions(self,now,now_data,amount=None):
        '''更新相关数据，每个时刻都更新'''
        if amount is not None:
            self.amount=amount
        #当前股票价格
        self.now_price=now_data[self.code].open
        #当前持仓市值
        self.hold_value=self.hold_amount*self.now_price
        #买入时所花总现金
        self.buy_value=self.buy_price*self.hold_amount
        #当前持仓收益
        self.hold_earning=self.hold_value-self.buy_value
        #收盘时更新当日最终持有量(对于明天来说即昨日持有量)
        if now.minute_order==242:
            self.last_amount=self.hold_amount

class star_data_order_records:
    def  __init__(self):
        self.order_records=[]

    def append(self,orders):
        '''追加交易记录，添加订单状态为0的订单
        使用时就使用self.order_records.append(self.breathing_orders)语句添加记录即可'''
        end_orders=[i for i in orders if i.order_alive==0]
        for order in end_orders:
            self.order_records.append(order)

class star_data_single_order(star_data_params):
    '''
    订单相关
    method可选limit或者market，默认limit
    life默认为当天结束
    price默认当前时刻收盘价
    action可选buy和sell，默认buy
    '''
    def __init__(
            self,
            order_code,
            order_amount,
            now,
            now_data,
            now_cash,
            now_cap,
            positions,
            total_sell_today,
            order_submit_time,
            order_price=None,
            order_method='limit',
            order_life=None,
            order_action='buy',
            order_alive=1,
            order_tail='unknown',
            order_remain=0
    ):
        star_data_params.__init__(self)
        #订单股票代码
        self.order_code=order_code
        #订单的下单量
        self.order_amount=order_amount
        #订单提交的时刻
        self.order_submit_time=order_submit_time
        #订单成交时刻（成交时再写入）
        self.order_success_time=None
        #订单的方式（限价为limit，市价为market，默认限价）
        self.order_method=order_method
        if order_price is not None:
            #订单的委托价格
            self.order_price=order_price
        else:
            self.order_price=now_data[order_code].close
        #订单的存续期（多少个时刻）
        if order_life is not None:
            self.order_life=order_life
        else:
            self.order_life=242-now.minute_order
        #订单的动作，买入还是卖出
        self.order_action=order_action
        #订单的生命状态，还在未成交的是1，过了存续期的，或者成交的，改为0
        self.order_alive=order_alive
        #订单消亡时的最终结局（未消亡时为unknown，成交的为success，过期失效的为die
        self.order_tail=order_tail
        #订单部分成交之后，剩余的量
        self.order_remain=order_remain
        #传入当前的现金、市值和持仓
        self.now_cash=now_cash
        self.now_cap=now_cap
        self.positions=positions
        self.total_sell_today=total_sell_today

    def order_success(self,now,way,now_data,now_cash,now_cap,positions,total_sell_today):
        '''订单成交，添加持仓、减少现金、增加市值、删除订单、添加交易记录'''
        if way=='low':
            actual_price=now_data[self.order_code].low*(1+self.trade_cost+self.slippage)
        elif way=='open':
            actual_price=now_data[self.order_code].open*(1+self.trade_cost+self.slippage)
        elif way=='high':
            actual_price=now_data[self.order_code].high*(1-self.trade_cost-self.slippage)
        elif way=='open_sell':
            actual_price=now_data[self.order_code].open*(1-self.trade_cost-self.slippage)
        #保证现金充足
        if way in ['low','open']:
            total_pay=min(actual_price*self.order_amount,now_cash)
        else:
            total_pay=actual_price*self.order_amount
        #自动整百，且成交量不会大于委托量
        actual_amount=min(np.floor(total_pay/actual_price/100)*100,self.order_amount)
        #t+1制度
        if self.order_action=='sell':
            left_amount=positions[self.order_code].get_last_amount()-total_sell_today[self.order_code]
            if left_amount<actual_amount:
                t_plus_1=1
            else:
                t_plus_1=0
            actual_amount=min(left_amount,actual_amount)
        total_pay=actual_amount*actual_price
        #现金减少或增加
        if way in ['low','open']:
            now_cash=now_cash-total_pay
            now_cap=now_cap+total_pay
        else:
            now_cash=now_cash+total_pay
            now_cap=now_cap-total_pay
        #写入成交价格
        self.deal_price=actual_price
        #写入成交量
        self.deal_amount=actual_amount
        #总成交金额
        self.deal_pay=total_pay
        #增加持仓或减少持仓
        if self.order_action=='buy':
            new_position=star_data_now_positions(
                self.order_code,actual_amount,actual_price)
        else:
            new_position=star_data_now_positions(
                self.order_code,-actual_amount,actual_price)
        if self.order_code in positions:
            positions[self.order_code]=positions[self.order_code]+new_position
        else:
            positions[self.order_code]=new_position
        self.order_remain=self.order_amount-actual_amount
        total_sell_today[self.order_code]=total_sell_today[self.order_code]+self.deal_amount
        self.order_success_time=now
        self.order_alive=0
        self.total_sell_today=total_sell_today
        self.now_cap=now_cap
        self.now_cash=now_cash
        self.positions=positions
        if self.order_remain==0:
            #如果全部成交
            self.order_tail='fully success'
        elif self.order_action=='sell' and t_plus_1==1:
            self.order_tail='t+1 partial success'
            self.order_remain=0
        elif self.order_action=='buy' and self.order_remain>0:
            self.order_tail='cash limit partial success'
        elif self.order_action=='sell' and t_plus_1==0:
            self.order_tail='fully success'
        else:
            self.order_tail='success'

    def order_remain_refresh(self):
        '''将订单余量，设定为委托量'''
        self.order_amount=self.order_remain

    def match_orders(self,now,now_data,last_closes,now_cash,now_cap,positions,total_sell_today):
        '''每个时刻执行一次，匹配订单，把能成交的成交了
        成交的动作为，添加持仓、减少现金、删除订单、添加交易记录'''
        self.order_life=self.order_life-1
        if self.order_life<0:
            #修改订单生命状态
            self.order_alive=0
            #修改订单结局
            self.order_tail='die'
            #归零剩余继续委托中的订单量
            self.order_remain=0
        else:
            if self.order_action=='buy':
                #判断是否涨停
                if now_data[self.order_code].low<last_closes[self.order_code]*1.098:
                    #限价买单，当前价格偏低时，可以成交
                    if self.order_method=='limit':
                        if self.order_price>=now_data[self.order_code].low:
                            self.order_success(
                                now,
                                way='low',
                                now_data=now_data,
                                now_cash=now_cash,
                                now_cap=now_cap,
                                positions=positions,
                                total_sell_today=total_sell_today
                            )
                    elif self.order_method=='market':
                        self.order_success(
                            now,
                            way='open',
                            now_data=now_data,
                            now_cash=now_cash,
                            now_cap=now_cap,
                            positions=positions,
                            total_sell_today=total_sell_today
                        )
            else:
                #判断是否有持仓
                if positions[self.order_code].hold_amount>0:
                    self.order_amount=min(self.order_amount,positions[self.order_code].hold_amount)
                    #判断是否跌停
                    if now_data[self.order_code].high>=last_closes[self.order_code]*0.902:
                        if self.order_method=='limit':
                            if self.order_price<=now_data[self.order_code].high:
                                self.order_success(
                                    now,
                                    way='high',
                                    now_data=now_data,
                                    now_cash=now_cash,
                                    now_cap=now_cap,
                                    positions=positions,
                                    total_sell_today=total_sell_today
                                )
                        elif self.order_method=='market':
                            self.order_success(
                                now,
                                way='open_sell',
                                now_data=now_data,
                                now_cash=now_cash,
                                now_cap=now_cap,
                                positions=positions,
                                total_sell_today=total_sell_today
                            )

class star_data(star_data_now_basic_info):
    '''数据部分中的核心数据类，包括交易日历、股票池、进入迭代器的时刻表、原始历史数据和迭代中历史数据'''
    def __init__(self):
        '''
        包括当前时间、当前分钟基础数据、当前持仓、当前订单、当前净值、当前持仓市值、当前现金、前收盘价
        '''
        #继承交易日历、原始数据、迭代开始时刻表
        star_data_now_basic_info.__init__(self)



class star_user_function(star_data_now_basic_info):
    '''用户的基本功能，例如买入、卖出、止盈止损、撤销订单等'''
    def __init__(self):
        star_data_now_basic_info.__init__(self)

    def buy_fixed_amount(
            self,
            code,
            amount,
            now,
            now_data,
            now_cash,
            now_cap,
            positions,
            total_sell_today,
            price=None,
            method='limit',
            life=None
    ):
        '''买入固定量'''
        amount=np.floor(amount/100)*100
        birth=star_data_single_order(
            order_code=code,
            order_amount=amount,
            now=now,
            now_data=now_data,
            now_cash=now_cash,
            now_cap=now_cap,
            positions=positions,
            total_sell_today=total_sell_today,
            order_submit_time=now,
            order_price=price,
            order_method=method,
            order_life=life,
            order_action='buy'
        )
        if self.breathing_orders_buy is None:
            self.breathing_orders_buy=[]
        self.breathing_orders_buy.append(birth)

    def sell_fixed_amount(
            self,
            code,
            amount,
            now,
            now_data,
            now_cash,
            now_cap,
            positions,
            total_sell_today,
            price=None,
            method='limit',
            life=None
    ):
        '''卖出固定量'''
        if code in self.positions:
            amount=np.floor(amount/100)*100
            birth=star_data_single_order(
                order_code=code,
                order_amount=amount,
                now=now,
                now_data=now_data,
                now_cash=now_cash,
                now_cap=now_cap,
                positions=positions,
                total_sell_today=total_sell_today,
                order_submit_time=now,
                order_price=price,
                order_method=method,
                order_life=life,
                order_action='sell'
            )
            if self.breathing_orders_sell is None:
                self.breathing_orders_sell=[]
            self.breathing_orders_sell.append(birth)

    def buy_cash_percent_user(
            self,
            code,
            percent,
            now,
            now_data,
            now_cash,
            now_cap,
            positions,
            total_sell_today,
            price=None,
            method='limit',
            life=None
    ):
        '''买入现金的百分比'''
        target_value=self.now_cash*percent
        target_amount=np.floor(target_value/self.now_data[code].close/100)*100
        birth=star_data_single_order(
            order_code=code,
            order_amount=target_amount,
            now=now,
            now_data=now_data,
            now_cash=now_cash,
            now_cap=now_cap,
            positions=positions,
            total_sell_today=total_sell_today,
            order_submit_time=now,
            order_price=price,
            order_method=method,
            order_life=life,
            order_action='buy'
        )
        if self.breathing_orders_buy is None:
            self.breathing_orders_buy=[]
        self.breathing_orders_buy.append(birth)

    def adjust_to_net_percent(
            self,
            code,
            percent,
            now,
            now_data,
            now_cash,
            now_cap,
            positions,
            total_sell_today,
            price=None,
            method='limit',
            life=None
    ):
        '''买入或卖出至当前净值的百分比'''
        target_net=self.now_net*percent
        target_net=target_net-self.positions[code].hold_value
        if target_net>0:
            target_amount=np.floor(target_net/self.now_data[code].close/100)*100
            birth=star_data_single_order(
                order_code=code,
                order_amount=target_amount,
                now=now,
                now_data=now_data,
                now_cash=now_cash,
                now_cap=now_cap,
                positions=positions,
                total_sell_today=total_sell_today,
                order_submit_time=now,
                order_price=price,
                order_method=method,
                order_life=life,
                order_action='buy'
            )
            if self.breathing_orders_buy is None:
                self.breathing_orders_buy=[]
            self.breathing_orders_buy.append(birth)
        elif target_net<0:
            target_amount=np.floor(-target_net/self.now_data[code].close/100)*100
            birth=star_data_single_order(
                order_code=code,
                order_amount=target_amount,
                now=now,
                now_data=now_data,
                now_cash=now_cash,
                now_cap=now_cap,
                positions=positions,
                total_sell_today=total_sell_today,
                order_submit_time=now,
                order_price=price,
                order_method=method,
                order_life=life,
                order_action='sell'
            )
            if self.breathing_orders_sell is None:
                self.breathing_orders_sell=[]
            self.breathing_orders_sell.append(birth)

    def cancel_order_buy(self,code):
        '''撤销某只股票的买入订单'''
        self.breathing_orders_buy=[i for i in self.breathing_orders_buy if i.order_code!=code]

    def cancel_order_sell(self,code):
        '''撤销某只股票的卖出订单'''
        self.breathing_orders_sell=[i for i in self.breathing_orders_sell if i.order_code!=code]

    def cancel_plenty_of_orders_buy(self,codes):
        '''撤销许多股票的买入订单'''
        self.breathing_orders_buy=[i for i in self.breathing_orders_buy if i.order_code not in codes]

    def cancel_plenty_of_orders_sell(self,codes):
        '''撤销许多股票的卖出订单'''
        self.breathing_orders_sell=[i for i in self.breathing_orders_sell if i.order_code not in codes]

    def cancel_all_orders_buy(self):
        '''撤销全部买入订单'''
        self.breathing_orders_buy=[]

    def cancel_all_orders_sell(self):
        '''撤销全部卖出订单'''
        self.breathing_orders_sell=[]


class star_comments_and_plot:
    def __init__(self,moon):
        '''moon是一个star类对象或star类子类对象，当运行结束后输入到这里'''
        self.moon=moon

    def get_net_series(self):
        '''获得净值序列'''
        time_series=list(self.moon.net_records.keys())
        net_series=list(self.moon.net_records.values())
        self.net_series=pd.DataFrame({'date':time_series,'net':net_series})
        self.net_series=self.net_series.set_index(['date'])
        self.net_series=self.net_series.net

    def float_to_percent(self,x,round_num=2):
        '''
        将小数转化为百分数形式
        由于转化后格式为字符串，所以最好仅在最终输出结果的时候使用这一函数
        运算中途使用此函数，可能会造成程序报错
        round_num是保留的小数位数，默认为2，即68.71%、35.42%这类的形式
        输入（主参数）：float
        输出：str
        '''
        x=round(x*100,round_num)
        x=''.join([str(x),'%'])
        return x

    def percent_to_float(self,x,round_num=4):
        '''
        将百分数（字符串形式的）转化为小数
        round_num为保留的小数位数
        输入（主参数）：str
        输出：float
        '''
        x=float(x[:-1])/100
        return x

    def comment_returns(self,series,percent=False):
        '''
        对基金的净值序列给出收益率方面的评价
        series为基金的净值序列，格式为pd.Series，注意，净值序列series的索引index应为时间
        在dataframe中具体表现为df.net、df['nav']等
        percent参数决定是否输出为百分数形式（百分数形式为str类型），默认输出小数
        输入（主参数）：pd.Series
        输出：float
        '''
        ret=(series.iloc[-1]-series.iloc[0])/series.iloc[0]
        if percent:
            ret=self.float_to_percent(ret)
        else:
            pass
        return ret

    def comment_yearly_returns(self,series,percent=False):
        '''
        对基金的净值序列给出年化收益率方面的评价
        series为基金的净值序列，格式为pd.Series，注意，净值序列series的索引index应为时间
        在dataframe中具体表现为df.net、df['nav']等
        percent参数决定是否输出为百分数形式（百分数形式为str类型），默认输出小数
        输入（主参数）：pd.Series
        输出：float
        '''
        duration=(series.index[-1].daily-series.index[0].daily).days
        year=duration/365
        ret=(series.iloc[-1]/series.iloc[0])**(1/year)-1
        if percent:
            ret=self.float_to_percent(ret)
        else:
            pass
        return ret

    def comment_drawbacks(self,series,percent=False):
        '''
        对基金的净值序列给出最大回撤率方面的评价
        series为基金的净值序列，格式为pd.Series，注意，净值序列series的索引index应为时间
        在dataframe中具体表现为df.net、df['nav']等
        percent参数决定是否输出为百分数形式（百分数形式为str类型），默认输出小数
        输入（主参数）：pd.Series
        输出：float
        '''
        draws=[]
        for i in range(1,len(series)-1):
            s=series[:i]
            smax=s.max()
            draws.append((smax-series[i])/smax)
        max_draw=max(draws)
        if percent:
            max_draw=self.float_to_percent(max_draw)
        else:
            pass
        return max_draw

    def comment_volatilitys(self,series,percent=False):
        '''
        对基金的净值序列给出波动率方面的评价
        series为基金的净值序列，格式为pd.Series，注意，净值序列series的索引index应为时间
        在dataframe中具体表现为df.net、df['nav']等
        percent参数决定是否输出为百分数形式（百分数形式为str类型），默认输出小数
        输入（主参数）：pd.Series
        输出：float
        '''
        series=series.pct_change()
        vol=np.std(series)*((250*242)**0.5)
        if percent:
            vol=self.float_to_percent(vol)
        else:
            pass
        return vol

    def comment_sharpes(self,series,risk_free_rate=0,percent=False):
        '''
        对基金的净值序列给出夏普比率方面的评价
        series为基金的净值序列，格式为pd.Series，注意，净值序列series的索引index应为时间
        在dataframe中具体表现为df.net、df['nav']等
        percent参数决定是否输出为百分数形式（百分数形式为str类型），默认输出小数
        输入（主参数）：pd.Series
        输出：float
        '''
        duration=(series.index[-1].daily-series.index[0].daily).days
        year=duration/365
        ret=(series.iloc[-1]/series.iloc[0])**(1/year)-1
        series=series.pct_change()
        vol=np.std(series)*((250*242)**0.5)
        sharpe=(ret-risk_free_rate)/vol
        if percent:
            sharpe=self.float_to_percent(sharpe)
        else:
            sharpe=round(sharpe,2)
        return sharpe

    def comment_minute_win_rate(self,series,percent=False):
        '''
        对基金的净值序列给出日度胜率方面的评价
        series为基金的净值序列，格式为pd.Series，注意，净值序列series的索引index应为时间
        在dataframe中具体表现为df.net、df['nav']等
        percent参数决定是否输出为百分数形式（百分数形式为str类型），默认输出小数
        输入（主参数）：pd.Series
        输出：float
        '''
        series=series.pct_change()
        wins=series[series>0]
        loses=series[series<0]
        daily_win_rate=len(wins)/(len(wins)+len(loses))
        if percent:
            daily_win_rate=self.float_to_percent(daily_win_rate)
        else:
            pass
        return daily_win_rate

    def comment_daily_win_rate(self,series,percent=False):
        '''
        对基金的净值序列给出日度胜率方面的评价
        series为基金的净值序列，格式为pd.Series，注意，净值序列series的索引index应为时间
        在dataframe中具体表现为df.net、df['nav']等
        percent参数决定是否输出为百分数形式（百分数形式为str类型），默认输出小数
        输入（主参数）：pd.Series
        输出：float
        '''
        series.index=[i.daily for i in list(series.index)]
        series=series.resample('D').last()
        series=series.pct_change()
        wins=series[series>0]
        loses=series[series<0]
        daily_win_rate=len(wins)/(len(wins)+len(loses))
        if percent:
            daily_win_rate=self.float_to_percent(daily_win_rate)
        else:
            pass
        return daily_win_rate

    def comment_on_fund_series(self,series):
        '''
        对基金的净值序列给出评价
        series为基金的净值序列，格式为pd.Series
        在dataframe中具体表现为df.net、df['nav']等
        对净值的评价如下
        收益率方面、年化收益率、回撤方面、波动率（年化）方面、夏普比率、分钟胜率、日度胜率
        输入（主参数）：pd.Series
        输出：pd.DataFrame
        '''
        def single_param(x):
            ret=self.comment_returns(series,percent=True)
            yearly_ret=self.comment_yearly_returns(series,percent=True)
            draw=self.comment_drawbacks(series,percent=True)
            vol=self.comment_volatilitys(series,percent=True)
            sharpe=self.comment_sharpes(series,percent=False)
            minute_win_rate=self.comment_minute_win_rate(series,percent=True)
            daily_win_rate=self.comment_daily_win_rate(series,percent=True)
            res=pd.DataFrame({
                '总收益率':[ret],
                '年化收益':[yearly_ret],
                '最大回撤':[draw],
                '年化波动':[vol],
                '年化夏普':[sharpe],
                '分钟胜率':[minute_win_rate],
                '日度胜率':[daily_win_rate]
            })
            return res
        comments=single_param(series)
        return comments

    def get_plotly_and_comment_table(self,df,tris):
        '''
        使用plotly绘制净值曲线图
        返回两个可向画布中添加的对象，一个是净值曲线子图，一个是基金评价表
        df为pd.DataFrame格式，索引为时间，包含两列，分别为基金净值列，和基准净值列
        输入（主参数）：pd.DataFrame
        输出：go.Scatter、go.Table
        '''
        df1=df.copy()
        df1.index=[i.to_datetime() for i in list(df1.index)]
        df1=df1.to_frame(name='net')
        scatter_nav=go.Scatter(
            x=df1.index,
            y=df1.net,
            mode='lines',
            name='策略净值走势',
            line={'color':'red'}
        )
        # scatter_bench=go.Scatter(
        #     x=df.index,
        #     y=df[bench],
        #     mode='lines',
        #     name=bench_name+'净值走势',
        #     line={'color':'grey'}
        # )
        comments=self.comment_on_fund_series(df)
        comments=comments.reset_index()
        comments.columns=['']+list(comments.columns)[1:]
        table=go.Table(
            header=dict(values=list(comments.columns)),
            cells=dict(values=list(comments.to_numpy().T)),
            name='策略评价'
        )
        table.cells.fill.color='rgb(255,232,232)'
        table.header.fill.color='rgb(255,232,232)'
        # column_rate=int(round((len(fund_name)/10),0))+0.5
        # table.columnwidth=[column_rate]+[1]*(comments.shape[1]-1)
        # table.cells.font.size=12
        # table.header.font.size=12
        # scatter=[scatter_nav,scatter_bench]
        tris=tris.assign(position_rate=tris.cap/tris.net)
        scatter_pos=go.Scatter(
            x=tris.date,
            y=tris.position_rate,
            mode='lines',
            name='仓位比例',
            line={'color':'red'}
        )
        scatter_net=go.Scatter(
            x=tris.date,
            y=tris.net,
            mode='lines',
            name='策略净值走势',
            line={'color':'red'}
        )
        scatter_cap=go.Scatter(
            x=tris.date,
            y=tris.cap,
            mode='lines',
            name='持仓市值走势',
            line={'color':'blue'}
        )
        scatter_cash=go.Scatter(
            x=tris.date,
            y=tris.cash,
            mode='lines',
            name='现金走势',
            line={'color':'grey'}
        )
        scatter=[scatter_net,scatter_cap,scatter_cash]
        return scatter_nav,table,scatter_pos,scatter

    def get_subplots(self,subplot_titles=False):
        '''
        绘制一张容纳子图和评价表的画布
        row为行数
        col为列数，此处列数默认为2列，即左边一列用于放置净值曲线，右边一列用于放置评价指标表
        subplot_titles为每个子图和评价表的标题的列表，默认为空
        如果要设置坐标轴，请传入净值曲线
        输入（主参数）：float
        输出：plotly.Figure
        '''
        if subplot_titles!=False:
            fig=make_subplots(
                rows=4,
                cols=1,
                specs=[[{'type':'domain'}],[{'type':'xy'}],[{'type':'xy'}],[{'type':'xy'}]],
                subplot_titles=subplot_titles
            )
        else:
            fig=make_subplots(
                rows=4,
                cols=1,
                specs=[[{'type':'domain'}],[{'type':'xy'}],[{'type':'xy'}],[{'type':'xy'}]]
            )
        fig.layout.plot_bgcolor='rgb(255,232,232)'
        # if df is not None:
        #     time_series=pd.date_range(df.index.min(),df.index.max()+relativedelta(years=2),freq='Y')
        #     time_series=pd.Series(list(map(lambda x:x-relativedelta(years=1)+relativedelta(days=1),time_series)))
        #     if time_series.shape[0]<5:
        #         time_series=pd.date_range(df.index.min(),df.index.max()+relativedelta(months=6),freq='Q')
        #         time_series=pd.Series(list(map(lambda x:x-relativedelta(months=3)+relativedelta(days=1),time_series)))
        #         if time_series.shape[0]<4:
        #             time_series=pd.date_range(df.index.min(),df.index.max()+relativedelta(months=2),freq='M')
        #             time_series=pd.Series(
        #                 list(map(lambda x:x-relativedelta(months=1)+relativedelta(days=1),time_series)))
        #             if time_series.shape[0]<4:
        #                 time_series=pd.date_range(df.index.min(),df.index.max()+relativedelta(days=14),freq='W')
        #                 time_series=pd.Series(
        #                     list(map(lambda x:x-relativedelta(days=7)+relativedelta(days=1),time_series)))
        # time_text=list(time_series.dt.strftime(date_format='%Y-%m-%d'))
        # fig.layout.xaxis.tickvals=time_series
        # fig.layout.xaxis.ticktext=time_text
        # fig.layout.xaxis.tickangle=60
        fig.layout.yaxis.domain=[0.23,0.95]
        fig.layout.yaxis2.domain=[0.11,0.22]
        fig.layout.yaxis3.domain=[0,0.1]
        return fig

    def add_scatter_and_table_to_figure(
            self,fig,scatters,tables,pos,tris,html_name='策略评价.html',width='100%',auto_open=True):
        '''
        将子图和表格添加到画布上
        fig为plotly.Figure，即子图和表格画布
        scatters为要添加的子图列表，可以为list或单个go.Scatter
        tables为要添加的表格列表，可以为list或单个go.Table
        html为最终画到html文件的路径
        auto_open，是否自动打开生成的html文件
        输入（主参数）：plotly.Figure、go.Scatter、go.Table
        输出：html文件
        '''
        if isinstance(scatters,list):
            height=1
            for scatter in scatters:
                fig.append_trace(scatter,row=2,col=1)
        else:
            height=1
            fig.append_trace(scatters,row=2,col=1)
        if isinstance(tables,list):
            for table in tables:
                fig.append_trace(table,row=1,col=1)
        else:
            fig.append_trace(tables,row=1,col=1)
        if isinstance(pos,list):
            for p in pos:
                fig.append_trace(p,row=3,col=1)
        else:
            fig.append_trace(pos,row=3,col=1)
        if isinstance(tris,list):
            for t in tris:
                fig.append_trace(t,row=4,col=1)
        else:
            fig.append_trace(tris,row=4,col=1)
        height='100%'
        if '.html' not in html_name:
            html_name=html_name+'.html'
        pio.write_html(fig,html_name,auto_open=auto_open,default_height=height,default_width=width)

    def plot(self,html_name='策略评价.html',width='100%',auto_open=True):
        '''封装好的，直接获得单一基金评价文件的函数'''
        self.get_net_series()
        scatter,table,pos,tris=self.get_plotly_and_comment_table(self.net_series,self.moon.net_cap_cash_table)
        fig=self.get_subplots()
        self.add_scatter_and_table_to_figure(
            fig,scatter,table,pos,tris,html_name=html_name,width=width,auto_open=auto_open)



class star(star_user_function):
    '''策略主体
    示例策略
    class lonely_star(star):
        def __init__(self):
            star.__init__(self)

        def planet(self):
            if self.now.daily>pd.Timestamp('2018-01-11'):
                for code in self.iter_data.keys():
                    if self.now_data[code].open>self.history_data[code].tail(242*5).close.quantile(0.99) and\
                        self.now_data[code].amount>self.history_data[code].tail(242*5).amount.quantile(0.01):
                        self.buy(code,100)
                    elif self.now_data[code].open<self.history_data[code].tail(242*5).close.quantile(0.99) and\
                        self.now_data[code].amount<self.history_data[code].tail(242*5).amount.quantile(0.01):
                        self.sell(code,100)

    ax=lonely_star()
    ax.set_backtest_period('2018-01-01','2021-04-30')
    ax.set_stock_list(['601615.SH','601877.SH'])
    ax.run()
    '''
    def __init__(self):
        #继承用户功能部分
        star_user_function.__init__(self)
        self.__new_year_poem='回家过年吧'
        print('recent dreams:\n',self.__new_year_poem)
        self.positions_records=[]

    def update_now_cap(self,now):
        '''更新当前持仓市值数据和总净值数据'''
        now_cap=0
        for k,v in self.positions.items():
            self.positions[k].update_now_positions(self.now,self.now_data)
            now_cap=now_cap+v.hold_value
        self.now_cap=now_cap
        self.now_net=self.now_cap+self.now_cash
        self.net_records[now]=self.now_net
        self.cap_records[now]=self.now_cap
        self.cash_records[now]=self.now_cash

    def net_cap_cash(self):
        '''输出净值、市值和现金表'''
        time_series=[i.to_datetime() for i in list(self.net_records.keys())]
        net_series=list(self.net_records.values())
        cap_series=list(self.cap_records.values())
        cash_series=list(self.cash_records.values())
        tris=pd.DataFrame({'date':time_series,'net':net_series,'cap':cap_series,'cash':cash_series})
        self.net_cap_cash_table=tris

    def show_positions_records(self):
        '''展示持仓记录表'''
        positions_records=pd.concat(self.positions_records)
        positions_records.index=['start']+[i.to_datetime() for i in list(positions_records.index)[1:]]
        self.positions_records_table=positions_records

    def buy(self,code,amount,price=None,method='limit',life=None):
        '''买入固定量'''
        self.buy_fixed_amount(
            code=code,
            amount=amount,
            now=self.now,
            now_data=self.now_data,
            now_cash=self.now_cash,
            now_cap=self.now_cap,
            positions=self.positions,
            total_sell_today=self.total_sell_today,
            price=price,
            method=method,
            life=life
        )

    def sell(self,code,amount,price=None,method='limit',life=None):
        '''卖出固定量'''
        self.sell_fixed_amount(
            code=code,
            amount=amount,
            now=self.now,
            now_data=self.now_data,
            now_cash=self.now_cash,
            now_cap=self.now_cap,
            positions=self.positions,
            total_sell_today=self.total_sell_today,
            price=price,
            method=method,
            life=life
        )

    def buy_cash_percent(self,code,percent,price=None,method='limit',life=None):
        '''买入现金的百分比'''
        self.buy_cash_percent_user(
            code=code,
            percent=percent,
            now=self.now,
            now_data=self.now_data,
            now_cash=self.now_cash,
            now_cap=self.now_cap,
            positions=self.positions,
            total_sell_today=self.total_sell_today,
            price=price,
            method=method,
            life=life
        )

    def to_net_percent(self,code,percent,price=None,method='limit',life=None):
        '''将持仓调整至精致的百分比'''
        self.adjust_to_net_percent(
            code=code,
            percent=percent,
            now=self.now,
            now_data=self.now_data,
            now_cash=self.now_cash,
            now_cap=self.now_cap,
            positions=self.positions,
            total_sell_today=self.total_sell_today,
            price=price,
            method=method,
            life=life
        )

    def planet(self):
        pass

    def run(self,html_name='策略评价.html'):
        '''每个时刻迭代运行的部分'''
        self.read_in_all_pri_data()
        self.positions_records_demo={k:[0] for k in self.stock_list}
        self.positions_records_demo=pd.DataFrame(self.positions_records_demo,index=['start'])
        self.positions_records.append(self.positions_records_demo)
        #今日累计卖出量
        self.total_sell_today={k:0 for k in self.stock_list}
        for now in tqdm.tqdm(self.tradedays):
            #更新当前时刻
            self.now=now
            #清零当日累积卖出量
            if now.minute_order==242:
                self.total_sell_today={k:0 for k in self.stock_list}
            #更新需要迭代的数据
            self.update_iter_data(now)
            #更新当前时刻基础数据
            self.update_minute_basic(now)
            #更新当前时刻持仓数据
            for k in self.positions.keys():
                self.positions[k].update_now_positions(now,self.now_data)
            positions_now={k:v.hold_amount for k,v in self.positions.items()}
            positions_now=pd.DataFrame(positions_now,index=[now])
            self.positions_records.append(positions_now)
            #更新当前时刻市值净值和现金
            self.update_now_cap(now)
            #执行卖出订单
            if self.breathing_orders_sell:
                for i in range(len(self.breathing_orders_sell)):
                    self.breathing_orders_sell[i].match_orders(
                        now,
                        self.now_data,
                        self.last_closes,
                        self.now_cash,
                        self.now_cap,
                        self.positions,
                        self.total_sell_today
                    )
                    self.now_cash=self.breathing_orders_sell[i].now_cash
                    self.now_cap=self.breathing_orders_sell[i].now_cap
                    self.now_net=self.now_cap+self.now_cash
                    self.positions=self.breathing_orders_sell[i].positions
                    self.total_sell_today=self.breathing_orders_sell[i].total_sell_today
                self.order_records.append(self.breathing_orders_sell)
                self.breathing_orders_sell=[i for i in self.breathing_orders_sell if i.order_alive==1]
                remain_orders=[i.order_remain_refresh() for i in self.breathing_orders_sell if i.order_remain>0]
                self.breathing_orders_sell.extend(remain_orders)
                self.update_now_cap(now)
            #执行买入订单
            if self.breathing_orders_buy:
                for i in range(len(self.breathing_orders_buy)):
                    self.breathing_orders_buy[i].match_orders(
                        now,
                        self.now_data,
                        self.last_closes,
                        self.now_cash,
                        self.now_cap,
                        self.positions,
                        self.total_sell_today
                    )
                    self.now_cash=self.breathing_orders_buy[i].now_cash
                    self.now_cap=self.breathing_orders_buy[i].now_cap
                    self.now_net=self.now_cap+self.now_cash
                    self.positions=self.breathing_orders_buy[i].positions
                self.order_records.append(self.breathing_orders_buy)
                self.breathing_orders_buy=[i for i in self.breathing_orders_buy if i.order_alive==1]
                remain_orders=[i.order_remain_refresh() for i in self.breathing_orders_buy if i.order_remain>0]
                self.breathing_orders_buy.extend(remain_orders)
                self.update_now_cap(now)
            #执行用户自己编写的策略
            self.planet()
        self.net_cap_cash()
        self.show_positions_records()
        #输出评价绘图
        boat_moon=star_comments_and_plot(self)
        boat_moon.plot(html_name=html_name)




