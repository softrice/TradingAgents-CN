# 克隆自聚宽文章：https://www.joinquant.com/post/74527
# 标题：红利价值策略：年化稳赚37%的低频高股息策略
# 作者：W一路向北
#
# 增强版 changelog:
# v2026.08.05  等权再平衡；买不起顺延；涨停打开延迟补位；可选宽止损(默认关)
# v2026.08.04  修正滑点/交易成本；20日动量综合打分

import pandas as pd
from jqdata import *

STRATEGY_VERSION = '2026.08.05'


def initialize(context):
    set_option("avoid_future_data", True)
    set_option('use_real_price', True)
    set_benchmark('000015.XSHG')
    set_slippage(PriceRelatedSlippage(0.001))
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0.0005,
            open_commission=0.0003,
            close_commission=0.0003,
            close_today_commission=0,
            min_commission=5,
        ),
        type='stock',
    )

    log.set_level('order', 'error')
    log.set_level('history', 'error')
    log.set_level('system', 'error')

    g.strategy_version = STRATEGY_VERSION
    g.sell_list = []
    g.target_list = []
    g.stock_num = 5
    g.high_limit_list = []
    g.strategy = 'hongli_momentum'
    g.filename = '红利动量增强.csv'

    g.momentum_days = 20
    g.dividend_weight = 0.85
    g.momentum_weight = 0.15
    g.min_select = 1

    # 可选宽止损：默认关闭（红利低频策略通常不需要 tight stop）
    g.enable_stop_loss = False
    g.stop_loss_drawdown = 0.25  # 单票从持仓最高价回撤 25% 触发

    # 月中补位状态
    g.score_df = None
    g.candidate_rank = []
    g.pending_replenish = False
    g.recently_sold = set()
    g.rebalance_date = None
    g.position_highs = {}

    run_daily(prepare_stock_list, '09:00')
    run_monthly(get_stock_list, 1, '09:01')
    run_monthly(my_trade, 1, '09:30')
    run_daily(check_limit_up, '10:00')
    run_daily(check_stop_loss, '14:00')
    run_daily(replenish_positions, '09:35')
    run_daily(replenish_positions, '14:50')


def get_min_lot(stock_code):
    if stock_code[:3] == '688':
        return 200
    return 100


def get_limit_ratio(stock_code, current_dt):
    date_str = str(current_dt)[:10]
    if stock_code[:3] == '688':
        return 0.20
    if stock_code[0] == '3' and date_str >= '2020-08-24':
        return 0.20
    return 0.10


def get_buy_limit_price(stock_code, close_price, current_dt):
    ratio = get_limit_ratio(stock_code, current_dt)
    return round(close_price * (1 + ratio), 2)


def calc_affordable_amount(stock_code, budget, limit_price):
    min_lot = get_min_lot(stock_code)
    if limit_price <= 0 or budget < limit_price * min_lot:
        return 0
    return min_lot * int(budget / limit_price / min_lot)


def is_unbuyable(current_data, stock_code):
    cd = current_data[stock_code]
    if cd.paused:
        return True
    if cd.last_price <= 0:
        return True
    if cd.high_limit > 0 and cd.last_price >= cd.high_limit * 0.998:
        return True
    return False


def get_current_slots_needed(context):
    return max(0, g.stock_num - len(context.portfolio.positions))


def update_position_high(stock_code, price):
    if price <= 0:
        return
    g.position_highs[stock_code] = max(g.position_highs.get(stock_code, price), price)


def pick_replenish_targets(context, hold_list, slots_needed):
    if slots_needed <= 0 or not g.candidate_rank:
        return []

    current_data = get_current_data()
    per_stock_value = context.portfolio.total_value / g.stock_num
    targets = []

    for code in g.candidate_rank:
        if len(targets) >= slots_needed:
            break
        if code in hold_list or code in g.recently_sold:
            continue
        if is_unbuyable(current_data, code):
            continue
        price = current_data[code].last_price
        if calc_affordable_amount(code, per_stock_value, price) <= 0:
            continue
        targets.append(code)

    return targets


def select_target_stocks(context, score_df, hold_list, stock_num):
    per_stock_budget = context.portfolio.total_value / stock_num
    target_list = []
    skipped = []

    for code in score_df.index:
        if len(target_list) >= stock_num:
            break

        if code in hold_list:
            target_list.append(code)
            continue

        close_df = get_price(
            code,
            end_date=context.previous_date,
            frequency='1d',
            count=1,
            fields=['close'],
            fq='pre',
            panel=False,
            skip_paused=False,
            fill_paused=True,
        )
        if close_df.empty:
            skipped.append((code, '无有效收盘价'))
            continue

        limit_price = get_buy_limit_price(code, close_df['close'].iloc[-1], context.current_dt)
        if calc_affordable_amount(code, per_stock_budget, limit_price) > 0:
            target_list.append(code)
        else:
            skipped.append((code, '单价 %.2f，预算 %.2f 不足买 %d 股' % (
                limit_price, per_stock_budget, get_min_lot(code))))

    if skipped:
        print('========== 买不起/跳过标的 ==========')
        for code, reason in skipped:
            print('%s: %s' % (code, reason))
        print('====================================')

    return target_list


def prepare_stock_list(context):
    g.high_limit_list = []
    g.hold_list = list(context.portfolio.positions)
    for stock_code, pos in context.portfolio.positions.items():
        update_position_high(stock_code, pos.price)

    if not g.hold_list:
        return

    df = get_price(
        g.hold_list,
        end_date=context.previous_date,
        frequency='daily',
        fields=['close', 'high_limit'],
        count=1,
        panel=False,
        fill_paused=False,
        skip_paused=False,
    ).dropna()
    df = df[df['close'] == df['high_limit']]
    g.high_limit_list = list(df.code)


def get_momentum_factor(context, stock_list, days=20):
    """
    20日简单价格动量（百分比）:
    momentum = (昨收 / 20个交易日前收盘 - 1) * 100
    """
    if not stock_list:
        return pd.DataFrame(columns=['momentum'])

    yesterday = context.previous_date
    df = get_price(
        stock_list,
        end_date=yesterday,
        frequency='daily',
        fields=['close'],
        count=days + 5,
        panel=False,
        skip_paused=False,
        fill_paused=True,
    )
    if df.empty:
        return pd.DataFrame(columns=['momentum'])

    momentum_list = []
    for code in stock_list:
        code_df = df[df['code'] == code].dropna()
        if len(code_df) >= days:
            start_price = code_df['close'].iloc[-days]
            end_price = code_df['close'].iloc[-1]
            momentum = (end_price / start_price - 1) * 100
        else:
            momentum = -999
        momentum_list.append({'code': code, 'momentum': momentum})

    return pd.DataFrame(momentum_list).set_index('code')


def build_score_df(context, stock_list):
    momentum_df = get_momentum_factor(context, stock_list, days=g.momentum_days)

    time1 = context.previous_date
    time0 = time1 - datetime.timedelta(days=365)
    interval = 1000
    list_len = len(stock_list)

    q = query(
        finance.STK_XR_XD.code,
        finance.STK_XR_XD.a_registration_date,
        finance.STK_XR_XD.bonus_amount_rmb,
    ).filter(
        finance.STK_XR_XD.a_registration_date >= time0,
        finance.STK_XR_XD.a_registration_date <= time1,
        finance.STK_XR_XD.code.in_(stock_list[:min(list_len, interval)]),
    )
    div_df = finance.run_query(q)

    if list_len > interval:
        for i in range(list_len // interval):
            q = query(
                finance.STK_XR_XD.code,
                finance.STK_XR_XD.a_registration_date,
                finance.STK_XR_XD.bonus_amount_rmb,
            ).filter(
                finance.STK_XR_XD.a_registration_date >= time0,
                finance.STK_XR_XD.a_registration_date <= time1,
                finance.STK_XR_XD.code.in_(stock_list[interval * (i + 1):min(list_len, interval * (i + 2))]),
            )
            div_df = pd.concat([div_df, finance.run_query(q)], ignore_index=True)

    dividend = div_df.fillna(0).set_index('code').groupby('code').sum()
    cap = get_fundamentals(
        query(valuation.code, valuation.market_cap).filter(valuation.code.in_(list(dividend.index))),
        date=time1,
    ).set_index('code')

    div_ratio_df = pd.concat([dividend, cap], axis=1, sort=False)
    div_ratio_df['dividend_ratio'] = (div_ratio_df['bonus_amount_rmb'] / 10000) / div_ratio_df['market_cap']
    score_df = div_ratio_df[['dividend_ratio']].join(momentum_df, how='inner')
    if score_df.empty:
        return score_df

    score_df['dividend_rank'] = score_df['dividend_ratio'].rank(pct=True, ascending=True)
    score_df['momentum_rank'] = score_df['momentum'].rank(pct=True, ascending=True)
    score_df['total_score'] = (
        g.dividend_weight * score_df['dividend_rank'] +
        g.momentum_weight * score_df['momentum_rank']
    )
    return score_df.sort_values('total_score', ascending=False)


def get_stock_list(context):
    g.sell_list = []
    g.target_list = []
    today = context.current_dt

    initial_list = get_all_securities('stock', today).index.tolist()
    initial_list = filter_new_stock(context, initial_list)
    initial_list = filter_kcb_stock(initial_list)
    initial_list = filter_st_stock(initial_list)
    initial_list = filter_paused_stock(initial_list)

    df = get_fundamentals(query(valuation.code).filter(
        valuation.code.in_(initial_list),
        valuation.pe_ratio.between(5, 50),
        indicator.inc_return.between(5, 100),
        indicator.inc_total_revenue_year_on_year.between(5, 100),
        indicator.inc_net_profit_year_on_year.between(10, 100),
    ))
    stock_list = list(df.code)

    dividend_list = get_dividend_ratio_filter_list(context, stock_list, False, 0.00, 1.0, 0.00)
    if not dividend_list:
        print('警告：无满足条件的股票，本次不交易')
        g.sell_list = [s for s in g.hold_list if s not in g.high_limit_list]
        return

    score_df = build_score_df(context, dividend_list)
    if score_df.empty:
        print('警告：综合打分数据为空，本次不交易')
        g.sell_list = [s for s in g.hold_list if s not in g.high_limit_list]
        return

    score_df = score_df[score_df['dividend_ratio'] > 0.03]
    if score_df.empty:
        print('警告：股息率过滤后无标的，本次不交易')
        g.sell_list = [s for s in g.hold_list if s not in g.high_limit_list]
        return

    hold_list = getattr(g, 'hold_list', list(context.portfolio.positions))
    target_list = select_target_stocks(context, score_df, hold_list, g.stock_num)
    select_count = max(g.min_select, min(g.stock_num, len(target_list)))
    g.target_list = target_list[:select_count]

    g.score_df = score_df
    g.candidate_rank = list(score_df.index)
    g.rebalance_date = context.current_dt.date()
    g.sell_list = [s for s in hold_list if s not in g.target_list and s not in g.high_limit_list]

    print('========== [%s] 综合打分详情 ==========' % g.strategy_version)
    print(score_df[['dividend_ratio', 'momentum', 'total_score']].head(10))
    print('选中标的:', g.target_list)
    print('卖出:', g.sell_list)


def my_trade(context):
    """月初等权调仓：先卖后买，对 target_list 全体 order_target_value。"""
    current_data = get_current_data()

    for s in g.sell_list:
        if s not in context.portfolio.positions:
            continue
        if current_data[s].last_price < current_data[s].high_limit:
            order_target_value(s, 0)
            g.position_highs.pop(s, None)

    if not g.target_list:
        return

    target_value = context.portfolio.total_value / len(g.target_list)
    print('========== 月初等权调仓 目标市值 %.2f ==========' % target_value)

    for s in g.target_list:
        if s in g.high_limit_list:
            print('保留涨停持仓 %s，暂不调仓' % s)
            continue
        if is_unbuyable(current_data, s) and s not in context.portfolio.positions:
            print('跳过买入 %s：涨停/停牌，稍后补位' % s)
            g.pending_replenish = True
            continue
        order_target_value(s, target_value)
        update_position_high(s, current_data[s].last_price)
        print('调仓 %s -> %.2f' % (s, target_value))

    if get_current_slots_needed(context) > 0:
        g.pending_replenish = True


def check_limit_up(context):
    current_data = get_current_data()
    if not g.high_limit_list:
        return

    for s in g.high_limit_list:
        if current_data[s].last_price < current_data[s].high_limit:
            order_target_value(s, 0)
            g.recently_sold.add(s)
            g.pending_replenish = True
            g.position_highs.pop(s, None)
            print(s, '涨停打开，卖出（待补位）')
        else:
            print(s, '涨停，继续持有')


def check_stop_loss(context):
    """可选：单票从持仓最高价回撤超过阈值则卖出。默认关闭。"""
    if not g.enable_stop_loss:
        return

    for s in list(context.portfolio.positions.keys()):
        if s in g.high_limit_list:
            continue

        pos = context.portfolio.positions[s]
        if pos.price <= 0:
            continue

        high_price = g.position_highs.get(s, pos.price)
        if high_price <= 0:
            continue

        drawdown = pos.price / high_price - 1
        if drawdown <= -g.stop_loss_drawdown:
            order_target_value(s, 0)
            g.recently_sold.add(s)
            g.pending_replenish = True
            g.position_highs.pop(s, None)
            print(s, '触发宽止损 回撤 %.1f%%，卖出' % (drawdown * 100))


def replenish_positions(context):
    slots_needed = get_current_slots_needed(context)
    if slots_needed <= 0:
        g.pending_replenish = False
        g.recently_sold = set()
        return

    if g.rebalance_date == context.current_dt.date() and context.current_dt.hour < 10:
        return

    if not g.candidate_rank:
        print('补位跳过：无缓存候选排名')
        return

    hold_list = list(context.portfolio.positions.keys())
    buy_list = pick_replenish_targets(context, hold_list, slots_needed)
    if not buy_list:
        print('补位待定：暂无可买标的，稍后重试')
        g.pending_replenish = True
        return

    per_stock_value = context.portfolio.total_value / g.stock_num
    current_data = get_current_data()

    print('========== 持仓补位 需补 %d 只 ==========' % slots_needed)
    for code in buy_list:
        if is_unbuyable(current_data, code):
            print('跳过 %s：涨停或停牌' % code)
            continue
        order_target_value(code, per_stock_value)
        update_position_high(code, current_data[code].last_price)
        print('补位 %s -> %.2f' % (code, per_stock_value))

    remaining = get_current_slots_needed(context)
    g.pending_replenish = remaining > 0
    if remaining == 0:
        g.recently_sold = set()
    else:
        print('仍缺 %d 只，后续继续尝试' % remaining)


def filter_paused_stock(stock_list):
    current_data = get_current_data()
    return [stock for stock in stock_list if not current_data[stock].paused]


def filter_st_stock(stock_list):
    current_data = get_current_data()
    return [
        stock for stock in stock_list
        if not current_data[stock].is_st
        and 'ST' not in current_data[stock].name
        and '*' not in current_data[stock].name
        and '退' not in current_data[stock].name
    ]


def filter_kcb_stock(stock_list):
    return [
        stock for stock in stock_list
        if stock[0] != '4' and stock[0] != '8' and stock[:2] != '68'
    ]


def filter_new_stock(context, stock_list):
    yesterday = context.previous_date
    return [
        stock for stock in stock_list
        if not yesterday - get_security_info(stock).start_date < datetime.timedelta(days=250)
    ]


def get_dividend_ratio_filter_list(context, stock_list, sort, p1, p2, threshold):
    time1 = context.previous_date
    time0 = time1 - datetime.timedelta(days=365)
    interval = 1000
    list_len = len(stock_list)

    q = query(
        finance.STK_XR_XD.code,
        finance.STK_XR_XD.a_registration_date,
        finance.STK_XR_XD.bonus_amount_rmb,
    ).filter(
        finance.STK_XR_XD.a_registration_date >= time0,
        finance.STK_XR_XD.a_registration_date <= time1,
        finance.STK_XR_XD.code.in_(stock_list[:min(list_len, interval)]),
    )
    df = finance.run_query(q)

    if list_len > interval:
        for i in range(list_len // interval):
            q = query(
                finance.STK_XR_XD.code,
                finance.STK_XR_XD.a_registration_date,
                finance.STK_XR_XD.bonus_amount_rmb,
            ).filter(
                finance.STK_XR_XD.a_registration_date >= time0,
                finance.STK_XR_XD.a_registration_date <= time1,
                finance.STK_XR_XD.code.in_(stock_list[interval * (i + 1):min(list_len, interval * (i + 2))]),
            )
            df = pd.concat([df, finance.run_query(q)], ignore_index=True)

    dividend = df.fillna(0).set_index('code').groupby('code').sum()
    cap = get_fundamentals(
        query(valuation.code, valuation.market_cap).filter(valuation.code.in_(list(dividend.index))),
        date=time1,
    ).set_index('code')

    df = pd.concat([dividend, cap], axis=1, sort=False)
    df['dividend_ratio'] = (df['bonus_amount_rmb'] / 10000) / df['market_cap']
    df = df.sort_values(by=['dividend_ratio'], ascending=sort)
    df = df[int(p1 * len(df)):int(p2 * len(df))]
    df = df[df['dividend_ratio'] > threshold]
    return list(df.index)
