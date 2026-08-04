# 克隆自聚宽文章：https://www.joinquant.com/post/74527
# 标题：红利价值策略：年化稳赚37%的低频高股息策略
# 作者：W一路向北
# 增强版：增加20日动量因子，综合打分选股
# 测试3：同时修正滑点+交易成本
# 测试4：买不起的高价股自动顺延，保证尽量持满 stock_num 只

import pandas as pd
from jqdata import *


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

    g.sell_list = []
    g.buy_df = pd.DataFrame(columns=['name', 'price', 'amount', 'value'])
    g.stock_num = 5
    g.high_limit_list = []
    g.strategy = 'hongli_momentum'
    g.filename = '红利动量增强.csv'

    g.first = 1
    g.out_cash = 0

    g.momentum_days = 20
    g.dividend_weight = 0.85
    g.momentum_weight = 0.15
    g.min_select = 1

    # 月中涨停打开卖出后的补位状态
    g.score_df = None
    g.candidate_rank = []
    g.pending_replenish = False
    g.recently_sold = set()
    g.rebalance_date = None

    run_daily(prepare_stock_list, '09:00')
    run_monthly(get_stock_list, 1, '09:01')
    run_monthly(my_trade, 1, '09:30')
    run_daily(check_limit_up, '10:00')
    # 涨停打开后不立刻追买：下午/次日再补，且跳过当日涨停
    run_daily(replenish_positions, '09:35')
    run_daily(replenish_positions, '14:50')


def get_min_lot(stock_code):
    """A股最小买入单位：科创板200股，其余100股。"""
    if stock_code[:3] == '688':
        return 200
    return 100


def get_limit_ratio(stock_code, current_dt):
    """估算涨停幅度，用于挂限价单。"""
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
    """根据预算计算可买股数（整手）。"""
    min_lot = get_min_lot(stock_code)
    if limit_price <= 0 or budget < limit_price * min_lot:
        return 0
    return min_lot * int(budget / limit_price / min_lot)


def is_unbuyable(current_data, stock_code):
    """当前无法买入：停牌、无报价、或涨停封板。"""
    cd = current_data[stock_code]
    if cd.paused:
        return True
    if cd.last_price <= 0:
        return True
    if cd.high_limit > 0 and cd.last_price >= cd.high_limit * 0.998:
        return True
    return False


def get_current_slots_needed(context):
    """距离目标持仓数还差几只。"""
    hold_count = len(context.portfolio.positions)
    return max(0, g.stock_num - hold_count)


def pick_replenish_targets(context, hold_list, slots_needed):
    """
    从月初缓存的候选排名中顺延补位。
    跳过：已持仓、当日刚卖出、涨停、停牌、买不起。
    """
    if slots_needed <= 0 or not g.candidate_rank:
        return []

    current_data = get_current_data()
    per_stock_value = context.portfolio.total_value / g.stock_num
    targets = []

    for code in g.candidate_rank:
        if len(targets) >= slots_needed:
            break
        if code in hold_list:
            continue
        if code in g.recently_sold:
            continue
        if is_unbuyable(current_data, code):
            continue

        price = current_data[code].last_price
        if calc_affordable_amount(code, per_stock_value, price) <= 0:
            continue
        targets.append(code)

    return targets


def select_target_stocks(context, score_df, hold_list, stock_num):
    """
    按综合得分从高到低选股；新进标的若买不起则顺延下一名。
    已持仓标的不受 affordability 限制。
    """
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
        amount = calc_affordable_amount(code, per_stock_budget, limit_price)
        if amount > 0:
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
    if len(g.hold_list) != 0:
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


def get_stock_list(context):
    g.buy_df = pd.DataFrame(columns=['name', 'price', 'amount', 'value'])
    yesterday = str(context.previous_date)
    today = context.current_dt

    initial_list = get_all_securities('stock', today).index.tolist()
    initial_list = filter_new_stock(context, initial_list)
    initial_list = filter_kcb_stock(initial_list)
    initial_list = filter_st_stock(initial_list)
    initial_list = filter_paused_stock(initial_list)

    stock_list = initial_list
    df = get_fundamentals(query(valuation.code).filter(
        valuation.code.in_(stock_list),
        valuation.pe_ratio.between(5, 50),
        indicator.inc_return.between(5, 100),
        indicator.inc_total_revenue_year_on_year.between(5, 100),
        indicator.inc_net_profit_year_on_year.between(10, 100),
    ))
    stock_list = list(df.code)

    dividend_list = get_dividend_ratio_filter_list(context, stock_list, False, 0.00, 1.0, 0.00)
    if not dividend_list:
        print("警告：无满足条件的股票，本次不交易")
        g.sell_list = [s for s in g.hold_list if s not in g.high_limit_list]
        return

    momentum_df = get_momentum_factor(context, dividend_list, days=g.momentum_days)

    time1 = context.previous_date
    time0 = time1 - datetime.timedelta(days=365)
    interval = 1000
    list_len = len(dividend_list)

    q = query(
        finance.STK_XR_XD.code,
        finance.STK_XR_XD.a_registration_date,
        finance.STK_XR_XD.bonus_amount_rmb,
    ).filter(
        finance.STK_XR_XD.a_registration_date >= time0,
        finance.STK_XR_XD.a_registration_date <= time1,
        finance.STK_XR_XD.code.in_(dividend_list[:min(list_len, interval)]),
    )
    div_df = finance.run_query(q)

    if list_len > interval:
        df_num = list_len // interval
        for i in range(df_num):
            q = query(
                finance.STK_XR_XD.code,
                finance.STK_XR_XD.a_registration_date,
                finance.STK_XR_XD.bonus_amount_rmb,
            ).filter(
                finance.STK_XR_XD.a_registration_date >= time0,
                finance.STK_XR_XD.a_registration_date <= time1,
                finance.STK_XR_XD.code.in_(dividend_list[interval * (i + 1):min(list_len, interval * (i + 2))]),
            )
            temp_df = finance.run_query(q)
            div_df = pd.concat([div_df, temp_df], ignore_index=True)

    dividend = div_df.fillna(0).set_index('code').groupby('code').sum()
    temp_list = list(dividend.index)

    q = query(valuation.code, valuation.market_cap).filter(valuation.code.in_(temp_list))
    cap = get_fundamentals(q, date=time1).set_index('code')

    div_ratio_df = pd.concat([dividend, cap], axis=1, sort=False)
    div_ratio_df['dividend_ratio'] = (div_ratio_df['bonus_amount_rmb'] / 10000) / div_ratio_df['market_cap']
    div_ratio_df = div_ratio_df[['dividend_ratio']]

    score_df = div_ratio_df.join(momentum_df, how='inner')
    if score_df.empty:
        print("警告：综合打分数据为空，本次不交易")
        g.sell_list = [s for s in g.hold_list if s not in g.high_limit_list]
        return

    score_df['dividend_rank'] = score_df['dividend_ratio'].rank(pct=True, ascending=True)
    score_df['momentum_rank'] = score_df['momentum'].rank(pct=True, ascending=True)
    score_df['total_score'] = (
        g.dividend_weight * score_df['dividend_rank'] +
        g.momentum_weight * score_df['momentum_rank']
    )
    score_df = score_df.sort_values('total_score', ascending=False)
    score_df = score_df[score_df['dividend_ratio'] > 0.03]

    if score_df.empty:
        print("警告：股息率过滤后无标的，本次不交易")
        g.sell_list = [s for s in g.hold_list if s not in g.high_limit_list]
        return

    hold_list = getattr(g, 'hold_list', list(context.portfolio.positions))
    target_list = select_target_stocks(context, score_df, hold_list, g.stock_num)
    select_count = max(g.min_select, min(g.stock_num, len(target_list)))
    target_list = target_list[:select_count]

    # 缓存排名，供月中涨停打开后补位
    g.score_df = score_df
    g.candidate_rank = list(score_df.index)
    g.rebalance_date = context.current_dt.date()

    print('========== 综合打分详情 ==========')
    print(score_df[['dividend_ratio', 'momentum', 'total_score']].head(10))
    print('==================================')
    print('选中标的:', target_list)

    g.sell_list = [s for s in hold_list if s not in target_list and s not in g.high_limit_list]
    buy_list = [s for s in target_list if s not in hold_list]

    value = context.portfolio.available_cash
    for s in g.sell_list:
        value += context.portfolio.positions[s].value

    if buy_list:
        value = value / len(buy_list)
        price_df = get_price(
            buy_list,
            end_date=yesterday,
            frequency='1d',
            count=1,
            fields=['close'],
            fq='pre',
            panel=False,
            skip_paused=False,
            fill_paused=True,
        ).set_index('code')

        rows = []
        for s in buy_list:
            limit_price = get_buy_limit_price(s, price_df.loc[s, 'close'], context.current_dt)
            amount = calc_affordable_amount(s, value * 1.05, limit_price)
            if amount <= 0:
                print('警告：%s 预算不足，跳过买入' % s)
                continue
            rows.append({
                'code': s,
                'name': get_security_info(s, yesterday).display_name,
                'price': limit_price,
                'amount': amount,
                'value': limit_price * amount,
            })

        if rows:
            g.buy_df = pd.DataFrame(rows).set_index('code')

    print('卖出', g.sell_list)
    print('———————————————————————————————————')
    print('红利动量', g.buy_df)
    print('———————————————————————————————————')


def my_trade(context):
    current_data = get_current_data()

    for s in g.sell_list:
        if current_data[s].last_price < current_data[s].high_limit:
            order_target_value(s, 0)

    for s in list(g.buy_df.index):
        amount = int(g.buy_df.loc[s, 'amount'])
        if amount <= 0:
            print('跳过买入 %s：数量为 0' % s)
            g.pending_replenish = True
            continue
        if is_unbuyable(current_data, s):
            print('跳过买入 %s：开盘涨停/停牌，稍后补位' % s)
            g.pending_replenish = True
            continue
        print('买入', [s, g.buy_df.loc[s, 'name']])
        order(s, amount, LimitOrderStyle(g.buy_df.loc[s, 'price']))
        print('———————————————————————————————————')

    if get_current_slots_needed(context) > 0:
        g.pending_replenish = True


def check_limit_up(context):
    current_data = get_current_data()

    if g.high_limit_list:
        for s in g.high_limit_list:
            if current_data[s].last_price < current_data[s].high_limit:
                order_target_value(s, 0)
                g.recently_sold.add(s)
                g.pending_replenish = True
                print(s, '涨停打开，卖出（待补位）')
                print('———————————————————————————————————')
            else:
                print(s, '涨停，继续持有')
                print('———————————————————————————————————')


def replenish_positions(context):
    """
    涨停打开卖出后的补位：不立刻在 10:00 追买，而是 09:35/14:50 尝试。
    从月初候选排名顺延，跳过当日涨停/停牌/刚卖出/买不起的标的。
    """
    slots_needed = get_current_slots_needed(context)
    if slots_needed <= 0:
        g.pending_replenish = False
        g.recently_sold = set()
        return

    # 月初调仓日上午 10:00 前不补位，避免和 my_trade 冲突
    if g.rebalance_date == context.current_dt.date() and context.current_dt.hour < 10:
        return

    if not g.candidate_rank:
        print('补位跳过：无缓存候选排名')
        return

    hold_list = list(context.portfolio.positions.keys())
    buy_list = pick_replenish_targets(context, hold_list, slots_needed)
    if not buy_list:
        print('补位待定：暂无可买标的（可能均在涨停/停牌），稍后重试')
        g.pending_replenish = True
        return

    per_stock_value = context.portfolio.total_value / g.stock_num
    current_data = get_current_data()

    print('========== 持仓补位 ==========')
    print('需补 %d 只，尝试买入: %s' % (slots_needed, buy_list))

    for code in buy_list:
        if is_unbuyable(current_data, code):
            print('跳过 %s：当前涨停或停牌' % code)
            continue
        order_target_value(code, per_stock_value)
        print('补位买入 %s，目标市值 %.2f' % (code, per_stock_value))

    remaining = get_current_slots_needed(context)
    g.pending_replenish = remaining > 0
    if remaining == 0:
        g.recently_sold = set()
    else:
        print('补位后仍缺 %d 只，后续继续尝试' % remaining)
    print('==============================')


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
        df_num = list_len // interval
        for i in range(df_num):
            q = query(
                finance.STK_XR_XD.code,
                finance.STK_XR_XD.a_registration_date,
                finance.STK_XR_XD.bonus_amount_rmb,
            ).filter(
                finance.STK_XR_XD.a_registration_date >= time0,
                finance.STK_XR_XD.a_registration_date <= time1,
                finance.STK_XR_XD.code.in_(stock_list[interval * (i + 1):min(list_len, interval * (i + 2))]),
            )
            temp_df = finance.run_query(q)
            df = pd.concat([df, temp_df], ignore_index=True)

    dividend = df.fillna(0).set_index('code').groupby('code').sum()
    temp_list = list(dividend.index)

    q = query(valuation.code, valuation.market_cap).filter(valuation.code.in_(temp_list))
    cap = get_fundamentals(q, date=time1).set_index('code')

    df = pd.concat([dividend, cap], axis=1, sort=False)
    df['dividend_ratio'] = (df['bonus_amount_rmb'] / 10000) / df['market_cap']
    df = df.sort_values(by=['dividend_ratio'], ascending=sort)
    df = df[int(p1 * len(df)):int(p2 * len(df))]
    df = df[df['dividend_ratio'] > threshold]
    return list(df.index)
