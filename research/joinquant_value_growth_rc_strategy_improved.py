# -*- coding: utf-8 -*-
"""
价值/成长轮动 + 多商品 + 国债 — 协方差风险预算(RC) 增强版
来源：小红书 @Ellery FIREway 策略改进 + 研究环境增强

在聚宽研究环境 (notebook) 中逐段运行，或整段粘贴运行。
依赖: jqdata, pandas, numpy, matplotlib

增强项:
  1. 协方差矩阵真·风险预算 (20/60 日协方差可切换)
  2. 商品腿拓宽为等权篮子 (豆粕 + 黄金 + 能源化工，自动跳过未上市标的)
  3. 债券预算可配置，默认对比 75:24:1 vs 60:30:10
  4. 价值/成长双弱时动态降股票风险预算
  5. 参数稳健性网格 (可选 RUN_GRID)
  6. 单边交易成本 + 最大单次换手约束
  7. 样本内(2013-2018) / 样本外(2019+) 分段评估
  8. 修复: 周频末交易日调仓、消除 bfill 前视、打印真实样本区间
  9. 价值/成长/基准改用全收益指数 (自动解析可用代码)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from jqdata import *
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 0. 全局配置
# =============================================================================

# --- 全收益指数候选 (按优先级，启动时自动解析第一个有数据的) ---
# 国证 R 指数: 480080/480081 = 成长/价值 100 全收益 (CNI 编制规则)
VALUE_CODE_CANDIDATES = [
    ('480081.XSHE', '国证价值100全收益'),
    ('480081.CNI',  '国证价值100全收益(CNI后缀)'),
    ('H980081.CSI', '国证价值100全收益(备选)'),
    ('980081.XSHE', '国证价值100价格指数(降级)'),
    ('399371.XSHE', '国证价值旧价格指数(降级)'),
]
GROWTH_CODE_CANDIDATES = [
    ('480080.XSHE', '国证成长100全收益'),
    ('480080.CNI',  '国证成长100全收益(CNI后缀)'),
    ('H980080.CSI', '国证成长100全收益(备选)'),
    ('980080.XSHE', '国证成长100价格指数(降级)'),
    ('399370.XSHE', '国证成长旧价格指数(降级)'),
]
BENCH_CODE_CANDIDATES = [
    ('H00905.CSI',  '中证500全收益'),
    ('H00905.XSHG', '中证500全收益(备选)'),
    ('000905.XSHG', '中证500价格指数(降级)'),
]

# 解析后的代码 (运行第 3 节时填充)
VALUE_CODE = None
GROWTH_CODE = None
BENCH_CODE = None
INDEX_DISPLAY = {}   # code -> 图例名称

# --- 商品 / 债券仍用 ETF (无统一全收益指数; ETF 前复权≈持有人全收益) ---
COMMODITY_CODES = [
    '159985.XSHE',              # 豆粕 ETF (~2019-12 上市)
    '518880.XSHG',              # 华安黄金 ETF
    '159981.XSHE',              # 能源化工 ETF (~2020-01 上市); 备选 159697/有色 ETF
]
BOND_CODE    = '511260.XSHG'    # 十年国债 ETF (前复权≈全收益)

# --- 风险预算方案 (name -> {stock, commodity, bond}) ---
BUDGET_SCENARIOS = {
    '75:24:1 (原版)':  {'stock': 0.75, 'commodity': 0.24, 'bond': 0.01},
    '60:30:10 (增强债)': {'stock': 0.60, 'commodity': 0.30, 'bond': 0.10},
}
ACTIVE_BUDGET = '60:30:10 (增强债)'   # 主回测使用的方案

# --- 策略参数 ---
LOOKBACK     = 20       # 轮动 / 波动率窗口
COV_LOOKBACK = 60       # 协方差估计窗口 (20 或 60)
THRESH       = 0.01     # 价值/成长轮动阈值
REBAL_FREQ   = 'W-FRI'  # 每周最后一个交易日调仓 (替代 [::5])
MIN_HISTORY  = max(LOOKBACK, COV_LOOKBACK) + 5

# --- 双弱降仓 ---
WEAK_BOTH_CUT   = True   # 价值&成长 20 日收益均 < 0 时降股票预算
WEAK_STOCK_SCALE = 0.40  # 降仓后股票预算乘数 (剩余预算按原比例分给商品/债)
WEAK_USE_MA     = True   # 额外: 价格低于 60 日均线也触发
MA_WINDOW       = 60

# --- 交易成本 & 换手 ---
ONE_WAY_COST_BPS = 10    # 单边 10bp
MAX_TURNOVER     = 0.30  # 单次调仓最大换手 (权重变化绝对值之和)，超出则部分调仓

# --- 样本区间 ---
DATA_START = '2013-07-18'
DATA_END   = '2026-08-16'
IS_END     = '2018-12-31'   # 样本内截止
OOS_START  = '2019-01-01'   # 样本外起始

# --- 稳健性网格 (较慢，按需开启) ---
RUN_GRID = True
GRID_LOOKBACK = [10, 20, 40]
GRID_THRESH   = [0.0, 0.005, 0.01, 0.02]
GRID_REBAL_DAYS = [5, 10]   # 仅网格时使用固定步长调仓

# --- 输出 ---
PRIMARY_SCENARIO = ACTIVE_BUDGET
SHOW_PLOTS = True


# =============================================================================
# 1. 工具函数
# =============================================================================

def get_close(codes, start, end, fq='pre'):
    """拉取收盘价宽表。指数全收益用 fq=None，ETF 用 fq='pre'。"""
    if isinstance(codes, str):
        codes = [codes]
    if not codes:
        return pd.DataFrame()
    df = get_price(
        codes, start_date=start, end_date=end,
        fields='close', frequency='daily', panel=False, fq=fq
    )
    return df.pivot(index='time', columns='code', values='close')


def probe_has_data(code, start, end, fq=None, min_bars=10):
    """探测代码在聚宽是否有足够数据。"""
    try:
        df = get_price(
            code, start_date=start, end_date=end,
            fields='close', frequency='daily', panel=False, fq=fq
        )
        if df is None or len(df) == 0:
            return False
        return df['close'].notna().sum() >= min_bars
    except Exception:
        return False


def resolve_index_code(candidates, start, end, asset_label):
    """从候选列表中解析第一个可用的全收益指数代码。"""
    for code, desc in candidates:
        if probe_has_data(code, start, end, fq=None):
            print(f'  ✓ {asset_label}: {code}  ({desc})')
            return code, desc
        print(f'  ✗ {asset_label}: {code} 无数据，尝试下一个...')
    raise ValueError(
        f'无法解析 {asset_label} 全收益指数。'
        f'请在研究环境运行 get_all_securities(types=["index"]) 查找后更新候选列表。'
    )


def fetch_mixed_close(index_codes, etf_codes, start, end):
    """分别拉取指数(不复权)与 ETF(前复权)，合并为宽表。"""
    frames = []
    for code in index_codes:
        df = get_close([code], start, end, fq=None)
        if not df.empty:
            frames.append(df)
    if etf_codes:
        df = get_close(etf_codes, start, end, fq='pre')
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how='outer')
    return out.sort_index()


def print_data_diagnostics(close_dict, label=''):
    """打印各标的真实可用区间。"""
    print(f"\n{'='*60}")
    print(f"数据诊断 {label}")
    print(f"{'='*60}")
    rows = []
    for name, s in close_dict.items():
        valid = s.dropna()
        if len(valid) == 0:
            rows.append((name, '无数据', '无数据', 0))
        else:
            rows.append((name, str(valid.index[0].date()), str(valid.index[-1].date()), len(valid)))
    diag = pd.DataFrame(rows, columns=['标的', '首日', '末日', '有效天数'])
    print(diag.to_string(index=False))
    return diag


def risk_budget_weights(cov, budgets, max_iter=500, tol=1e-9):
    """
    真·风险预算: 迭代求解使风险贡献 RC_i ∝ budget_i。
    cov: (n,n), budgets: (n,) 和为 1
    """
    budgets = np.asarray(budgets, dtype=float)
    n = len(budgets)
    diag = np.diag(cov)
    w = budgets / np.sqrt(np.maximum(diag, 1e-16))
    w = w / w.sum()

    for _ in range(max_iter):
        sigma_p = np.sqrt(w @ cov @ w)
        if sigma_p < 1e-14:
            break
        mrc = cov @ w
        rc = w * mrc / sigma_p
        rc = np.maximum(rc, 1e-16)
        w_new = w * budgets * sigma_p / rc
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


def calc_risk_contribution(w, cov):
    """计算各资产风险贡献及占比。"""
    w = np.asarray(w, dtype=float)
    sigma_p = np.sqrt(w @ cov @ w)
    if sigma_p < 1e-14:
        return np.zeros_like(w), sigma_p
    mrc = cov @ w
    rc = w * mrc / sigma_p
    return rc, sigma_p


def inverse_vol_weights(budget, vol_row):
    """简化版 inverse-vol 配权 (用于对比)。"""
    assets = list(budget.keys())
    raw = np.array([budget[a] / max(vol_row[a], 1e-12) for a in assets])
    return raw / raw.sum()


def get_rebal_dates(index, freq=REBAL_FREQ):
    """按 pandas 频率取调仓日 (默认每周五/该周最后交易日)。"""
    s = pd.Series(1, index=index)
    return s.resample(freq).last().dropna().index


def get_rebal_dates_step(index, step):
    """网格搜索用: 固定步长调仓。"""
    return index[::step]


def apply_turnover_cap(w_old, w_target, max_turnover):
    """超出最大换手时，按比例向目标权重移动。"""
    w_old = np.asarray(w_old, dtype=float)
    w_target = np.asarray(w_target, dtype=float)
    delta = w_target - w_old
    turnover = np.abs(delta).sum()
    if turnover <= max_turnover or turnover < 1e-12:
        return w_target.copy(), turnover
    alpha = max_turnover / turnover
    w_new = w_old + alpha * delta
    w_new = w_new / w_new.sum()
    return w_new, np.abs(w_new - w_old).sum()


def calc_metrics(port_ret, rf=0.025):
    """年化收益、最大回撤、夏普(减无风险)、卡玛。"""
    port_ret = port_ret.dropna()
    if len(port_ret) < 2:
        return dict(ann=np.nan, mdd=np.nan, sharpe=np.nan, calmar=np.nan, days=0)
    nav = (1 + port_ret).cumprod()
    years = len(nav) / 252
    ann = nav.iloc[-1] ** (1 / years) - 1
    mdd = (nav / nav.cummax() - 1).min()
    excess = port_ret - rf / 252
    sharpe = excess.mean() / (excess.std() + 1e-12) * np.sqrt(252)
    calmar = ann / abs(mdd) if mdd != 0 else np.nan
    return dict(ann=ann, mdd=mdd, sharpe=sharpe, calmar=calmar, days=len(port_ret), nav=nav)


def build_commodity_return(close, commodity_codes, ret_index):
    """商品篮子: 等权平均日收益，对齐到 ret_index。"""
    rets = close[commodity_codes].pct_change()
    return rets.mean(axis=1).reindex(ret_index)


def build_rotation_signal(close, value_code, growth_code, lookback, thresh, ret_index):
    """价值/成长轮动信号: 1=价值, -1=成长。"""
    ret20_v = close[value_code] / close[value_code].shift(lookback) - 1
    ret20_g = close[growth_code] / close[growth_code].shift(lookback) - 1
    diff = ret20_v - ret20_g

    signal = pd.Series(np.nan, index=diff.index)
    signal[diff > thresh] = 1
    signal[diff < -thresh] = -1
    signal = signal.ffill().fillna(1)
    signal = signal.reindex(ret_index).ffill().fillna(1)

    # 日收益必须与 signal 同索引，避免 1592 vs 1593 广播错误
    daily_ret = close.pct_change().reindex(ret_index)
    stock_ret = pd.Series(
        np.where(signal.values == 1, daily_ret[value_code].values, daily_ret[growth_code].values),
        index=ret_index,
    )

    return signal, stock_ret, ret20_v.reindex(ret_index), ret20_g.reindex(ret_index)


def build_weak_mask(ret20_v, ret20_g, close, value_code, growth_code, ret_index):
    """双弱 / 均线破位 -> 降股票预算。"""
    mask = pd.Series(False, index=ret_index)
    if WEAK_BOTH_CUT:
        mask |= (ret20_v < 0) & (ret20_g < 0)
    if WEAK_USE_MA:
        ma_v = close[value_code].rolling(MA_WINDOW).mean()
        ma_g = close[growth_code].rolling(MA_WINDOW).mean()
        below_ma = (close[value_code] < ma_v) & (close[growth_code] < ma_g)
        mask |= below_ma.reindex(ret_index).fillna(False)
    return mask


def effective_budget(base_budget, weak_today):
    """根据双弱信号调整风险预算。"""
    b = dict(base_budget)
    if not weak_today:
        return b
    stock_b = b['stock'] * WEAK_STOCK_SCALE
    freed = b['stock'] - stock_b
    other_sum = b['commodity'] + b['bond']
    if other_sum < 1e-12:
        b['stock'] = stock_b
        return b
    b['stock'] = stock_b
    b['commodity'] += freed * (b['commodity'] / other_sum)
    b['bond'] += freed * (b['bond'] / other_sum)
    return b


# =============================================================================
# 2. 核心回测引擎
# =============================================================================

def run_backtest(
    close,
    commodity_codes,
    budget,
    lookback=LOOKBACK,
    cov_lookback=COV_LOOKBACK,
    thresh=THRESH,
    rebal_dates=None,
    one_way_cost_bps=ONE_WAY_COST_BPS,
    max_turnover=MAX_TURNOVER,
    use_true_rc=True,
    label='',
):
    """
    返回 dict: port_ret, nav, w_hist, rc_hist, turnover_hist, metrics, ...
    """
    assets = ['stock', 'commodity', 'bond']
    budget_vec_base = np.array([budget[a] for a in assets])

    # 对齐: 所有腿均有价格
    cols_needed = [VALUE_CODE, GROWTH_CODE, BOND_CODE] + commodity_codes
    close = close[cols_needed].copy()
    valid_mask = close.notna().all(axis=1)
    close = close.loc[valid_mask]
    if len(close) < MIN_HISTORY + 10:
        raise ValueError(f'[{label}] 有效样本过短: {len(close)} 天')

    ret = close.pct_change().dropna()
    ret_index = ret.index

    _, stock_ret, ret20_v, ret20_g = build_rotation_signal(
        close, VALUE_CODE, GROWTH_CODE, lookback, thresh, ret_index
    )
    commodity_ret = build_commodity_return(close, commodity_codes, ret_index)
    bond_ret = ret[BOND_CODE]

    r = pd.DataFrame({
        'stock': stock_ret,
        'commodity': commodity_ret,
        'bond': bond_ret,
    }, index=ret_index)

    weak_mask = build_weak_mask(ret20_v, ret20_g, close, VALUE_CODE, GROWTH_CODE, ret_index)

    if rebal_dates is None:
        rebal_dates = get_rebal_dates(ret_index)
    else:
        rebal_dates = pd.Index(rebal_dates).intersection(ret_index)

    # 预计算每日目标权重 (真 RC 或 inverse-vol)
    target_w = pd.DataFrame(index=ret_index, columns=assets, dtype=float)
    rc_check = pd.DataFrame(index=ret_index, columns=assets, dtype=float)

    for date in ret_index:
        b_dict = effective_budget(budget, bool(weak_mask.loc[date]))
        b_vec = np.array([b_dict[a] for a in assets])

        hist = r.loc[:date].tail(cov_lookback)
        if len(hist) < lookback:
            continue

        vol_row = hist.tail(lookback).std()

        if use_true_rc and len(hist) >= cov_lookback:
            cov = hist.cov().values
            # 正定修正
            cov = cov + np.eye(len(assets)) * 1e-8
            w = risk_budget_weights(cov, b_vec)
            rc, _ = calc_risk_contribution(w, cov)
            rc_check.loc[date] = rc / (rc.sum() + 1e-12)
        else:
            vol_dict = {a: vol_row[a] for a in assets}
            inv_budget = {a: b_dict[a] for a in assets}
            w = inverse_vol_weights(inv_budget, vol_dict)
            rc_check.loc[date] = np.nan

        target_w.loc[date] = w

    # 从第一个有效权重日开始，不做 bfill 前视
    first_valid = target_w.dropna(how='any').index.min()
    if pd.isna(first_valid):
        raise ValueError(f'[{label}] 无法估计权重')
    target_w = target_w.loc[first_valid:]
    r = r.loc[first_valid:]
    rebal_dates = rebal_dates.intersection(r.index)

    port_ret = pd.Series(0.0, index=r.index)
    cost_ret = pd.Series(0.0, index=r.index)
    w_hist = pd.DataFrame(index=r.index, columns=assets, dtype=float)
    turnover_hist = pd.Series(0.0, index=r.index)

    cur_w = target_w.iloc[0].values.astype(float)
    cost_rate = one_way_cost_bps / 10000.0

    for date, rrow in r.iterrows():
        rvec = rrow.values.astype(float)

        if date in rebal_dates:
            w_tgt = target_w.loc[date].values.astype(float)
            w_new, turnover = apply_turnover_cap(cur_w, w_tgt, max_turnover)
            # 成本: 单边 bp × 换手 (权重变化绝对值之和)
            cost = turnover * cost_rate
            cost_ret.loc[date] = cost
            turnover_hist.loc[date] = turnover
            cur_w = w_new
        else:
            cur_w = cur_w * (1 + rvec)
            cur_w = cur_w / cur_w.sum()

        port_ret.loc[date] = np.dot(cur_w, rvec) - cost_ret.loc[date]
        w_hist.loc[date] = cur_w

    metrics = calc_metrics(port_ret)
    metrics['label'] = label
    metrics['start'] = r.index[0]
    metrics['end'] = r.index[-1]

    return dict(
        port_ret=port_ret,
        nav=metrics['nav'],
        w_hist=w_hist,
        rc_hist=rc_check.loc[r.index],
        turnover_hist=turnover_hist,
        cost_ret=cost_ret,
        r=r,
        signal=build_rotation_signal(close, VALUE_CODE, GROWTH_CODE, lookback, thresh, r.index)[0],
        weak_mask=weak_mask.loc[r.index],
        metrics=metrics,
    )


# =============================================================================
# 3. 拉取数据
# =============================================================================

print('正在解析全收益指数代码...')
VALUE_CODE, VALUE_NAME = resolve_index_code(VALUE_CODE_CANDIDATES, DATA_START, DATA_END, '价值')
GROWTH_CODE, GROWTH_NAME = resolve_index_code(GROWTH_CODE_CANDIDATES, DATA_START, DATA_END, '成长')
BENCH_CODE, BENCH_NAME = resolve_index_code(BENCH_CODE_CANDIDATES, DATA_START, DATA_END, '基准')
INDEX_DISPLAY = {
    VALUE_CODE: VALUE_NAME,
    GROWTH_CODE: GROWTH_NAME,
    BENCH_CODE: BENCH_NAME,
}

print('\n正在拉取数据...')
index_codes = [VALUE_CODE, GROWTH_CODE]
etf_codes = [BOND_CODE] + COMMODITY_CODES
raw_close = fetch_mixed_close(index_codes, etf_codes, DATA_START, DATA_END)
bench_raw = get_close([BENCH_CODE], DATA_START, DATA_END, fq=None)

diag_parts = {}
for code in index_codes + etf_codes:
    if code in raw_close.columns:
        diag_parts[f'{code} ({INDEX_DISPLAY.get(code, "ETF")})'] = raw_close[code]
if BENCH_CODE in bench_raw.columns:
    diag_parts[f'{BENCH_CODE} ({BENCH_NAME})'] = bench_raw[BENCH_CODE]
print_data_diagnostics(diag_parts, '(原始各标的)')

# 商品篮子: 只用已有数据的标的
available_commodities = [c for c in COMMODITY_CODES if c in raw_close.columns and raw_close[c].notna().sum() > 252]
print(f'\n商品篮子实际使用: {available_commodities}')
if not available_commodities:
    raise ValueError('无可用商品 ETF，请检查 COMMODITY_CODES')

# 组合回测对齐起点 = 所有必需标的均有数据的第一个日期
required = [VALUE_CODE, GROWTH_CODE, BOND_CODE] + available_commodities
aligned_close = raw_close[required].dropna()
print(f'\n组合对齐后样本: {aligned_close.index[0].date()} ~ {aligned_close.index[-1].date()} ({len(aligned_close)} 天)')

bench_close = bench_raw[BENCH_CODE]
bench_close = bench_close.reindex(aligned_close.index).ffill()
bench_nav = bench_close / bench_close.iloc[0]
bench_ret = bench_close.pct_change().dropna()


# =============================================================================
# 4. 主回测: 多预算方案 + RC vs Inverse-Vol 对比
# =============================================================================

results = {}

for scen_name, budget in BUDGET_SCENARIOS.items():
    for use_rc, rc_label in [(True, 'RC'), (False, 'InvVol')]:
        key = f'{scen_name} | {rc_label}'
        print(f'\n运行: {key} ...')
        try:
            res = run_backtest(
                aligned_close.copy(),
                available_commodities,
                budget,
                use_true_rc=use_rc,
                label=key,
            )
            results[key] = res
        except Exception as e:
            print(f'  跳过 {key}: {e}')

primary_key = f'{PRIMARY_SCENARIO} | RC'
if primary_key not in results:
    if not results:
        raise RuntimeError('所有回测均失败，请检查上方报错信息')
    primary_key = next(iter(results))
primary = results[primary_key]
m = primary['metrics']

print(f"\n{'='*60}")
print(f'主策略: {primary_key}')
print(f'样本: {m["start"].date()} ~ {m["end"].date()}')
print(f'年化收益: {m["ann"]*100:.2f}%')
print(f'最大回撤: {m["mdd"]*100:.2f}%')
print(f'夏普比率: {m["sharpe"]:.3f} (Rf=2.5%)')
print(f'卡玛比率: {m["calmar"]:.3f}')
print(f'累计交易成本拖累: {primary["cost_ret"].sum()*100:.2f}% (占初始单位净值)')

# 对比表
print(f"\n{'='*60}")
print('方案对比')
print(f"{'='*60}")
rows = []
for k, res in results.items():
    mm = res['metrics']
    rows.append({
        '方案': k,
        '年化%': round(mm['ann'] * 100, 2),
        '回撤%': round(mm['mdd'] * 100, 2),
        '夏普': round(mm['sharpe'], 3),
        '卡玛': round(mm['calmar'], 3),
    })
print(pd.DataFrame(rows).to_string(index=False))

# RC 达成度 (主策略)
rc_mean = primary['rc_hist'].mean()
print(f"\n主策略平均风险贡献占比 (RC 目标 {PRIMARY_SCENARIO}):")
target_b = BUDGET_SCENARIOS[ACTIVE_BUDGET]
for a in ['stock', 'commodity', 'bond']:
    tgt = target_b[a]
    act = rc_mean[a]
    print(f'  {a}: 目标 {tgt*100:.1f}%  实际 {act*100:.1f}%  偏差 {(act-tgt)*100:+.1f}pp')


# =============================================================================
# 5. 样本内 / 样本外
# =============================================================================

def split_metrics(res, is_end=IS_END, oos_start=OOS_START):
    pr = res['port_ret']
    is_ret = pr.loc[:is_end]
    oos_ret = pr.loc[oos_start:]
    return calc_metrics(is_ret), calc_metrics(oos_ret)

print(f"\n{'='*60}")
print('样本内 (<=2018) vs 样本外 (2019+)')
print(f"{'='*60}")
is_m, oos_m = split_metrics(primary)
print(f"样本内: 年化 {is_m['ann']*100:.2f}%  回撤 {is_m['mdd']*100:.2f}%  夏普 {is_m['sharpe']:.3f}")
print(f"样本外: 年化 {oos_m['ann']*100:.2f}%  回撤 {oos_m['mdd']*100:.2f}%  夏普 {oos_m['sharpe']:.3f}")

# 基准对比 (全样本 & OOS)
years = len(primary['nav']) / 252
bench_nav_aligned = bench_nav.reindex(primary['nav'].index).ffill()
bench_ann = bench_nav_aligned.iloc[-1] ** (1 / years) - 1
bench_mdd = (bench_nav_aligned / bench_nav_aligned.cummax() - 1).min()
bench_sharpe = calc_metrics(bench_ret.reindex(primary['port_ret'].index).fillna(0))['sharpe']
print(f"\n基准({BENCH_NAME}): 年化 {bench_ann*100:.2f}%  回撤 {bench_mdd*100:.2f}%  夏普 {bench_sharpe:.3f}")


# =============================================================================
# 6. 参数稳健性网格
# =============================================================================

if RUN_GRID:
    print(f"\n{'='*60}")
    print('参数稳健性网格 (RC + 主预算方案)...')
    print(f"{'='*60}")
    grid_rows = []
    base_budget = BUDGET_SCENARIOS[ACTIVE_BUDGET]

    for lb in GRID_LOOKBACK:
        for th in GRID_THRESH:
            for rb in GRID_REBAL_DAYS:
                try:
                    rd = get_rebal_dates_step(
                        aligned_close.dropna().pct_change().dropna().index, rb
                    )
                    gres = run_backtest(
                        aligned_close.copy(),
                        available_commodities,
                        base_budget,
                        lookback=lb,
                        thresh=th,
                        rebal_dates=rd,
                        use_true_rc=True,
                        label='grid',
                    )
                    gm = gres['metrics']
                    grid_rows.append({
                        'LOOKBACK': lb,
                        'THRESH': th,
                        'REBAL': rb,
                        '年化%': round(gm['ann'] * 100, 2),
                        '回撤%': round(gm['mdd'] * 100, 2),
                        '夏普': round(gm['sharpe'], 3),
                        '卡玛': round(gm['calmar'], 3),
                    })
                except Exception:
                    pass

    grid_df = pd.DataFrame(grid_rows)
    if len(grid_df):
        print(grid_df.sort_values('夏普', ascending=False).to_string(index=False))
        print(f"\n网格统计: 年化 {grid_df['年化%'].mean():.2f}%±{grid_df['年化%'].std():.2f}%, "
              f"夏普 {grid_df['夏普'].mean():.3f}±{grid_df['夏普'].std():.3f}")
    else:
        print('网格无有效结果')


# =============================================================================
# 7. 年度收益
# =============================================================================

nav = primary['nav']
print('\n年度收益 (主策略):')
annual = (nav.resample('A').last() / nav.resample('A').first() - 1) * 100
print(annual.round(2))


# =============================================================================
# 8. 绘图
# =============================================================================

if SHOW_PLOTS:
    nav_assets = aligned_close / aligned_close.iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].plot(nav_assets[VALUE_CODE], label=INDEX_DISPLAY.get(VALUE_CODE, '价值'), linewidth=1.2)
    axes[0].plot(nav_assets[GROWTH_CODE], label=INDEX_DISPLAY.get(GROWTH_CODE, '成长'), linewidth=1.2)
    for c in available_commodities:
        axes[0].plot(nav_assets[c], label=f'{c}(ETF)', linewidth=1.0, alpha=0.8)
    axes[0].plot(nav_assets[BOND_CODE], label='十年国债ETF', linewidth=1.2)
    axes[0].set_title('各标的净值 (归一化; 股=全收益指数, 商品/债=ETF前复权)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(bench_nav.reindex(nav.index), label=INDEX_DISPLAY.get(BENCH_CODE, '中证500全收益'), alpha=0.7, linewidth=1.2, color='orange')
    for k, res in results.items():
        if 'RC' in k and '75:24:1' in k:
            axes[1].plot(res['nav'], label=k, alpha=0.6, linewidth=1.2, linestyle='--')
    axes[1].plot(nav, label=f'主策略 {primary_key}', linewidth=2.5, color='red')
    axes[1].axvline(pd.Timestamp(IS_END), color='gray', linestyle=':', alpha=0.8, label='样本内/外分界')
    axes[1].set_title('RC 组合 vs 基准')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 权重时序
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    w_plot = primary['w_hist']
    ax2.stackplot(
        w_plot.index,
        w_plot['stock'], w_plot['commodity'], w_plot['bond'],
        labels=['股票', '商品篮子', '债券'], alpha=0.7
    )
    ax2.set_title('实际权重时序 (含漂移 + 成本)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 风险贡献 vs 目标
    fig3, ax3 = plt.subplots(figsize=(14, 4))
    rc_plot = primary['rc_hist'].rolling(20).mean()
    ax3.plot(rc_plot['stock'], label='股票 RC')
    ax3.plot(rc_plot['commodity'], label='商品 RC')
    ax3.plot(rc_plot['bond'], label='债券 RC')
    tb = BUDGET_SCENARIOS[ACTIVE_BUDGET]
    ax3.axhline(tb['stock'], color='C0', linestyle='--', alpha=0.5)
    ax3.axhline(tb['commodity'], color='C1', linestyle='--', alpha=0.5)
    ax3.axhline(tb['bond'], color='C2', linestyle='--', alpha=0.5)
    ax3.set_title('滚动平均风险贡献占比 vs 目标预算 (虚线)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 收益贡献分解
    contrib = primary['w_hist'].shift(1) * primary['r']
    cum_contrib = contrib.cumsum()
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
    ax4a.stackplot(
        cum_contrib.index,
        cum_contrib['stock'], cum_contrib['commodity'], cum_contrib['bond'],
        labels=['股票', '商品', '债券'], alpha=0.7
    )
    ax4a.set_title('累计收益贡献分解')
    ax4a.legend(loc='upper left')
    ax4a.grid(True, alpha=0.3)
    cc = contrib.sum()
    ax4b.pie(cc.values, labels=['股票', '商品', '债券'], autopct='%1.1f%%', startangle=90)
    ax4b.set_title('收益贡献占比')
    plt.tight_layout()
    plt.show()

    print('\n各资产累计收益贡献占比:')
    for name, c in cc.items():
        print(f'  {name}: {c:.4f} ({c / cc.sum() * 100:.1f}%)')

print('\n完成。如需加速网格或关闭绘图: RUN_GRID=False, SHOW_PLOTS=False')
