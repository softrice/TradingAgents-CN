# -*- coding: utf-8 -*-
"""
价值/成长轮动 + 多商品 + 国债 — 协方差风险预算(RC) 增强版
来源：小红书 @Ellery FIREway 策略改进 + 研究环境增强

在聚宽研究环境 (notebook) 中整段粘贴运行。
依赖: jqdata, pandas, numpy, matplotlib

数据模式 (可同时跑多种对比):
  - index:      价值/成长/基准用指数 (fq=None)
  - etf_v100:   国证价值100/成长100 ETF (159263/159259, 跟踪980081/980080)
  - etf_proxy:  长历史风格代理 (510880/159915, 仅作补充对照)
  - etf:        同 etf_v100 的别名
  商品/债券始终为 ETF 前复权。
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

# --- 数据模式 ---
# index=指数; etf_v100=国证价值100/成长100 ETF; etf_proxy=长历史风格代理(对照)
DATA_MODES = ['index', 'etf_v100', 'etf_proxy']
PRIMARY_DATA_MODE = 'etf_v100'      # 网格 / 详图默认: 真·跟踪 ETF
COMPARE_MODES = True

# --- 指数模式 (fq=None) ---
# 注: 399371/399370 = 国证1000价值/成长(旧); 980081/980080 = 国证价值100/成长100(新)
VALUE_INDEX_CANDIDATES = [
    ('480081.XSHE', '国证价值100全收益'),
    ('480081.CNI',  '国证价值100全收益(CNI)'),
    ('H980081.CSI', '国证价值100全收益(备选)'),
    ('980081.XSHE', '国证价值100价格指数'),
    ('399371.XSHE', '国证1000价值(旧,非价值100)'),
]
GROWTH_INDEX_CANDIDATES = [
    ('480080.XSHE', '国证成长100全收益'),
    ('480080.CNI',  '国证成长100全收益(CNI)'),
    ('H980080.CSI', '国证成长100全收益(备选)'),
    ('980080.XSHE', '国证成长100价格指数'),
    ('399370.XSHE', '国证1000成长(旧,非成长100)'),
]
BENCH_INDEX_CANDIDATES = [
    ('H00905.CSI',  '中证500全收益'),
    ('H00905.XSHG', '中证500全收益(备选)'),
    ('000905.XSHG', '中证500价格指数'),
]

# --- 国证价值100 / 成长100 官方 ETF (易方达, 2025 上市) ---
VALUE_V100_ETF_CANDIDATES = [
    ('159263.XSHE', '国证价值100ETF(159263→980081)'),
]
GROWTH_V100_ETF_CANDIDATES = [
    ('159259.XSHE', '国证成长100ETF(159259→980080)'),
]

# --- 长历史风格代理 (与价值100/成长100 成分不同, 仅对照) ---
VALUE_ETF_PROXY_CANDIDATES = [
    ('510880.XSHG', '红利ETF(价值代理,非价值100)'),
    ('512040.XSHG', '价值ETF'),
    ('159913.XSHE', '价值ETF(深)'),
]
GROWTH_ETF_PROXY_CANDIDATES = [
    ('159915.XSHE', '创业板ETF(成长代理,非成长100)'),
    ('159949.XSHE', '创业板50ETF'),
    ('159967.XSHE', '成长ETF'),
]
BENCH_ETF_CANDIDATES = [
    ('510500.XSHG', '中证500ETF前复权'),
]

ETF_MODE_SPECS = {
    'etf_v100': {
        'value': VALUE_V100_ETF_CANDIDATES,
        'growth': GROWTH_V100_ETF_CANDIDATES,
        'stock_type': '国证价值100/成长100 ETF(前复权)',
    },
    'etf_proxy': {
        'value': VALUE_ETF_PROXY_CANDIDATES,
        'growth': GROWTH_ETF_PROXY_CANDIDATES,
        'stock_type': '风格代理ETF(前复权,非价值100/成长100)',
    },
}
MIN_ALIGNED_DAYS_WARN = 504    # 对齐样本少于 2 年时提示

COMMODITY_CODES = [
    '159985.XSHE',              # 豆粕 ETF
    '518880.XSHG',              # 黄金 ETF
    '159981.XSHE',              # 能化 ETF; 备选 159697
]
BOND_CODE = '511260.XSHG'       # 十年国债 ETF

BUDGET_SCENARIOS = {
    '75:24:1 (原版)':  {'stock': 0.75, 'commodity': 0.24, 'bond': 0.01},
    '60:30:10 (增强债)': {'stock': 0.60, 'commodity': 0.30, 'bond': 0.10},
}
ACTIVE_BUDGET = '60:30:10 (增强债)'

LOOKBACK     = 20
COV_LOOKBACK = 60
THRESH       = 0.01
REBAL_FREQ   = 'W-FRI'
MIN_HISTORY  = max(LOOKBACK, COV_LOOKBACK) + 5

WEAK_BOTH_CUT    = True
WEAK_STOCK_SCALE = 0.40
WEAK_USE_MA      = True
MA_WINDOW        = 60

# --- 交易成本 & 换手 ---
# ONE_WAY_COST_BPS: 单边费率，单位 bp(基点)。1bp = 万分之一 = 0.01%
#   万一手续费 → ONE_WAY_COST_BPS = 1
#   万三       → 3 ;  万五 → 5 ;  万十(0.10%) → 10
ONE_WAY_COST_BPS = 1     # 默认: 万一 (单边 0.01%)
MAX_TURNOVER     = 0.30  # 单次调仓最大换手 (权重变化绝对值之和)

DATA_START = '2013-07-18'
DATA_END   = '2026-08-16'
IS_END     = '2018-12-31'
OOS_START  = '2019-01-01'

RUN_GRID = True                     # 仅对 PRIMARY_DATA_MODE 跑网格
GRID_LOOKBACK = [10, 20, 40]
GRID_THRESH   = [0.0, 0.005, 0.01, 0.02]
GRID_REBAL_DAYS = [5, 10]

PRIMARY_SCENARIO = ACTIVE_BUDGET
SHOW_PLOTS = True


# =============================================================================
# 1. 工具函数
# =============================================================================

def get_close(codes, start, end, fq='pre'):
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


def resolve_code(candidates, start, end, asset_label, fq=None):
    mode_hint = 'ETF前复权' if fq == 'pre' else '指数'
    for code, desc in candidates:
        if probe_has_data(code, start, end, fq=fq):
            print(f'  ✓ {asset_label}({mode_hint}): {code}  ({desc})')
            return code, desc
        print(f'  ✗ {asset_label}: {code} 无数据，尝试下一个...')
    raise ValueError(f'无法解析 {asset_label}，请更新候选列表或在聚宽查 code')


def fetch_mixed_close(stock_codes, stock_fq, etf_codes, start, end):
    """股票侧(指数或ETF) + 商品/债 ETF 合并宽表。"""
    frames = []
    for code in stock_codes:
        df = get_close([code], start, end, fq=stock_fq)
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
    budgets = np.asarray(budgets, dtype=float)
    w = budgets / np.sqrt(np.maximum(np.diag(cov), 1e-16))
    w = w / w.sum()
    for _ in range(max_iter):
        sigma_p = np.sqrt(w @ cov @ w)
        if sigma_p < 1e-14:
            break
        mrc = cov @ w
        rc = np.maximum(w * mrc / sigma_p, 1e-16)
        w_new = w * budgets * sigma_p / rc
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


def calc_risk_contribution(w, cov):
    w = np.asarray(w, dtype=float)
    sigma_p = np.sqrt(w @ cov @ w)
    if sigma_p < 1e-14:
        return np.zeros_like(w), sigma_p
    mrc = cov @ w
    return w * mrc / sigma_p, sigma_p


def inverse_vol_weights(budget, vol_row):
    assets = list(budget.keys())
    raw = np.array([budget[a] / max(vol_row[a], 1e-12) for a in assets])
    return raw / raw.sum()


def get_rebal_dates(index, freq=REBAL_FREQ):
    s = pd.Series(1, index=index)
    return s.resample(freq).last().dropna().index


def get_rebal_dates_step(index, step):
    return index[::step]


def apply_turnover_cap(w_old, w_target, max_turnover):
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
    rets = close[commodity_codes].pct_change()
    return rets.mean(axis=1).reindex(ret_index)


def build_rotation_signal(close, value_code, growth_code, lookback, thresh, ret_index):
    ret20_v = close[value_code] / close[value_code].shift(lookback) - 1
    ret20_g = close[growth_code] / close[growth_code].shift(lookback) - 1
    diff = ret20_v - ret20_g

    signal = pd.Series(np.nan, index=diff.index)
    signal[diff > thresh] = 1
    signal[diff < -thresh] = -1
    signal = signal.ffill().fillna(1)
    signal = signal.reindex(ret_index).ffill().fillna(1)

    daily_ret = close.pct_change().reindex(ret_index)
    stock_ret = pd.Series(
        np.where(signal.values == 1, daily_ret[value_code].values, daily_ret[growth_code].values),
        index=ret_index,
    )
    return signal, stock_ret, ret20_v.reindex(ret_index), ret20_g.reindex(ret_index)


def build_weak_mask(ret20_v, ret20_g, close, value_code, growth_code, ret_index):
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
# 2. 数据集准备 & 回测引擎
# =============================================================================

def normalize_mode(mode):
    """'etf' 视为 etf_v100 别名。"""
    return 'etf_v100' if mode == 'etf' else mode


def prepare_dataset(mode, start=DATA_START, end=DATA_END):
    """解析代码、拉数据、对齐样本。返回 dataset dict。"""
    mode = normalize_mode(mode)
    print(f"\n{'#'*60}")
    print(f'# 数据模式: {mode.upper()}')
    print(f"{'#'*60}")

    if mode == 'index':
        stock_fq = None
        bench_fq = None
        print('正在解析指数代码...')
        value_code, value_name = resolve_code(VALUE_INDEX_CANDIDATES, start, end, '价值', fq=None)
        growth_code, growth_name = resolve_code(GROWTH_INDEX_CANDIDATES, start, end, '成长', fq=None)
        bench_code, bench_name = resolve_code(BENCH_INDEX_CANDIDATES, start, end, '基准', fq=None)
        rotation_switch_cost = False
        stock_type = '指数(价格/全收益, fq=None)'
    elif mode in ETF_MODE_SPECS:
        spec = ETF_MODE_SPECS[mode]
        stock_fq = 'pre'
        bench_fq = 'pre'
        print(f'正在解析 ETF 代码 (前复权) — {spec["stock_type"]}...')
        value_code, value_name = resolve_code(spec['value'], start, end, '价值', fq='pre')
        growth_code, growth_name = resolve_code(spec['growth'], start, end, '成长', fq='pre')
        bench_code, bench_name = resolve_code(BENCH_ETF_CANDIDATES, start, end, '基准', fq='pre')
        rotation_switch_cost = ETF_ROTATION_SWITCH_COST
        stock_type = spec['stock_type']
    else:
        raise ValueError(f'未知模式: {mode}，可用 index / etf_v100 / etf_proxy')

    display = {
        value_code: value_name,
        growth_code: growth_name,
        bench_code: bench_name,
    }

    etf_codes = [BOND_CODE] + COMMODITY_CODES
    raw_close = fetch_mixed_close([value_code, growth_code], stock_fq, etf_codes, start, end)
    bench_raw = get_close([bench_code], start, end, fq=bench_fq)

    diag = {}
    for code in [value_code, growth_code] + etf_codes:
        if code in raw_close.columns:
            diag[f'{code} ({display.get(code, "ETF")})'] = raw_close[code]
    if bench_code in bench_raw.columns:
        diag[f'{bench_code} ({bench_name})'] = bench_raw[bench_code]
    print_data_diagnostics(diag, f'[{mode}] 原始各标的')

    available_commodities = [
        c for c in COMMODITY_CODES
        if c in raw_close.columns and raw_close[c].notna().sum() > 252
    ]
    print(f'\n[{mode}] 商品篮子: {available_commodities}')
    if not available_commodities:
        raise ValueError(f'[{mode}] 无可用商品 ETF')

    required = [value_code, growth_code, BOND_CODE] + available_commodities
    aligned_close = raw_close[required].dropna()
    n_aligned = len(aligned_close)
    print(f'[{mode}] 组合对齐: {aligned_close.index[0].date()} ~ {aligned_close.index[-1].date()} ({n_aligned} 天)')
    print(f'[{mode}] 股票腿={stock_type}; 轮动切换成本={"开" if rotation_switch_cost else "关"}')
    if mode == 'etf_v100' and n_aligned < MIN_ALIGNED_DAYS_WARN:
        print(f'  ⚠ 国证价值100/成长100 ETF 上市晚(159263≈2025-06, 159259≈2025-08)，'
              f'对齐样本仅 {n_aligned} 天；长周期请对照 etf_proxy / index 模式')

    bench_close = bench_raw[bench_code].reindex(aligned_close.index).ffill()
    bench_nav = bench_close / bench_close.iloc[0]
    bench_ret = bench_close.pct_change().dropna()

    return dict(
        mode=mode,
        value_code=value_code,
        growth_code=growth_code,
        bench_code=bench_code,
        display=display,
        aligned_close=aligned_close,
        available_commodities=available_commodities,
        bench_close=bench_close,
        bench_nav=bench_nav,
        bench_ret=bench_ret,
        bench_name=bench_name,
        rotation_switch_cost=rotation_switch_cost,
        stock_type=stock_type,
    )


def run_backtest(
    dataset,
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
    close = dataset['aligned_close'].copy()
    commodity_codes = dataset['available_commodities']
    value_code = dataset['value_code']
    growth_code = dataset['growth_code']
    charge_rotation = dataset['rotation_switch_cost']

    assets = ['stock', 'commodity', 'bond']
    cols_needed = [value_code, growth_code, BOND_CODE] + commodity_codes
    close = close[cols_needed].copy()
    close = close.loc[close.notna().all(axis=1)]
    if len(close) < MIN_HISTORY + 10:
        raise ValueError(f'[{label}] 有效样本过短: {len(close)} 天')

    ret = close.pct_change().dropna()
    ret_index = ret.index

    signal, stock_ret, ret20_v, ret20_g = build_rotation_signal(
        close, value_code, growth_code, lookback, thresh, ret_index
    )
    commodity_ret = build_commodity_return(close, commodity_codes, ret_index)
    bond_ret = ret[BOND_CODE]

    r = pd.DataFrame({'stock': stock_ret, 'commodity': commodity_ret, 'bond': bond_ret}, index=ret_index)
    weak_mask = build_weak_mask(ret20_v, ret20_g, close, value_code, growth_code, ret_index)

    if rebal_dates is None:
        rebal_dates = get_rebal_dates(ret_index)
    else:
        rebal_dates = pd.Index(rebal_dates).intersection(ret_index)

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
            cov = hist.cov().values + np.eye(len(assets)) * 1e-8
            w = risk_budget_weights(cov, b_vec)
            rc, _ = calc_risk_contribution(w, cov)
            rc_check.loc[date] = rc / (rc.sum() + 1e-12)
        else:
            w = inverse_vol_weights({a: b_dict[a] for a in assets}, {a: vol_row[a] for a in assets})
            rc_check.loc[date] = np.nan
        target_w.loc[date] = w

    first_valid = target_w.dropna(how='any').index.min()
    if pd.isna(first_valid):
        raise ValueError(f'[{label}] 无法估计权重')
    target_w = target_w.loc[first_valid:]
    r = r.loc[first_valid:]
    signal = signal.loc[first_valid:]
    rebal_dates = rebal_dates.intersection(r.index)

    port_ret = pd.Series(0.0, index=r.index)
    cost_ret = pd.Series(0.0, index=r.index)
    rot_cost_ret = pd.Series(0.0, index=r.index)
    w_hist = pd.DataFrame(index=r.index, columns=assets, dtype=float)
    turnover_hist = pd.Series(0.0, index=r.index)
    cost_rate = one_way_cost_bps / 10000.0
    cur_w = target_w.iloc[0].values.astype(float)
    prev_signal = signal.iloc[0]

    for date, rrow in r.iterrows():
        rvec = rrow.values.astype(float)
        sig = signal.loc[date]

        if date in rebal_dates:
            w_tgt = target_w.loc[date].values.astype(float)
            w_new, turnover = apply_turnover_cap(cur_w, w_tgt, max_turnover)
            cost = turnover * cost_rate
            cost_ret.loc[date] = cost
            turnover_hist.loc[date] = turnover
            cur_w = w_new
        else:
            cur_w = cur_w * (1 + rvec)
            cur_w = cur_w / cur_w.sum()

        # ETF 模式: 价值↔成长切换日，股票腿近似全额换手
        if charge_rotation and sig != prev_signal:
            rot_turn = abs(cur_w[0]) * 2.0
            rot_cost = rot_turn * cost_rate
            rot_cost_ret.loc[date] = rot_cost
            cost_ret.loc[date] += rot_cost
        prev_signal = sig

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
        rot_cost_ret=rot_cost_ret,
        r=r,
        signal=signal,
        weak_mask=weak_mask.loc[r.index],
        metrics=metrics,
    )


def run_all_scenarios(dataset):
    results = {}
    for scen_name, budget in BUDGET_SCENARIOS.items():
        for use_rc, rc_label in [(True, 'RC'), (False, 'InvVol')]:
            key = f'{scen_name} | {rc_label}'
            print(f"  [{dataset['mode']}] 运行: {key} ...")
            try:
                results[key] = run_backtest(dataset, budget, use_true_rc=use_rc, label=key)
            except Exception as e:
                print(f'    跳过: {e}')
    return results


def split_metrics(res, is_end=IS_END, oos_start=OOS_START):
    pr = res['port_ret']
    return calc_metrics(pr.loc[:is_end]), calc_metrics(pr.loc[oos_start:])


def print_mode_summary(dataset, results, primary_key):
    primary = results[primary_key]
    m = primary['metrics']
    mode = dataset['mode']
    print(f"\n{'='*60}")
    print(f'[{mode.upper()}] 主策略: {primary_key}')
    print(f"样本: {m['start'].date()} ~ {m['end'].date()}")
    print(f"股票: {dataset['value_code']} / {dataset['growth_code']} ({dataset['stock_type']})")
    print(f"基准: {dataset['bench_code']} ({dataset['bench_name']})")
    print(f"年化: {m['ann']*100:.2f}%  回撤: {m['mdd']*100:.2f}%  夏普: {m['sharpe']:.3f}  卡玛: {m['calmar']:.3f}")
    print(f"调仓成本拖累: {primary['cost_ret'].sum()*100:.2f}%", end='')
    if dataset['rotation_switch_cost']:
        print(f" (含轮动切换 {primary['rot_cost_ret'].sum()*100:.2f}%)")
    else:
        print()

    rows = []
    for k, res in results.items():
        mm = res['metrics']
        rows.append({'方案': k, '年化%': round(mm['ann']*100, 2), '回撤%': round(mm['mdd']*100, 2),
                     '夏普': round(mm['sharpe'], 3), '卡玛': round(mm['calmar'], 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    years = len(primary['nav']) / 252
    bnav = dataset['bench_nav'].reindex(primary['nav'].index).ffill()
    bench_ann = bnav.iloc[-1] ** (1 / years) - 1
    bench_mdd = (bnav / bnav.cummax() - 1).min()
    bench_sharpe = calc_metrics(dataset['bench_ret'].reindex(primary['port_ret'].index).fillna(0))['sharpe']
    print(f"基准: 年化 {bench_ann*100:.2f}%  回撤 {bench_mdd*100:.2f}%  夏普 {bench_sharpe:.3f}")
    return primary


def compare_modes(mode_store, primary_scenario=PRIMARY_SCENARIO):
    if len(mode_store) < 2:
        return
    print(f"\n{'='*60}")
    print('多模式对比 (主方案 RC)')
    print(f"{'='*60}")
    rows = []
    for mode, pack in mode_store.items():
        ds = pack['dataset']
        pk = f'{primary_scenario} | RC'
        if pk not in pack['results']:
            pk = next(iter(pack['results']))
        m = pack['results'][pk]['metrics']
        rows.append({
            '模式': mode,
            '样本起': str(m['start'].date()),
            '样本止': str(m['end'].date()),
            '年化%': round(m['ann']*100, 2),
            '回撤%': round(m['mdd']*100, 2),
            '夏普': round(m['sharpe'], 3),
            '价值标的': ds['value_code'],
            '成长标的': ds['growth_code'],
        })
    print(pd.DataFrame(rows).to_string(index=False))


# =============================================================================
# 3. 主流程
# =============================================================================

if isinstance(DATA_MODES, str):
    DATA_MODES = [DATA_MODES]

mode_store = {}

for raw_mode in DATA_MODES:
    mode = normalize_mode(raw_mode)
    if mode in mode_store:
        print(f'\n[{mode}] 已在 mode_store 中，跳过重复运行')
        continue
    ds = prepare_dataset(mode)
    print(f"\n[{mode}] 开始回测...")
    results = run_all_scenarios(ds)
    if not results:
        print(f'[{mode}] 全部失败，跳过')
        continue

    pk = f'{PRIMARY_SCENARIO} | RC'
    if pk not in results:
        pk = next(iter(results))
    primary = print_mode_summary(ds, results, pk)

    is_m, oos_m = split_metrics(primary)
    print(f"  样本内(<=2018): 年化 {is_m['ann']*100:.2f}% (可能 nan — 样本不足)")
    print(f"  样本外(2019+):  年化 {oos_m['ann']*100:.2f}%  回撤 {oos_m['mdd']*100:.2f}%  夏普 {oos_m['sharpe']:.3f}")

    mode_store[mode] = dict(dataset=ds, results=results, primary_key=pk, primary=primary)

if not mode_store:
    raise RuntimeError('所有数据模式均失败')

if COMPARE_MODES and len(mode_store) >= 2:
    compare_modes(mode_store)

# --- 网格 (仅 PRIMARY_DATA_MODE) ---
if RUN_GRID and PRIMARY_DATA_MODE in mode_store:
    pack = mode_store[PRIMARY_DATA_MODE]
    ds = pack['dataset']
    print(f"\n{'='*60}")
    print(f'参数网格 [{PRIMARY_DATA_MODE}] RC + {ACTIVE_BUDGET}...')
    print(f"{'='*60}")
    grid_rows = []
    base_budget = BUDGET_SCENARIOS[ACTIVE_BUDGET]
    idx = ds['aligned_close'].dropna().pct_change().dropna().index
    for lb in GRID_LOOKBACK:
        for th in GRID_THRESH:
            for rb in GRID_REBAL_DAYS:
                try:
                    gres = run_backtest(
                        ds, base_budget, lookback=lb, thresh=th,
                        rebal_dates=get_rebal_dates_step(idx, rb),
                        use_true_rc=True, label='grid',
                    )
                    gm = gres['metrics']
                    grid_rows.append({'LOOKBACK': lb, 'THRESH': th, 'REBAL': rb,
                                      '年化%': round(gm['ann']*100, 2), '回撤%': round(gm['mdd']*100, 2),
                                      '夏普': round(gm['sharpe'], 3), '卡玛': round(gm['calmar'], 3)})
                except Exception:
                    pass
    grid_df = pd.DataFrame(grid_rows)
    if len(grid_df):
        print(grid_df.sort_values('夏普', ascending=False).to_string(index=False))
        print(f"\n网格: 年化 {grid_df['年化%'].mean():.2f}%±{grid_df['年化%'].std():.2f}%, "
              f"夏普 {grid_df['夏普'].mean():.3f}±{grid_df['夏普'].std():.3f}")

# --- 年度收益 (PRIMARY_DATA_MODE) ---
if PRIMARY_DATA_MODE in mode_store:
    nav = mode_store[PRIMARY_DATA_MODE]['primary']['nav']
    print(f'\n年度收益 [{PRIMARY_DATA_MODE}]:')
    print(((nav.resample('A').last() / nav.resample('A').first() - 1) * 100).round(2))

# --- 绘图 (PRIMARY_DATA_MODE; 可选叠加 index 曲线) ---
if SHOW_PLOTS and PRIMARY_DATA_MODE in mode_store:
    pack = mode_store[PRIMARY_DATA_MODE]
    ds, results, pk = pack['dataset'], pack['results'], pack['primary_key']
    primary = pack['primary']
    nav = primary['nav']
    aligned = ds['aligned_close']
    nav_assets = aligned / aligned.iloc[0]
    vc, gc = ds['value_code'], ds['growth_code']

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    axes[0].plot(nav_assets[vc], label=ds['display'][vc], linewidth=1.2)
    axes[0].plot(nav_assets[gc], label=ds['display'][gc], linewidth=1.2)
    for c in ds['available_commodities']:
        axes[0].plot(nav_assets[c], label=f'{c}(ETF)', alpha=0.8)
    axes[0].plot(nav_assets[BOND_CODE], label='国债ETF')
    axes[0].set_title(f'各标的净值 [{PRIMARY_DATA_MODE}] {ds["stock_type"]}')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ds['bench_nav'].reindex(nav.index), label=ds['bench_name'], color='orange', alpha=0.7)
    overlay_colors = {'index': 'gray', 'etf_v100': 'green', 'etf_proxy': 'blue'}
    for om, pack in mode_store.items():
        if om == PRIMARY_DATA_MODE:
            continue
        opk = pack['primary_key']
        axes[1].plot(
            pack['results'][opk]['nav'], label=f'{om}',
            linestyle='--', alpha=0.55, color=overlay_colors.get(om, None),
        )
    axes[1].plot(nav, label=f'主策略 {pk}', linewidth=2.5, color='red')
    axes[1].set_title('组合 vs 基准 (虚线=另一数据模式)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    w_plot = primary['w_hist']
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.stackplot(w_plot.index, w_plot['stock'], w_plot['commodity'], w_plot['bond'],
                  labels=['股票', '商品', '债券'], alpha=0.7)
    ax2.set_title(f'权重时序 [{PRIMARY_DATA_MODE}]')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    contrib = primary['w_hist'].shift(1) * primary['r']
    cc = contrib.sum()
    print(f'\n[{PRIMARY_DATA_MODE}] 收益贡献: stock {cc["stock"]/cc.sum()*100:.1f}%  '
          f'commodity {cc["commodity"]/cc.sum()*100:.1f}%  bond {cc["bond"]/cc.sum()*100:.1f}%')

print('\n完成。模式: index / etf_v100(159263+159259) / etf_proxy(510880+159915); RUN_GRID / SHOW_PLOTS 可调')
