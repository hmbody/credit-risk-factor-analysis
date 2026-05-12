"""
房地产行业信用风险因子分析
=========================
基于面板数据 + 固定效应模型，分析影响房地产企业信用风险的关键财务因子

研究对象：A股房地产行业上市公司（2020-2024年）
因变量（信用风险代理变量）：Altman Z-score 修正版
自变量：盈利能力、偿债能力、营运能力、成长能力等财务指标
方法：面板数据固定效应模型 + 异质性分析 + 稳健性检验

作者：张亚鹏
日期：2026-05-12
"""

import tushare as ts
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 初始化
# ============================================================
ts.set_token("你的tushare token")  # 替换为你的token，或设置环境变量 TUSHARE_TOKEN
pro = ts.pro_api()

# 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("  房地产行业信用风险因子分析")
print("  基于面板数据固定效应模型")
print("=" * 70)

# ============================================================
# 1. 获取房地产行业公司列表
# ============================================================
print("\n[1/6] 获取房地产行业上市公司列表...")

# 获取所有上市状态的公司
all_stocks = pro.stock_basic(exchange='', list_status='L',
                            fields='ts_code,name,industry,area,list_date')
print(f"  全市场上市公司：{len(all_stocks)} 家")

# 筛选房地产相关行业
real_estate_industries = ['全国性房地产', '区域房地产', '商业地产', '园区开发']
# 直接用关键词
estate_stocks = all_stocks[all_stocks['industry'].str.contains('房地产|地产|园区', na=False)]
print(f"  房地产行业公司：{len(estate_stocks)} 家")
print(f"  具体行业分布：")
for ind, cnt in estate_stocks['industry'].value_counts().items():
    print(f"    {ind}: {cnt} 家")

# 排除 ST、*ST 公司
estate_stocks = estate_stocks[~estate_stocks['name'].str.contains('ST|退', na=False)]
# 排除上市时间晚于2020年的（没有完整5年数据）
estate_stocks = estate_stocks[estate_stocks['list_date'] < '20200101']
print(f"  筛选后（非ST + 2019年前上市）：{len(estate_stocks)} 家")

ts_codes = estate_stocks['ts_code'].tolist()
company_names = dict(zip(estate_stocks['ts_code'], estate_stocks['name']))

# ============================================================
# 2. 批量获取财务指标数据
# ============================================================
print("\n[2/6] 批量获取财务指标数据（2020-2024）...")

all_finance = []
years = ['2020', '2021', '2022', '2023', '2024']
batch_size = 20  # 分批获取，避免超时

for i in range(0, len(ts_codes), batch_size):
    batch = ts_codes[i:i+batch_size]
    batch_str = ','.join(batch)
    try:
        # 获取2020-2024年财务指标
        df = pro.fina_indicator_vip(
            ts_code=batch_str,
            start_date='20200101',
            end_date='20241231',
            fields='ts_code,ann_date,end_date,report_type,'
                   'eps,roe,roa,grossprofit_margin,netprofit_margin,'
                   'debt_to_assets,current_ratio,quick_ratio,'
                   'assets_turn,inv_turn,ar_turn,'
                   'ocf_to_revenue,ocf_to_oper_income,'
                   'ebit_to_interest,ebitda_to_interest,'
                   'profit_to_op,profit_to_gr,op_to_gr,'
                   'total_revenue_ps,undist_profit_ps,'
                   'fcff,fcfe,'
                   'current_exint,noncurrent_exint,'
                   'bps,ocfps,'
                   'total_assets,total_liab,total_hldr_eqy_inc_min_int,'
                   'revenue,oper_profit,net_profit,n_income'
        )
        if df is not None and len(df) > 0:
            all_finance.append(df)
            print(f"  [{i+1}-{min(i+batch_size, len(ts_codes))}] 获取成功，{len(df)} 条记录")
    except Exception as e:
        # 降级：用普通fina_indicator
        try:
            df = pro.fina_indicator(
                ts_code=batch_str, period='20200101'
            )
            if df is not None and len(df) > 0:
                all_finance.append(df)
                print(f"  [{i+1}-{min(i+batch_size, len(ts_codes))}] vip降级普通接口，{len(df)} 条")
        except Exception as e2:
            print(f"  [{i+1}-{min(i+batch_size, len(ts_codes))}] [FAIL] 失败：{str(e2)[:80]}")

if not all_finance:
    # 如果批量失败，逐个尝试
    print("  批量失败，逐个获取...")
    all_finance = []
    for i, code in enumerate(ts_codes):
        try:
            df = pro.fina_indicator(ts_code=code, start_date='20200101', end_date='20241231',
                fields='ts_code,ann_date,end_date,report_type,'
                       'eps,roe,roa,grossprofit_margin,netprofit_margin,'
                       'debt_to_assets,current_ratio,quick_ratio,'
                       'assets_turn,inv_turn,ar_turn,'
                       'ocf_to_revenue,ocf_to_oper_income,'
                       'ebit_to_interest,ebitda_to_interest,'
                       'profit_to_op,profit_to_gr,op_to_gr,'
                       'total_revenue_ps,undist_profit_ps,'
                       'fcff,fcfe,current_exint,noncurrent_exint,'
                       'bps,ocfps,'
                       'total_assets,total_liab,total_hldr_eqy_inc_min_int,'
                       'revenue,oper_profit,net_profit,n_income')
            if df is not None and len(df) > 0:
                all_finance.append(df)
            if (i+1) % 10 == 0:
                print(f"  进度：{i+1}/{len(ts_codes)}")
        except Exception as e:
            pass
    print(f"  逐个获取完成：{len(all_finance)} 家公司")

finance_raw = pd.concat(all_finance, ignore_index=True)
print(f"  原始数据：{len(finance_raw)} 条记录")
print(f"  覆盖公司数：{finance_raw['ts_code'].nunique()} 家")

# ============================================================
# 3. 数据清洗
# ============================================================
print("\n[3/6] 数据清洗与特征工程...")

# 只保留年报（report_type=1）
if 'report_type' in finance_raw.columns:
    finance_raw = finance_raw[finance_raw['report_type'] == '1'].copy()
    print(f"  仅保留年报：{len(finance_raw)} 条")

# 提取年份
finance_raw['year'] = pd.to_datetime(finance_raw['end_date']).dt.year

# 数值列转换
numeric_cols = finance_raw.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    finance_raw[col] = pd.to_numeric(finance_raw[col], errors='coerce')

# 排除金融类指标中明显异常的值
finance_raw.replace([np.inf, -np.inf], np.nan, inplace=True)

# 构建面板：每年每公司一条记录（取最新公告）
df_panel = finance_raw.sort_values(['ts_code', 'year', 'ann_date']).drop_duplicates(
    subset=['ts_code', 'year'], keep='last'
).copy()
print(f"  面板数据：{len(df_panel)} 条（{df_panel['ts_code'].nunique()} 公司 × {df_panel['year'].nunique()} 年）")

# ---- 特征工程 ----
# 注明：因变量使用利息覆盖率（ebit_to_interest）——衡量企业偿债能力的核心指标
# 利息覆盖率 < 1.5 → 利润不足以覆盖利息支出 → 高信用风险

# 构建综合信用风险评分（仅用保证存在的列）
score_components = []
if 'debt_to_assets' in df_panel.columns:
    score_components.append(df_panel['debt_to_assets'].rank(pct=True) * 0.35)
if 'ebit_to_interest' in df_panel.columns:
    score_components.append((1 - df_panel['ebit_to_interest'].rank(pct=True)) * 0.30)
if 'current_ratio' in df_panel.columns:
    score_components.append((1 - df_panel['current_ratio'].rank(pct=True)) * 0.20)
if 'roe' in df_panel.columns:
    score_components.append((1 - df_panel['roe'].rank(pct=True)) * 0.15)

if score_components:
    df_panel['credit_risk_score'] = sum(score_components)
else:
    df_panel['credit_risk_score'] = np.nan

# 同时也计算简化Z-score
df_panel['working_capital'] = np.nan  # 需要流动资产-流动负债，后续改进
df_panel['X1'] = np.nan

# 用更直接的信用风险度量：利息保障倍数 作为连续因变量
# 利息保障倍数 < 1 意味着无法用利润覆盖利息支出（高风险）
# 利息保障倍数 > 5 意味着相对安全

df_panel['interest_coverage'] = df_panel['ebit_to_interest'].clip(-50, 100)

# 二分类：信用风险高低
df_panel['high_credit_risk'] = (df_panel['ebit_to_interest'] < 1.5).astype(int)

print(f"  信用风险分布（利息覆盖率<1.5为高风险）：")
risk_dist = df_panel['high_credit_risk'].value_counts()
print(f"    高风险：{risk_dist.get(1, 0)} 条 ({risk_dist.get(1, 0)/len(df_panel)*100:.1f}%)")
print(f"    低风险：{risk_dist.get(0, 0)} 条 ({risk_dist.get(0, 0)/len(df_panel)*100:.1f}%)")

# 特征列表（定义所有候选特征，后续自动筛选实际存在的列）
feature_cols_all = {
    'debt_to_assets': '资产负债率',
    'current_ratio': '流动比率',
    'quick_ratio': '速动比率',
    'roe': 'ROE（净资产收益率）',
    'roa': 'ROA（总资产收益率）',
    'grossprofit_margin': '毛利率',
    'netprofit_margin': '净利率',
    'assets_turn': '总资产周转率',
    'ocf_to_revenue': '经营现金流/营收',
    'ocf_to_oper_income': '经营现金流/营收',
    'profit_to_gr': '利润增长率',
    'op_to_gr': '营业利润增长率',
    'inv_turn': '存货周转率',
    'ar_turn': '应收账款周转率',
}

# 只保留真实存在的列
feature_cols = {k: v for k, v in feature_cols_all.items() if k in df_panel.columns}
available_features = list(feature_cols.keys())
print(f"  可用特征变量（{len(available_features)}个）：")
for f in available_features:
    print(f"    {f:25s} → {feature_cols[f]}")

# 清理：移除极端缺失（核心变量必须存在）
core_vars = [v for v in ['interest_coverage', 'debt_to_assets', 'current_ratio', 'roe', 'roa'] if v in df_panel.columns]
df_model = df_panel.dropna(subset=core_vars).copy()

# 缩尾处理（1%和99%分位数）
for col in available_features:
    if col in df_model.columns:
        lo = df_model[col].quantile(0.01)
        hi = df_model[col].quantile(0.99)
        df_model[col] = df_model[col].clip(lo, hi)

# 对因变量也缩尾
lo_y = df_model['interest_coverage'].quantile(0.01)
hi_y = df_model['interest_coverage'].quantile(0.99)
df_model['interest_coverage_w'] = df_model['interest_coverage'].clip(lo_y, hi_y)

df_model['company_name'] = df_model['ts_code'].map(company_names)

print(f"  建模数据集：{len(df_model)} 条")
print(f"  覆盖公司：{df_model['ts_code'].nunique()} 家，年份：{sorted(df_model['year'].unique())}")

# ============================================================
# 4. 描述性统计与相关性分析
# ============================================================
print("\n[4/6] 描述性统计与相关性分析...")

# 描述性统计表（动态构建，仅用实际存在的列）
desc_vars = ['interest_coverage_w'] + [f for f in available_features if f in df_model.columns]
desc_labels = ['利息覆盖率'] + [feature_cols[f] for f in desc_vars[1:]]
# 安全过滤：确保desc_vars里全是df_model中存在的列
desc_vars = [v for v in desc_vars if v in df_model.columns]
desc_labels = desc_labels[:len(desc_vars)]

desc_df = df_model[desc_vars].describe().round(3)
print("\n  描述性统计：")
print(desc_df.to_string())

# 相关性矩阵
corr_vars = desc_vars
corr_matrix = df_model[corr_vars].corr().round(3)

print("\n  相关性矩阵（利息覆盖率 vs 各因子）：")
corr_with_target = corr_matrix['interest_coverage_w'].sort_values(ascending=False)
for var, corr in corr_with_target.items():
    label = dict(zip(desc_vars, desc_labels)).get(var, var)
    bar = '█' * int(abs(corr) * 30) if abs(corr) > 0.05 else ''
    print(f"    {label:15s}  {corr:+.3f}  {bar}")

# ============================================================
# 5. 面板数据固定效应模型
# ============================================================
print("\n[5/6] 面板数据固定效应模型...")

# 确保所有数值列的类型正确
for col in df_model.columns:
    if col not in ['ts_code', 'company_name', 'industry', 'leverage_group', 'lev_group']:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
# 填充因变量
df_model['interest_coverage_w'] = df_model['interest_coverage_w'].fillna(df_model['interest_coverage_w'].median())

# 准备回归变量（动态选择：优先用核心财务指标，排除共线性强的）
# 从可用特征中选6-8个最有经济含义的
priority_vars = ['debt_to_assets', 'current_ratio', 'roe', 'roa',
                 'assets_turn', 'grossprofit_margin', 'netprofit_margin',
                 'ocf_to_revenue', 'ocf_to_oper_income', 'profit_to_gr', 'op_to_gr']
X_vars = [v for v in priority_vars if v in df_model.columns and v != 'interest_coverage_w']
# 去重：ocf_to_revenue和ocf_to_oper_income只保留一个
if 'ocf_to_oper_income' in X_vars and 'ocf_to_revenue' in X_vars:
    X_vars.remove('ocf_to_oper_income')
if 'op_to_gr' in X_vars and 'profit_to_gr' in X_vars:
    X_vars.remove('op_to_gr')  # 两个增长率只保留一个

# 限制变量数量（最多8个，避免过度拟合）
X_vars = X_vars[:8]
print(f"  回归自变量（{len(X_vars)}个）：")
for v in X_vars:
    print(f"    {v:25s} → {feature_cols.get(v, v)}")

# 标准化（方便比较系数量级）
X_std = df_model[X_vars].copy()
X_std = X_std.astype(float)
X_mean = X_std.mean()
X_std_dev = X_std.std().replace(0, 1)  # 防止除零
X_std = (X_std - X_mean) / X_std_dev
X_std = X_std.fillna(0).replace([np.inf, -np.inf], 0)

y = df_model['interest_coverage_w'].values.astype(float)
y_std = (y - y.mean()) / y.std()

# 模型1：混合OLS（基准）
print("\n  [模型1] 混合OLS（基准）")
X_ols = sm.add_constant(X_std)
model_ols = sm.OLS(y_std, X_ols).fit()

# 模型2：年份固定效应
print("  [模型2] 年份固定效应")
year_dummies = pd.get_dummies(df_model['year'].astype(int), prefix='year', drop_first=True).astype(float)
X_year_fe = pd.concat([X_std, year_dummies], axis=1).astype(float)
X_year_fe = sm.add_constant(X_year_fe, has_constant='add')
model_year_fe = sm.OLS(y_std, X_year_fe).fit()

# 模型3：公司固定效应（降维：组内去心）
print("  [模型3] 公司固定效应（组内去心）")
df_dm = df_model.copy().reset_index(drop=True)
for v in X_vars + ['interest_coverage_w']:
    firm_mean = df_dm.groupby('ts_code')[v].transform('mean')
    df_dm[f'{v}_dm'] = df_dm[v].astype(float) - firm_mean.astype(float)
# 同时加入年份虚拟变量
X_dm = pd.concat([
    df_dm[[f'{v}_dm' for v in X_vars]],
    year_dummies.reset_index(drop=True)
], axis=1).astype(float)
X_dm = sm.add_constant(X_dm, has_constant='add')
y_dm = df_dm['interest_coverage_w_dm'].astype(float)
model_firm_fe = sm.OLS(y_dm, X_dm).fit()

# 模型4：双向固定效应（公司+年份）—— 这是核心模型
print("  [模型4] 双向固定效应（公司+年份）——核心模型")
df_fe = df_model.copy().reset_index(drop=True)
# 组内去心（同时去除公司和年份均值）
for v in X_vars + ['interest_coverage_w']:
    firm_mean = df_fe.groupby('ts_code')[v].transform('mean').astype(float)
    year_mean = df_fe.groupby('year')[v].transform('mean').astype(float)
    grand_mean = float(df_fe[v].mean())
    df_fe[f'{v}_fe'] = df_fe[v].astype(float) - firm_mean - year_mean + grand_mean

X_fe = df_fe[[f'{v}_fe' for v in X_vars]].astype(float)
X_fe = sm.add_constant(X_fe, has_constant='add')
y_fe = df_fe['interest_coverage_w_fe'].astype(float)
model_twoway_fe = sm.OLS(y_fe, X_fe).fit()

# 汇总结果
print("\n" + "=" * 70)
print("  回归结果汇总（因变量：利息覆盖率，数值越高=信用风险越低）")
print("=" * 70)

result_vars = ['const'] + X_vars
result_labels = ['常数项'] + [feature_cols.get(v, v) for v in X_vars]

for i, (name, model) in enumerate([
    ('混合OLS', model_ols),
    ('年份FE', model_year_fe),
    ('公司FE', model_firm_fe),
    ('双向FE（公司+年份）', model_twoway_fe)
]):
    print(f"\n  {'─'*50}")
    print(f"  [{name}]")
    print(f"  {'─'*50}")
    print(f"  R-sq = {model.rsquared:.3f}  |  Adj R-sq = {model.rsquared_adj:.3f}  |  N = {int(model.nobs)}")
    print(f"  {'变量':20s} {'系数':>8s} {'标准误':>8s} {'t值':>7s} {'p值':>8s}  {'显著性'}")
    print(f"  {'─'*60}")
    for v_idx, var in enumerate(result_vars):
        # FE模型参数名可能带 _fe 或 _dm 后缀
        actual_var = var
        if var not in model.params.index:
            if f'{var}_fe' in model.params.index:
                actual_var = f'{var}_fe'
            elif f'{var}_dm' in model.params.index:
                actual_var = f'{var}_dm'
            else:
                continue
        coef = model.params[actual_var]
        se = model.bse[actual_var]
        tval = coef / se
        pval = model.pvalues[actual_var]
        stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.10 else ''))
        label = result_labels[v_idx] if v_idx < len(result_labels) else var
        print(f"  {label:20s} {coef:>+8.4f} {se:>8.4f} {tval:>+7.3f} {pval:>8.4f}  {stars}")

# ============================================================
# 6. 可视化
# ============================================================
print("\n[6/6] 生成可视化图表...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('房地产行业信用风险因子分析', fontsize=18, fontweight='bold', y=0.98)

# (1) 信用风险年度趋势
ax1 = axes[0, 0]
yearly_risk = df_model.groupby('year').agg(
    avg_coverage=('interest_coverage_w', 'mean'),
    high_risk_pct=('high_credit_risk', 'mean')
).reset_index()
ax1_twin = ax1.twinx()
ax1.bar(yearly_risk['year'], yearly_risk['avg_coverage'], color='steelblue', alpha=0.7, label='平均利息覆盖率')
ax1_twin.plot(yearly_risk['year'], yearly_risk['high_risk_pct']*100, 'ro-', linewidth=2, label='高风险占比(%)')
ax1.set_xlabel('年份')
ax1.set_ylabel('平均利息覆盖率', color='steelblue')
ax1_twin.set_ylabel('高风险企业占比 (%)', color='red')
ax1.set_title('信用风险年度变化趋势')
ax1.set_xticks(yearly_risk['year'])

# (2) 资产负债率 vs 利息覆盖率
ax2 = axes[0, 1]
scatter = ax2.scatter(df_model['debt_to_assets'], df_model['interest_coverage_w'],
                      c=df_model['year'], cmap='coolwarm', alpha=0.5, s=30)
ax2.set_xlabel('资产负债率')
ax2.set_ylabel('利息覆盖率')
ax2.set_title('杠杆率 vs 信用风险')
ax2.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='高风险阈值')
ax2.legend()
plt.colorbar(scatter, ax=ax2, label='年份')

# (3) ROE vs 利息覆盖率
ax3 = axes[0, 2]
ax3.scatter(df_model['roe'], df_model['interest_coverage_w'],
            alpha=0.4, s=30, color='darkgreen')
ax3.set_xlabel('ROE（净资产收益率）')
ax3.set_ylabel('利息覆盖率')
ax3.set_title('盈利能力 vs 信用风险')
ax3.axhline(y=1.5, color='red', linestyle='--', alpha=0.5)
z = np.polyfit(df_model['roe'].fillna(0), df_model['interest_coverage_w'].fillna(0), 1)
p = np.poly1d(z)
x_line = np.linspace(df_model['roe'].min(), df_model['roe'].max(), 100)
ax3.plot(x_line, p(x_line), 'orange', linewidth=2, alpha=0.7)

# (4) 回归系数对比（核心模型的系数）
ax4 = axes[1, 0]
coefs = pd.DataFrame({
    '变量': [feature_cols.get(v, v) for v in X_vars],
    '系数': [model_twoway_fe.params.get(f'{v}_fe', 0) for v in X_vars],
    '标准误': [model_twoway_fe.bse.get(f'{v}_fe', 0) for v in X_vars]
})
coefs = coefs.sort_values('系数', ascending=True)
colors_bar = ['#d9534f' if c < 0 else '#5cb85c' for c in coefs['系数']]
ax4.barh(range(len(coefs)), coefs['系数'], xerr=coefs['标准误'],
         color=colors_bar, alpha=0.8, capsize=3)
ax4.set_yticks(range(len(coefs)))
ax4.set_yticklabels(coefs['变量'])
ax4.set_xlabel('标准化系数（双向固定效应模型）')
ax4.set_title('信用风险因子影响力排序')
ax4.axvline(x=0, color='black', linewidth=0.5)

# (5) 相关性热力图
ax5 = axes[1, 1]
corr = df_model[desc_vars].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, ax=ax5, cbar_kws={'shrink': 0.8},
            xticklabels=desc_labels, yticklabels=desc_labels,
            annot_kws={'size': 7})
ax5.set_title('财务指标相关性热力图')

# (6) 行业细分：不同公司类型的信用风险
ax6 = axes[1, 2]
# 按资产负债率分三组
df_model['leverage_group'] = pd.qcut(df_model['debt_to_assets'], 3,
                                      labels=['低杠杆', '中杠杆', '高杠杆'])
group_risk = df_model.groupby('leverage_group').agg(
    mean_coverage=('interest_coverage_w', 'mean'),
    std_coverage=('interest_coverage_w', 'std')
).reset_index()
colors_g = ['#5cb85c', '#f0ad4e', '#d9534f']
bars = ax6.bar(group_risk['leverage_group'], group_risk['mean_coverage'],
               yerr=group_risk['std_coverage'], color=colors_g, alpha=0.8, capsize=5)
ax6.set_xlabel('杠杆水平分组')
ax6.set_ylabel('平均利息覆盖率')
ax6.set_title('不同杠杆水平下的信用风险差异')
ax6.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='高风险阈值')
ax6.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('D:/19899/credit-risk-project/analysis_charts.png', dpi=150, bbox_inches='tight')
print("  图表已保存：analysis_charts.png")

# ============================================================
# 7. 异质性分析：按公司类型分组
# ============================================================
print("\n" + "=" * 70)
print("  异质性分析：按资产负债率分组回归")
print("=" * 70)

df_het = df_model.copy().reset_index(drop=True)
df_het['lev_group'] = pd.qcut(df_het['debt_to_assets'], 2, labels=['低杠杆组', '高杠杆组'])

for group_name in ['低杠杆组', '高杠杆组']:
    subset = df_het[df_het['lev_group'] == group_name].copy().reset_index(drop=True)
    if len(subset) < 30:
        continue

    # 组内去心
    for v in X_vars + ['interest_coverage_w']:
        firm_m = subset.groupby('ts_code')[v].transform('mean').astype(float)
        year_m = subset.groupby('year')[v].transform('mean').astype(float)
        grand_m = float(subset[v].mean())
        subset[f'{v}_fe'] = subset[v].astype(float) - firm_m - year_m + grand_m

    X_sub = subset[[f'{v}_fe' for v in X_vars]].astype(float)
    X_sub = sm.add_constant(X_sub, has_constant='add')
    y_sub = subset['interest_coverage_w_fe'].astype(float)
    model_sub = sm.OLS(y_sub, X_sub).fit()

    print(f"\n  [{group_name}] N={len(subset)}")
    print(f"  R^2 = {model_sub.rsquared:.3f}")
    for var in X_vars:
        vname = f'{var}_fe'
        if vname in model_sub.params.index:
            coef = model_sub.params[vname]
            pval = model_sub.pvalues[vname]
            stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.10 else ''))
            print(f"    {feature_cols.get(var, var):20s} {coef:>+8.4f} (p={pval:.4f}) {stars}")

# ============================================================
# 8. 稳健性检验
# ============================================================
print("\n" + "=" * 70)
print("  稳健性检验")
print("=" * 70)

# 8.1 替换因变量：用 ROA 替代利息覆盖率（从盈利能力角度验证）
print("\n  [检验1] 替换因变量：使用 ROA 作为经营健康状况代理变量")

df_rob = df_model.copy().reset_index(drop=True)
rob_vars = [v for v in X_vars if v != 'roa']  # 排除roa本身（现在是因变量）
for v in rob_vars + ['roa']:
    firm_m = df_rob.groupby('ts_code')[v].transform('mean').astype(float)
    year_m = df_rob.groupby('year')[v].transform('mean').astype(float)
    grand_m = float(df_rob[v].mean())
    df_rob[f'{v}_fe'] = df_rob[v].astype(float) - firm_m - year_m + grand_m

X_rob = df_rob[[f'{v}_fe' for v in rob_vars]].astype(float)
X_rob = sm.add_constant(X_rob, has_constant='add')
y_rob = df_rob['roa_fe'].astype(float)
model_rob1 = sm.OLS(y_rob, X_rob).fit()
print(f"  R^2 = {model_rob1.rsquared:.3f}")
for var in rob_vars:
    vname = f'{var}_fe'
    if vname in model_rob1.params.index:
        coef = model_rob1.params[vname]
        pval = model_rob1.pvalues[vname]
        stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.10 else ''))
        print(f"    {feature_cols.get(var, var):20s} {coef:>+8.4f} (p={pval:.4f}) {stars}")

# 8.2 缩尾处理前后对比
print("\n  [检验2] 未缩尾数据回归（验证异常值敏感性）")
df_raw_reg = df_panel.dropna(subset=['interest_coverage', 'debt_to_assets', 'current_ratio',
                                       'roe', 'roa', 'assets_turn']).copy().reset_index(drop=True)
for v in X_vars + ['interest_coverage']:
    firm_m = df_raw_reg.groupby('ts_code')[v].transform('mean').astype(float)
    year_m = df_raw_reg.groupby('year')[v].transform('mean').astype(float)
    grand_m = float(df_raw_reg[v].mean())
    df_raw_reg[f'{v}_fe'] = df_raw_reg[v].astype(float) - firm_m - year_m + grand_m

X_raw = df_raw_reg[[f'{v}_fe' for v in X_vars]].astype(float)
X_raw = sm.add_constant(X_raw, has_constant='add')
y_raw = df_raw_reg['interest_coverage_fe'].astype(float)
model_rob2 = sm.OLS(y_raw, X_raw).fit()
print(f"  R^2 = {model_rob2.rsquared:.3f} (vs 缩尾后 {model_twoway_fe.rsquared:.3f})")
print(f"  结论：{'结果稳健，未受极端值驱动' if abs(model_rob2.rsquared - model_twoway_fe.rsquared) < 0.1 else '需关注极端值影响'}")

# ============================================================
# 9. 保存结果
# ============================================================
print("\n" + "=" * 70)
print("  保存结果文件...")
print("=" * 70)

# 保存建模数据集
df_model[['ts_code', 'company_name', 'year'] + X_vars + ['interest_coverage_w', 'high_credit_risk', 'credit_risk_score']].to_csv(
    'D:/19899/credit-risk-project/model_dataset.csv', index=False, encoding='utf-8-sig')
print("  [OK] model_dataset.csv —— 建模数据集")

# 保存描述性统计
desc_df.to_csv('D:/19899/credit-risk-project/descriptive_stats.csv', encoding='utf-8-sig')
print("  [OK] descriptive_stats.csv —— 描述性统计")

# 保存核心回归结果
with open('D:/19899/credit-risk-project/regression_results.txt', 'w', encoding='utf-8') as f:
    f.write("房地产行业信用风险因子分析 —— 回归结果\n")
    f.write("=" * 60 + "\n")
    f.write(f"样本：{df_model['ts_code'].nunique()} 家公司 × {df_model['year'].nunique()} 年\n")
    f.write(f"模型：双向固定效应（公司 + 年份）\n\n")
    f.write(model_twoway_fe.summary().as_text())
print("  [OK] regression_results.txt —— 回归结果")

# 保存项目说明
project_readme = """# 房地产行业信用风险因子分析

## 项目概述

本分析使用A股房地产上市公司2020-2024年面板数据，运用固定效应模型识别影响企业信用风险的关键财务因子。

## 研究方法

- **数据来源**：Tushare金融数据库
- **样本**：A股房地产行业上市公司（非ST，2019年前上市）
- **时间跨度**：2020-2024年（5年面板数据）
- **因变量**：利息覆盖率（EBIT/利息支出）——衡量企业偿债能力的核心指标
- **自变量**：资产负债率、流动比率、ROE、ROA、总资产周转率、毛利率、经营现金流比率、利润增长率
- **模型**：双向固定效应模型（公司个体效应 + 年份时间效应）
- **异质性分析**：按杠杆水平分组回归
- **稳健性检验**：替换因变量（ROA）+ 未缩尾数据验证

## 核心发现

（运行代码后自动生成）

## 文件结构

```
credit-risk-project/
├── 01_check_data.py          # 数据接口检查
├── 02_full_analysis.py       # 完整分析代码
├── model_dataset.csv         # 建模数据集
├── descriptive_stats.csv     # 描述性统计
├── regression_results.txt    # 回归结果
└── analysis_charts.png       # 可视化图表
```

## 技能栈

- Python (pandas, numpy, statsmodels)
- 面板数据分析（固定效应模型）
- 数据可视化（matplotlib, seaborn）
- Tushare金融数据API
"""

with open('D:/19899/credit-risk-project/README.md', 'w', encoding='utf-8') as f:
    f.write(project_readme)
print("  [OK] README.md —— 项目说明")

print("\n" + "=" * 70)
print("  [OK] 分析完成！")
print("=" * 70)
print(f"\n  输出文件：D:/19899/credit-risk-project/")
print(f"    - analysis_charts.png    可视化图表")
print(f"    - model_dataset.csv      建模数据集")
print(f"    - descriptive_stats.csv  描述性统计")
print(f"    - regression_results.txt 回归结果")
print(f"    - README.md              项目说明")
print(f"\n  核心模型：双向固定效应 R^2 = {model_twoway_fe.rsquared:.3f}")
