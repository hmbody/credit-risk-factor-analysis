"""
检查 tushare 可用数据
"""
import tushare as ts

ts.set_token("你的tushare token")  # 替换为你的token
pro = ts.pro_api()

print("=" * 60)
print("1. 股票基本信息（测试拉取）")
print("=" * 60)
try:
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry,area,list_date')
    print(f"上市公司总数：{len(df)}")
    print(f"行业分布（前20）：")
    print(df['industry'].value_counts().head(20))
    print(f"\n样本数据：")
    print(df.head(5))
except Exception as e:
    print(f"❌ stock_basic 失败：{e}")

print("\n" + "=" * 60)
print("2. 资产负债表")
print("=" * 60)
try:
    df = pro.balancesheet(ts_code='000001.SZ', start_date='20220101', end_date='20221231',
                          fields='ts_code,ann_date,f_ann_date,end_date,report_type,total_assets,total_liab,total_hldr_eqy_inc_min_int')
    print(f"平安银行资产负债表：{len(df)} 条")
    print(df.head())
except Exception as e:
    print(f"❌ balancesheet 失败：{e}")

print("\n" + "=" * 60)
print("3. 利润表")
print("=" * 60)
try:
    df = pro.income(ts_code='000001.SZ', start_date='20220101', end_date='20221231',
                    fields='ts_code,ann_date,end_date,report_type,total_revenue,oper_profit,net_profit')
    print(f"平安银行利润表：{len(df)} 条")
    print(df.head())
except Exception as e:
    print(f"❌ income 失败：{e}")

print("\n" + "=" * 60)
print("4. 现金流量表")
print("=" * 60)
try:
    df = pro.cashflow(ts_code='000001.SZ', start_date='20220101', end_date='20221231',
                      fields='ts_code,ann_date,end_date,report_type,cash_recp_sg_and_rs')
    print(f"平安银行现金流量表：{len(df)} 条")
    print(df.head())
except Exception as e:
    print(f"❌ cashflow 失败：{e}")

print("\n" + "=" * 60)
print("5. 日线行情")
print("=" * 60)
try:
    df = pro.daily(ts_code='000001.SZ', start_date='20220101', end_date='20220110')
    print(f"平安银行日线：{len(df)} 条")
    print(df.head())
except Exception as e:
    print(f"❌ daily 失败：{e}")

print("\n" + "=" * 60)
print("6. 财务指标（衍生指标）")
print("=" * 60)
try:
    df = pro.fina_indicator(ts_code='000001.SZ', start_date='20220101', end_date='20221231')
    print(f"平安银行财务指标：{len(df)} 条，共 {len(df.columns)} 列")
    print(f"关键列名：")
    for c in df.columns[:30]:
        print(f"  {c}")
except Exception as e:
    print(f"❌ fina_indicator 失败：{e}")

print("\n" + "=" * 60)
print("7. 宏观经济数据")
print("=" * 60)
try:
    df = pro.cn_gdp(start_q='2020Q1', end_q='2025Q4')
    print(f"GDP 数据：{len(df)} 条")
    print(df.head())
except Exception as e:
    print(f"❌ cn_gdp 失败：{e}")
