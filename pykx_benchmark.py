"""
DataFrame Library Performance Test Framework
Comparing: pandas, pyKX, and polars
"""

import pandas as pd
import polars as pl
import time
import tracemalloc
from typing import Dict, List, Tuple
import numpy as np

# Note: pyKX requires installation and q/kdb+ setup
# pip install pykx
try:
    import pykx as kx
    PYKX_AVAILABLE = True
except ImportError:
    PYKX_AVAILABLE = False
    print("Warning: pyKX not available. Install with: pip install pykx")

# Test Results Storage
results = {
    'pandas': {},
    'polars': {},
    'pykx': {}
}

# Helper Functions
def measure_performance(func, *args, **kwargs) -> Tuple[float, float, any]:
    """Measure execution time and memory usage"""
    tracemalloc.start()
    start_time = time.time()
    
    result = func(*args, **kwargs)
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    execution_time = end_time - start_time
    memory_mb = peak / 1024 / 1024
    
    return execution_time, memory_mb, result

# Generate Test Dataset
print("=" * 60)
print("GENERATING TEST DATASET")
print("=" * 60)

np.random.seed(42)
n_rows = 1_000_000

test_data = {
    'id': np.arange(n_rows),
    'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows),
    'value': np.random.randn(n_rows) * 100,
    'quantity': np.random.randint(1, 100, n_rows),
    'date': pd.date_range('2020-01-01', periods=n_rows, freq='1min')
}

print(f"Dataset: {n_rows:,} rows x {len(test_data)} columns")
print()

# ============================================================================
# TEST CASE 1: Data Loading and Creation
# ============================================================================
print("=" * 60)
print("TEST CASE 1: Data Loading and Creation")
print("=" * 60)

# Pandas
time_pd, mem_pd, df_pd = measure_performance(pd.DataFrame, test_data)
results['pandas']['loading'] = {'time': time_pd, 'memory': mem_pd}
print(f"pandas:  {time_pd:.4f}s | {mem_pd:.2f} MB")

# Polars
time_pl, mem_pl, df_pl = measure_performance(pl.DataFrame, test_data)
results['polars']['loading'] = {'time': time_pl, 'memory': mem_pl}
print(f"polars:  {time_pl:.4f}s | {mem_pl:.2f} MB")

# pyKX
if PYKX_AVAILABLE:
    time_kx, mem_kx, df_kx = measure_performance(kx.toq, df_pd)
    results['pykx']['loading'] = {'time': time_kx, 'memory': mem_kx}
    print(f"pyKX:    {time_kx:.4f}s | {mem_kx:.2f} MB")
else:
    results['pykx']['loading'] = {'time': None, 'memory': None}
    print(f"pyKX:    N/A (not installed)")

print()

# ============================================================================
# TEST CASE 2: Filtering Operations
# ============================================================================
print("=" * 60)
print("TEST CASE 2: Filtering Operations (value > 0 AND category in ['A','B'])")
print("=" * 60)

# Pandas
def filter_pandas():
    return df_pd[(df_pd['value'] > 0) & (df_pd['category'].isin(['A', 'B']))]

time_pd, mem_pd, result_pd = measure_performance(filter_pandas)
results['pandas']['filtering'] = {'time': time_pd, 'memory': mem_pd, 'rows': len(result_pd)}
print(f"pandas:  {time_pd:.4f}s | {mem_pd:.2f} MB | {len(result_pd):,} rows")

# Polars
def filter_polars():
    return df_pl.filter(
        (pl.col('value') > 0) & (pl.col('category').is_in(['A', 'B']))
    )

time_pl, mem_pl, result_pl = measure_performance(filter_polars)
results['polars']['filtering'] = {'time': time_pl, 'memory': mem_pl, 'rows': len(result_pl)}
print(f"polars:  {time_pl:.4f}s | {mem_pl:.2f} MB | {len(result_pl):,} rows")

# pyKX
if PYKX_AVAILABLE:
    def filter_pykx():
        return df_kx.select(kx.q('{[t] select from t where value > 0, category in `A`B}'))
    
    time_kx, mem_kx, result_kx = measure_performance(filter_pykx)
    results['pykx']['filtering'] = {'time': time_kx, 'memory': mem_kx, 'rows': len(result_kx.pd())}
    print(f"pyKX:    {time_kx:.4f}s | {mem_kx:.2f} MB | {len(result_kx.pd()):,} rows")
else:
    results['pykx']['filtering'] = {'time': None, 'memory': None, 'rows': None}
    print(f"pyKX:    N/A")

print()

# ============================================================================
# TEST CASE 3: GroupBy and Aggregation
# ============================================================================
print("=" * 60)
print("TEST CASE 3: GroupBy Aggregation (by category)")
print("=" * 60)

# Pandas
def groupby_pandas():
    return df_pd.groupby('category').agg({
        'value': ['mean', 'sum', 'std'],
        'quantity': ['sum', 'max']
    })

time_pd, mem_pd, result_pd = measure_performance(groupby_pandas)
results['pandas']['groupby'] = {'time': time_pd, 'memory': mem_pd}
print(f"pandas:  {time_pd:.4f}s | {mem_pd:.2f} MB")

# Polars
def groupby_polars():
    return df_pl.group_by('category').agg([
        pl.col('value').mean().alias('value_mean'),
        pl.col('value').sum().alias('value_sum'),
        pl.col('value').std().alias('value_std'),
        pl.col('quantity').sum().alias('quantity_sum'),
        pl.col('quantity').max().alias('quantity_max')
    ])

time_pl, mem_pl, result_pl = measure_performance(groupby_polars)
results['polars']['groupby'] = {'time': time_pl, 'memory': mem_pl}
print(f"polars:  {time_pl:.4f}s | {mem_pl:.2f} MB")

# pyKX
if PYKX_AVAILABLE:
    def groupby_pykx():
        return kx.q('''
            {[t] 
                select 
                    value_mean:avg value,
                    value_sum:sum value,
                    value_std:dev value,
                    quantity_sum:sum quantity,
                    quantity_max:max quantity
                by category from t
            }
        ''', df_kx)
    
    time_kx, mem_kx, result_kx = measure_performance(groupby_pykx)
    results['pykx']['groupby'] = {'time': time_kx, 'memory': mem_kx}
    print(f"pyKX:    {time_kx:.4f}s | {mem_kx:.2f} MB")
else:
    results['pykx']['groupby'] = {'time': None, 'memory': None}
    print(f"pyKX:    N/A")

print()

# ============================================================================
# TEST CASE 4: Join Operations
# ============================================================================
print("=" * 60)
print("TEST CASE 4: Join Operations (self-join on category)")
print("=" * 60)

# Create smaller datasets for join
join_size = 100_000
df_pd_small = df_pd.head(join_size)
df_pl_small = df_pl.head(join_size)

# Pandas
def join_pandas():
    df_left = df_pd_small[['category', 'value']].copy()
    df_right = df_pd_small[['category', 'quantity']].copy()
    return df_left.merge(df_right, on='category', how='inner')

time_pd, mem_pd, result_pd = measure_performance(join_pandas)
results['pandas']['join'] = {'time': time_pd, 'memory': mem_pd, 'rows': len(result_pd)}
print(f"pandas:  {time_pd:.4f}s | {mem_pd:.2f} MB | {len(result_pd):,} rows")

# Polars
def join_polars():
    df_left = df_pl_small.select(['category', 'value'])
    df_right = df_pl_small.select(['category', 'quantity'])
    return df_left.join(df_right, on='category', how='inner')

time_pl, mem_pl, result_pl = measure_performance(join_polars)
results['polars']['join'] = {'time': time_pl, 'memory': mem_pl, 'rows': len(result_pl)}
print(f"polars:  {time_pl:.4f}s | {mem_pl:.2f} MB | {len(result_pl):,} rows")

# pyKX
if PYKX_AVAILABLE:
    df_kx_small = kx.toq(df_pd_small)
    def join_pykx():
        return kx.q('''
            {[t]
                left: select category, value from t;
                right: select category, quantity from t;
                left ij right
            }
        ''', df_kx_small)
    
    time_kx, mem_kx, result_kx = measure_performance(join_pykx)
    results['pykx']['join'] = {'time': time_kx, 'memory': mem_kx, 'rows': len(result_kx.pd())}
    print(f"pyKX:    {time_kx:.4f}s | {mem_kx:.2f} MB | {len(result_kx.pd()):,} rows")
else:
    results['pykx']['join'] = {'time': None, 'memory': None, 'rows': None}
    print(f"pyKX:    N/A")

print()

# ============================================================================
# TEST CASE 5: Complex Query (filter + groupby + sort)
# ============================================================================
print("=" * 60)
print("TEST CASE 5: Complex Query (filter > 50 + groupby + sort)")
print("=" * 60)

# Pandas
def complex_pandas():
    return (df_pd[df_pd['value'] > 50]
            .groupby('category')
            .agg({'value': 'mean', 'quantity': 'sum'})
            .sort_values('value', ascending=False))

time_pd, mem_pd, result_pd = measure_performance(complex_pandas)
results['pandas']['complex'] = {'time': time_pd, 'memory': mem_pd}
print(f"pandas:  {time_pd:.4f}s | {mem_pd:.2f} MB")

# Polars
def complex_polars():
    return (df_pl
            .filter(pl.col('value') > 50)
            .group_by('category')
            .agg([
                pl.col('value').mean().alias('value_mean'),
                pl.col('quantity').sum().alias('quantity_sum')
            ])
            .sort('value_mean', descending=True))

time_pl, mem_pl, result_pl = measure_performance(complex_polars)
results['polars']['complex'] = {'time': time_pl, 'memory': mem_pl}
print(f"polars:  {time_pl:.4f}s | {mem_pl:.2f} MB")

# pyKX
if PYKX_AVAILABLE:
    def complex_pykx():
        return kx.q('''
            {[t]
                `value_mean xdesc select 
                    value_mean:avg value,
                    quantity_sum:sum quantity
                by category from t where value > 50
            }
        ''', df_kx)
    
    time_kx, mem_kx, result_kx = measure_performance(complex_pykx)
    results['pykx']['complex'] = {'time': time_kx, 'memory': mem_kx}
    print(f"pyKX:    {time_kx:.4f}s | {mem_kx:.2f} MB")
else:
    results['pykx']['complex'] = {'time': None, 'memory': None}
    print(f"pyKX:    N/A")

print()

# ============================================================================
# FINAL REPORT
# ============================================================================
print("=" * 60)
print("PERFORMANCE REPORT")
print("=" * 60)
print()

test_cases = ['loading', 'filtering', 'groupby', 'join', 'complex']
test_names = [
    'Data Loading',
    'Filtering',
    'GroupBy Aggregation',
    'Join Operations',
    'Complex Query'
]

# Summary Table
print("EXECUTION TIME COMPARISON (seconds)")
print("-" * 60)
print(f"{'Test Case':<25} {'pandas':>10} {'polars':>10} {'pyKX':>10}")
print("-" * 60)

total_times = {'pandas': 0, 'polars': 0, 'pykx': 0}

for test, name in zip(test_cases, test_names):
    pd_time = results['pandas'][test]['time']
    pl_time = results['polars'][test]['time']
    kx_time = results['pykx'][test]['time']
    
    total_times['pandas'] += pd_time if pd_time else 0
    total_times['polars'] += pl_time if pl_time else 0
    total_times['pykx'] += kx_time if kx_time else 0
    
    pd_str = f"{pd_time:.4f}" if pd_time else "N/A"
    pl_str = f"{pl_time:.4f}" if pl_time else "N/A"
    kx_str = f"{kx_time:.4f}" if kx_time else "N/A"
    
    print(f"{name:<25} {pd_str:>10} {pl_str:>10} {kx_str:>10}")

print("-" * 60)
print(f"{'TOTAL':<25} {total_times['pandas']:>10.4f} {total_times['polars']:>10.4f} {total_times['pykx']:>10.4f}")
print()

# Memory Usage
print("PEAK MEMORY USAGE (MB)")
print("-" * 60)
print(f"{'Test Case':<25} {'pandas':>10} {'polars':>10} {'pyKX':>10}")
print("-" * 60)

for test, name in zip(test_cases, test_names):
    pd_mem = results['pandas'][test]['memory']
    pl_mem = results['polars'][test]['memory']
    kx_mem = results['pykx'][test]['memory']
    
    pd_str = f"{pd_mem:.2f}" if pd_mem else "N/A"
    pl_str = f"{pl_mem:.2f}" if pl_mem else "N/A"
    kx_str = f"{kx_mem:.2f}" if kx_mem else "N/A"
    
    print(f"{name:<25} {pd_str:>10} {pl_str:>10} {kx_str:>10}")

print()

# Winner Analysis
print("=" * 60)
print("ANALYSIS & RECOMMENDATIONS")
print("=" * 60)
print()

# Calculate speed ratios
speed_comparison = []
for test in test_cases:
    if results['pandas'][test]['time'] and results['polars'][test]['time']:
        ratio = results['pandas'][test]['time'] / results['polars'][test]['time']
        speed_comparison.append(ratio)

if speed_comparison:
    avg_speedup = np.mean(speed_comparison)
    print(f"📊 Polars is on average {avg_speedup:.2f}x faster than pandas")
    print()

print("🏆 WINNERS BY CATEGORY:")
print()

for test, name in zip(test_cases, test_names):
    times = {}
    if results['pandas'][test]['time']:
        times['pandas'] = results['pandas'][test]['time']
    if results['polars'][test]['time']:
        times['polars'] = results['polars'][test]['time']
    if results['pykx'][test]['time']:
        times['pyKX'] = results['pykx'][test]['time']
    
    if times:
        winner = min(times.items(), key=lambda x: x[1])
        print(f"  {name:<25} → {winner[0]} ({winner[1]:.4f}s)")

print()
print("📝 KEY FINDINGS:")
print()
print("• pandas: Mature, extensive ecosystem, slower for large datasets")
print("• polars: Fastest for most operations, modern API, parallel execution")
print("• pyKX: Requires kdb+ setup, excellent for time-series and financial data")
print()
print("💡 RECOMMENDATION:")
print()
if avg_speedup > 2:
    print("  Consider migrating to Polars for performance-critical applications.")
    print("  Polars shows significant speed advantages across most operations.")
else:
    print("  Polars offers moderate performance improvements over pandas.")
    print("  Choose based on your specific needs and existing codebase.")
print()
print("=" * 60)
