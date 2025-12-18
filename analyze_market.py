import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have loaded your data
def analyze_market_characteristics(btc_data, eth_data):
    """
    Compare market characteristics between BTC and ETH.
    
    Args:
        btc_data: DataFrame with BTC LOB data
        eth_data: DataFrame with ETH LOB data
    """
    
    print("="*70)
    print("MARKET CHARACTERISTICS COMPARISON: BTC vs ETH")
    print("="*70)
    
    # 1. Price Statistics
    print("\n PRICE STATISTICS")
    print("-"*70)
    
    for name, data in [('BTC', btc_data), ('ETH', eth_data)]:
        avg_price = data['midpoint'].mean()
        price_std = data['midpoint'].std()
        price_range = data['midpoint'].max() - data['midpoint'].min()
        
        print(f"\n{name}:")
        print(f"  Average Price:    ${avg_price:>12,.2f}")
        print(f"  Std Deviation:    ${price_std:>12,.2f}")
        print(f"  Price Range:      ${price_range:>12,.2f}")
    
    # 2. Returns and Volatility
    print("\n\n RETURNS AND VOLATILITY")
    print("-"*70)
    
    btc_returns = btc_data['midpoint'].pct_change().dropna()
    eth_returns = eth_data['midpoint'].pct_change().dropna()
    
    btc_vol = btc_returns.std() * np.sqrt(86400)  # Annualized (assuming 1-second data)
    eth_vol = eth_returns.std() * np.sqrt(86400)
    
    print(f"\nBTC:")
    print(f"  Mean Return:      {btc_returns.mean():>12.6f}")
    print(f"  Volatility:       {btc_returns.std():>12.6f}")
    print(f"  Annualized Vol:   {btc_vol:>12.2%}")
    print(f"  Skewness:         {btc_returns.skew():>12.4f}")
    print(f"  Kurtosis:         {btc_returns.kurtosis():>12.4f}")
    
    print(f"\nETH:")
    print(f"  Mean Return:      {eth_returns.mean():>12.6f}")
    print(f"  Volatility:       {eth_returns.std():>12.6f}")
    print(f"  Annualized Vol:   {eth_vol:>12.2%}")
    print(f"  Skewness:         {eth_returns.skew():>12.4f}")
    print(f"  Kurtosis:         {eth_returns.kurtosis():>12.4f}")
    
    vol_ratio = eth_vol / btc_vol
    print(f"\n ETH is {vol_ratio:.2f}x {'MORE' if vol_ratio > 1 else 'LESS'} volatile than BTC")
    
    # 3. Spread Statistics
    print("\n\n SPREAD STATISTICS")
    print("-"*70)
    
    for name, data in [('BTC', btc_data), ('ETH', eth_data)]:
        avg_spread = data['spread'].mean()
        median_spread = data['spread'].median()
        spread_std = data['spread'].std()
        
        # Relative spread (spread / midpoint)
        rel_spread = (data['spread'] / data['midpoint']).mean()
        
        print(f"\n{name}:")
        print(f"  Average Spread:   ${avg_spread:>12,.2f}")
        print(f"  Median Spread:    ${median_spread:>12,.2f}")
        print(f"  Std Deviation:    ${spread_std:>12,.2f}")
        print(f"  Relative Spread:  {rel_spread:>12.4%}")
    
    spread_ratio = eth_data['spread'].mean() / btc_data['spread'].mean()
    print(f"\n ETH spreads are {spread_ratio:.2f}x {'WIDER' if spread_ratio > 1 else 'TIGHTER'} than BTC")
    
    # 4. Trading Opportunity Assessment
    print("\n\n TRADING OPPORTUNITY ASSESSMENT")
    print("-"*70)
    
    # Count large price movements (potential opportunities)
    btc_large_moves = (abs(btc_returns) > btc_returns.std() * 2).sum()
    eth_large_moves = (abs(eth_returns) > eth_returns.std() * 2).sum()
    
    btc_opp_rate = btc_large_moves / len(btc_returns)
    eth_opp_rate = eth_large_moves / len(eth_returns)
    
    print(f"\nLarge Price Movements (> 2σ):")
    print(f"  BTC: {btc_large_moves:>8,} ({btc_opp_rate:>6.2%})")
    print(f"  ETH: {eth_large_moves:>8,} ({eth_opp_rate:>6.2%})")
    
    # Spread-to-volatility ratio (higher = better for market making)
    btc_ratio = btc_data['spread'].mean() / (btc_returns.std() * btc_data['midpoint'].mean())
    eth_ratio = eth_data['spread'].mean() / (eth_returns.std() * eth_data['midpoint'].mean())
    
    print(f"\nSpread-to-Volatility Ratio:")
    print(f"  BTC: {btc_ratio:>8.4f}")
    print(f"  ETH: {eth_ratio:>8.4f}")
    print(f"\n  {'BTC' if btc_ratio > eth_ratio else 'ETH'} offers better risk/reward for market making")
    
    # 5. Autocorrelation (predictability)
    print("\n\n PREDICTABILITY (Autocorrelation)")
    print("-"*70)
    
    btc_autocorr = btc_returns.autocorr(lag=1)
    eth_autocorr = eth_returns.autocorr(lag=1)
    
    print(f"\nFirst-order autocorrelation:")
    print(f"  BTC: {btc_autocorr:>8.4f}")
    print(f"  ETH: {eth_autocorr:>8.4f}")
    
    if abs(btc_autocorr) > abs(eth_autocorr):
        print(f"\n  BTC shows {'stronger momentum' if btc_autocorr > 0 else 'stronger mean reversion'}")
    else:
        print(f"\n  ETH shows {'stronger momentum' if eth_autocorr > 0 else 'stronger mean reversion'}")
    
    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY & IMPLICATIONS FOR MARKET MAKING")
    print("="*70)
    
    print(f"\n1. VOLATILITY:")
    if vol_ratio > 1.2:
        print(f"   ETH is significantly more volatile ({vol_ratio:.2f}x)")
        print(f"   Higher inventory risk on ETH")
        print(f"   Agents need better risk management on ETH")
    elif vol_ratio < 0.8:
        print(f"   BTC is more volatile ({1/vol_ratio:.2f}x)")
        print(f"   Higher inventory risk on BTC")
    else:
        print(f"   Similar volatility (ratio: {vol_ratio:.2f})")
        print(f"   Comparable risk profiles")
    
    print(f"\n2. SPREADS:")
    if spread_ratio > 1.2:
        print(f"   ETH has wider spreads ({spread_ratio:.2f}x)")
        print(f"   More profit potential per trade on ETH")
        print(f"   But possibly lower execution probability")
    elif spread_ratio < 0.8:
        print(f"   BTC has wider spreads ({1/spread_ratio:.2f}x)")
        print(f"   More profit potential per trade on BTC")
    else:
        print(f"   Similar spreads (ratio: {spread_ratio:.2f})")
    
    print(f"\n3. EXPECTED PERFORMANCE:")
    if vol_ratio > 1.2 and spread_ratio > 1.2:
        print(f"   ETH: Higher risk, higher reward")
        print(f"   A-S might perform BETTER on ETH (adaptive spreads)")
        print(f"   RL agents need good risk management")
    elif vol_ratio > 1.2 and spread_ratio < 1.2:
        print(f"   ETH: Higher risk, similar reward")
        print(f"   Agents likely perform WORSE on ETH")
        print(f"   Volatility not compensated by spread")
    elif vol_ratio < 1.2 and spread_ratio > 1.2:
        print(f"   ETH: Similar risk, higher reward")
        print(f"   Agents likely perform BETTER on ETH")
        print(f"   Ideal conditions for market making")
    else:
        print(f"   Similar risk/reward profile")
        print(f"   Expect similar performance on both assets")
    
    print("\n" + "="*70)
    
    return {
        'volatility_ratio': vol_ratio,
        'spread_ratio': spread_ratio,
        'btc_vol': btc_vol,
        'eth_vol': eth_vol,
        'btc_spread': btc_data['spread'].mean(),
        'eth_spread': eth_data['spread'].mean()
    }




def plot_comparison(btc_data, eth_data):
    """Create comprehensive comparison plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Market Characteristics Comparison: BTC vs ETH', 
                 fontsize=16, fontweight='bold')
    
    # 1. Price evolution
    ax = axes[0, 0]
    ax.plot(btc_data['midpoint'].values / btc_data['midpoint'].iloc[0], 
            label='BTC', alpha=0.7, linewidth=1)
    ax.plot(eth_data['midpoint'].values / eth_data['midpoint'].iloc[0], 
            label='ETH', alpha=0.7, linewidth=1)
    ax.set_title('Normalized Price Evolution')
    ax.set_ylabel('Normalized Price')
    ax.set_xlabel('Timestep')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Returns distribution
    ax = axes[0, 1]
    btc_returns = btc_data['midpoint'].pct_change().dropna()
    eth_returns = eth_data['midpoint'].pct_change().dropna()
    
    ax.hist(btc_returns, bins=100, alpha=0.6, label='BTC', density=True)
    ax.hist(eth_returns, bins=100, alpha=0.6, label='ETH', density=True)
    ax.set_title('Returns Distribution')
    ax.set_xlabel('Return')
    ax.set_ylabel('Density')
    ax.set_xlim([-0.01, 0.01])  # Focus on main body
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Rolling volatility
    ax = axes[0, 2]
    window = 1000
    btc_vol = btc_returns.rolling(window).std()
    eth_vol = eth_returns.rolling(window).std()
    
    ax.plot(btc_vol, label='BTC', alpha=0.7, linewidth=1)
    ax.plot(eth_vol, label='ETH', alpha=0.7, linewidth=1)
    ax.set_title(f'Rolling Volatility ({window} steps)')
    ax.set_ylabel('Volatility')
    ax.set_xlabel('Timestep')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Spread distribution
    ax = axes[1, 0]
    ax.hist(btc_data['spread'], bins=50, alpha=0.6, label='BTC', density=True)
    ax.hist(eth_data['spread'], bins=50, alpha=0.6, label='ETH', density=True)
    ax.set_title('Spread Distribution')
    ax.set_xlabel('Spread ($)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Relative spread over time
    ax = axes[1, 1]
    btc_rel_spread = (btc_data['spread'] / btc_data['midpoint']).rolling(1000).mean()
    eth_rel_spread = (eth_data['spread'] / eth_data['midpoint']).rolling(1000).mean()
    
    ax.plot(btc_rel_spread, label='BTC', alpha=0.7, linewidth=1)
    ax.plot(eth_rel_spread, label='ETH', alpha=0.7, linewidth=1)
    ax.set_title('Rolling Relative Spread (1000 steps)')
    ax.set_ylabel('Spread / Midpoint')
    ax.set_xlabel('Timestep')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    stats_text = "SUMMARY STATISTICS\n" + "="*40 + "\n\n"
    
    stats_text += "Volatility:\n"
    stats_text += f"  BTC: {btc_returns.std():>8.6f}\n"
    stats_text += f"  ETH: {eth_returns.std():>8.6f}\n"
    stats_text += f"  Ratio: {eth_returns.std()/btc_returns.std():>8.2f}x\n\n"
    
    stats_text += "Average Spread:\n"
    stats_text += f"  BTC: ${btc_data['spread'].mean():>8.2f}\n"
    stats_text += f"  ETH: ${eth_data['spread'].mean():>8.2f}\n"
    stats_text += f"  Ratio: {eth_data['spread'].mean()/btc_data['spread'].mean():>8.2f}x\n\n"
    
    stats_text += "Relative Spread:\n"
    btc_rel = (btc_data['spread'] / btc_data['midpoint']).mean()
    eth_rel = (eth_data['spread'] / eth_data['midpoint']).mean()
    stats_text += f"  BTC: {btc_rel:>8.4%}\n"
    stats_text += f"  ETH: {eth_rel:>8.4%}\n"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('btc_vs_eth_characteristics.png', dpi=300, bbox_inches='tight')
    print("\n Figure saved: btc_vs_eth_characteristics.png")
    plt.show()
