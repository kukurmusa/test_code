import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BarraRiskModel:
    """
    Barra Risk Model Implementation
    
    Key Formulas:
    1. Factor Model: R_i,t = α_i + Σ(β_i,k * F_k,t) + ε_i,t
    2. Portfolio Risk: σ_p² = w'Xfω'X' + w'Dw (factor risk + specific risk)
    3. Factor Returns: F_t = (X'Ω⁻¹X)⁻¹X'Ω⁻¹R_t
    """
    
    def __init__(self, lookback_window=252, lambda_factor=0.97, lambda_specific=0.84):
        self.lookback_window = lookback_window
        self.lambda_factor = lambda_factor  # Exponential decay for factor covariance
        self.lambda_specific = lambda_specific  # Exponential decay for specific risk
        
    def generate_dummy_data(self, n_assets=100, n_periods=500, n_industries=10):
        """Generate realistic dummy financial data"""
        np.random.seed(42)
        
        # Generate dates
        dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='D')
        
        # Asset identifiers
        assets = [f'STOCK_{i:03d}' for i in range(n_assets)]
        
        # Industry classification
        industries = [f'IND_{i}' for i in range(n_industries)]
        industry_map = {asset: np.random.choice(industries) for asset in assets}
        
        # Generate fundamental data (relatively stable)
        market_caps = np.random.lognormal(15, 2, n_assets) * 1e6  # Market cap in millions
        book_values = market_caps * np.random.uniform(0.3, 3.0, n_assets)
        earnings = market_caps * np.random.uniform(0.02, 0.15, n_assets)
        
        # Generate price returns with factor structure
        # Market factor
        market_returns = np.random.normal(0.0005, 0.015, n_periods)
        
        # Style factor returns
        size_factor = np.random.normal(0.0001, 0.008, n_periods)
        value_factor = np.random.normal(0.0002, 0.012, n_periods)
        momentum_factor = np.random.normal(0, 0.010, n_periods)
        
        # Industry factor returns
        industry_returns = {ind: np.random.normal(0, 0.008, n_periods) 
                           for ind in industries}
        
        # Generate asset returns using factor model
        returns_data = []
        exposures_data = []
        
        for i, asset in enumerate(assets):
            # Factor exposures (relatively stable with some variation)
            market_beta = np.random.normal(1.0, 0.3)
            size_exposure = np.log(market_caps[i] / np.median(market_caps))
            value_exposure = np.log(book_values[i] / market_caps[i])
            momentum_exposure = np.random.normal(0, 0.5)
            
            # Industry exposure (dummy variable)
            industry_exposure = {ind: 1.0 if industry_map[asset] == ind else 0.0 
                               for ind in industries}
            
            # Generate returns using factor model
            specific_risk = np.random.normal(0, 0.02, n_periods)
            
            asset_returns = (market_beta * market_returns + 
                           size_exposure * size_factor +
                           value_exposure * value_factor +
                           momentum_exposure * momentum_factor +
                           sum(industry_exposure[ind] * industry_returns[ind] 
                               for ind in industries) +
                           specific_risk)
            
            returns_data.append(asset_returns)
            
            # Store exposures
            exp_dict = {
                'asset': asset,
                'market_beta': market_beta,
                'size': size_exposure,
                'value': value_exposure,
                'momentum': momentum_exposure,
                'industry': industry_map[asset],
                'market_cap': market_caps[i],
                'book_value': book_values[i],
                'earnings': earnings[i]
            }
            exposures_data.append(exp_dict)
        
        # Create DataFrames
        self.returns = pd.DataFrame(returns_data, 
                                  index=assets, 
                                  columns=dates).T
        
        self.exposures = pd.DataFrame(exposures_data).set_index('asset')
        
        # Create industry dummy variables
        for ind in industries:
            self.exposures[f'ind_{ind}'] = (self.exposures['industry'] == ind).astype(float)
        
        # Factor returns (true factors used in simulation)
        self.true_factors = pd.DataFrame({
            'market': market_returns,
            'size': size_factor,
            'value': value_factor,
            'momentum': momentum_factor,
            **{f'ind_{ind}': industry_returns[ind] for ind in industries}
        }, index=dates)
        
        print(f"Generated data:")
        print(f"- {n_assets} assets over {n_periods} periods")
        print(f"- {n_industries} industries")
        print(f"- Returns shape: {self.returns.shape}")
        print(f"- Exposures shape: {self.exposures.shape}")
        
        return self.returns, self.exposures
    
    def build_factor_exposures_matrix(self, date=None):
        """
        Build factor exposures matrix X
        Formula: X is N×K matrix where N=assets, K=factors
        """
        if date is None:
            date = self.returns.index[-1]
        
        # Select factor columns
        factor_cols = ['market_beta', 'size', 'value', 'momentum']
        industry_cols = [col for col in self.exposures.columns if col.startswith('ind_')]
        
        all_factors = factor_cols + industry_cols
        
        X = self.exposures[all_factors].values
        
        return X, all_factors
    
    def estimate_factor_returns(self, end_date=None, method='wls'):
        """
        Estimate factor returns using cross-sectional regression
        Formula: F_t = (X'Ω⁻¹X)⁻¹X'Ω⁻¹R_t
        
        Where:
        - F_t: Factor returns at time t
        - X: Factor exposures matrix
        - Ω: Specific risk covariance matrix (diagonal)
        - R_t: Asset returns at time t
        """
        if end_date is None:
            end_date = self.returns.index[-1]
        
        # Get factor exposures
        X, factor_names = self.build_factor_exposures_matrix(end_date)
        
        # Initialize results
        factor_returns = []
        residuals = []
        
        # Estimate for each time period
        for date in self.returns.index:
            if date > end_date:
                break
                
            R_t = self.returns.loc[date].values
            
            if method == 'ols':
                # Ordinary Least Squares
                try:
                    beta = np.linalg.lstsq(X, R_t, rcond=None)[0]
                except np.linalg.LinAlgError:
                    beta = np.zeros(X.shape[1])
            
            elif method == 'wls':
                # Weighted Least Squares (weight by market cap)
                weights = np.sqrt(self.exposures['market_cap'].values)
                W = np.diag(weights)
                
                try:
                    XTW = X.T @ W
                    beta = np.linalg.solve(XTW @ X, XTW @ R_t)
                except np.linalg.LinAlgError:
                    beta = np.zeros(X.shape[1])
            
            # Calculate residuals
            residual = R_t - X @ beta
            
            factor_returns.append(beta)
            residuals.append(residual)
        
        # Convert to DataFrames
        self.factor_returns = pd.DataFrame(
            factor_returns, 
            index=self.returns.index[:len(factor_returns)],
            columns=factor_names
        )
        
        self.residuals = pd.DataFrame(
            residuals,
            index=self.returns.index[:len(residuals)],
            columns=self.returns.columns
        )
        
        return self.factor_returns, self.residuals
    
    def calculate_factor_covariance(self, method='exponential'):
        """
        Calculate factor covariance matrix
        Formula: Ω_f = Σ(λ^(T-t) * F_t * F_t') / Σ(λ^(T-t))
        """
        if not hasattr(self, 'factor_returns'):
            raise ValueError("Must estimate factor returns first")
        
        T = len(self.factor_returns)
        
        if method == 'exponential':
            # Exponentially weighted covariance
            weights = np.array([self.lambda_factor**(T-1-t) for t in range(T)])
            weights = weights / weights.sum()
            
            # Demean factor returns
            factor_mean = (self.factor_returns * weights[:, np.newaxis]).sum(axis=0)
            factor_demeaned = self.factor_returns - factor_mean
            
            # Calculate covariance
            cov_matrix = np.zeros((len(self.factor_returns.columns), 
                                 len(self.factor_returns.columns)))
            
            for t in range(T):
                f_t = factor_demeaned.iloc[t].values
                cov_matrix += weights[t] * np.outer(f_t, f_t)
            
        elif method == 'sample':
            # Sample covariance
            cov_matrix = self.factor_returns.cov().values
        
        self.factor_covariance = pd.DataFrame(
            cov_matrix,
            index=self.factor_returns.columns,
            columns=self.factor_returns.columns
        )
        
        return self.factor_covariance
    
    def calculate_specific_risk(self, method='exponential'):
        """
        Calculate specific risk (idiosyncratic risk)
        Formula: σ²_i = Σ(λ^(T-t) * ε²_i,t) / Σ(λ^(T-t))
        """
        if not hasattr(self, 'residuals'):
            raise ValueError("Must estimate factor returns first")
        
        T = len(self.residuals)
        
        if method == 'exponential':
            # Exponentially weighted specific variance
            weights = np.array([self.lambda_specific**(T-1-t) for t in range(T)])
            weights = weights / weights.sum()
            
            # Calculate weighted variance for each asset
            specific_var = np.zeros(len(self.residuals.columns))
            
            for i, asset in enumerate(self.residuals.columns):
                residual_series = self.residuals[asset].values
                # Demean residuals
                residual_mean = (residual_series * weights).sum()
                residual_demeaned = residual_series - residual_mean
                
                # Calculate variance
                specific_var[i] = (weights * residual_demeaned**2).sum()
        
        elif method == 'sample':
            # Sample variance
            specific_var = self.residuals.var().values
        
        self.specific_risk = pd.Series(
            np.sqrt(specific_var),
            index=self.residuals.columns
        )
        
        return self.specific_risk
    
    def calculate_portfolio_risk(self, weights):
        """
        Calculate portfolio risk decomposition
        Formula: σ²_p = w'XΩ_fX'w + w'Dw
        
        Where:
        - w: Portfolio weights
        - X: Factor exposures matrix
        - Ω_f: Factor covariance matrix
        - D: Specific risk diagonal matrix
        """
        if not hasattr(self, 'factor_covariance'):
            raise ValueError("Must calculate factor covariance first")
        
        # Ensure weights are aligned with assets
        if isinstance(weights, dict):
            weights = pd.Series(weights)
        
        # Align weights with returns columns
        weights = weights.reindex(self.returns.columns, fill_value=0)
        w = weights.values
        
        # Get factor exposures
        X, factor_names = self.build_factor_exposures_matrix()
        
        # Factor risk: w'XΩ_fX'w
        factor_exposure = X.T @ w  # K×1 vector
        factor_risk = factor_exposure.T @ self.factor_covariance.values @ factor_exposure
        
        # Specific risk: w'Dw
        specific_risk_contrib = (w**2 * self.specific_risk.values**2).sum()
        
        # Total portfolio risk
        total_risk = factor_risk + specific_risk_contrib
        portfolio_vol = np.sqrt(total_risk)
        
        # Risk decomposition
        risk_decomp = {
            'total_risk': total_risk,
            'portfolio_volatility': portfolio_vol,
            'factor_risk': factor_risk,
            'specific_risk': specific_risk_contrib,
            'factor_risk_pct': factor_risk / total_risk * 100,
            'specific_risk_pct': specific_risk_contrib / total_risk * 100
        }
        
        # Factor contributions
        factor_contributions = {}
        for i, factor in enumerate(factor_names):
            factor_contrib = (factor_exposure[i]**2 * 
                            self.factor_covariance.iloc[i, i])
            factor_contributions[factor] = factor_contrib
        
        return risk_decomp, factor_contributions
    
    def fit_model(self, returns=None, exposures=None):
        """Fit the complete Barra risk model"""
        if returns is not None:
            self.returns = returns
        if exposures is not None:
            self.exposures = exposures
        
        print("Fitting Barra Risk Model...")
        
        # Step 1: Estimate factor returns
        print("1. Estimating factor returns...")
        self.estimate_factor_returns()
        
        # Step 2: Calculate factor covariance
        print("2. Calculating factor covariance matrix...")
        self.calculate_factor_covariance()
        
        # Step 3: Calculate specific risk
        print("3. Calculating specific risk...")
        self.calculate_specific_risk()
        
        print("Model fitting complete!")
        
        return self
    
    def generate_risk_report(self, portfolio_weights=None):
        """Generate comprehensive risk report"""
        
        # Default equal-weight portfolio if none provided
        if portfolio_weights is None:
            n_assets = len(self.returns.columns)
            portfolio_weights = pd.Series(1/n_assets, index=self.returns.columns)
        
        # Calculate portfolio risk
        risk_decomp, factor_contrib = self.calculate_portfolio_risk(portfolio_weights)
        
        # Create report
        report = {
            'model_summary': {
                'assets': len(self.returns.columns),
                'factors': len(self.factor_returns.columns),
                'time_periods': len(self.returns),
                'lookback_window': self.lookback_window
            },
            'portfolio_risk': risk_decomp,
            'factor_contributions': factor_contrib,
            'top_specific_risks': self.specific_risk.nlargest(10).to_dict(),
            'factor_volatilities': np.sqrt(np.diag(self.factor_covariance.values)),
            'avg_specific_risk': self.specific_risk.mean()
        }
        
        return report

# Example usage and demonstration
if __name__ == "__main__":
    # Initialize model
    barra = BarraRiskModel()
    
    # Generate dummy data
    print("Generating dummy data...")
    returns, exposures = barra.generate_dummy_data(n_assets=50, n_periods=252)
    
    # Fit the model
    barra.fit_model()
    
    # Create sample portfolio (tech-heavy)
    portfolio_weights = pd.Series(0, index=returns.columns)
    # Concentrate in first 10 assets
    portfolio_weights.iloc[:10] = 0.1
    
    # Generate risk report
    print("\nGenerating risk report...")
    report = barra.generate_risk_report(portfolio_weights)
    
    # Display results
    print("\n" + "="*50)
    print("BARRA RISK ANALYTICS REPORT")
    print("="*50)
    
    print(f"\nModel Summary:")
    for key, value in report['model_summary'].items():
        print(f"  {key}: {value}")
    
    print(f"\nPortfolio Risk Analysis:")
    risk = report['portfolio_risk']
    print(f"  Total Portfolio Volatility: {risk['portfolio_volatility']:.4f} ({risk['portfolio_volatility']*100:.2f}%)")
    print(f"  Factor Risk: {risk['factor_risk']:.6f} ({risk['factor_risk_pct']:.1f}%)")
    print(f"  Specific Risk: {risk['specific_risk']:.6f} ({risk['specific_risk_pct']:.1f}%)")
    
    print(f"\nTop Factor Contributions:")
    factor_contrib = report['factor_contributions']
    sorted_factors = sorted(factor_contrib.items(), key=lambda x: x[1], reverse=True)
    for factor, contrib in sorted_factors[:5]:
        print(f"  {factor}: {contrib:.6f} ({contrib/risk['total_risk']*100:.1f}%)")
    
    print(f"\nTop Specific Risks:")
    for asset, risk_val in list(report['top_specific_risks'].items())[:5]:
        print(f"  {asset}: {risk_val:.4f}")
    
    print(f"\nAverage Specific Risk: {report['avg_specific_risk']:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Factor returns time series
    barra.factor_returns[['market_beta', 'size', 'value', 'momentum']].plot(
        ax=axes[0,0], title='Factor Returns Time Series'
    )
    axes[0,0].set_ylabel('Returns')
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Factor correlation heatmap
    factor_corr = barra.factor_returns.corr()
    sns.heatmap(factor_corr, annot=True, cmap='coolwarm', center=0,
                ax=axes[0,1], cbar_kws={'shrink': 0.8})
    axes[0,1].set_title('Factor Correlation Matrix')
    
    # 3. Specific risk distribution
    axes[1,0].hist(barra.specific_risk, bins=20, alpha=0.7, edgecolor='black')
    axes[1,0].set_title('Specific Risk Distribution')
    axes[1,0].set_xlabel('Specific Risk')
    axes[1,0].set_ylabel('Frequency')
    
    # 4. Portfolio risk decomposition
    risk_components = ['Factor Risk', 'Specific Risk']
    risk_values = [risk['factor_risk_pct'], risk['specific_risk_pct']]
    
    axes[1,1].pie(risk_values, labels=risk_components, autopct='%1.1f%%',
                  colors=['lightblue', 'lightcoral'])
    axes[1,1].set_title('Portfolio Risk Decomposition')
    
    plt.tight_layout()
    plt.show()
    
    print("\nModel validation - Factor R²:")
    # Calculate R² for each factor
    for factor in ['market_beta', 'size', 'value', 'momentum']:
        true_factor = barra.true_factors[factor] if factor in barra.true_factors.columns else None
        estimated_factor = barra.factor_returns[factor]
        
        if true_factor is not None:
            # Align indices
            common_idx = true_factor.index.intersection(estimated_factor.index)
            if len(common_idx) > 0:
                corr = np.corrcoef(true_factor.loc[common_idx], 
                                 estimated_factor.loc[common_idx])[0,1]
                r_squared = corr**2
                print(f"  {factor}: R² = {r_squared:.3f}")