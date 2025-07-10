import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb

# SHAP for feature importance
# try:
#    import shap
#    SHAP_AVAILABLE = True
# except ImportError:
#    SHAP_AVAILABLE = False
#    print("SHAP not available. Install with: pip install shap")

class AdvancedSP500Predictor:
    def __init__(self, symbol='^GSPC', lookback_days=1000):
        """
        Initialize the S&P 500 predictor with pipeline-based processing
        
        Args:
            symbol: Yahoo Finance symbol for S&P 500
            lookback_days: Number of days to look back (more reliable than period strings)
        """
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.raw_data = {}
        self.processed_data = None
        self.feature_pipeline = None
        self.model = None
        self.feature_names = []
        
    def fetch_market_data(self):
        """Fetch multiple market data sources"""
        print(f"Fetching {self.lookback_days} days of market data...")
        
        # Calculate start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        # Primary data sources
        symbols = {
            'SP500': self.symbol,
            'VIX': '^VIX',  # Market Volatility Index
            'TNX': '^TNX',  # 10-year Treasury
            'DXY': 'DX-Y.NYB',  # US Dollar Index
            'GOLD': 'GC=F',  # Gold futures
            'OIL': 'CL=F',   # Oil futures
        }
        
        for name, symbol in symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    self.raw_data[name] = data
                    print(f"✓ {name}: {len(data)} days")
                else:
                    print(f"✗ {name}: No data available")
                    
            except Exception as e:
                print(f"✗ {name}: Error fetching data - {e}")
        
        if 'SP500' not in self.raw_data:
            raise ValueError("Could not fetch S&P 500 data")
            
        print(f"Successfully fetched data for {len(self.raw_data)} sources")
        return self.raw_data
    
    def create_price_features(self, df, prefix=''):
        """Create price-based features using vectorized operations"""
        features = pd.DataFrame(index=df.index)
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume'] if 'Volume' in df.columns else None
        
        # Returns (multiple options)
        features[f'{prefix}return_1d'] = close.pct_change(1)
        features[f'{prefix}return_5d'] = close.pct_change(5)
        features[f'{prefix}return_10d'] = close.pct_change(10)
        features[f'{prefix}return_20d'] = close.pct_change(20)
        
        # Log returns
        features[f'{prefix}log_return'] = np.log(close / close.shift(1))
        
        # Price position features
        features[f'{prefix}high_low_ratio'] = high / low
        features[f'{prefix}close_high_ratio'] = close / high
        features[f'{prefix}close_low_ratio'] = close / low
        
        # Rolling statistics (using multiple windows)
        windows = [5, 10, 20, 50]
        
        for window in windows:
            # Moving averages
            ma = close.rolling(window=window, min_periods=window//2).mean()
            features[f'{prefix}ma_{window}_ratio'] = close / ma
            features[f'{prefix}ma_{window}_diff'] = (close - ma) / close
            
            # Volatility
            vol = features[f'{prefix}return_1d'].rolling(window=window, min_periods=window//2).std()
            features[f'{prefix}volatility_{window}d'] = vol
            
            # Price momentum
            features[f'{prefix}momentum_{window}d'] = close / close.shift(window) - 1
            
            # High-low range volatility
            hl_range = (high - low) / close
            features[f'{prefix}hl_volatility_{window}d'] = hl_range.rolling(window=window, min_periods=window//2).mean()
        
        # Volume features (if available)
        if volume is not None:
            features[f'{prefix}volume_ma_ratio'] = volume / volume.rolling(window=20, min_periods=10).mean()
            features[f'{prefix}volume_std'] = volume.rolling(window=20, min_periods=10).std()
            
            # Price-volume relationship
            features[f'{prefix}pv_corr'] = features[f'{prefix}return_1d'].rolling(window=20, min_periods=10).corr(
                volume.pct_change().rolling(window=20, min_periods=10).mean()
            )
        
        return features
    
    def create_technical_indicators(self, df, prefix=''):
        """Create technical indicators using efficient calculations"""
        features = pd.DataFrame(index=df.index)
        close = df['Close']
        
        # RSI(Relative Strength Index: magnitude of recent price changes)
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14, min_periods=7).mean()
        avg_loss = loss.rolling(window=14, min_periods=7).mean()
        
        rs = avg_gain / avg_loss
        features[f'{prefix}rsi'] = 100 - (100 / (1 + rs))
        
        # MACD(Moving Average Convergence Divergence: trend-following momentum indicator)
        exp12 = close.ewm(span=12).mean()
        exp26 = close.ewm(span=26).mean()
        features[f'{prefix}macd'] = exp12 - exp26
        features[f'{prefix}macd_signal'] = features[f'{prefix}macd'].ewm(span=9).mean()
        features[f'{prefix}macd_histogram'] = features[f'{prefix}macd'] - features[f'{prefix}macd_signal']
        
        # Bollinger Bands(upper and lower bands around a moving average)
        bb_window = 20
        bb_ma = close.rolling(window=bb_window, min_periods=bb_window//2).mean()
        bb_std = close.rolling(window=bb_window, min_periods=bb_window//2).std()
        
        features[f'{prefix}bb_upper'] = bb_ma + (bb_std * 2)
        features[f'{prefix}bb_lower'] = bb_ma - (bb_std * 2)
        features[f'{prefix}bb_position'] = (close - features[f'{prefix}bb_lower']) / (features[f'{prefix}bb_upper'] - features[f'{prefix}bb_lower'])
        features[f'{prefix}bb_width'] = (features[f'{prefix}bb_upper'] - features[f'{prefix}bb_lower']) / bb_ma
        
        # Stochastic Oscillator(closing price relative to high-low range)
        low_min = df['Low'].rolling(window=14, min_periods=7).min()
        high_max = df['High'].rolling(window=14, min_periods=7).max()
        features[f'{prefix}stoch_k'] = 100 * (close - low_min) / (high_max - low_min)
        features[f'{prefix}stoch_d'] = features[f'{prefix}stoch_k'].rolling(window=3, min_periods=2).mean()
        
        return features
    
    def create_cross_asset_features(self):
        """Create features from relationships between different assets"""
        features = pd.DataFrame()
        
        # Ensure we have enough data to create features
        if len(self.raw_data) < 2:
            return features
        
        # Get common date range
        common_dates = None
        for data in self.raw_data.values():
            if common_dates is None:
                common_dates = data.index
            else:
                common_dates = common_dates.intersection(data.index)
        
        if len(common_dates) == 0:
            return features
        
        # Align all data to common dates
        aligned_data = {}
        for name, data in self.raw_data.items():
            aligned_data[name] = data.reindex(common_dates)
        
        features = pd.DataFrame(index=common_dates)
        
        # VIX features (fear gauge)
        if 'VIX' in aligned_data:
            vix_close = aligned_data['VIX']['Close']
            features['vix_level'] = vix_close
            features['vix_change'] = vix_close.pct_change()
            features['vix_ma_ratio'] = vix_close / vix_close.rolling(window=20, min_periods=10).mean()
            
            # VIX-SPX relationship
            if 'SP500' in aligned_data:
                sp_returns = aligned_data['SP500']['Close'].pct_change()
                features['vix_sp_corr'] = vix_close.rolling(window=20, min_periods=10).corr(sp_returns)
        
        # Interest rate features
        if 'TNX' in aligned_data:
            tnx_close = aligned_data['TNX']['Close']
            features['tnx_level'] = tnx_close
            features['tnx_change'] = tnx_close.diff()
            features['tnx_momentum'] = tnx_close / tnx_close.shift(20) - 1
        
        # Dollar strength
        if 'DXY' in aligned_data:
            dxy_close = aligned_data['DXY']['Close']
            features['dxy_level'] = dxy_close
            features['dxy_change'] = dxy_close.pct_change()
            features['dxy_momentum'] = dxy_close / dxy_close.shift(20) - 1
        
        # Commodity features
        for commodity in ['GOLD', 'OIL']:
            if commodity in aligned_data:
                comm_close = aligned_data[commodity]['Close']
                features[f'{commodity.lower()}_return'] = comm_close.pct_change()
                features[f'{commodity.lower()}_momentum'] = comm_close / comm_close.shift(20) - 1
        
        return features
    
    def build_feature_matrix(self, target_days=30):
        """Build complete feature matrix using pipeline approach"""
        print("Building feature matrix...")
        
        if 'SP500' not in self.raw_data:
            raise ValueError("SP500 data not available")
        
        sp500_data = self.raw_data['SP500'].copy()
        
        # Create feature components
        price_features = self.create_price_features(sp500_data, prefix='sp_')
        tech_features = self.create_technical_indicators(sp500_data, prefix='sp_')
        cross_features = self.create_cross_asset_features()
        
        # Combine all features
        feature_dfs = [price_features, tech_features]
        if not cross_features.empty:
            feature_dfs.append(cross_features)
        
        # Merge on common index
        combined_features = feature_dfs[0]
        for df in feature_dfs[1:]:
            combined_features = combined_features.join(df, how='inner')
        
        # Create target variable
        sp500_prices = sp500_data['Close'].reindex(combined_features.index)
        target = sp500_prices.shift(-target_days) / sp500_prices - 1
        
        # Remove samples where target is not available
        valid_mask = ~target.isna()
        combined_features = combined_features[valid_mask]
        target = target[valid_mask]
        
        # Handle remaining NaN values using forward fill
        combined_features = combined_features.fillna(method='ffill').fillna(method='bfill')
        
        # Replace/Remove any remaining NaN rows
        final_mask = ~combined_features.isna().any(axis=1)
        combined_features = combined_features[final_mask]
        target = target[final_mask]
        
        # Remove infinite values
        inf_mask = np.isinf(combined_features).any(axis=1)
        if inf_mask.any():
            combined_features = combined_features[~inf_mask]
            target = target[~inf_mask]
        
        # Reset index to ensure dates are aligned
        self.processed_data = {
            'features': combined_features,
            'target': target,
            'feature_names': list(combined_features.columns),
            'dates': combined_features.index
        }
        
        print(f"Feature matrix shape: {combined_features.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Date range: {combined_features.index[0].date()} to {combined_features.index[-1].date()}")
        
        return self.processed_data
    
    def create_preprocessing_pipeline(self):
        """Create preprocessing pipeline"""
        
        # Separate features by type for different preprocessing
        price_features = [col for col in self.processed_data['feature_names'] if 'return' in col or 'momentum' in col]
        ratio_features = [col for col in self.processed_data['feature_names'] if 'ratio' in col or 'position' in col]
        level_features = [col for col in self.processed_data['feature_names'] if 'level' in col or 'vix' in col or 'rsi' in col]
        other_features = [col for col in self.processed_data['feature_names'] 
                         if col not in price_features + ratio_features + level_features]
        
        # Create preprocessing steps
        preprocessors = []
        
        if price_features:
            preprocessors.append(('price_scaler', RobustScaler(), price_features))
        if ratio_features:
            preprocessors.append(('ratio_scaler', MinMaxScaler(), ratio_features))
        if level_features:
            preprocessors.append(('level_scaler', RobustScaler(), level_features))
        if other_features:
            preprocessors.append(('other_scaler', RobustScaler(), other_features))
        
        self.feature_pipeline = ColumnTransformer(preprocessors, remainder='passthrough')
        
        print(f"Created preprocessing pipeline with {len(preprocessors)} components")
        return self.feature_pipeline
    
    def train_model_with_cv(self, n_splits=5):
        """Train model using time series cross-validation"""
        if self.processed_data is None:
            raise ValueError("No processed data. Call build_feature_matrix() first.")
        
        X = self.processed_data['features']
        y = self.processed_data['target']
        
        print(f"Training with {len(X)} samples and {X.shape[1]} features")
        
        # Create time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # XGBoost model with regularization settings
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42
        )
        
        # Create full pipeline
        if self.feature_pipeline is None:
            self.create_preprocessing_pipeline()
        
        pipeline = Pipeline([
            ('preprocessor', self.feature_pipeline),
            ('model', model)
        ])
        
        # Cross-validation scores
        cv_scores = []
        fold_predictions = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"Training fold {fold + 1}/{n_splits}")
            
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Fit pipeline
            pipeline.fit(X_train_fold, y_train_fold)
            
            # Predict
            y_pred_fold = pipeline.predict(X_val_fold)
            
            # Score
            score = r2_score(y_val_fold, y_pred_fold)
            cv_scores.append(score)
            
            fold_predictions.append({
                'dates': X_val_fold.index,
                'actual': y_val_fold,
                'predicted': y_pred_fold,
                'score': score
            })
        
        # Train final model on all data
        self.model = pipeline.fit(X, y)
        
        print(f"\nCross-validation R² scores: {cv_scores}")
        print(f"Mean CV R²: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return cv_scores, fold_predictions
    
    def create_comprehensive_plots(self, cv_results=None):
        """Create comprehensive visualization plots in a single window"""
        importances = None
        if self.model is None or self.processed_data is None:
            print("Model or data not available for plotting.")
            return

        # Prepare data
        X = self.processed_data['features']
        y = self.processed_data['target']
        y_pred = self.model.predict(X)
        dates = self.processed_data['dates']
        feature_names = self.processed_data['feature_names']

        # Feature importance (from XGBoost)
        try:
            importances = self.model.named_steps['model'].feature_importances_
        except Exception:
            importances = np.zeros(len(feature_names))

        # Cross-validation R² scores
        cv_scores = []
        if cv_results is not None and isinstance(cv_results, tuple):
            cv_scores = cv_results[0]

        # Cumulative returns
        actual_cum = (1 + y).cumprod()
        pred_cum = (1 + pd.Series(y_pred, index=y.index)).cumprod()

        # Rolling correlation
        rolling_corr = pd.Series(y_pred, index=y.index).rolling(60, min_periods=20).corr(y)

        # Set up the figure
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))
        fig.suptitle("Advanced S&P 500 Prediction Model Diagnostics", fontsize=18, fontweight='bold')

        # 1. Cross-validation R² Scores (Bar Plot)
        ax = axs[0, 0]
        if cv_scores:
            ax.bar(range(1, len(cv_scores) + 1), cv_scores, color='skyblue')
            ax.set_title("Cross-validation R² Scores")
            ax.set_xlabel("Fold")
            ax.set_ylabel("R² Score")
            ax.axhline(np.mean(cv_scores), color='red', linestyle='--', label='Mean R²')
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No CV results", ha='center', va='center')
            ax.set_axis_off()

        # 2. Feature Importance (Bar Plot)
        ax = axs[0, 1]
        sorted_idx = np.argsort(importances)[::-1][:15]
        ax.barh(np.array(feature_names)[sorted_idx][::-1], importances[sorted_idx][::-1], color='teal')
        ax.set_title("Top 15 Feature Importances")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.invert_yaxis()

        # 3. Actual vs. Predicted Returns (Scatter Plot)
        ax = axs[1, 0]
        sns.scatterplot(x=y, y=y_pred, ax=ax, alpha=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax.set_title("Actual vs. Predicted Returns")
        ax.set_xlabel("Actual Return")
        ax.set_ylabel("Predicted Return")

        # 4. Residuals Distribution (Histogram)
        ax = axs[1, 1]
        residuals = y - y_pred
        sns.histplot(residuals, bins=40, kde=True, ax=ax, color='purple')
        ax.set_title("Residuals Distribution")
        ax.set_xlabel("Residual (Actual - Predicted)")
        ax.set_ylabel("Frequency")

        # 5. Cumulative Returns Comparison (Line Plot)
        ax = axs[2, 0]
        ax.plot(dates, actual_cum, label='Actual', color='black')
        ax.plot(dates, pred_cum, label='Predicted', color='orange', alpha=0.8)
        ax.set_title("Cumulative Returns Comparison")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend()

        # 6. Rolling Correlation (Line Plot)
        ax = axs[2, 1]
        ax.plot(dates, rolling_corr, color='green')
        ax.set_title("Rolling 60-Day Correlation (Actual vs. Predicted)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Correlation")
        ax.axhline(0, color='gray', linestyle='--', lw=1)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()
        

    def make_prediction(self, days_ahead=30):
        """Make prediction for future returns"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Get latest features
        latest_features = self.processed_data['features'].iloc[-1:].copy()
        
        # Make prediction
        prediction = self.model.predict(latest_features)[0]
        
        # Get current price
        current_price = self.raw_data['SP500']['Close'].iloc[-1]
        predicted_price = current_price * (1 + prediction)
        
        # Calculate confidence intervals (simple approach using historical errors)
        # In practice, you might want to use more sophisticated methods
        X = self.processed_data['features']
        y = self.processed_data['target']
        all_predictions = self.model.predict(X)
        prediction_std = np.std(y - all_predictions)
        
        confidence_lower = prediction - 1.96 * prediction_std
        confidence_upper = prediction + 1.96 * prediction_std
        
        print(f"\n{days_ahead}-Day Prediction Results:")
        print(f"Current S&P 500 Price: ${current_price:.2f}")
        print(f"Predicted Return: {prediction*100:.2f}%")
        print(f"95% Confidence Interval: [{confidence_lower*100:.2f}%, {confidence_upper*100:.2f}%]")
        print(f"Predicted Price: ${predicted_price:.2f}")
        print(f"Price Range: ${current_price * (1 + confidence_lower):.2f} - ${current_price * (1 + confidence_upper):.2f}")
        
        return {
            'prediction': prediction,
            'predicted_price': predicted_price,
            'confidence_interval': (confidence_lower, confidence_upper),
            'current_price': current_price
        }
    
    def run_complete_analysis(self, target_days=30):
        """Run the complete analysis pipeline"""
        print("Starting Advanced S&P 500 Prediction Analysis")
        print("=" * 60)
        
        # 1. Fetch data
        self.fetch_market_data()
        
        # 2. Build features
        self.build_feature_matrix(target_days=target_days)
        
        # 3. Train with cross-validation
        cv_results = self.train_model_with_cv()
        
        # 5. Make prediction
        prediction_results = self.make_prediction(target_days)

        # 4. Create visualizations
        self.create_comprehensive_plots(cv_results)
        
        return cv_results, prediction_results


# 1500 days lookback 
if __name__ == "__main__":
    try:
        # Initialize with more data
        predictor = AdvancedSP500Predictor(lookback_days=2000)  
        
        # Run complete analysis
        cv_results, prediction_results = predictor.run_complete_analysis(target_days=30)
        
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        cv_scores, _ = cv_results
        print(f"Model Performance: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f} R²")
        print(f"Prediction: {prediction_results['prediction']*100:.2f}% return")
        print(f"Target Price: ${prediction_results['predicted_price']:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying with reduced lookback period...")
        try:
            predictor = AdvancedSP500Predictor(lookback_days=800)
            cv_results, prediction_results = predictor.run_complete_analysis(target_days=21)
        except Exception as e2:
            print(f"Second attempt failed: {e2}")

