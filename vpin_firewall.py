"""
Microservice: VPIN Cybersecurity Firewall
Architecture Component: Noise Reduction & HFT Spoofing Defense
Mathematical/Theoretical Purpose:
Implements the VPIN model. Transitions data from chronological to volume-time. 
Updated to safely align volume bucket arrays with aggregated 1h dataframe lengths,
preventing negative padding index errors.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Tuple

class VPINFirewall:
    def __init__(self, volume_bucket_size: int = 50000, rolling_window: int = 50):
        self.volume_bucket_size = volume_bucket_size
        self.rolling_window = rolling_window

    def apply_bulk_volume_classification(self, price_series: pd.Series, volume_series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        if price_series.empty or volume_series.empty:
            return np.array([]), np.array([])
            
        price_diff = price_series.diff().fillna(0)
        std_dev = price_diff.std()
        
        if std_dev == 0 or np.isnan(std_dev):
            std_dev = 1e-8
            
        prob_buy = norm.cdf(price_diff / std_dev)
        buy_volume = volume_series * prob_buy
        sell_volume = volume_series * (1 - prob_buy)
        
        return buy_volume.values, sell_volume.values

    def calculate_vpin(self, buy_vol: np.ndarray, sell_vol: np.ndarray) -> pd.Series:
        if len(buy_vol) == 0:
            return pd.Series([])
            
        total_vol = buy_vol + sell_vol
        cumulative_vol = np.cumsum(total_vol)
        
        if len(cumulative_vol) == 0:
            return pd.Series([])

        max_cum_vol = cumulative_vol[-1]
        if max_cum_vol < self.volume_bucket_size:
            return pd.Series(np.zeros(len(buy_vol)))
            
        num_buckets = int(max_cum_vol // self.volume_bucket_size)
        if num_buckets < self.rolling_window:
            return pd.Series(np.zeros(len(buy_vol)))
            
        bucket_indices = np.searchsorted(cumulative_vol, np.arange(1, num_buckets + 1) * self.volume_bucket_size)
        
        vpin_values = []
        for i in range(self.rolling_window, len(bucket_indices)):
            start_idx = bucket_indices[i - self.rolling_window]
            end_idx = bucket_indices[i]
            
            window_buy = np.sum(buy_vol[start_idx:end_idx])
            window_sell = np.sum(sell_vol[start_idx:end_idx])
            
            imbalance = np.abs(window_buy - window_sell)
            total_window_vol = window_buy + window_sell
            
            vpin = imbalance / total_window_vol if total_window_vol > 0 else 0
            vpin_values.append(vpin)
            
        if not vpin_values:
            return pd.Series(np.zeros(len(buy_vol)))
            
        # FIX: Handle cases where 1h aggregated volume creates more buckets than rows
        diff = len(buy_vol) - len(vpin_values)
        if diff > 0:
            final_vpin = np.pad(vpin_values, (diff, 0), 'constant', constant_values=0)
        else:
            # Truncate the oldest buckets to perfectly match the dataframe length
            final_vpin = vpin_values[-len(buy_vol):]
            
        return pd.Series(final_vpin)

    def extract_cdf_zscore(self, vpin_series: pd.Series) -> float:
        if vpin_series.empty:
            return 0.0
            
        vpin_array = vpin_series.to_numpy()
        vpin_array = vpin_array[vpin_array > 0]
        
        if len(vpin_array) < 2:
            return 0.0
            
        mean_vpin = np.mean(vpin_array)
        std_vpin = np.std(vpin_array)
        current_vpin = vpin_series.iloc[-1]
        
        if std_vpin == 0 or np.isnan(std_vpin):
            return 0.0
            
        z_score = (current_vpin - mean_vpin) / std_vpin
        return float(norm.cdf(z_score))

    def execute_filter(self, df_close: pd.DataFrame, df_volume: pd.DataFrame) -> dict:
        results = {}
        for col in df_close.columns:
            buy_v, sell_v = self.apply_bulk_volume_classification(df_close[col], df_volume[col])
            vpin_series = self.calculate_vpin(buy_v, sell_v)
            toxicity_score = self.extract_cdf_zscore(vpin_series)
            results[col] = {
                'toxicity_zscore': toxicity_score,
                'adversarial_state': toxicity_score > 0.95
            }
        return results