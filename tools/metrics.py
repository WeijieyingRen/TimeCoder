"""
Evaluation Metrics Calculation Tools

Includes MSE, RMSE, MAE, R² and other metrics
"""
import numpy as np
from typing import Dict, Tuple


def calculate_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Squared Error (MSE)"""
    return np.mean((actual - predicted) ** 2)


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE)"""
    mse = calculate_mse(actual, predicted)
    return np.sqrt(mse)


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE)"""
    return np.mean(np.abs(actual - predicted))


def calculate_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Coefficient of Determination (R²)"""
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    r2 = 1 - (ss_res / ss_tot)
    return r2


def calculate_all_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate all evaluation metrics

    Returns:
        Dictionary containing all metrics
    """
    return {
        'mse': calculate_mse(actual, predicted),
        'rmse': calculate_rmse(actual, predicted),
        'mae': calculate_mae(actual, predicted),
        'r2': calculate_r2(actual, predicted)
    }


def calculate_metrics_with_stats(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """
    Calculate all metrics and their standard deviations

    Returns:
        Dictionary containing (mean, std) for each metric
    """
    metrics = calculate_all_metrics(actual, predicted)

    # For single samples, standard deviation is 0
    result = {}
    for name, value in metrics.items():
        result[name] = (value, 0.0)

    return result


def compute_statistics_from_multiple_runs(all_predictions: list, all_actuals: list) -> Dict[str, Tuple[float, float]]:
    """
    Calculate statistics from multiple runs

    Args:
        all_predictions: List of predicted values from multiple runs
        all_actuals: List of actual values from multiple runs

    Returns:
        Dictionary containing (mean, std) for all metrics
    """
    if len(all_predictions) != len(all_actuals):
        raise ValueError("Number of predicted and actual values must be equal")

    if len(all_predictions) == 0:
        return {}

    # Calculate metrics for each run
    all_metrics = []
    for pred, act in zip(all_predictions, all_actuals):
        metrics = calculate_all_metrics(np.array(act), np.array(pred))
        all_metrics.append(metrics)

    # Calculate statistics
    result = {}
    for metric_name in ['mse', 'rmse', 'mae', 'r2']:
        values = [m[metric_name] for m in all_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=0)  # Population standard deviation
        result[metric_name] = (mean_val, std_val)

    return result
