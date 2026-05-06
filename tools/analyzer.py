"""
Performance Analyzer - Performance Analysis and Feedback Generation

Used to analyze model performance, diagnose issues, and generate improvement suggestions
"""
import numpy as np
from typing import Dict, List, Tuple


class PerformanceAnalyzer:
    """Performance Analyzer"""

    def __init__(self):
        self.history = []

    def analyze(self, predictions: np.ndarray, actuals: np.ndarray,
                generation: int) -> Dict:
        """
        Analyze performance and generate feedback

        Args:
            predictions: Predicted values
            actuals: Actual values
            generation: Current generation number

        Returns:
            Dictionary containing performance metrics and improvement suggestions
        """
        # Calculate basic metrics
        mse = np.mean((predictions - actuals) ** 2)
        rmse = sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))

        # R²
        ss_res = np.sum((predictions - actuals) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Diagnose issues
        issues = []
        suggestions = []

        # Feature-level diagnosis (select feature names based on dataset)
        feature_names = []

        # Try to get feature names from environment (if available)
        if hasattr(self, 'env') and hasattr(self.env, 'get_feature_names'):
            feature_names = self.env.get_feature_names()
        else:
            # Infer dataset type based on dimensions
            if len(predictions.shape) == 2:
                state_dim = predictions.shape[1]
                if state_dim == 16:
                    # flood_ts
                    feature_names = [
                        'WATER_0', 'WATER_1', 'WATER_2', 'WATER_3', 'WATER_4', 'WATER_5', 'WATER_6',
                        'TIDE', 'GATE_0', 'GATE_1', 'GATE_2', 'GATE_3', 'GATE_4',
                        'PUMP_0', 'PUMP_1', 'RAIN'
                    ]
                elif state_dim == 4:
                    # cancer_ts
                    feature_names = ['tumor_volume', 'chemo_concentration', 'chemo_dosage', 'radio_dosage']
                elif state_dim == 5:
                    # covid_ts
                    feature_names = ['susceptible', 'exposed', 'infected', 'recovered', 'intervention']
                elif state_dim == 10:
                    # surgery_ts
                    feature_names = ['MBP', 'SBP', 'DBP', 'SpO2', 'HR', 'ETCO2', 'EPI', 'NEPI', 'PHEN', 'VASO']
                else:
                    # Use generic names by default
                    feature_names = [f'feature_{i}' for i in range(state_dim)]

        if len(predictions.shape) == 2 and feature_names:
            feature_mse = np.mean((predictions - actuals) ** 2, axis=0)
            issues.append(f'Feature-level MSE breakdown:')
            for i, (name, fmse) in enumerate(zip(feature_names, feature_mse)):
                if fmse > 0.5:
                    issues.append(f'  - {name}: {fmse:.4f} (CRITICAL - needs urgent improvement)')
                    suggestions.append(f'PRIORITY: Focus on improving {name} predictions - model is failing badly')
                    suggestions.append(f'For {name}: Add stronger interactions with related features and improve dynamics modeling')
                elif fmse > 0.2:
                    issues.append(f'  - {name}: {fmse:.4f} (needs improvement)')
                    suggestions.append(f'Improve {name} predictions by fine-tuning coefficients and adding interactions')
                else:
                    issues.append(f'  - {name}: {fmse:.4f} (good)')

        # 1. MSE diagnosis - more specific suggestions
        if mse > 0.5:
            issues.append(f'CRITICAL: High overall MSE ({mse:.4f}) - model is severely underfitting')
            suggestions.append('URGENT: Add state interaction terms between adjacent water stations')
            suggestions.append('IMPLEMENT: Complex water flow dynamics with upstream-downstream coupling')
            suggestions.append('ADD: Gate-pump-water station couplings with multiplicative interactions')
            suggestions.append('INCREASE: Model complexity with more parameters and nonlinear terms')
            suggestions.append('USE: Quadratic and cubic terms for feature interactions')
        elif mse > 0.2:
            issues.append(f'Moderate overall MSE ({mse:.4f}) - model fit needs improvement')
            suggestions.append('Fine-tune the water flow coefficients for better accuracy')
            suggestions.append('Add nonlinear terms like quadratic or cubic interactions')
            suggestions.append('Adjust tide coupling strength - stronger for downstream stations')
            suggestions.append('Optimize rain effect coefficients - ensure proportional distribution')

        # 2. Variance diagnosis - more specific suggestions
        pred_std = np.std(predictions)
        actual_std = np.std(actuals)
        if pred_std < actual_std * 0.5:
            issues.append(f'Low variance (pred={pred_std:.4f}, actual={actual_std:.4f}) - predictions too smooth')
            suggestions.append('Add multiplicative interaction terms between states')
            suggestions.append('Reduce damping coefficients in the water stations')
            suggestions.append('Introduce more nonlinear responses to gates and pumps')
            suggestions.append('Add sinusoidal terms for tide effects')
        elif pred_std > actual_std * 1.5:
            issues.append(f'High variance (pred={pred_std:.4f}, actual={actual_std:.4f}) - predictions too volatile')
            suggestions.append('Increase damping on water station dynamics')
            suggestions.append('Add smoothing terms to gate and pump inputs')
            suggestions.append('Clip extreme values in the state derivatives')
            suggestions.append('Reduce the strength of nonlinear terms')

        # 3. Bias diagnosis - more specific suggestions
        mean_error = np.mean(predictions - actuals)
        if abs(mean_error) > 0.3:
            if mean_error > 0:
                issues.append(f'Systematic overestimation (bias={mean_error:.4f})')
                suggestions.append('Add negative correction terms to water station dynamics')
                suggestions.append('Reduce positive coefficients in the water flow equations')
                suggestions.append('Adjust gate and pump effect coefficients downward')
                suggestions.append('Consider adding bias correction for tide effects')
            else:
                issues.append(f'Systematic underestimation (bias={mean_error:.4f})')
                suggestions.append('Add positive driving terms to water station dynamics')
                suggestions.append('Increase positive coefficients in the water flow equations')
                suggestions.append('Adjust gate and pump effect coefficients upward')
                suggestions.append('Enhance rain effect on water levels')

        # 4. R² diagnosis
        if r2 < 0.5:
            issues.append(f'Low R² ({r2:.3f}) - model explains little variance')
            suggestions.append('Consider adding missing state variables or interactions')
        elif r2 < 0.7:
            issues.append(f'Moderate R² ({r2:.3f}) - room for improvement')
            suggestions.append('Try capturing additional dynamics')

        # 5. Outlier detection
        errors = predictions - actuals
        outlier_ratio = np.sum(np.abs(errors) > 2 * np.std(errors)) / len(errors)
        if outlier_ratio > 0.1:
            issues.append(f'High outlier ratio ({outlier_ratio:.1%})')
            suggestions.append('Add robustness terms or clip extreme values')

        # Execute statistical descriptor analysis
        statistical_descriptors = {}

        # Temporal lag detection
        lag_result = self.temporal_lag_detection(predictions, actuals)
        statistical_descriptors['temporal_lag'] = lag_result
        if lag_result['lag_detected']:
            issues.append(f'Temporal lag detected: {lag_result["lag_value"]} steps')
            suggestions.extend(lag_result['suggestions'])

        # Trend analysis
        trend_result = self.trend_analysis(predictions, actuals)
        statistical_descriptors['trend'] = trend_result
        if trend_result['trend_detected']:
            issues.append(f'Trend mismatch: {trend_result["direction"]}')
            suggestions.extend(trend_result['suggestions'])

        # Periodicity detection
        periodicity_result = self.periodicity_detection(predictions, actuals)
        statistical_descriptors['periodicity'] = periodicity_result
        if periodicity_result['periodicity_detected']:
            issues.append(f'Periodicity mismatch: actual {periodicity_result["estimated_period"]}')
            suggestions.extend(periodicity_result['suggestions'])

        # Abrupt change detection
        abrupt_change_result = self.abrupt_change_detection(predictions, actuals)
        statistical_descriptors['abrupt_change'] = abrupt_change_result
        if abrupt_change_result['abrupt_changes_detected']:
            issues.append(f'Abrupt changes not captured: {len(abrupt_change_result["change_points"])} points')
            suggestions.extend(abrupt_change_result['suggestions'])

        # Systematic bias quantification
        bias_result = self.systematic_bias_quantification(predictions, actuals)
        statistical_descriptors['bias'] = bias_result
        if bias_result['bias_detected']:
            issues.append(f'Systematic bias: {bias_result["direction"]}')
            suggestions.extend(bias_result['suggestions'])

        # Calculate overall score
        scores = []
        for descriptor_name, descriptor_result in statistical_descriptors.items():
            if 'score' in descriptor_result:
                scores.append(descriptor_result['score'])

        if scores:
            overall_score = np.mean(scores)
        else:
            overall_score = 0.0

        # Record history
        self.history.append({
            'generation': generation,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'score': overall_score,
            'issues': issues.copy(),
            'suggestions': suggestions.copy(),
            'statistical_descriptors': statistical_descriptors.copy()
        })

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'score': overall_score,
            'issues': issues,
            'suggestions': suggestions,
            'pred_std': pred_std,
            'actual_std': actual_std,
            'mean_error': mean_error,
            'statistical_descriptors': statistical_descriptors
        }

    def get_trend_analysis(self, window: int = 5) -> Dict:
        """
        Analyze recent trends

        Args:
            window: Analysis window size

        Returns:
            Trend analysis results
        """
        if len(self.history) < 2:
            return {'trend': 'insufficient_data', 'suggestions': []}

        recent = self.history[-window:]
        mse_values = [h['mse'] for h in recent]

        # Calculate trend
        if len(mse_values) >= 2:
            mse_change = mse_values[-1] - mse_values[0]
            mse_change_pct = mse_change / mse_values[0] * 100 if mse_values[0] > 0 else 0
        else:
            mse_change = 0
            mse_change_pct = 0

        # Determine trend
        if mse_change < -0.01:
            trend = 'improving'
        elif mse_change > 0.01:
            trend = 'degrading'
        else:
            trend = 'stable'

        # Detect stagnation
        unique_mse = len(set([round(m, 4) for m in mse_values]))
        stagnation = unique_mse == 1

        # Generate suggestions
        suggestions = []
        if stagnation:
            suggestions.append('Performance stagnating - try different model structure')
            suggestions.append('Consider adding new state interactions')
        if trend == 'degrading':
            suggestions.append('Performance degrading - consider simpler model')
            suggestions.append('Return to previous best configuration')
        if trend == 'improving':
            suggestions.append('Good progress - continue current direction')

        return {
            'trend': trend,
            'mse_change': mse_change,
            'mse_change_pct': mse_change_pct,
            'stagnation': stagnation,
            'suggestions': suggestions
        }

    def temporal_lag_detection(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict:
        """
        Temporal lag detection: Use cross-correlation to identify systematic delays in model response

        Args:
            predictions: Predicted values
            actuals: Actual values

        Returns:
            Dictionary containing lag information and score
        """
        if len(predictions) < 2 or len(actuals) < 2:
            return {
                'lag_detected': False,
                'lag_value': 0,
                'confidence': 0.0,
                'score': 0.0,
                'suggestions': []
            }

        # Calculate cross-correlation
        max_lag = min(10, len(predictions) // 2)
        lags = np.arange(-max_lag, max_lag + 1)
        correlations = []

        for lag in lags:
            if lag < 0:
                corr = np.corrcoef(predictions[-lag:], actuals[:lag])[0, 1]
            elif lag > 0:
                corr = np.corrcoef(predictions[:-lag], actuals[lag:])[0, 1]
            else:
                corr = np.corrcoef(predictions, actuals)[0, 1]

            if not np.isnan(corr):
                correlations.append(corr)
            else:
                correlations.append(0.0)

        # Find position of maximum correlation value
        max_corr_idx = np.argmax(np.abs(correlations))
        max_lag_value = lags[max_corr_idx]
        max_corr = correlations[max_corr_idx]

        # Calculate confidence
        confidence = np.abs(max_corr)
        lag_detected = confidence > 0.5 and max_lag_value != 0

        # Calculate score: Based on confidence and lag detection result
        if not lag_detected:
            score = 1.0  # No lag detected, high score
        else:
            # More severe lag, lower score
            lag_magnitude = abs(max_lag_value)
            score = max(0.0, 1.0 - (lag_magnitude * 0.1) - (1 - confidence) * 0.5)

        # Generate suggestions - more specific suggestions
        suggestions = []
        if lag_detected:
            if max_lag_value > 0:
                suggestions.append(f'Model lags behind actuals by {max_lag_value} steps')
                suggestions.append(f'Add lagged terms of previous {max_lag_value} states')
                suggestions.append('Implement memory terms like dx_dt[i] += 0.3 * state[i-1]')
                suggestions.append('Consider adding delay buffers for water station dynamics')
                suggestions.append('Introduce delayed gate/pump effects')
            else:
                suggestions.append(f'Model predicts ahead of actuals by {-max_lag_value} steps')
                suggestions.append('Reduce forward-looking terms in the model')
                suggestions.append('Increase damping on future state predictions')
                suggestions.append('Avoid using current state for future predictions')

        return {
            'lag_detected': lag_detected,
            'lag_value': max_lag_value,
            'confidence': confidence,
            'score': score,
            'suggestions': suggestions
        }

    def trend_analysis(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict:
        """
        Trend analysis: Perform linear regression to detect linear trends and return related information

        Args:
            predictions: Predicted values
            actuals: Actual values

        Returns:
            Dictionary containing trend information and score
        """
        if len(predictions) < 3 or len(actuals) < 3:
            return {
                'trend_detected': False,
                'slope_magnitude': 0.0,
                'direction': 'none',
                'r_squared': 0.0,
                'score': 0.0,
                'suggestions': []
            }

        # Prepare data
        x = np.arange(len(predictions)).reshape(-1, 1)

        # Handle 2D arrays: calculate mean of all features
        if len(predictions.shape) == 2:
            y_pred = np.mean(predictions, axis=1).reshape(-1, 1)
            y_actual = np.mean(actuals, axis=1).reshape(-1, 1)
        else:
            y_pred = predictions.reshape(-1, 1)
            y_actual = actuals.reshape(-1, 1)

        # Linear regression
        from sklearn.linear_model import LinearRegression

        # Predicted value trends
        model_pred = LinearRegression().fit(x, y_pred)
        slope_pred = model_pred.coef_[0, 0]
        r2_pred = model_pred.score(x, y_pred)

        # Actual value trends
        model_actual = LinearRegression().fit(x, y_actual)
        slope_actual = model_actual.coef_[0, 0]
        r2_actual = model_actual.score(x, y_actual)

        # Calculate trend difference
        slope_diff = slope_pred - slope_actual
        slope_magnitude = abs(slope_diff)

        # Determine direction
        if slope_diff > 0.01:
            direction = 'overestimating'
        elif slope_diff < -0.01:
            direction = 'underestimating'
        else:
            direction = 'matching'

        trend_detected = slope_magnitude > 0.01 and (r2_pred > 0.3 or r2_actual > 0.3)

        # Calculate score: Based on trend difference and R² values
        if not trend_detected:
            score = 1.0  # Trends match, high score
        else:
            # Smaller trend difference, higher score
            score = max(0.0, 1.0 - slope_magnitude * 10 - (1 - max(r2_pred, r2_actual)) * 0.5)

        # Generate suggestions
        suggestions = []
        if trend_detected:
            if direction == 'overestimating':
                suggestions.append('Model overestimates trend - consider adjusting slope parameters')
                suggestions.append('Add negative feedback terms to counteract upward bias')
            elif direction == 'underestimating':
                suggestions.append('Model underestimates trend - consider increasing slope parameters')
                suggestions.append('Add positive feedback terms to capture upward momentum')

        return {
            'trend_detected': trend_detected,
            'slope_magnitude': slope_magnitude,
            'direction': direction,
            'r_squared': max(r2_pred, r2_actual),
            'score': score,
            'suggestions': suggestions
        }

    def periodicity_detection(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict:
        """
        Periodicity detection: Use autocorrelation to identify cyclic patterns and return related information

        Args:
            predictions: Predicted values
            actuals: Actual values

        Returns:
            Dictionary containing periodicity information and score
        """
        if len(predictions) < 10 or len(actuals) < 10:
            return {
                'periodicity_detected': False,
                'estimated_period': 0,
                'residual_periodicity': 0.0,
                'score': 0.0,
                'suggestions': []
            }

        # Calculate autocorrelation
        def compute_autocorrelation(data, max_lag):
            autocorrs = []
            for lag in range(1, max_lag + 1):
                corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorrs.append(corr)
                else:
                    autocorrs.append(0.0)
            return autocorrs

        max_lag = min(20, len(predictions) // 2)
        actual_autocorr = compute_autocorrelation(actuals, max_lag)
        pred_autocorr = compute_autocorrelation(predictions, max_lag)

        # Find main period
        if actual_autocorr:
            actual_period = np.argmax(actual_autocorr) + 1 if max(actual_autocorr) > 0.3 else 0
        else:
            actual_period = 0

        if pred_autocorr:
            pred_period = np.argmax(pred_autocorr) + 1 if max(pred_autocorr) > 0.3 else 0
        else:
            pred_period = 0

        # Calculate residual periodicity
        residual_periodicity = abs(actual_period - pred_period) / max(actual_period, 1)

        periodicity_detected = actual_period > 0 and abs(actual_period - pred_period) > 1

        # Calculate score: Based on residual periodicity
        if actual_period == 0:
            score = 1.0  # No periodicity detected, high score
        else:
            # Smaller residual, higher score
            score = max(0.0, 1.0 - residual_periodicity)

        # Generate suggestions
        suggestions = []
        if periodicity_detected:
            suggestions.append(f'Actual period: {actual_period}, Predicted period: {pred_period} - consider adjusting periodicity terms')
            suggestions.append('Add sinusoidal or periodic components to capture cyclical patterns')

        return {
            'periodicity_detected': periodicity_detected,
            'estimated_period': actual_period,
            'residual_periodicity': residual_periodicity,
            'score': score,
            'suggestions': suggestions
        }

    def abrupt_change_detection(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict:
        """
        Abrupt change detection: Identify sudden changes by analyzing absolute differences between consecutive points

        Args:
            predictions: Predicted values
            actuals: Actual values

        Returns:
            Dictionary containing abrupt change information and score
        """
        if len(predictions) < 3 or len(actuals) < 3:
            return {
                'abrupt_changes_detected': False,
                'change_points': [],
                'magnitude': 0.0,
                'score': 0.0,
                'suggestions': []
            }

        # Calculate consecutive differences
        actual_diff = np.abs(np.diff(actuals))
        pred_diff = np.abs(np.diff(predictions))

        # Detect change points
        threshold = np.mean(actual_diff) + 2 * np.std(actual_diff)
        change_points = np.where(actual_diff > threshold)[0]

        # Calculate difference between predicted and actual
        diff_diff = np.abs(actual_diff - pred_diff)
        magnitude = np.mean(diff_diff) if len(diff_diff) > 0 else 0.0

        abrupt_changes_detected = len(change_points) > 0 and magnitude > np.mean(actual_diff)

        # Calculate score: Based on abrupt change detection result and difference magnitude
        if len(change_points) == 0:
            score = 1.0  # No change points, high score
        else:
            # Smaller difference, higher score
            avg_actual_diff = np.mean(actual_diff)
            if avg_actual_diff > 0:
                score = max(0.0, 1.0 - (magnitude / avg_actual_diff))
            else:
                score = 1.0

        # Generate suggestions
        suggestions = []
        if abrupt_changes_detected:
            suggestions.append(f'Detected {len(change_points)} abrupt changes not captured by model')
            suggestions.append('Add nonlinear terms or threshold-based mechanisms to capture sudden changes')
            suggestions.append('Consider using piecewise models or change point detection')

        return {
            'abrupt_changes_detected': abrupt_changes_detected,
            'change_points': change_points.tolist(),
            'magnitude': magnitude,
            'score': score,
            'suggestions': suggestions
        }

    def systematic_bias_quantification(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict:
        """
        Systematic bias quantification: Detect consistent overestimation or underestimation and return related information

        Args:
            predictions: Predicted values
            actuals: Actual values

        Returns:
            Dictionary containing bias information and score
        """
        if len(predictions) < 5 or len(actuals) < 5:
            return {
                'bias_detected': False,
                'bias_magnitude': 0.0,
                'direction': 'none',
                'significant_bias_ratio': 0.0,
                'score': 0.0,
                'suggestions': []
            }

        # Calculate error
        errors = predictions - actuals
        mean_error = np.mean(errors)
        bias_magnitude = abs(mean_error)

        # Determine direction
        if mean_error > 0.1:
            direction = 'overestimation'
        elif mean_error < -0.1:
            direction = 'underestimation'
        else:
            direction = 'none'

        # Calculate significant bias ratio
        significant_bias_threshold = 0.2 * np.std(actuals)
        significant_bias_count = np.sum(np.abs(errors) > significant_bias_threshold)
        significant_bias_ratio = significant_bias_count / len(errors)

        bias_detected = bias_magnitude > 0.1 or significant_bias_ratio > 0.3

        # Calculate score: Based on bias magnitude and significant bias ratio
        actual_std = np.std(actuals)
        if actual_std > 0:
            # Ratio of bias relative to actual value standard deviation
            normalized_bias = bias_magnitude / actual_std
            score = max(0.0, 1.0 - normalized_bias - significant_bias_ratio)
        else:
            score = 1.0

        # Generate suggestions
        suggestions = []
        if bias_detected:
            if direction == 'overestimation':
                suggestions.append('Model systematically overestimates - consider adding negative bias correction')
                suggestions.append('Adjust model parameters to reduce positive bias')
            elif direction == 'underestimation':
                suggestions.append('Model systematically underestimates - consider adding positive bias correction')
                suggestions.append('Adjust model parameters to reduce negative bias')
            if significant_bias_ratio > 0.3:
                suggestions.append(f'{significant_bias_ratio:.1%} of predictions show significant bias')
                suggestions.append('Consider adding adaptive bias correction mechanisms')

        return {
            'bias_detected': bias_detected,
            'bias_magnitude': bias_magnitude,
            'direction': direction,
            'significant_bias_ratio': significant_bias_ratio,
            'score': score,
            'suggestions': suggestions
        }


class CodeQualityAnalyzer:
    """Code Quality Analyzer"""

    def analyze(self, code: str) -> Dict:
        """
        Analyze code quality

        Args:
            code: Generated code

        Returns:
            Code quality report
        """
        feedback = {
            'syntax_valid': False,
            'dimension_match': False,
            'numerical_stability': False,
            'issues': [],
            'suggestions': []
        }

        # 1. Syntax check
        try:
            compile(code, '<string>', 'exec')
            feedback['syntax_valid'] = True
        except SyntaxError as e:
            feedback['issues'].append(f'Syntax error: {e}')
            feedback['suggestions'].append('Fix syntax errors before execution')
            return feedback

        # 2. Dimension consistency check
        if 'np.zeros_like(state)' not in code:
            feedback['issues'].append('Dimension mismatch risk')
            feedback['suggestions'].append('Use np.zeros_like(state) to ensure correct dimensions')
        else:
            feedback['dimension_match'] = True

        # 3. Numerical stability check
        unstable_patterns = []

        if 'np.exp(' in code and 'np.clip(' not in code:
            unstable_patterns.append('np.exp without clipping')
            feedback['issues'].append('Potential numerical overflow in exp()')
            feedback['suggestions'].append('Use np.clip() or np.exp(np.clip(...))')

        if 'np.log(' in code and 'np.maximum(' not in code:
            unstable_patterns.append('np.log without protection')
            feedback['issues'].append('Potential log of zero or negative')
            feedback['suggestions'].append('Use np.log(np.maximum(x, small_value))')

        if '1.0 /' in code or '/ state' in code:
            unstable_patterns.append('Division without protection')
            feedback['issues'].append('Potential division by zero')
            feedback['suggestions'].append('Add small epsilon to denominator')

        if not unstable_patterns:
            feedback['numerical_stability'] = True

        # 4. Best practice check
        if 'import numpy' not in code and 'import numpy as np' not in code:
            feedback['issues'].append('NumPy not imported')
            feedback['suggestions'].append('Add: import numpy as np')

        if 'return dx_dt' not in code:
            feedback['issues'].append('Missing return statement')
            feedback['suggestions'].append('Add: return dx_dt')

        return feedback


class FeedbackGenerator:
    """Feedback Generator - Integrate All Analyses"""

    def __init__(self):
        self.performance_analyzer = PerformanceAnalyzer()
        self.code_analyzer = CodeQualityAnalyzer()

    def generate_feedback(self, predictions: np.ndarray, actuals: np.ndarray,
                         code: str, generation: int) -> str:
        """
        Generate complete feedback

        Args:
            predictions: Predicted values
            actuals: Actual values
            code: Generated code
            generation: Current generation number

        Returns:
            Feedback text
        """
        # Performance analysis
        perf_feedback = self.performance_analyzer.analyze(predictions, actuals, generation)

        # Code quality analysis
        code_feedback = self.code_analyzer.analyze(code)

        # Trend analysis
        trend_analysis = self.performance_analyzer.get_trend_analysis()

        # Build feedback text
        feedback_text = f"""
Performance Analysis (Generation {generation}):

Metrics:
- MSE: {perf_feedback['mse']:.6f}
- RMSE: {perf_feedback['rmse']:.6f}
- MAE: {perf_feedback['mae']:.6f}
- R²: {perf_feedback['r2']:.3f}
- Score: {perf_feedback.get('score', 0.0):.3f}

Issues Identified:
"""

        # Add performance issues
        for issue in perf_feedback['issues']:
            feedback_text += f"- [Performance] {issue}\n"

        # Add code issues
        for issue in code_feedback['issues']:
            feedback_text += f"- [Code Quality] {issue}\n"

        feedback_text += "\nSuggestions for Improvement:\n"

        # Add performance suggestions
        for suggestion in perf_feedback['suggestions']:
            feedback_text += f"- {suggestion}\n"

        # Add code suggestions
        for suggestion in code_feedback['suggestions']:
            feedback_text += f"- {suggestion}\n"

        # Add statistical descriptor analysis
        if 'statistical_descriptors' in perf_feedback:
            feedback_text += "\nStatistical Descriptors Analysis:\n"
            stats = perf_feedback['statistical_descriptors']

            # Temporal lag detection
            if stats.get('temporal_lag', {}).get('lag_detected'):
                lag_info = stats['temporal_lag']
                feedback_text += f"- Temporal Lag: {lag_info['lag_value']} steps (confidence: {lag_info['confidence']:.2f})\n"

            # Trend analysis
            if stats.get('trend', {}).get('trend_detected'):
                trend_info = stats['trend']
                feedback_text += f"- Trend: {trend_info['direction']} (slope: {trend_info['slope_magnitude']:.4f})\n"

            # Periodicity detection
            if stats.get('periodicity', {}).get('periodicity_detected'):
                periodicity_info = stats['periodicity']
                feedback_text += f"- Periodicity: Actual {periodicity_info['estimated_period']} steps\n"

            # Abrupt change detection
            if stats.get('abrupt_change', {}).get('abrupt_changes_detected'):
                abrupt_info = stats['abrupt_change']
                feedback_text += f"- Abrupt Changes: {len(abrupt_info['change_points'])} points detected\n"

            # Systematic bias quantification
            if stats.get('bias', {}).get('bias_detected'):
                bias_info = stats['bias']
                feedback_text += f"- Systematic Bias: {bias_info['direction']} (magnitude: {bias_info['bias_magnitude']:.4f})\n"

        # Add trend analysis
        feedback_text += f"\nOverall Trend: {trend_analysis['trend']}\n"
        if trend_analysis.get('mse_change', 0) != 0:
            feedback_text += f"MSE Change: {trend_analysis.get('mse_change_pct', 0):+.2f}%\n"

        for suggestion in trend_analysis['suggestions']:
            feedback_text += f"- {suggestion}\n"

        return feedback_text

    def get_feedback_for_prompt(self, predictions: np.ndarray, actuals: np.ndarray,
                               code: str, generation: int) -> Dict:
        """
        Generate feedback dictionary for prompt use

        Returns:
            Feedback dictionary that can be used to build prompts
        """
        perf_feedback = self.performance_analyzer.analyze(predictions, actuals, generation)
        code_feedback = self.code_analyzer.analyze(code)
        trend_analysis = self.performance_analyzer.get_trend_analysis()

        feedback_dict = {
            'metrics': {
                'mse': perf_feedback['mse'],
                'rmse': perf_feedback['rmse'],
                'mae': perf_feedback['mae'],
                'r2': perf_feedback['r2'],
                'score': perf_feedback.get('score', 0.0)
            },
            'issues': perf_feedback['issues'] + code_feedback['issues'],
            'suggestions': perf_feedback['suggestions'] + code_feedback['suggestions'],
            'trend': trend_analysis['trend'],
            'mse_change_pct': trend_analysis['mse_change_pct']
        }

        # Add statistical descriptor information
        if 'statistical_descriptors' in perf_feedback:
            feedback_dict['statistical_descriptors'] = perf_feedback['statistical_descriptors']

        return feedback_dict


def sqrt(x):
    """Helper function to calculate square root"""
    return np.sqrt(x)
