"""
Evaluator Module - Execute Model Evaluation and Testing
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any


def print_baseline_results(strategy, env_name):
    """Print baseline prediction results"""
    if not hasattr(strategy, 'test_predictions') or not hasattr(strategy, 'test_actuals'):
        print("No prediction data available")
        return

    predictions = strategy.test_predictions
    actuals = strategy.test_actuals

    print(f"\n{'='*60}")
    print(f"Baseline Prediction Results - {env_name}")
    print(f"{'='*60}")

    if len(predictions) > 0 and len(actuals) > 0:
        n_steps_per_sample = 10
        n_total_entries = len(predictions)
        n_samples = n_total_entries // n_steps_per_sample

        if n_samples == 0:
            n_samples = n_total_entries
            n_steps_per_sample = 1

        print(f"Number of prediction samples: {n_samples}")
        print(f"Prediction steps per sample: {n_steps_per_sample}")

        if len(predictions.shape) == 2:
            n_features = predictions.shape[1]
            print(f"Number of features: {n_features}")

            feature_names = []
            if hasattr(strategy.env, 'get_feature_names'):
                feature_names = strategy.env.get_feature_names()

            vital_indices = [0, 1, 2, 3, 4]
            if n_features > 5:
                vital_indices.append(n_features - 1)

            for fi in vital_indices:
                if fi >= n_features:
                    continue

                fname = feature_names[fi] if fi < len(feature_names) else f"Feature{fi}"

                pred_finals = []
                actual_finals = []
                for s in range(n_samples):
                    idx = s * n_steps_per_sample + (n_steps_per_sample - 1)
                    if idx < len(predictions):
                        pred_finals.append(predictions[idx, fi])
                        actual_finals.append(actuals[idx, fi])

                pred_finals = np.array(pred_finals)
                actual_finals = np.array(actual_finals) if len(actual_finals) > 0 else np.array([])

                if len(pred_finals) == 0:
                    continue

                print(f"\n{fname} (Index {fi}) - Step 10 Prediction:")
                print(f"  Predicted: Mean={np.mean(pred_finals):.2f}, Std={np.std(pred_finals):.2f}")
                print(f"  Actual: Mean={np.mean(actual_finals):.2f}, Std={np.std(actual_finals):.2f}")

                print(f"  First 10 samples comparison:")
                for j in range(min(10, len(pred_finals))):
                    print(f"    Sample{j}: Pred={pred_finals[j]:.2f}, Actual={actual_finals[j]:.2f}")

                mae = np.mean(np.abs(pred_finals - actual_finals))
                rmse = np.sqrt(np.mean((pred_finals - actual_finals) ** 2))
                print(f"  MAE={mae:.2f}, RMSE={rmse:.2f}")
        else:
            print("Prediction data dimension abnormal")
    else:
        print("No valid prediction data")


def plot_curves(strategy, env_name, save_dir="experiments/results/plots"):
    """Plot input and output curves (supports 10-step sequence prediction)"""
    if not hasattr(strategy, 'test_predictions') or not hasattr(strategy, 'test_actuals'):
        print("No data available for plotting")
        return

    predictions = strategy.test_predictions
    actuals = strategy.test_actuals

    if len(predictions) == 0 or len(actuals) == 0:
        print("No valid data")
        return

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    n_features = predictions.shape[1] if len(predictions.shape) == 2 else 1
    n_plots = min(n_features, 6)  # Plot up to 6 features

    # Calculate subplot layout
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols

    plt.figure(figsize=(14, 4 * n_rows))

    feature_names = []
    if hasattr(strategy.env, 'get_feature_names'):
        feature_names = strategy.env.get_feature_names()

    # Check if sequence mode (with 10-step prediction)
    is_sequence_mode = False
    full_pred_sequence = None
    full_true_sequence = None

    if hasattr(strategy, '_saved_states') and hasattr(strategy, '_saved_full_sequence'):
        initial_states = strategy._saved_states
        full_target_sequences = strategy._saved_full_sequence

        # Check if data shape matches sequence mode
        if (len(initial_states) > 0 and len(full_target_sequences) > 0 and
            len(full_target_sequences.shape) == 3):

            n_samples = len(initial_states)
            # Rebuild full prediction sequence
            full_pred_sequence = []
            # For simplicity, use first sample to demonstrate
            sample_idx = 0
            if sample_idx < n_samples:
                # Try to get best model from strategy and rerun
                if hasattr(strategy, 'best_code') and strategy.best_code:
                    try:
                        from models.strategies import TimeCoderStrategy

                        # Clean code
                        code = strategy.best_code
                        if '```python' in code:
                            code = code.split('```python')[1]
                        if '```' in code:
                            code = code.split('```')[0]

                        namespace = {}
                        exec(code, namespace)
                        d_state__dt = namespace.get('d_state__dt')

                        if d_state__dt:
                            # Rebuild first sample's complete sequence
                            current_state = initial_states[sample_idx].copy()
                            pred_sequence = [current_state.copy()]

                            for step in range(10):
                                dx_dt = d_state__dt(current_state)
                                if dx_dt is not None:
                                    if hasattr(dx_dt, 'detach'):
                                        dx_dt = dx_dt.detach().numpy()
                                    dx_dt_array = np.array(dx_dt)
                                    if dx_dt_array.shape == current_state.shape:
                                        current_state = current_state + dx_dt_array
                                        pred_sequence.append(current_state.copy())

                            if len(pred_sequence) == 11:
                                full_pred_sequence = np.array(pred_sequence)
                                full_true_sequence = full_target_sequences[sample_idx]
                                is_sequence_mode = True
                    except Exception as e:
                        print(f"Error rebuilding sequence: {e}")

    if is_sequence_mode and full_pred_sequence is not None:
        # Plot sequence mode - complete 10-step prediction
        print("\nPlotting complete 10-step prediction sequence (Sample 1)")

        for i in range(n_plots):
            plt.subplot(n_rows, n_cols, i + 1)

            # Prepare data: initial state + 10-step prediction
            x = range(len(full_pred_sequence))
            pred_data = full_pred_sequence[:, i]
            true_data = full_true_sequence[:, i]

            # Plot curves
            plt.plot(x, true_data, 'b-o', label='True', linewidth=2, markersize=4, alpha=0.7)
            plt.plot(x, pred_data, 'r--s', label='Pred', linewidth=2, markersize=4, alpha=0.7)

            # Mark initial state
            plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
            plt.text(0.5, plt.ylim()[0], 'Initial', ha='center', va='bottom',
                    fontsize=8, alpha=0.7)

            # Set title
            if i < len(feature_names):
                title = f"{feature_names[i]}"
            else:
                title = f"Feature {i}"
            plt.title(title, fontsize=12)
            plt.xlabel('Prediction Step (0=Initial, 1-10=Predicted)', fontsize=10)
            plt.ylabel('Value', fontsize=10)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)

            # Set x-axis ticks
            plt.xticks(x, [f'{t}' for t in x])
    else:
        # Plot normal mode (single-step prediction)
        print("\nPlotting single-step prediction mode")

        # Use saved state sequences for Euler integration
        pred_states = None
        true_states = None

        if hasattr(strategy, '_saved_states') and hasattr(strategy, '_saved_next_states'):
            states = strategy._saved_states
            next_states = strategy._saved_next_states

            if len(states) > len(predictions):
                # Cumulative prediction: start from initial state, apply derivatives step by step
                pred_states = np.zeros_like(states[:len(predictions) + 1])
                pred_states[0] = states[0]  # Initial state

                for t in range(len(predictions)):
                    # Simple Euler integration
                    pred_states[t + 1] = pred_states[t] + predictions[t]

                # True state sequence (including initial and subsequent states)
                true_states = np.zeros_like(pred_states)
                true_states[0] = states[0]
                for t in range(len(predictions)):
                    true_states[t + 1] = next_states[t]

        if pred_states is None or true_states is None:
            # If no state sequences, plot derivative comparison
            pred_states = predictions
            true_states = actuals
            print("\nNote: Currently plotting derivative comparison (rate of change), not state sequence comparison")

        for i in range(n_plots):
            plt.subplot(n_rows, n_cols, i + 1)

            # Get current feature data
            if len(pred_states.shape) == 2:
                pred_data = pred_states[:, i]
                actual_data = true_states[:, i]
            else:
                pred_data = pred_states
                actual_data = true_states

            # Plot curves
            x = range(len(pred_data))
            plt.plot(x, actual_data, 'b-', label='True', linewidth=2, alpha=0.7)
            plt.plot(x, pred_data, 'r--', label='Pred', linewidth=2, alpha=0.7)

            # Set title
            if i < len(feature_names):
                title = f"{feature_names[i]}"
            else:
                title = f"Feature {i}"
            plt.title(title, fontsize=12)
            plt.xlabel('Time Step', fontsize=10)
            plt.ylabel('Value', fontsize=10)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    plot_file = save_path / f"{env_name}_curves.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nCurve plot saved to: {plot_file}")
    plt.show()


def run_evaluation(env, strategy, llm, config) -> Dict[str, Any]:
    """
    Run complete evaluation pipeline

    Args:
        env: Environment object
        strategy: Strategy object
        llm: LLM client
        config: Configuration object

    Returns:
        Dictionary containing evaluation results
    """
    generations = config.run.generations

    # Initialize strategy
    strategy.initialize(env, llm, config)

    # Evolution loop
    for gen in range(generations):
        strategy.evolve(generation=gen)

    # Test evaluation
    test_results = strategy.evaluate_test()

    # Print baseline results
    print_baseline_results(strategy, env.env_name)

    # Save prediction data
    save_predictions(strategy, env.env_name)

    return {
        'env_name': env.env_name,
        'method_name': strategy.name,
        'seed': env.seed,
        'test_mse': test_results['mse'],
        'test_rmse': test_results['rmse'],
        'test_mae': test_results['mae']
    }


def save_predictions(strategy, env_name):
    """Save prediction data to file (including training and test sets, preserving 15-min input history)"""
    save_dir = Path("experiments/results")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Saving Prediction Data")
    print(f"{'='*60}")

    # ===== 0. Save input history data (first 15 minutes) =====
    if hasattr(strategy, 'env') and strategy.env is not None:
        try:
            # Get test set input data (first 15 minutes history)
            test_input = strategy.env.data.get('test_input', {})
            if test_input:
                input_history = []
                feature_order = [
                    'ARTD', 'ARTM', 'ARTS', 'ETCO2 (mmHg)', 'HR',
                    'INTERVENTION_BOLUS', 'INTERVENTION_INFUSION',
                    'MED_aminocaproic acid', 'MED_amiodarone', 'MED_cisatracurium',
                    'MED_clevidipine', 'MED_dexamethasone', 'MED_dexmedetomidine',
                    'MED_dobutamine', 'MED_dopamine', 'MED_ephedrine', 'MED_epinephrine',
                    'MED_esmolol', 'MED_fentanyl', 'MED_hydromorphone', 'MED_ketamine',
                    'MED_labetalol', 'MED_milrinone', 'MED_morphine', 'MED_nicardipine',
                    'MED_nitroglycerin', 'MED_norepinephrine', 'MED_ondansetron',
                    'MED_oxytocin', 'MED_phenylephrine', 'MED_propofol', 'MED_remifentanil',
                    'MED_sufentanil', 'MED_sugammadex', 'MED_tranexamic acid',
                    'MED_vasopressin', 'RR'
                ]

                # Build input history: (n_samples, 15, 37)
                first_key = list(test_input.values())[0]
                n_samples = first_key.shape[1]

                for sidx in range(n_samples):
                    sample_history = []
                    for t in range(15):
                        state_t = []
                        for key in feature_order:
                            if key in test_input:
                                val = test_input[key]
                                if len(val.shape) == 2:
                                    state_t.append(val[t, sidx])
                        if state_t:
                            sample_history.append(np.array(state_t))

                    if len(sample_history) == 15:
                        input_history.append(np.array(sample_history))

                if len(input_history) > 0:
                    input_history_arr = np.array(input_history)
                    input_file = save_dir / f"{env_name}_test_input_history.npy"
                    np.save(input_file, input_history_arr)
                    print(f"\n[Input History] First 15 minutes complete history: {input_file} (shape: {input_history_arr.shape})")
        except Exception as e:
            print(f"\n[Warning] Cannot save input history: {e}")

    # ===== 1. Save training set predictions =====
    if hasattr(strategy, 'best_train_predictions') and len(strategy.best_train_predictions) > 0:
        train_pred_file = save_dir / f"{env_name}_train_full_predictions.npy"
        np.save(train_pred_file, strategy.best_train_predictions)
        print(f"\n[Training Set] Complete prediction sequence: {train_pred_file} (shape: {strategy.best_train_predictions.shape})")

        if hasattr(strategy, 'best_train_initial_states') and strategy.best_train_initial_states is not None:
            train_states_file = save_dir / f"{env_name}_train_initial_states.npy"
            np.save(train_states_file, strategy.best_train_initial_states)
            print(f"[Training Set] Initial states: {train_states_file} (shape: {strategy.best_train_initial_states.shape})")

        if hasattr(strategy, 'best_train_targets') and strategy.best_train_targets is not None:
            train_targets_file = save_dir / f"{env_name}_train_targets.npy"
            np.save(train_targets_file, strategy.best_train_targets)
            print(f"[Training Set] Target sequences: {train_targets_file} (shape: {strategy.best_train_targets.shape})")

        if hasattr(strategy, 'best_train_state_preds') and len(strategy.best_train_state_preds) > 0:
            train_flat_pred_file = save_dir / f"{env_name}_train_predictions.npy"
            train_flat_actual_file = save_dir / f"{env_name}_train_actuals.npy"
            np.save(train_flat_pred_file, strategy.best_train_state_preds)
            np.save(train_flat_actual_file, strategy.best_train_state_actuals)
            print(f"[Training Set] Flattened predictions: {train_flat_pred_file} (shape: {strategy.best_train_state_preds.shape})")
            print(f"[Training Set] Flattened actuals: {train_flat_actual_file} (shape: {strategy.best_train_state_actuals.shape})")

    # ===== 2. Save test set predictions =====
    if hasattr(strategy, 'best_test_predictions') and len(strategy.best_test_predictions) > 0:
        test_pred_file = save_dir / f"{env_name}_test_full_predictions.npy"
        np.save(test_pred_file, strategy.best_test_predictions)
        print(f"\n[Test Set] Complete prediction sequence: {test_pred_file} (shape: {strategy.best_test_predictions.shape})")

        if hasattr(strategy, 'best_test_initial_states') and strategy.best_test_initial_states is not None:
            test_states_file = save_dir / f"{env_name}_test_initial_states.npy"
            np.save(test_states_file, strategy.best_test_initial_states)
            print(f"[Test Set] Initial states: {test_states_file} (shape: {strategy.best_test_initial_states.shape})")

        if hasattr(strategy, 'best_test_targets') and strategy.best_test_targets is not None:
            test_targets_file = save_dir / f"{env_name}_test_targets.npy"
            np.save(test_targets_file, strategy.best_test_targets)
            print(f"[Test Set] Target sequences: {test_targets_file} (shape: {strategy.best_test_targets.shape})")

    # Save flattened test set predictions
    if hasattr(strategy, 'test_predictions') and hasattr(strategy, 'test_actuals'):
        predictions = strategy.test_predictions
        actuals = strategy.test_actuals

        if len(predictions) > 0 and len(actuals) > 0:
            test_flat_pred_file = save_dir / f"{env_name}_test_predictions.npy"
            test_flat_actual_file = save_dir / f"{env_name}_test_actuals.npy"
            np.save(test_flat_pred_file, predictions)
            np.save(test_flat_actual_file, actuals)
            print(f"\n[Test Set] Flattened predictions: {test_flat_pred_file} (shape: {predictions.shape})")
            print(f"[Test Set] Flattened actuals: {test_flat_actual_file} (shape: {actuals.shape})")

    print(f"\n{'='*60}")
    print(f"Saving Complete!")
    print(f"{'='*60}")
