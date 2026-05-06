"""
Strategy Definitions - TimeCoder Complete Implementation (with Feedback Mechanism) - Fixed Version
"""
# Important: Import torch first to avoid Windows DLL issues
import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np
from pathlib import Path


class Strategy(ABC):
    """Base Strategy Class"""

    def __init__(self, name: str):
        self.name = name
        self.env = None
        self.llm = None
        self.config = None
        self.best_code = None
        self.best_params = None
        self.best_mse = float('inf')
        self.best_state_mse = float('inf')
        self.history = []
        self.feedback_history = []
        self.evaluate_state_sequence = False

    def initialize(self, env, llm, config):
        """Initialize strategy"""
        self.env = env
        self.llm = llm
        self.config = config
        self.evaluate_state_sequence = getattr(config.run, 'evaluate_state_sequence', False)

    @abstractmethod
    def evolve(self, generation: int):
        """Evolve one generation"""
        pass

    @abstractmethod
    def evaluate_test(self) -> Dict:
        """Evaluate test set"""
        pass

    def _execute_and_evaluate(self, code: str, test_mode: bool = False) -> tuple:
        """Execute code and evaluate, returning derivative predictions and state predictions (supports 10-step sequence prediction)"""
        import time
        start_time = time.time()

        try:
            # Clean code
            if '```python' in code:
                code = code.split('```python')[1]
            if '```' in code:
                code = code.split('```')[0]

            code_lines = code.split('\n')
            clean_code_lines = []
            in_function = False
            for line in code_lines:
                if not line.strip():
                    continue
                if 'def d_state__dt' in line:
                    in_function = True
                if in_function:
                    clean_code_lines.append(line)
                if in_function and 'return' in line.strip() and 'dx_dt' in line.strip():
                    break
            clean_code = '\n'.join(clean_code_lines)

            # Pre-import common libraries in namespace to resolve function-level import issues
            import numpy as np
            namespace = {
                'np': np,
                'numpy': np,
                '__builtins__': __builtins__
            }
            exec(clean_code, namespace)

            d_state__dt = namespace.get('d_state__dt')
            if d_state__dt is None:
                return np.array([]), np.array([]), np.array([]), np.array([])

            # Verify function executes correctly (using dynamic dimensions)
            env_name = self.env.env_name
            if env_name == 'flood_ts':
                test_dim = 5
            elif env_name == 'cancer_ts':
                test_dim = 4
            elif env_name == 'covid_ts':
                test_dim = 5
            elif env_name == 'surgery_ts':
                test_dim = 10
            else:
                test_dim = 37  # Default value

            test_state = np.zeros(test_dim)

            # Verify function executes correctly (lenient validation, allowing some minor issues)
            try:
                test_result = d_state__dt(test_state)
                if test_result is None:
                    print(f"[Warning] Function returned None, continuing to try running...")
                elif np.any(np.isnan(test_result)) or np.any(np.isinf(test_result)):
                    print(f"[Warning] Function returned NaN/Inf, continuing to try running...")
                else:
                    print(f"[Validation Passed] Function output shape: {test_result.shape}")
            except (IndexError, KeyError, TypeError, ValueError) as e:
                # LLM-generated code may have index errors, but may work fine during actual execution
                print(f"[Warning] Error during validation ({type(e).__name__}: {e}), but will continue trying on actual data...")

            data = self.env.data
            self._saved_states = None
            self._saved_next_states = None
            self._saved_full_sequence = None  # Save complete sequence for plotting

            states = None
            next_states = None
            full_target_sequences = None

            # Debug: Print data structure
            print(f"[Data Debug] Data dictionary keys: {list(data.keys())}")
            if 'train_input' in data:
                print(f"[Data Debug] Found train_input format")
                print(f"[Data Debug] train_input keys: {list(data['train_input'].keys())[:5]}...")
                first_val = next(iter(data['train_input'].values()))
                print(f"[Data Debug] First feature shape: {first_val.shape}")
            elif 'states' in data:
                print(f"[Data Debug] Using old format (states)")
            else:
                print(f"[Data Debug] Unknown format!")

            if 'states' in data:
                # Old data format - convert to sequence mode!
                print(f"[Data Debug] Converting old format (states/next_states) to sequence mode...")

                raw_states = data['states']
                if 'next_states' in data:
                    raw_next_states = data['next_states']
                else:
                    raw_next_states = raw_states[1:]
                    raw_states = raw_states[:-1]

                # Get output window size (from config or data inference)
                # Try to infer from data name, default 24
                env_name = self.env.env_name
                if env_name == 'surgery_ts':
                    output_window = 20
                elif env_name in ['flood_ts', 'cancer_ts', 'covid_ts']:
                    output_window = 24
                else:
                    output_window = 11  # Default value

                input_window = output_window  # Input window = Output window

                print(f"[Data Debug] Raw data: raw_states.shape={raw_states.shape}")
                print(f"[Data Debug] Using windows: input={input_window}, output={output_window}")

                # Use sliding window to build sequence data
                initial_states_list = []
                target_sequences_list = []

                total_samples = len(raw_states)
                max_start_idx = total_samples - input_window - output_window

                if max_start_idx <= 0:
                    print(f"[Data Debug] Insufficient data! Need at least {input_window + output_window} samples, only have {total_samples}")
                    # Fallback to single-step mode
                    states = raw_states[:min(100, len(raw_states))]
                    next_states = raw_next_states[:min(100, len(raw_next_states))] if len(raw_next_states) > 0 else None
                else:
                    # Limit sample count to avoid memory issues
                    max_samples = min(max_start_idx, 1000)  # Maximum 1000 sequence samples
                    step = max(1, max_start_idx // max_samples)

                    sample_indices = list(range(0, max_start_idx, step))[:max_samples]

                    for start_idx in sample_indices:
                        # Initial state: last state of input window
                        initial_state = raw_states[start_idx + input_window - 1]
                        initial_states_list.append(initial_state)

                        # Target sequence: all states of output window
                        target_seq = []
                        for t in range(output_window):
                            target_state = raw_states[start_idx + input_window + t]
                            target_seq.append(target_state)
                        target_sequences_list.append(target_seq)

                    if initial_states_list:
                        states = np.array(initial_states_list)
                        full_target_sequences = np.array(target_sequences_list)
                        print(f"[Data Debug] Successfully built sequence: {len(initial_states_list)} samples")
                        print(f"[Data Debug]    states.shape={states.shape}")
                        print(f"[Data Debug]    targets.shape={full_target_sequences.shape}")
                    else:
                        # Fallback to single-step mode
                        states = raw_states[:min(100, len(raw_states))]
                        next_states = raw_next_states[:min(100, len(raw_next_states))] if len(raw_next_states) > 0 else None
                        print(f"[Data Debug] Build failed, falling back to single-step mode")
            elif 'train_input' in data and 'train_output' in data:
                # Our new format - directly has train_input/test_input/train_output/test_output
                initial_states_list = []
                target_sequences_list = []

                if test_mode:
                    inputs_dict = data['test_input']
                    targets_dict = data['test_output']
                else:
                    inputs_dict = data['train_input']
                    targets_dict = data['train_output']

                # Get feature order
                env_name = self.env.env_name
                feature_order = self._get_feature_order(env_name, inputs_dict)

                # Get first input data to check shape
                first_key = next(iter(inputs_dict.values()))
                # Our data shape is [time, samples]
                if len(first_key.shape) == 2:
                    # [time, samples] -> [samples, time]
                    pass
                n_samples = first_key.shape[1]

                # Use full samples during training (6548 samples)
                if not test_mode:
                    max_samples = n_samples  # Use all training samples
                    print(f"[Training Mode] Using full {max_samples}/{n_samples} samples (complete training)")
                else:
                    max_samples = n_samples  # Test with all
                    print(f"[Test Mode] Using all {max_samples} samples")

                for sample_idx in range(max_samples):
                    # Print progress every 100 samples
                    if sample_idx % 100 == 0:
                        print(f"[Progress] Processing sample {sample_idx}/{max_samples} ({sample_idx/max_samples*100:.1f}%)")

                    # Build initial state - take last time step of input
                    state_parts = []
                    for key in feature_order:
                        if key in inputs_dict:
                            val = inputs_dict[key]
                            if len(val.shape) == 2:
                                val_at_last_time = val[-1, sample_idx]
                                state_parts.append(val_at_last_time)

                    # Dynamically get output window size
                    first_target_key = next(iter(targets_dict.values()))
                    output_window = first_target_key.shape[0]  # Get actual number of time steps

                    # Build complete target sequence - dynamically adapt to different output_window
                    target_sequence = []
                    for t in range(output_window):
                        state_parts_t = []
                        for key in feature_order:
                            if key in targets_dict:
                                val = targets_dict[key]
                                if len(val.shape) == 2:
                                    val_at_t = val[t, sample_idx]
                                    state_parts_t.append(val_at_t)
                        if state_parts_t:
                            state_t = np.array(state_parts_t)
                            target_sequence.append(state_t)

                    if state_parts and len(target_sequence) == output_window:
                        state = np.array(state_parts)
                        initial_states_list.append(state)
                        target_sequences_list.append(target_sequence)

                if initial_states_list:
                    states = np.array(initial_states_list)
                    full_target_sequences = np.array(target_sequences_list)
                    print(f"[Data Debug] Successfully built sequence: states.shape={states.shape}, targets.shape={full_target_sequences.shape}")
                else:
                    print(f"[Data Debug] initial_states_list is empty! Build failed")
                    print(f"[Data Debug] Check feature_order length: {len(feature_order)}")
                    print(f"[Data Debug] Check inputs_dict key count: {len(inputs_dict)}")
            elif 'result' in data:
                # New data format - sequence mode
                initial_states_list = []
                target_sequences_list = []

                result = data['result']
                data_key = 'test_data' if test_mode and 'test_data' in result else 'train_data'

                if data_key in result:
                    inputs_dict, targets_dict = result[data_key]

                    # Get feature order
                    env_name = self.env.env_name
                    feature_order = self._get_feature_order(env_name, inputs_dict)

                    # Get first input data to check shape
                    first_val = next(iter(inputs_dict.values()))
                    batch_size = first_val.shape[0]
                    max_samples = batch_size

                    for sample_idx in range(max_samples):
                        # Build initial state - take last time step of input
                        state_parts = []
                        for key in feature_order:
                            if key in inputs_dict:
                                val = inputs_dict[key]
                                if len(val.shape) == 3:
                                    # [batch, features, time] -> [sample_idx, :, -1]
                                    state_parts.append(val[sample_idx, :, -1])

                        # Build complete target sequence - dynamically adapt to different output_window
                        first_target_val = next(iter(targets_dict.values()))
                        output_window = first_target_val.shape[2] if len(first_target_val.shape) == 3 else first_target_val.shape[0]
                        target_sequence = []
                        for t in range(output_window):
                            state_parts_t = []
                            for key in feature_order:
                                if key in targets_dict:
                                    val = targets_dict[key]
                                    if len(val.shape) == 3:
                                        state_parts_t.append(val[sample_idx, :, t])
                            if state_parts_t:
                                import torch
                                state_t = torch.cat(state_parts_t, dim=0).numpy()
                                target_sequence.append(state_t)

                        if state_parts and target_sequence:
                            import torch
                            state = torch.cat(state_parts, dim=0).numpy()
                            initial_states_list.append(state)
                            target_sequences_list.append(target_sequence)

                    if initial_states_list:
                        states = np.array(initial_states_list)
                        full_target_sequences = np.array(target_sequences_list)

            if states is None or len(states) == 0:
                return np.array([]), np.array([]), np.array([]), np.array([])

            # If sequence mode, perform multi-step prediction
            if full_target_sequences is not None:
                print(f"[Data Debug] Entering sequence mode! full_target_sequences.shape={full_target_sequences.shape}")

                # Batch processing - avoid memory overflow
                batch_size = 100  # Smaller batch size
                n_samples = len(states)

                # Only save complete sequence during test mode for plotting
                if test_mode:
                    full_predicted_sequences = []
                else:
                    full_predicted_sequences = None

                sequence_predictions_flat = []
                sequence_actuals_flat = []
                failed_samples = 0  # Record count of failed samples

                for batch_start in range(0, n_samples, batch_size):
                    batch_end = min(batch_start + batch_size, n_samples)

                    # Print batch progress
                    print(f"[Batch Progress] Processing batch {batch_start//batch_size + 1}/{(n_samples-1)//batch_size + 1} (samples {batch_start}-{batch_end})")

                    # Process current batch
                    for i in range(batch_start, batch_end):
                        try:
                            current_state = states[i].copy()
                            target_sequence = full_target_sequences[i]

                            # Dynamically get number of prediction steps (output_window - 1, since it already includes initial state)
                            n_prediction_steps = len(target_sequence) - 1

                            # Perform Euler integration prediction (dynamic steps)
                            predicted_sequence = [current_state.copy()]
                            for step in range(n_prediction_steps):
                                dx_dt = d_state__dt(current_state)
                                if dx_dt is not None:
                                    if hasattr(dx_dt, 'detach'):
                                        dx_dt = dx_dt.detach().numpy()
                                    dx_dt_array = np.array(dx_dt)
                                    if dx_dt_array.shape == current_state.shape:
                                        current_state = current_state + dx_dt_array
                                        predicted_sequence.append(current_state.copy())

                            if len(predicted_sequence) == len(target_sequence):  # Dynamic length check
                                if test_mode and full_predicted_sequences is not None:
                                    full_predicted_sequences.append(predicted_sequence)

                                for step in range(n_prediction_steps):
                                    sequence_predictions_flat.append(predicted_sequence[step + 1])
                                    sequence_actuals_flat.append(target_sequence[step + 1])

                        except Exception as e:
                            failed_samples += 1
                            if failed_samples <= 3:  # Only print details of first 3 errors
                                print(f"[Warning] Sample {i} prediction failed: {type(e).__name__}: {str(e)[:100]}")

                # Report failure statistics
                if failed_samples > 0:
                    print(f"[Statistics] {failed_samples}/{n_samples} samples failed prediction ({failed_samples/n_samples*100:.1f}%)")
                else:
                    print(f"[Statistics] All {n_samples} samples predicted successfully!")

                    # Clean up memory after batch processing completes
                    import gc
                    gc.collect()

                # Save complete sequence (only in test mode)
                if test_mode:
                    self._saved_states = states
                    self._saved_full_sequence = full_target_sequences
                    self._full_predicted_sequences = full_predicted_sequences
                    print(f"[Debug] Test mode - Saved complete prediction sequence: {len(full_predicted_sequences) if full_predicted_sequences else 0} samples")
                else:
                    print(f"[Debug] Training mode - Not saving complete sequence to instance attributes (save memory)")

                if len(sequence_predictions_flat) > 0:
                    return (np.array([]), np.array([]),
                            np.array(sequence_predictions_flat), np.array(sequence_actuals_flat))

            # If not sequence mode, fall back to single-step mode
            print(f"[Data Debug] full_target_sequences is None, entering single-step mode")
            derivative_predictions = []
            derivative_actuals = []
            state_predictions = []
            state_actuals = []

            for i in range(len(states)):
                state = states[i]
                next_state = next_states[i] if next_states is not None else None
                if next_state is None:
                    continue

                try:
                    dx_dt = d_state__dt(state)
                    if dx_dt is not None:
                        if hasattr(dx_dt, 'detach'):
                            dx_dt = dx_dt.detach().numpy()
                        if isinstance(dx_dt, (np.ndarray, list)) and len(dx_dt) > 0:
                            dx_dt_array = np.array(dx_dt)
                            if dx_dt_array.shape == state.shape:
                                derivative_predictions.append(dx_dt_array)
                                derivative_actuals.append(next_state - state)

                                pred_next_state = state + dx_dt_array
                                state_predictions.append(pred_next_state)
                                state_actuals.append(next_state)
                except Exception as e:
                    pass

            if len(derivative_predictions) == 0:
                elapsed = time.time() - start_time
                print(f"[Evaluation Complete] Time taken: {elapsed:.1f}s, No valid predictions")
                return np.array([]), np.array([]), np.array([]), np.array([])

            elapsed = time.time() - start_time
            print(f"[Evaluation Complete] Time taken: {elapsed:.1f}s, Predictions: {len(state_predictions)}")

            return (np.array(derivative_predictions), np.array(derivative_actuals),
                    np.array(state_predictions), np.array(state_actuals))

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[Execution Error] Time taken: {elapsed:.1f}s, Error: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]), np.array([]), np.array([]), np.array([])

    def _get_feature_order(self, env_name: str, inputs_dict: Dict) -> List[str]:
        """Get feature order"""
        if env_name == 'flood_ts':
            return ['WATER', 'TIDE', 'GATE', 'PUMP', 'RAIN']
        elif env_name == 'cancer_ts':
            return ['tumor_volume', 'chemo_concentration', 'chemo_dosage', 'radio_dosage']
        elif env_name == 'covid_ts':
            return ['susceptible', 'exposed', 'infected', 'recovered', 'intervention']
        elif env_name == 'surgery_ts':
            return ['MBP', 'SBP', 'DBP', 'SpO2', 'HR', 'ETCO2', 'EPI', 'NEPI', 'PHEN', 'VASO']
        elif env_name == 'medications_ts':
            return [
                'ARTS', 'ARTD', 'ARTM', 'ETCO2 (mmHg)', 'HR', 'RR',
                'MED_propofol', 'MED_ephedrine', 'MED_sugammadex', 'MED_epinephrine',
                'MED_hydromorphone', 'MED_ondansetron', 'MED_tranexamic acid',
                'MED_phenylephrine', 'MED_norepinephrine', 'MED_nitroglycerin',
                'MED_vasopressin', 'MED_cisatracurium', 'MED_labetalol',
                'MED_dexamethasone', 'MED_fentanyl', 'MED_amiodarone',
                'MED_morphine', 'MED_dobutamine', 'MED_remifentanil', 'MED_oxytocin'
            ]
        elif env_name == 'hospital_ts':
            return ['icu_occupied', 'standard_occupied', 'total_patients']
        elif env_name == 'mover_ts':
            return [
                'ARTS', 'ARTD', 'ARTM', 'ETCO2 (mmHg)', 'HR', 'RR',
                'MED_propofol', 'MED_ephedrine', 'MED_sugammadex', 'MED_epinephrine',
                'MED_hydromorphone', 'MED_ondansetron', 'MED_tranexamic acid',
                'MED_phenylephrine', 'MED_norepinephrine', 'MED_nitroglycerin',
                'MED_vasopressin', 'MED_cisatracurium', 'MED_labetalol',
                'MED_dexamethasone', 'MED_fentanyl', 'MED_amiodarone',
                'MED_morphine', 'MED_dobutamine', 'MED_remifentanil', 'MED_oxytocin'
            ]
        elif env_name == 'medication_intervention':
            return [
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
        else:
            return list(inputs_dict.keys())

    def _calculate_mse(self, predictions, actuals) -> float:
        """Calculate MSE"""
        if len(predictions) == 0:
            return float('inf')
        return float(np.mean((predictions - actuals) ** 2))

    def _calculate_rmse(self, predictions, actuals) -> float:
        """Calculate RMSE"""
        if len(predictions) == 0:
            return float('inf')
        return float(np.sqrt(np.mean((predictions - actuals) ** 2)))

    def _calculate_mae(self, predictions, actuals) -> float:
        """Calculate MAE"""
        if len(predictions) == 0:
            return float('inf')
        return float(np.mean(np.abs(predictions - actuals)))

    def evaluate_test(self) -> Dict:
        """Evaluate test set and collect complete predictions for training and test sets"""
        if self.best_code is None:
            return {
                'mse': float('inf'),
                'rmse': float('inf'),
                'mae': float('inf')
            }

        print(f"\n{'='*80}")
        print(f"[Evaluation] Running test set...")
        print(f"{'='*80}")

        # ===== 1. Run test set =====
        deriv_preds, deriv_actuals, state_preds, state_actuals = self._execute_and_evaluate(
            self.best_code, test_mode=True
        )

        if self.evaluate_state_sequence and len(state_preds) > 0:
            self.test_predictions = state_preds
            self.test_actuals = state_actuals
            primary_mse = self._calculate_mse(state_preds, state_actuals)
            primary_rmse = self._calculate_rmse(state_preds, state_actuals)
            primary_mae = self._calculate_mae(state_preds, state_actuals)
        else:
            self.test_predictions = deriv_preds
            self.test_actuals = deriv_actuals
            primary_mse = self._calculate_mse(deriv_preds, deriv_actuals)
            primary_rmse = self._calculate_rmse(deriv_preds, deriv_actuals)
            primary_mae = self._calculate_mae(deriv_preds, deriv_actuals)

        # Save test set complete sequence
        print(f"[Debug] Checking complete prediction sequence...")
        print(f"[Debug] _full_predicted_sequences exists: {hasattr(self, '_full_predicted_sequences')}")
        if hasattr(self, '_full_predicted_sequences'):
            print(f"[Debug] _full_predicted_sequences length: {len(self._full_predicted_sequences) if self._full_predicted_sequences else 0}")

        if hasattr(self, '_full_predicted_sequences') and len(self._full_predicted_sequences) > 0:
            self.best_test_predictions = np.array(self._full_predicted_sequences)
            self.best_test_initial_states = self._saved_states.copy() if hasattr(self, '_saved_states') and self._saved_states is not None else None
            self.best_test_targets = self._saved_full_sequence.copy() if hasattr(self, '_saved_full_sequence') and self._saved_full_sequence is not None else None
            print(f"[Debug] Set best_test_predictions (shape: {self.best_test_predictions.shape})")
        else:
            print(f"[Debug] _full_predicted_sequences is empty or does not exist! Cannot save complete sequence")

        print(f"[Test Set] MSE={primary_mse:.4f}, RMSE={primary_rmse:.4f}, MAE={primary_mae:.4f}")

        # ===== 2. Run training set to collect predictions =====
        print(f"\n{'='*80}")
        print(f"[Evaluation] Running training set with best model to collect predictions...")
        print(f"{'='*80}")

        # Temporarily switch to training mode run
        # We need to temporarily modify env's data to run training set
        original_data = self.env.data
        try:
            # Swap training and test data
            if 'train_input' in original_data and 'train_output' in original_data:
                temp_train_data = {
                    'train_input': original_data['train_input'],
                    'train_output': original_data['train_output'],
                    'test_input': original_data['train_input'],  # Treat training set as test set
                    'test_output': original_data['train_output']
                }
                self.env.data = temp_train_data

                # Run training set (via test_mode=True to save complete predictions)
                train_deriv_preds, train_deriv_actuals, train_state_preds, train_state_actuals = self._execute_and_evaluate(
                    self.best_code, test_mode=True
                )

                # Save training set complete predictions
                if hasattr(self, '_full_predicted_sequences') and len(self._full_predicted_sequences) > 0:
                    self.best_train_predictions = np.array(self._full_predicted_sequences)
                    self.best_train_initial_states = self._saved_states.copy() if hasattr(self, '_saved_states') else None
                    self.best_train_targets = self._saved_full_sequence.copy() if hasattr(self, '_saved_full_sequence') else None
                    self.best_train_state_preds = train_state_preds.copy() if len(train_state_preds) > 0 else np.array([])
                    self.best_train_state_actuals = train_state_actuals.copy() if len(train_state_actuals) > 0 else np.array([])

                    if len(train_state_preds) > 0:
                        train_mse = self._calculate_mse(train_state_preds, train_state_actuals)
                        train_rmse = self._calculate_rmse(train_state_preds, train_state_actuals)
                        train_mae = self._calculate_mae(train_state_preds, train_state_actuals)
                        print(f"[Training Set] MSE={train_mse:.4f}, RMSE={train_rmse:.4f}, MAE={train_mae:.4f}")
                        print(f"[Training Set] Saved prediction sample count: {len(self.best_train_predictions)}")

        finally:
            # Restore original data
            self.env.data = original_data

        print(f"{'='*80}")

        return {
            'mse': primary_mse,
            'rmse': primary_rmse,
            'mae': primary_mae
        }


class TimeCoderStrategy(Strategy):
    """TimeCoder Strategy (with feedback mechanism and intelligent backtracking)"""

    def __init__(self):
        super().__init__('TimeCoder')
        self.stagnation_counter = 0
        self.last_improvement_gen = 0

    def evolve(self, generation: int):
        """Evolve one generation using feedback mechanism and intelligent backtracking"""
        from llm.prompts import PromptTemplates
        from tools.analyzer import FeedbackGenerator
        import os
        import time
        import random

        gen_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"[{self.name}] Starting generation {generation}...")
        print(f"{'='*80}\n")

        feedback_generator = FeedbackGenerator()
        feedback_generator.performance_analyzer.env = self.env

        stagnation_threshold = getattr(self.config.run, 'stagnation_threshold', 5)

        use_backtracking = False
        if generation > 0 and self.stagnation_counter >= stagnation_threshold:
            use_backtracking = True
            print(f"[{self.name}] Detected stagnation for {self.stagnation_counter} generations, triggering backtracking!")
            self.stagnation_counter = 0

        previous_feedback = None
        if self.feedback_history:
            previous_feedback = self.feedback_history[-1]

        task_prompt = PromptTemplates.get_task_prompt(
            self.env.env_name,
            self.config.run.generations,
            generation,
            feedback=previous_feedback
        )

        # Force randomness - add random seed and timestamp to prompt!
        random_seed = random.randint(1000, 9999)
        current_timestamp = int(time.time())
        task_prompt += f"\n\nCRITICAL DEBUG INFO: RANDOM_SEED={random_seed}, TIMESTAMP={current_timestamp}"
        task_prompt += "\nYOU MUST GENERATE COMPLETELY DIFFERENT CODE THAN ANY PREVIOUS VERSION!"

        if use_backtracking and self.best_code:
            task_prompt += "\n\nCRITICAL: Previous model has stagnated! Please completely change the model structure!\n"

        system_prompt = PromptTemplates.get_system_prompt(env_name=self.env.env_name)
        skeleton = PromptTemplates.get_skeleton_code(self.env.env_name)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt + "\n\n" + skeleton}
        ]

        print(f"\n{'='*80}")
        print(f"[{self.name}] Gen {generation} - Starting LLM code generation...")
        print(f"[{self.name}] Prompt length: {len(task_prompt)} characters")
        print(f"{'='*80}\n")

        try:
            print(f"[{self.name}] Waiting for LLM response (timeout: 300s)...")
            code = self.llm.generate(messages)

            print(f"\n{'='*80}")
            print(f"[{self.name}] LLM code generation complete! Length: {len(code)} chars")
            print(f"[{self.name}] Code preview (first 200 chars):")
            print(code[:200] + "..." if len(code) > 200 else code)
            print(f"{'='*80}\n")

            print(f"[{self.name}] Evaluating code performance (using full 6548 training samples)...")
            deriv_preds, deriv_actuals, state_preds, state_actuals = self._execute_and_evaluate(code)

            if self.evaluate_state_sequence and len(state_preds) > 0:
                primary_mse = self._calculate_mse(state_preds, state_actuals)
                primary_rmse = self._calculate_rmse(state_preds, state_actuals)
                primary_mae = self._calculate_mae(state_preds, state_actuals)
                feedback_preds, feedback_actuals = state_preds, state_actuals
                mse_type = "State MSE"
            else:
                primary_mse = self._calculate_mse(deriv_preds, deriv_actuals)
                primary_rmse = self._calculate_rmse(deriv_preds, deriv_actuals)
                primary_mae = self._calculate_mae(deriv_preds, deriv_actuals)
                feedback_preds, feedback_actuals = deriv_preds, deriv_actuals
                mse_type = "Deriv MSE"

            deriv_mse = self._calculate_mse(deriv_preds, deriv_actuals) if len(deriv_preds) > 0 else float('inf')
            state_mse = self._calculate_mse(state_preds, state_actuals) if len(state_preds) > 0 else float('inf')

            feedback = feedback_generator.generate_feedback(
                feedback_preds, feedback_actuals, code, generation
            )

            feedback_dict = feedback_generator.get_feedback_for_prompt(
                feedback_preds, feedback_actuals, code, generation
            )
            score = feedback_dict['metrics'].get('score', 0.0)

            code_dir = f"experiments/code/{self.env.env_name}"
            os.makedirs(code_dir, exist_ok=True)
            code_file = f"{code_dir}/gen_{generation}.py"
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)

            self.feedback_history.append(feedback)

            if self.evaluate_state_sequence:
                comparison_mse = state_mse
                best_comparison = self.best_state_mse
            else:
                comparison_mse = deriv_mse
                best_comparison = self.best_mse

            if comparison_mse < best_comparison:
                improvement = (best_comparison - comparison_mse) / best_comparison * 100 if best_comparison != float('inf') else 100

                if self.evaluate_state_sequence:
                    self.best_state_mse = comparison_mse
                else:
                    self.best_mse = comparison_mse

                self.best_code = code

                self.history.append({
                    'generation': generation,
                    'mse': comparison_mse,
                    'deriv_mse': deriv_mse,
                    'state_mse': state_mse,
                    'score': score,
                    'code': code
                })
                self.stagnation_counter = 0
                self.last_improvement_gen = generation
                print(f"[{self.name}] Gen {generation}: New best {mse_type} = {comparison_mse:.6f}, RMSE = {primary_rmse:.6f}, MAE = {primary_mae:.6f} (improvement +{improvement:.1f}%)")
            else:
                self.stagnation_counter += 1
                print(f"[{self.name}] Gen {generation}: {mse_type} = {comparison_mse:.6f}, RMSE = {primary_rmse:.6f}, MAE = {primary_mae:.6f} (best = {best_comparison:.6f})")

        except Exception as e:
            print(f"[{self.name}] Gen {generation}: Error - {e}")
            import traceback
            traceback.print_exc()
            if self.best_code is None:
                from llm.prompts import PromptTemplates
                self.best_code = PromptTemplates.get_skeleton_code(self.env.env_name)

        # ===== Generation {generation} Complete =====
        gen_elapsed = time.time() - gen_start_time
        print(f"\n{'='*80}")
        print(f"[{self.name}] Generation {generation} complete! Total time: {gen_elapsed:.1f}s ({gen_elapsed/60:.1f}min)")
        print(f"{'='*80}\n")

        # ===== Memory cleanup - release intermediate data after each generation (but keep last one for evaluate_test) =====
        import gc
        # Note: Do not clear _full_predicted_sequences, _saved_states, _saved_full_sequence
        # These will be used in evaluate_test() and cleaned up after that function completes
        gc.collect()


def get_strategy(name: str) -> Strategy:
    """Get strategy instance"""
    strategies = {
        'TimeCoder': TimeCoderStrategy()
    }

    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}")

    return strategies[name]
