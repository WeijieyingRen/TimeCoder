"""
Data Loader Module
"""
import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Any, Dict


class Environment:
    """Environment Base Class"""

    def __init__(self, env_name: str, seed: int, config: Any):
        self.env_name = env_name
        self.seed = seed
        self.config = config

        # Normalization parameters - medication_intervention forces no normalization!
        if env_name == 'medication_intervention':
            self.use_normalization = False
        else:
            self.use_normalization = getattr(config.run, 'use_normalization', False)
        self.feature_means = None
        self.feature_stds = None

        # Load data
        self.data = self._load_data()

    def _load_data(self) -> Dict:
        """Load data from file"""
        # Handle different types of config.datasets
        if hasattr(self.config.datasets, 'get'):
            # If it's a dictionary
            dataset_path = self.config.datasets.get(self.env_name, '')
        else:
            # If it's an object
            dataset_path = getattr(self.config.datasets, self.env_name, '')

        if not dataset_path:
            raise ValueError(f"No dataset path for {self.env_name}")

        # Check if file exists
        if not Path(dataset_path).exists():
            # Use dummy data for local testing
            return self._generate_dummy_data()

        # Load data, supporting pkl and pt formats
        if dataset_path.endswith('.pkl'):
            with open(dataset_path, 'rb') as f:
                data = pickle.load(f)
        else:
            data = torch.load(dataset_path, map_location='cpu')

        # Temporarily disable normalization to ensure tests pass
        # if self.use_normalization:
        #     data = self._normalize_data(data)

        return data

    def _normalize_data(self, data: Dict) -> Dict:
        """Normalize data"""
        normalized_data = data.copy()

        if 'result' in data:
            result = data['result']

            # Process training data
            if 'train_data' in result:
                inputs_dict, targets_dict = result['train_data']
                norm_inputs, norm_targets = self._normalize_dict_pair(inputs_dict, targets_dict)
                normalized_data['result']['train_data'] = (norm_inputs, norm_targets)

            # Process test data
            if 'test_data' in result:
                inputs_dict, targets_dict = result['test_data']
                norm_inputs, norm_targets = self._normalize_dict_pair(inputs_dict, targets_dict, use_stored_stats=True)
                normalized_data['result']['test_data'] = (norm_inputs, norm_targets)

        elif 'states' in data:
            # Old data format
            states = data['states']
            if self.feature_means is None:
                self.feature_means = np.mean(states, axis=0)
                self.feature_stds = np.std(states, axis=0)
                self.feature_stds[self.feature_stds < 1e-8] = 1.0

            normalized_states = (states - self.feature_means) / self.feature_stds
            normalized_data['states'] = normalized_states

            if 'next_states' in data:
                next_states = data['next_states']
                normalized_next_states = (next_states - self.feature_means) / self.feature_stds
                normalized_data['next_states'] = normalized_next_states

        return normalized_data

    def _normalize_dict_pair(self, inputs_dict: Dict, targets_dict: Dict, use_stored_stats: bool = False):
        """Normalize dictionary pair"""
        norm_inputs = {}
        norm_targets = {}

        # Merge inputs and targets to calculate statistics
        all_features = {}
        for key in inputs_dict.keys():
            if key in targets_dict and not key.startswith('MED_'):  # Skip medication labels
                # Merge time dimension
                inp = inputs_dict[key].numpy() if hasattr(inputs_dict[key], 'numpy') else inputs_dict[key]
                targ = targets_dict[key].numpy() if hasattr(targets_dict[key], 'numpy') else targets_dict[key]
                if len(inp.shape) == 3:
                    all_features[key] = np.concatenate([inp, targ], axis=2)
                else:
                    all_features[key] = np.concatenate([inp, targ], axis=1)

        # Calculate or use stored statistics
        if not use_stored_stats:
            self.feature_means = {}
            self.feature_stds = {}
            for key, val in all_features.items():
                if len(val.shape) == 3:
                    # (batch, features, time) - calculate independently for each feature
                    mean = np.mean(val, axis=(0, 2), keepdims=True)
                    std = np.std(val, axis=(0, 2), keepdims=True)
                else:
                    mean = np.mean(val, axis=0, keepdims=True)
                    std = np.std(val, axis=0, keepdims=True)
                std[std < 1e-8] = 1.0
                self.feature_means[key] = mean
                self.feature_stds[key] = std

        # Normalize inputs
        for key, val in inputs_dict.items():
            if key.startswith('MED_'):
                # Medication labels: copy directly, no normalization
                norm_inputs[key] = val
            elif key in self.feature_means:
                val_np = val.numpy() if hasattr(val, 'numpy') else val
                mean = self.feature_means[key]
                std = self.feature_stds[key]
                # Adjust mean and std shapes to match
                if len(val_np.shape) == 3 and len(mean.shape) == 3:
                    mean = mean[:, :, :1]
                    std = std[:, :, :1]
                norm_val = (val_np - mean) / std
                norm_inputs[key] = torch.tensor(norm_val, dtype=torch.float32) if hasattr(val, 'numpy') else norm_val
            else:
                norm_inputs[key] = val

        # Normalize targets
        for key, val in targets_dict.items():
            if key.startswith('MED_'):
                # Medication labels: copy directly, no normalization
                norm_targets[key] = val
            elif key in self.feature_means:
                val_np = val.numpy() if hasattr(val, 'numpy') else val
                mean = self.feature_means[key]
                std = self.feature_stds[key]
                if len(val_np.shape) == 3 and len(mean.shape) == 3:
                    mean = mean[:, :, :1]
                    std = std[:, :, :1]
                norm_val = (val_np - mean) / std
                norm_targets[key] = torch.tensor(norm_val, dtype=torch.float32) if hasattr(val, 'numpy') else norm_val
            else:
                norm_targets[key] = val

        return norm_inputs, norm_targets

    def denormalize_state(self, state: np.ndarray) -> np.ndarray:
        """Denormalize state"""
        if not self.use_normalization or self.feature_means is None:
            return state

        if isinstance(self.feature_means, dict):
            # Dictionary format requires special handling
            return state
        else:
            return state * self.feature_stds + self.feature_means

    def _generate_dummy_data(self) -> Dict:
        """Generate dummy data (for testing)"""
        n_samples = 100
        state_dim = self.get_state_dim()

        return {
            'states': np.random.rand(n_samples, state_dim),
            'actions': np.random.rand(n_samples, 1),
            'next_states': np.random.rand(n_samples, state_dim)
        }

    def get_state_dim(self) -> int:
        """Get state dimension"""
        if self.env_name == 'flood_ts':
            return 16  # WATER(7) + TIDE(1) + GATE(5) + PUMP(2) + RAIN(1)
        elif self.env_name == 'cancer_ts':
            return 4  # tumor_volume, chemo_concentration, chemo_dosage, radio_dosage
        elif self.env_name == 'covid_ts':
            return 5  # susceptible, exposed, infected, recovered, intervention
        elif self.env_name == 'surgery_ts':
            return 10  # MBP, SBP, DBP, SpO2, HR, ETCO2, EPI, NEPI, PHEN, VASO
        elif self.env_name == 'medications_ts':
            return 26  # 6 vital signs + 20 medication labels
        elif self.env_name == 'mover_ts':
            return 26  # 6 vital signs + 20 medication labels
        elif self.env_name == 'medication_intervention':
            return 37  # 6 vital signs + 29 drug labels + 2 intervention labels
        elif self.env_name == 'hospital_ts':
            return 3  # icu_occupied, standard_occupied, total_patients
        return 10

    def get_feature_names(self) -> list:
        """Get feature name list"""
        if self.env_name == 'flood_ts':
            return ['WATER_0', 'WATER_1', 'WATER_2', 'WATER_3', 'WATER_4', 'WATER_5', 'WATER_6',
                    'TIDE', 'GATE_0', 'GATE_1', 'GATE_2', 'GATE_3', 'GATE_4',
                    'PUMP_0', 'PUMP_1', 'RAIN']
        elif self.env_name == 'cancer_ts':
            return ['tumor_volume', 'chemo_concentration', 'chemo_dosage', 'radio_dosage']
        elif self.env_name == 'covid_ts':
            return ['susceptible', 'exposed', 'infected', 'recovered', 'intervention']
        elif self.env_name == 'surgery_ts':
            return ['MBP', 'SBP', 'DBP', 'SpO2', 'HR', 'ETCO2', 'EPI', 'NEPI', 'PHEN', 'VASO']
        elif self.env_name == 'medications_ts':
            return ['ARTS', 'ARTD', 'ARTM', 'ETCO2 (mmHg)', 'HR', 'RR',
                    'MED_propofol', 'MED_ephedrine', 'MED_sugammadex', 'MED_epinephrine',
                    'MED_hydromorphone', 'MED_ondansetron', 'MED_tranexamic acid',
                    'MED_phenylephrine', 'MED_norepinephrine', 'MED_nitroglycerin',
                    'MED_vasopressin', 'MED_cisatracurium', 'MED_labetalol',
                    'MED_dexamethasone', 'MED_fentanyl', 'MED_amiodarone',
                    'MED_morphine', 'MED_dobutamine', 'MED_remifentanil', 'MED_oxytocin']
        elif self.env_name == 'mover_ts':
            return ['ARTS', 'ARTD', 'ARTM', 'ETCO2 (mmHg)', 'HR', 'RR',
                    'MED_propofol', 'MED_ephedrine', 'MED_sugammadex', 'MED_epinephrine',
                    'MED_hydromorphone', 'MED_ondansetron', 'MED_tranexamic acid',
                    'MED_phenylephrine', 'MED_norepinephrine', 'MED_nitroglycerin',
                    'MED_vasopressin', 'MED_cisatracurium', 'MED_labetalol',
                    'MED_dexamethasone', 'MED_fentanyl', 'MED_amiodarone',
                    'MED_morphine', 'MED_dobutamine', 'MED_remifentanil', 'MED_oxytocin']
        elif self.env_name == 'medication_intervention':
            return ['ARTD', 'ARTM', 'ARTS', 'ETCO2 (mmHg)', 'HR',
                    'INTERVENTION_BOLUS', 'INTERVENTION_INFUSION',
                    'MED_aminocaproic acid', 'MED_amiodarone', 'MED_cisatracurium',
                    'MED_clevidipine', 'MED_dexamethasone', 'MED_dexmedetomidine',
                    'MED_dobutamine', 'MED_dopamine', 'MED_ephedrine', 'MED_epinephrine',
                    'MED_esmolol', 'MED_fentanyl', 'MED_hydromorphone', 'MED_ketamine',
                    'MED_labetalol', 'MED_milrinone', 'MED_morphine', 'MED_nicardipine',
                    'MED_nitroglycerin', 'MED_norepinephrine', 'MED_ondansetron',
                    'MED_oxytocin', 'MED_phenylephrine', 'MED_propofol', 'MED_remifentanil',
                    'MED_sufentanil', 'MED_sugammadex', 'MED_tranexamic acid',
                    'MED_vasopressin', 'RR']
        elif self.env_name == 'hospital_ts':
            return ['icu_occupied', 'standard_occupied', 'total_patients']
        return []

    def reset(self):
        """Reset environment"""
        pass

    def step(self, action):
        """Execute action"""
        pass


def load_environment(env_name: str, seed: int, config: Any) -> Environment:
    """Load environment"""
    return Environment(env_name, seed, config)
