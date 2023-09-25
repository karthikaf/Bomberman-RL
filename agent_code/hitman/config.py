import numpy as np
from typing import Any, Dict, List


class GameState():
    def __init__(self, all_features: Dict[str, List[Any]]):
        for k, v in all_features.items():
            assert len(v) > 0, f"Unsupported: Feature '{k}' is an empty list."

        self.all_features = all_features

    def get_index(self, features: Dict[str, Any]):
        assert set(features.keys()) == set(self.all_features.keys()), \
            f"Missing features {set(self.all_features.keys()) - set(features.keys())}."

        idx = 0
        shift = 1
        for k in self.all_features.keys():
            idx += shift * self.all_features[k].index(features[k])
            shift *= len(self.all_features[k])

        return idx

    def get_state(self, idx):

        state = {}
        value_lengths = [len(v) for v in list(self.all_features.values())]

        for i, k in enumerate(list(self.all_features.keys())[::-1]):
            shift = int(np.prod(value_lengths[:-(1+i)]))
            feature_index = idx // shift
            idx = idx % shift
            state[k] = self.all_features[k][feature_index]

        return state

    def __len__(self):
        return np.prod([len(v) for v in list(self.all_features.values())])
