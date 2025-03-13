from typing import Any, Dict, List, Union

import torch


class metric:
    def single_seq_eval(self, chains) -> float:
        raise NotImplementedError

    def __call__(
        self, chains: Union[Dict, List[Dict]], extra_info: Any = None
    ) -> Union[float, torch.tensor]:
        if isinstance(chains, dict):
            return self.single_seq_eval(chains, extra_info)

        reward = torch.zeros(len(chains))
        for i, chain in enumerate(chains):
            reward[i] = self.single_seq_eval(chain)

        return reward
