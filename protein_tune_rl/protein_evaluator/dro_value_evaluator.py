import pandas as pd
import torch
import torch.distributed as dist

from protein_tune_rl import logger
from protein_tune_rl.collator import create_collator
from protein_tune_rl.dataloader import create_dataloader
from protein_tune_rl.dataset import create_dataset
from protein_tune_rl.models import create_model
from protein_tune_rl.protein_evaluator.evaluator import Evaluator
from protein_tune_rl.tokenizer import create_tokenizer


class DROValueEvaluator(Evaluator):
    def __init__(self, config):
        assert dist.get_world_size() == 1
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = self.config["evaluator"]["batch_size"]
        self.model_name = self.config["evaluator"]["model_name"]

        self.dataset = create_dataset(
            name=self.config['dataset']['name'],
            data_directory=self.config['dataset']['data_directory'],
            chain=self.config["dataset"]["chain"],
            region=self.config["dataset"]["region"],
        )

        self.dataloader = create_dataloader(
            self.dataset, batch_size=self.batch_size, shuffle=False
        )

        self.tokenizer = create_tokenizer(
            name=self.config['tokenizer']['name'],
            tokenizer_config=self.config['tokenizer']['tokenizer_config'],
        )

        self.collator = create_collator(
            name=self.config['collator']['name'],
            model_name='gpt2',
            tokenizer=self.tokenizer,
        )

        self.value = create_model(
            name="iglm_w_linear_head",
            hf_config=self.config['value_model']['dir'],
            vocab_size=self.tokenizer.vocab_size,
        ).to(self.device)

        self.value.load_state_dict(
            torch.load(
                "/p/vast1/hayes56/optlm/dro/norm_ss_perc_sheet/2025_04_10_11_15_47_dro_steps_100000_bs_1024_lr_0.0001/value_model_50000.bin",
                weights_only=True,
            )
        )

    def run(self, output_dir):

        eval_df = pd.read_csv(self.config['dataset']['data_directory'])
        scores = []
        self.value.eval()

        for batch_number, batch in enumerate(iter(self.dataloader)):

            tokenized_batch = self.collator(batch, eval=True)

            value_attention_mask = torch.ones(tokenized_batch['prompts'].shape) * (
                tokenized_batch['prompts'] != 0
            )
            pred_scores = (
                self.value(
                    tokenized_batch['prompts'].to(self.device),
                    attention_mask=value_attention_mask.to(self.device),
                )
                .float()
                .flatten()
                .cpu()
            )

            logger.info(f"batch {batch_number}; completed")

            scores += list(pred_scores.detach().numpy())

        eval_df['predicted_score'] = scores

        eval_df.to_csv(f"{output_dir}/{self.model_name}_eval.csv")

        return eval_df
