from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel


def create_model(name, hf_config=None, vocab_size=None, train_all_params=False):

    if name.lower() == "gpt2":
        return _create_gpt2_model(vocab_size, name)
    if name.lower() == "iglm":
        from protein_tune_rl.models.decoder import Decoder

        model = AutoModelForCausalLM.from_pretrained(hf_config)
        model.resize_token_embeddings(vocab_size)

        return Decoder(model, name)

    if name.lower() == "iglm_w_linear_head":
        from protein_tune_rl.models.decoder import DecoderWithLinearHead

        model = AutoModelForCausalLM.from_pretrained(hf_config)
        model.resize_token_embeddings(vocab_size)

        return DecoderWithLinearHead(model, name, train_all_params)


def _create_gpt2_model(vocab_size, name):
    from protein_tune_rl.models.decoder import Decoder

    config = GPT2Config()
    config.vocab_size = vocab_size
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(vocab_size)

    return Decoder(model, name)
