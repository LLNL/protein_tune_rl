from transformers import GPT2Config, GPT2LMHeadModel, AutoModelForCausalLM

def create_model(model_name, hf_config, tokenizer):

    if model_name.lower() == "gpt2":
        from protein_tune_rl.models.decoder import Decoder
        config = GPT2Config()
        config.vocab_size = tokenizer.vocab_size
        model = GPT2LMHeadModel(config)
        model.resize_token_embeddings(tokenizer.vocab_size)

        return Decoder(model, model_name)


        
    