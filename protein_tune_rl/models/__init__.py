from transformers import GPT2Config, GPT2LMHeadModel, AutoModelForCausalLM

def create_model(name, hf_config=None, vocab_size=None):

    if name.lower() == "gpt2":
        from protein_tune_rl.models.decoder import Decoder
        config = GPT2Config()
        config.vocab_size = vocab_size
        model = GPT2LMHeadModel(config)
        model.resize_token_embeddings(vocab_size)

        return Decoder(model, name)



        
    