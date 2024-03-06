from src.seq2seq.encoder import EncoderRNN
from src.seq2seq.decoder import BahdanauAttnDecoderRNN, LuongAttnDecoderRNN
import click
from src.seq2seq.train import train
from src.seq2seq.data import get_dataloader, device
from src.seq2seq.evaluate import compute_bleu_score, evaluate_first_n_pairs
from src.seq2seq.sampler import GreedySampling, TemperatureSampling


@click.command()
@click.option(
    "--attention_mechanism",
    default="bahdanau",
    type=click.Choice(["bahdanau", "luong"]),
)
@click.option(
    "--sampling_type", default="greedy", type=click.Choice(["greedy", "temperature"])
)
@click.option("--temperature", default=0.5, type=float)
def main(attention_mechanism, sampling_type, temperature):
    hidden_size = 128
    batch_size = 32

    input_lang, output_lang, train_dataloader, test_dataloader = get_dataloader(
        batch_size
    )

    decoders = {
        "bahdanau": BahdanauAttnDecoderRNN,
        "luong": LuongAttnDecoderRNN,
    }

    samplers = {
        "greedy": GreedySampling(),
        "temperature": TemperatureSampling(temperature=temperature),
    }

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = decoders[attention_mechanism](hidden_size, output_lang.n_words).to(device)
    train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)

    sampler = samplers[sampling_type]
    encoder.eval()
    decoder.eval()
    bleu_score = compute_bleu_score(
        encoder, decoder, test_dataloader, input_lang, output_lang, sampler
    )
    print(f"BLEU score: {bleu_score}")

    # evaluate randomly to see the results
    evaluate_first_n_pairs(encoder, decoder, sampler, n=50)


if __name__ == "__main__":
    main()
