from src.encoder import EncoderRNN
from src.decoder import AttnDecoderRNN, LuongAttnDecoderRNN
import click
from src.train import train
from src.data import get_dataloader, device
from src.evaluate import compute_bleu_score, evaluate_first_n_pairs
from src.sampler import GreedySampling, TemperatureSampling


@click.command()
@click.option(
    "--attention_mechanism",
    default="bahdanau",
    type=click.Choice(["bahdanau", "luong"]),
)
@click.option(
    "--sampling_type", default="greedy", type=click.Choice(["greedy", "temperature"])
)
def main(attention_mechanism, sampling_type):
    hidden_size = 128
    batch_size = 32

    input_lang, output_lang, train_dataloader, test_dataloader = get_dataloader(
        batch_size
    )

    decoders = {
        "bahdanau": AttnDecoderRNN,
        "luong": LuongAttnDecoderRNN,
    }

    samplers = {
        "greedy": GreedySampling,
        "temperature": TemperatureSampling,
    }

    sampler = samplers[sampling_type]()
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = decoders[attention_mechanism](hidden_size, output_lang.n_words).to(device)
    train(train_dataloader, encoder, decoder, 10, print_every=5, plot_every=5)

    sampler = GreedySampling()
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
