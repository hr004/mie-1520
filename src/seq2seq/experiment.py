from src.seq2seq.encoder import EncoderRNN
from src.seq2seq.decoder import DecoderRNN

from src.seq2seq.train import train
from src.seq2seq.data import get_dataloader, device
from src.seq2seq.evaluate import compute_bleu_score
from src.seq2seq.sampler import GreedySampling, TemperatureSampling

hidden_size = 128
batch_size = 32


def main():
    input_lang, output_lang, train_dataloader, test_dataloader = get_dataloader(
        batch_size
    )

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, output_lang.n_words).to(device)

    train(train_dataloader, encoder, decoder, 10, print_every=5, plot_every=5)

    sampler = TemperatureSampling(0.5)

    encoder.eval()
    decoder.eval()
    bleu_score = compute_bleu_score(
        encoder, decoder, test_dataloader, input_lang, output_lang, sampler
    )
    print(f"BLEU score: {bleu_score}")


if __name__ == "__main__":
    main()
