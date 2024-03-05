import torch
import random
from src.seq2seq.data import input_lang, output_lang, pairs, EOS_token, tensorFromSentence
from sacrebleu.metrics import BLEU


def evaluate(encoder, decoder, sentence, input_lang, output_lang, sampler):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(
            encoder_outputs, encoder_hidden
        )

        topi = sampler.sample(decoder_outputs)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


def evaluate_first_n_pairs(encoder, decoder, sampler, n=10):
    for i in range(n):
        # pair = random.choice(pairs)
        pair = pairs[i]
        print(">", pair[0])
        print("=", pair[1])
        output_words, _ = evaluate(
            encoder, decoder, pair[0], input_lang, output_lang, sampler
        )
        output_sentence = " ".join(output_words)
        print("<", output_sentence)
        print("")


def compute_bleu_score(encoder, decoder, test_pairs, input_lang, output_lang, sampler):
    """
    Compute the BLEU score for the given encoder, decoder, and test pairs.

    Parameters:
    encoder (EncoderRNN): The encoder model.
    decoder (AttnDecoderRNN): The decoder model.
    test_pairs (list): A list of test pairs, where each pair is a tuple of input and target sequences.

    Returns:
    float: The BLEU score.
    """

    references = []
    hypotheses = []

    for pair in test_pairs:
        references.append(pair[1])
        output_words, _ = evaluate(
            encoder, decoder, pair[0], input_lang, output_lang, sampler
        )
        output_sentence = " ".join(output_words)
        hypotheses.append(output_sentence)

    bleu = BLEU().corpus_score(hypotheses, [references])
    return bleu.score
