import sys

sys.path.insert(1, 'src/')
import utilities
from src import config

SARC_DATA = config.SARC_DATA
SARC_MAIN = config.SARC_MAIN
SARC_POL = config.SARC_POL

from fastai.text import *


def create_csv_labels(f_name='bunch_data.csv'):
    if not os.path.exists(f_name):
        learn_data = []

        for index, label in enumerate(train_labels):
            ancestor = train_docs['ancestors'][index][0]
            response1 = train_docs['responses'][index][0]
            response2 = train_docs['responses'][index][1]
            label1 = label[0]
            label2 = label[1]
            learn_data.append([label1, ancestor + ' - ' + response1])
            learn_data.append([label2, ancestor + ' - ' + response2])

        with open(f_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(learn_data)


if __name__ == "__main__":
    train_path = SARC_MAIN + 'train-balanced.csv'
    test_path = SARC_MAIN + 'test-balanced.csv'
    comment_path = SARC_MAIN + 'comments.json'

    train_docs, test_docs, train_labels, test_labels = \
        utilities.load_sarc_responses(train_path, test_path, comment_path)

    create_csv_labels()

    if not os.path.exists('lm_databunch'):
        ml_bunch = TextLMDataBunch.from_csv(path='.', csv_name='bunch_data.csv')

    data_lm = load_data('.', 'lm_databunch', bs=bs)

    learn_lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)

    bs = 64

    lr = 2e-2
    lr *= bs / 48

    learn_lm.to_fp16()
    learn_lm.fit_one_cycle(1, lr * 10, moms=(0.8, 0.7))
    learn_lm.unfreeze()
    print("Unfrozen Layers!")
    learn_lm.fit_one_cycle(10, lr, moms=(0.8, 0.7))

    learn_lm.save('fine_tuned')
    learn_lm.save_encoder('fine_tuned_enc')
