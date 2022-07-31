import os.path

from gensim.models.word2vec import Word2Vec

from callback import callback
import config


def train_word2vec(sentences):
    model_name = "word2vec.model"
    path = os.path.join(config.MODEL_PATH, model_name)
    if os.path.exists(path):
        model = Word2Vec.load(path)
    else:
        print('Training Word2Vec model')
        model = Word2Vec(sentences, window=8, min_count=5, size=config.W2V_SIZE,
                                    sg=1, hs=0, alpha=0.025, min_alpha=1e-4, negative=5,
                                    ns_exponent=0.75, compute_loss=True, callbacks=[callback()],
                                    seed=1234, iter=20, workers=4)
        model.save(path)
    return model.wv

