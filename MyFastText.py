from gensim.models.fasttext import FastText as FT_gensim
from gensim.test.utils import datapath
import os

def create_my_model(path=os.getcwd()+'/new_train.csv'):
    print('Using gensim libraries...')
    
    # Set file names for train and test data
    corpus_file = datapath(path)
    model = FT_gensim(size=100)
    
    print('build the vocabulary...')
    model.build_vocab(corpus_file=corpus_file)
    
    print('train the model...')
    model.train(
        corpus_file=corpus_file, 
        epochs=10,
        total_examples=model.corpus_count, 
        total_words=model.corpus_total_words,
        model='skipgram',
        iter=20,
        threads=1,
        min_count=3)
    
    model.save('./fasttext_model/model_FT.model')
    
if __name__ == "__main__":
    # execute only if run as a script
    create_my_model()
    exit()