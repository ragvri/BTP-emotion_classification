# return [x_train:np, language:int], [y_classification:int, y_intenisty:float]
import fastText
import numpy as np
import re
import json
import emoji
import pandas as pd


class Generator:
    def __init__(self, file_path, ft, labels2Idx: dict, emotions_dictionary: str = None):
        '''
        file_path: path of file to be used by the generator
        ft: fastText loaded model
        labels2Idx: dictionary mapping labels to ids
        emotions_dictionary: path to emotions dictionary
        '''
        self.file_path = file_path
        self.ft = ft
        self.labels2Idx = labels2Idx
        self.count = 0
        self.emotions_dictionary = emotions_dictionary

    def twitter_tokenizer(self, line: str, emotions_dictionary: str = None):
        '''
        line is the tweet to tokenize. 
        emotions_dictionary: is the file path to the emotion dictionary. It is used to tokenize 
        emoticons
        '''
        line = emoji.demojize(line)
        line = re.sub(r'http\S+', 'URL', line)
        line = re.sub('@[\w_]+', 'USER_MENTION', line)
        line = re.sub('\|LBR\|', '', line)
        line = re.sub('\.\.\.+', '...', line)
        line = re.sub('!!+', '!!', line)
        line = re.sub('\?\?+', '??', line)
        words = re.compile(
            '[\U00010000-\U0010ffff]|[\w-]+|[^ \w\U00010000-\U0010ffff]+', re.UNICODE).findall(line.strip())
        words = [w.strip() for w in words if w.strip() != '']

        if emotions_dictionary is not None:
            words_emojified = []
            json_file = open(emotions_dictionary)
            json_str = json_file.read()
            json_data = json.loads(json_str)  # emotions dict
            for word in words:
                if word in json_data:
                    new_word = json_data[word]
                    words_emojified.append(new_word)
                else:
                    words_emojified.append(word)
            return words_emojified

        return(words)

    def total_data(self):
        data = pd.read_csv(self.file_path, sep='\t', header=None)
        return len(data)

    def process_features(self, data, seq_len):
        features = np.zeros((seq_len, 300), dtype=np.float32)
        tweet = self.twitter_tokenizer(data, self.emotions_dictionary)
        for j in range(min(seq_len, len(tweet))):
            features[j] = self.ft.get_word_vector(tweet[j])
        return features

    def generate(self, batch_size: int, seq_len: int = 75):
        data_len = self.total_data()
        data = pd.read_csv(self.file_path, sep='\t', header=None)
        # print(self.count)
        while True:
            batch_features_ft = np.zeros(
                (batch_size, seq_len, 300), dtype=np.float32)
            batch_class = np.zeros((batch_size, 1), dtype=int)
            batch_intensity = np.zeros((batch_size, 1), dtype=float)

            for i in range(batch_size):
                # print(self.count)
                try:
                    batch_features_ft[i] = self.process_features(
                        data[0].loc[self.count], seq_len)
                    batch_class[i][0] = self.labels2Idx[data[1].loc[self.count]]
                    batch_intensity[i][0] = data[2].loc[self.count]
               
                except Exception as e:
                    # print(data[0].loc[self.count])
                    print(self.count, data_len)
                    # exit()
                self.count = (self.count + 1) % data_len

            yield ([batch_features_ft, 0], [batch_class, batch_intensity])


if __name__ == "__main__":

    ft = fastText.load_model(
        '/home1/zishan/WordEmbeddings/FastText/wiki.hi.bin')
    labels2Idx = {'SADNESS': 0, 'FEAR/ANXIETY': 1, 'SYMPATHY/PENSIVENESS': 2, 'JOY': 3,
                  'OPTIMISM': 4, 'NO-EMOTION': 5, 'DISGUST': 6, 'ANGER': 7, 'SURPRISE': 8}
    train_generator = Generator(
        '/home1/zishan/raghav/emotion/MultilingualMultitask/dataset/news_hindi/data_with_intensity_hindi.txt', ft, labels2Idx)
    x, y = next(train_generator.generate(4))
    print(type(x[0]))
    # print(x[0].shape)
    # print(x[1])
    # print(y[0].shape)
    # print(y[1].shape)
    # next(train_generator.generate(4))
    # next(train_generator.generate(4))
    # line = "http:google.com @ragvri :) happy day "
    # print(twitter_tokenizer(line, '/home1/zishan/raghav/emotion/MultilingualMultitask/dataset/emoji_dictionary.json'))
