# For reading in data, cleaning data, and splitting data
import nltk.data
import string
from string import punctuation

from csv import writer

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

class Joke:
    def __init__(self, joke):
        if joke.find('Edit :') != -1:
            joke = joke[:joke.find('Edit :')]
        if joke.find('EDIT :') != -1:
            joke = joke[:joke.find('EDIT:')]
        if joke.find('edit :') != -1:
            joke = joke[:joke.find('edit :')]
        
        reddit_parts = joke.split('_____')
        reddit_title = reddit_parts[0]
        reddit_body = reddit_parts[1]
        # add punctuation if title does not end with punctuation
        # this is a hacky approach because not every title should end with a period
        if reddit_body.startswith(reddit_title):
            reddit_body = reddit_body[len(reddit_title):] # Remove title from reddit post body
        if not reddit_title.endswith(tuple(punctuation)):
            reddit_title = reddit_title + '.'
        # self.joke = reddit_title + " " + reddit_body
        
        tokenized_body = tokenizer.tokenize(reddit_body)
        # body: every sentence except the last one
        if len(tokenized_body) > 0:
            self.body = reddit_title + " "
            self.body += " ".join(tokenized_body[:-1])
            # punchline: last sentence
            self.punchline = tokenized_body[-1]
        # NOTE: relies on tsv properly reading empty cells in first column
        else:
            self.punchline = reddit_title
            self.body = ""

    def __str__(self):
        return 'BODY: ' + self.body + '\nPUNCHLINE: ' + self.punchline + '\n'


# Meant for reading from datasets/data/reddit_full tsv files
def read_tsv(filename):
    with open(filename, "r") as f:
        funny = []
        unfunny = []
        for line in f:
            line = line.replace('"', '').replace('”', '').replace('“', '').replace('&#x200B;', '').replace('&nbsp;', ' ').replace('’', "'").replace("‘", "'")
            line = line.split(',', 3)
            if len(line[3]) >= 10 and (line[3][-10:-1] == '[removed]' or line[3][-10:-1] == '[deleted]'):
                continue # Ignore reddit posts that have been removed
            line[3] = line[3].strip()
            for c in string.punctuation:
                if c != '_':
                    line[3] = line[3].replace(c, ' ' + c + ' ')
            if (line[1] == "0"):
                joke = Joke(line[3].strip())
                unfunny.append([joke.body, joke.punchline])
            else:
                joke = Joke(line[3].strip())
                funny.append([joke.body, joke.punchline])
        return funny, unfunny

# This successfully splits funny and unfunny jokes.
# Further preprocessing: punctuation, split body and punchline?

if __name__ == '__main__':
    funny, unfunny = read_tsv('../datasets/data/reddit_full/train.tsv')
    test_funny, test_unfunny = read_tsv('../datasets/data/reddit_full/test.tsv')
    # print(len(funny), len(unfunny))
    fields = ['body', 'punchline']
    dir = '../datasets/data/reddit_preprocessed/'
    files = ['funny.tsv', 'unfunny.tsv', 'test_funny.tsv', 'test_unfunny.tsv']
    jokes = [funny, unfunny, test_funny, test_unfunny]
    for file, joke_type in zip(files, jokes):
        with open(dir + file, 'w') as f:
            w = writer(f, delimiter='\t')
            w.writerow(fields)
            num = 0
            for joke in joke_type:
                if joke[0] != "":
                    w.writerow(joke)
                else:
                    num += 1
        print('There were ' + str(num) + ' jokes with empty bodies')
