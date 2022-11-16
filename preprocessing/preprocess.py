# For reading in data, cleaning data, and splitting data
import nltk

class Joke:
    def __init__(self, joke):
        if joke.find('Edit:') != -1:
            joke = joke[:joke.find('Edit:')]
        if joke.find('EDIT:') != -1:
            joke = joke[:joke.find('EDIT:')]
        self.joke = joke
        parts = self.joke.split('_____')
        self.body = parts[0]
        self.punchline = parts[1]
        if self.punchline.startswith(self.body):
            self.punchline = self.punchline[len(self.body):] # Remove title from reddit post body
        # TODO: Split into body and punchline via last sentence rather than "_____"

    def __str__(self):
        return 'BODY: ' + self.body + '\nPUNCHLINE: ' + self.punchline + '\n'


# Meant for reading from datasets/data/reddit_full tsv files
def read_tsv(filename):
    with open(filename, "r") as f:
        funny = []
        unfunny = []
        for line in f:
            line = line.replace('"', '').replace('”', '').replace('“', '').split(',', 3)
            if len(line[3]) >= 10 and (line[3][-10:-1] == '[removed]' or line[3][-10:-1] == '[deleted]'):
                continue # Ignore reddit posts that have been removed
            if (line[1] == "0"):
                unfunny.append(Joke(line[3].strip()))
            else:
                funny.append(Joke(line[3].strip()))
        return funny, unfunny

# This successfully splits funny and unfunny jokes.
# Further preprocessing: punctuation, split body and punchline?

if __name__ == '__main__':
    funny, unfunny = read_tsv('../datasets/data/reddit_full/train.tsv')
    print(len(funny), len(unfunny))
    for joke in funny:
        print(joke)