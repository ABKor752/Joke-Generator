# For reading in data, cleaning data, and splitting data

class Joke:
    def __init__(self, joke):
        self.joke = joke
        parts = joke.split('_____')
        self.body = parts[0]
        self.punchline = parts[1]
        # TODO: Split into body and punchline via last sentence rather than "_____"

    def __str__(self):
        return 'BODY: ' + self.body + '\nPUNCHLINE: ' + self.punchline + '\n'


# Meant for reading from datasets/data/reddit_full tsv files
def read_tsv(filename):
    with open(filename, "r") as f:
        funny = []
        unfunny = []
        for line in f:
            line = line.split(',', 3)
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
    for joke in funny:
        print(joke)