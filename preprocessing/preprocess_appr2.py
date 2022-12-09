import csv
import random
from transformers import pipeline

NUM_PARAPHRASED_PUNCHLINES = 5
PATH_TO_FUNNY_TRAIN = '../datasets/data/reddit_preprocessed/funny.tsv'
PATH_TO_UNFUNNY_TRAIN = '../datasets/data/reddit_preprocessed/unfunny.tsv'
PATH_TO_FUNNY_TEST = '../datasets/data/reddit_preprocessed/test_funny.tsv'
PATH_TO_UNFUNNY_TEST = '../datasets/data/reddit_preprocessed/test_unfunny.tsv'
PATH_TO_HUMOR_DETECTION_MODEL = "../humor_detector/model/"
PATH_TO_DATA_WRITE_DIR = '../datasets/data/reddit_preprocessed/appr2/'

class QualityControlPipeline:
    
    def __init__(self, type):
        assert type in ['captions', 'questions', 'sentences']
        self.pipe = pipeline('text2text-generation', model=f'ibm/qcpg-{type}')
        self.ranges = {
            'captions': {'lex': [0, 90], 'syn': [0, 80], 'sem': [0, 95]},
            'sentences': {'lex': [0, 100], 'syn': [0, 80], 'sem': [0, 95]},
            'questions': {'lex': [0, 90], 'syn': [0, 75], 'sem': [0, 95]}
        }[type]

    def __call__(self, text, lexical, syntactic, semantic, **kwargs):
        assert all([0 <= val <= 1 for val in [lexical, syntactic, semantic]]), \
                 f' control values must be between 0 and 1, got {lexical}, {syntactic}, {semantic}'
        names = ['semantic_sim', 'lexical_div', 'syntactic_div']
        control = [int(5 * round(val * 100 / 5)) for val in [semantic, lexical, syntactic]]
        control ={name: max(min(val , self.ranges[name[:3]][1]), self.ranges[name[:3]][0]) for name, val in zip(names, control)}
        control = [f'COND_{name.upper()}_{control[name]}' for name in names]
        assert all(cond in self.pipe.tokenizer.additional_special_tokens for cond in control)
        text = ' '.join(control) + text if isinstance(text, str) else [' '.join(control) for t in text]
        return self.pipe(text, **kwargs)

# Meant for reading PREPROCESSED reddit datasets
def read_tsv(filename):
    with open(filename, "r") as f:
        paraphraser = QualityControlPipeline('sentences')
        # TODO: make sure to get the LOGITS and not the discrete class!
        # probably have to do it here and not in the loop through the tsv
        humor_predictor = pipeline("sentiment-analysis", model=PATH_TO_HUMOR_DETECTION_MODEL)

        tsv_file = csv.reader(f, delimiter='\t')
        # skip header
        next(tsv_file)
        examples = []
        for line in tsv_file:
            body = line[0]
            inc = 0.5 / NUM_PARAPHRASED_PUNCHLINES
            for i in range(NUM_PARAPHRASED_PUNCHLINES):
                new_punchline = paraphraser(line[1], lexical=0.2 + inc*i, syntactic=1 - inc*i, semantic=0.2 + inc*i)[0]['generated_text']
                # print(new_punchline)
                whole_joke = body + ' ' + new_punchline
                # TODO: make sure to get the LOGITS and not the discrete class!
                humor_level = humor_predictor(whole_joke)[0]['score'] if humor_predictor(whole_joke)[0]['label'] == 'POSITIVE' else 1 - humor_predictor(whole_joke)[0]['score']
                examples.append([body, new_punchline, humor_level])
        return examples

# def augment_jokes(bodies, punchlines):
#     assert(len(bodies) == len(punchlines), "mismatch in number of bodies and number of punchlines")
    

#     for i in range(len(bodies)):


#     return augmented_jokes

if __name__ == '__main__':
    train = read_tsv(PATH_TO_FUNNY_TRAIN) + read_tsv(PATH_TO_UNFUNNY_TRAIN)
    test = read_tsv(PATH_TO_FUNNY_TEST) + read_tsv(PATH_TO_UNFUNNY_TEST)

    random.shuffle(train)
    random.shuffle(test)

    fields = ['body', 'punchline', 'humor_level']
    files = ['train.tsv', 'test.tsv']
    jokes = [train, test]
    for file, joke_type in zip(files, jokes):
        with open(PATH_TO_DATA_WRITE_DIR + file, 'w') as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(fields)
            num = 0
            for joke in joke_type:
                w.writerow(joke)