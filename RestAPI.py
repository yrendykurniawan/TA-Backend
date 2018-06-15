#!flask/bin/python
from flask import Flask, jsonify
import facebook
import requests
from html.parser import HTMLParser
from cucco import Cucco
import collections, re
import nltk
from hmmtagger import MainTagger
from tokenization import *
from utils.files import get_full_path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


token = '421088441576297|zZlbhj3mvFGSUtUwz0-YitI0FQ8'

app = Flask(__name__)

tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]

mt = None

def init_tag():
    global mt
    if mt is None:
        mt = MainTagger(get_full_path("resource/Lexicon.trn") , get_full_path("resource/Ngram.trn"), 0, 3, 3, 0, 0, False, 0.2, 0, 500.0, 1)

def do_tag(asd):
    responsed =asd
    lines = responsed.strip().split("\n")

    result = []
    try:
        init_tag()
        for l in lines:

            if len(l) == 0: continue
            out = sentence_extraction(cleaning(l))

            for o in out:

                strtag = " ".join(tokenisasi_kalimat(o)).strip()

                result += [" ".join(mt.taggingStr(strtag))]

    except:
        return "Error Exception"
    #print("\n".join(result))
    return result

class spellCheck:
    # Peter Norvig Spelling Correction Algorithm based on Bayes' Theorem
    # http://norvig.com/spell-correct.html
    def train(self, features):
        model = collections.defaultdict(lambda: 1)
        for f in features:
            model[f] += 1
        return model

    def __init__(self):
        text_files = 'kumpulan_data/spellingset.txt'
        file = open(text_files, encoding="utf8").read()
        self.NWORDS = self.train(self.words(file))
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'

    def words(self, text): return re.findall('[a-z]+', text.lower())

    def edits1(self, word):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits for c in self.alphabet if b]
        inserts = [a + c + b for a, b in splits for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def known_edits2(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.NWORDS)

    def known(self, words): return set(w for w in words if w in self.NWORDS)

    def correct(self, word):
        candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
        return max(candidates, key=self.NWORDS.get)


class jalanSpellCheck:
    def __init__(self):
        self.__katadasar = [line.replace('\n', '') for line in open('kumpulan_data/rootword.txt').read().splitlines()]
        self.p = 0
        self.n = 0
        self.nn = 0

    def correctSpelling(self, text):
        sc = spellCheck()
        return sc.correct(text) if text not in self.__katadasar else text  # for t in text.split()

class slangWordCorrect:
    def jalan(self, text):
        # Alamat file
        slangWords_files = 'kumpulan_data/slangword.txt'
        # Open file
        file = open(slangWords_files, encoding="utf8").readlines()
        # Bikin dictionary buat menampung dari file
        listWord = {}
        # Load file ke dictionary
        creds = [cred.strip() for cred in file]
        for cred in creds:
            kataSalah, kataBenar = cred.split(':')
            listWord[kataSalah] = kataBenar

        # print(listWord)

        # split the words based on whitespace
        sentence_list = nltk.word_tokenize(text)
        # print(sentence_list)
        # make a place where we can build our new sentence
        new_sentence = []

        reformed = [listWord[word] if word in listWord else word for word in sentence_list]
        reformed = " ".join(reformed)
        return reformed

def read_dataset(fname, t_type):
    dataset = []
    f = open(fname, 'r', encoding="utf8")
    line = f.readline()
    while line != '':
        dataset.append([line, t_type])
        line = f.readline()
    f.close()

    return dataset

@app.route('/tasks/<int:task_id>', methods=['GET'])
def get_tasks(task_id):
    abc = []

    graph = facebook.GraphAPI(access_token=token, version=2.10)
    node = "/%s" % task_id

    video = graph.request(node + '/comments?fields=id,message,comment_count,'
                                 'reactions.type(LIKE).limit(0).summary(total_count).as(like),'
                                 'reactions.type(LOVE).limit(0).summary(total_count).as(love),'
                                 'reactions.type(WOW).limit(0).summary(total_count).as(wow),'
                                 'reactions.type(HAHA).limit(0).summary(total_count).as(haha),'
                                 'reactions.type(SAD).limit(0).summary(total_count).as(sad),'
                                 'reactions.type(ANGRY).limit(0).summary(total_count).as(angry)')
    #video = graph.request(node + '?fields='
    #                            'reactions.type(LIKE).limit(0).summary(total_count).as(like),'
    #                           'reactions.type(LOVE).limit(0).summary(total_count).as(love),'
    #                             'reactions.type(WOW).limit(0).summary(total_count).as(wow),'
    #                             'reactions.type(HAHA).limit(0).summary(total_count).as(haha),'
    #                             'reactions.type(SAD).limit(0).summary(total_count).as(sad),'
    #                             'reactions.type(ANGRY).limit(0).summary(total_count).as(angry)')

    # Wrap this block in a while loop so we can keep paginating requests until
    # finished.

    # Baca dataset
    joy_feel = read_dataset(get_full_path("dataset/isear/pp/filter/joy.txt"), 'joy')
    disgust_feel = read_dataset(get_full_path("dataset/isear/pp/filter/disgust.txt"), 'disgust')
    sadness_feel = read_dataset(get_full_path("dataset/isear/pp/filter/sadness.txt"), 'sadness')
    anger_feel = read_dataset(get_full_path("dataset/isear/pp/filter/anger.txt"), 'anger')
    fear_feel = read_dataset(get_full_path("dataset/isear/pp/filter/fear.txt"), 'fear')
    surprise_feel = read_dataset(get_full_path("dataset/isear/pp/filter/surpriseExtra.txt"), 'surprise')

    # filter away words that are less than 3 letters to form the training data
    dataku = []
    for (words, sentiment) in joy_feel + disgust_feel + sadness_feel + anger_feel + fear_feel + surprise_feel:
        dataku.append((words.rstrip(), sentiment))

    lines = []
    labels = []
    for words, sentiment in dataku:
        html_parser = HTMLParser()

        lines.append(html_parser.unescape(words))
        labels.append(sentiment)

    headlines, labels = lines, labels

    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(
            ngram_range=(2, 3), min_df=1, max_df=0.8,
            stop_words=frozenset(["saya", "sedang", "lagi", "adalah", "di", "dari", "karena", "dan",
                                  "dengan", "ke", "yang", "untuk", "itu", "orang"])
        )),
        ('tfidf_transformer', TfidfTransformer()),
        ('classifier', MultinomialNB())
    ])
    pipeline.fit(headlines, labels)
    angerx = 0
    joyx = 0
    surprisex = 0
    sadnessx = 0
    fearx = 0
    disgustx = 0
    while (True):
        try:
            # print("Get post comments data :")
            for each_video in video['data']:
                if each_video['message'] != "":
                    # connect to database
                    init_tag()
                    html_parser = HTMLParser()
                    spell_check = jalanSpellCheck()
                    koreksi_slang = slangWordCorrect()
                    cucco = Cucco()

                    kata = cucco.replace_emojis( each_video['message'] )

                    # Escape HTML
                    kata = html_parser.unescape(each_video['message'])
                    kata = " ".join(kata.split())

                    # Hapus emoji
                    kata = cucco.replace_emojis(kata)

                    normalizations = [
                        'remove_extra_white_spaces'
                    ]

                    # Hapus extra spasi
                    kata = cucco.normalize(kata, normalizations)

                    kata= kata.replace('/', " ")

                    # Conver ke lowercase
                    kata = kata.lower()

                    # Hapus repeating character yang lebih dari 2
                    kata = re.sub(r'(.)\1+', r'\1\1', kata)

                    # Proses ,. yang sisa jadi 2
                    kata = kata.replace("..", ".")
                    kata = kata.replace(",,", ",")
                    kata = kata.replace("!!", "!")
                    kata = kata.replace("??", "?")



                    # Tambahkan spasi habis titik
                    rx = r"\.(?=\S)"
                    kata = re.sub(rx, ". ", kata)

                    # Slang correction
                    kata = koreksi_slang.jalan(kata)

                    # Spellcheck error
                    #tampung_kata_1 = []
                    #tampung_1 = kata.split()
                    #for word in tampung_1:
                    #    tampung_kata_1.append(spell_check.correctSpelling(word))
                    #kata = " ".join(tampung_kata_1)
                    asdqwe = kata

                    # Check apakah ada tanda baca di akhir
                    if (re.match('.*[^.?!]$', kata) is not None) == True:
                        kata = kata + " ."

                    resultx = do_tag(kata)
                    kata = " ".join(resultx)


                    #print(words)
                    #xxx = "".join([" " + i for i in words]).strip()

                    #kata = xxx



                    if kata != "":
                        linesz = []
                        linesz.append(kata)
                        words = []
                        for y in linesz:
                            lines = y.split()
                            for x in lines:
                                word = x.split("/")
                                chars_to_remove = set((",", "IN", "CC", "SC", "CDO", "CDC", "CDP", "CDI",
                                                       "DT", "MD", "OP", "CP", "SYM", "."
                                                       ))
                                if word[1] not in chars_to_remove:
                                    words.append(word[0] + "_" + word[1])
                            resultx = "".join([" " + i for i in words]).strip()
                            #print(resultx)

                        cobaa = []
                        cobaa.append(resultx)
                        for x in pipeline.predict(cobaa):
                            hasilx = x
                        if hasilx == 'anger':
                            angerx = angerx + 1
                        elif hasilx == 'joy':
                            joyx = joyx + 1
                        elif hasilx == 'sadness':
                            sadnessx = sadnessx + 1
                        elif hasilx == 'fear':
                            fearx = fearx + 1
                        elif hasilx == 'disgust':
                            disgustx = disgustx + 1
                        elif hasilx == 'surprise':
                            surprisex = surprisex + 1

                        comments_data = {
                            'id': each_video['id'],
                            'komen': each_video['message'],
                            'asdqwe':asdqwe,
                            'komen_edit': resultx,
                            'prediksi' : hasilx,
                            'like_count': each_video['like']['summary']['total_count'],
                            'love_count':each_video['love']['summary']['total_count'],
                            'wow_count': each_video['wow']['summary']['total_count'],
                            'haha_count': each_video['haha']['summary']['total_count'],
                            'sad_count' : each_video['sad']['summary']['total_count'],
                            'angry_count': each_video['angry']['summary']['total_count']
                        }

                    abc.append(comments_data)
            # Attempt to make a request to the next page of data, if it exists.
            video = requests.get(video['paging']['next']).json()
        except KeyError:
            # When there are no more pages (['paging']['next']), break from the
            # loop and end the script.
            break

    ctrku = {
        'anger' : angerx,
        'joy' : joyx,
        'sadness' : sadnessx,
        'fear' : fearx,
        'surprise' : surprisex,
        'disgust' : disgustx
        }

    #comments_data = {
    #    'id' : video['comment_count'],
    #    'video_like' : video['like']['summary']['total_count'],
    #    'video_love': video['love']['summary']['total_count'],
    #    'video_wow': video['wow']['summary']['total_count'],
    #    'video_haha': video['haha']['summary']['total_count'],
    #    'video_sad': video['sad']['summary']['total_count'],
    #    'video_angry': video['angry']['summary']['total_count']
    #    }
    abc.append(comments_data)

    return jsonify({'tasks': abc},{'ASD' : ctrku})

if __name__ == '__main__':
    app.run(debug=True)