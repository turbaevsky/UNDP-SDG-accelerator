#!/usr/bin/python3

from flask import Flask, jsonify, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import urllib.request
import spacy
import pdfplumber
import os
import re
from tqdm import tqdm

from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from werkzeug.routing import Rule
app.url_rule_class = lambda path, **options: Rule('/nlp' + path, **options)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load(fn):
    print(f'loading file...')
    #urllib.request.urlretrieve(path, f"{cnt}.pdf")
    txt, no = 'start', 0
    text = ''
    with pdfplumber.open(fn) as pdf:
        while txt != '':
            try:
                page = pdf.pages[no]
                text += page.extract_text()
                no += 1
                #print('.', end='')
            except Exception as e:
                print(e)
                break
    
    print('writting...')
    #print(text)
    out = app.config['UPLOAD_FOLDER'] + '/text.txt'
    with open(out, 'w') as textfile:
        textfile.write(text)
        

def clean(fn):
    with open(fn, 'r') as f:
        r = f.read()
        r = re.sub(r'[.]{2,}','',r)
        r = re.sub(r'[ ]+',' ',r)
        ##r = re.sub(r'\d{2} \d\.\d+','\n',r)
        r = re.sub(r'cid:\d+','\n',r)
        r = re.sub(r'\n',' ',r)
        ##r = re.sub(r'\d+.\d+\W+Goal \d+:','\nGoal: ',r)
        return r


def rule3(text):
    doc = nlp(text)
    sent = []
    for token in doc:
        # look for prepositions
        if token.pos_=='ADP':
            phrase = ''
            # if its head word is a noun
            if token.head.pos_=='NOUN':
                # append noun and preposition to phrase
                phrase += token.head.text
                phrase += ' '+token.text
                # check the nodes to the right of the preposition
                for right_tok in token.rights:
                    # append if it is a noun or proper noun
                    if (right_tok.pos_ in ['NOUN','PROPN']):
                        phrase += ' '+right_tok.text
                if len(phrase)>2:
                    sent.append(phrase)
    return sent


def search(doc, words, maybe, stop):
    for i, sent in enumerate(doc.sents):
        st = [s.text.lower() for s in sent]
        
        if all([w in st for w in words]) and \
        any([m in st for m in maybe]) and \
        not any([s in st for s in stop]) and \
        len(st)<100:
            #print(i,'=>', sent)
            return i

    
def entities(d):
    ent = []
    query = [[['poverty','all'], ['no','zero','end','goal','progress'], ['conclusion']],
        [['hunger','food','security'], ['no','zero','end','goal'], ['conclusion','annexes']],
        [['ensure','healthy','lives'], ['promote'], ['conclusion','annexes']],
        [['ensure','quality','education'], ['promote'], ['conclusion','annexes']],
        [['gender','equality'], ['women'], ['conclusion','annexes']],
        [['water','sanitation'], ['ensure'], ['conclusion','annexes']],
        [['modern','energy'], ['ensure'], ['conclusion','annexes']],
        [['economic','growth'], ['promote'], ['conclusion','annexes']],
        [['resilient','infrastructure'], ['promote'], ['conclusion','annexes']],
        [['reduce','inequality'], ['countries'], ['conclusion','annexes']],
        [['human','settlements'], ['safe'], ['conclusion','annexes']],    
        [['consumption','production'], ['pattern'], ['conclusion','annexes']],    
        [['climate','change'], ['combat'], ['conclusion','annexes']],
        [['oceans','seas','marines'], ['resources'], ['conclusion','annexes']],
        [['ecosystems','forests'], ['protect'], ['conclusion','annexes']],
        [['justice','societies'], ['promote'], ['conclusion','annexes']],
        [['revitalize','partnership'], ['development'], ['conclusion','annexes']]]
    print('finding entries...')
    for q in tqdm(query):
        ent.append(search(d, q[0], q[1], q[2])) 
    print(f'done:{ent}')
    return ent
    
    
def summ(d):
    doc = nlp(d)
    keyword = []
    stopwords = list(STOP_WORDS)
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    for token in doc:
        if(token.text in stopwords or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            keyword.append(token.text)
    freq_word = Counter(keyword)

    max_freq = Counter(keyword).most_common(1)[0][1]
    for word in freq_word.keys():  
        freq_word[word] = (freq_word[word]/max_freq)
    freq_word.most_common(5)

    sent_strength={}
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent]+=freq_word[word.text]
                else:
                    sent_strength[sent]=freq_word[word.text]
                
    summarized_sentences = nlargest(3, sent_strength, key=sent_strength.get)
    # summary
    final_sentences = [ w.text for w in summarized_sentences ]
    #summary = ' '.join(final_sentences)
    #print(summary)
    return final_sentences
    
    
def insight(d):
    global nlp
    ins = {}
    pos = entities(d)
    nlp = spacy.load("en_core_web_sm")
    for i in tqdm(range(len(pos)-1)):
        label = f'Goal {i+1}'
        start = pos[i] if pos[i] is not None else None
        end = pos[i+1] if (pos[i+1] is not None and pos[i] is not None and pos[i]<pos[i+1]) else pos[i]+50 if pos[i] is not None else None
        sl = ""
        if start is not None and end is not None:
            for j, s in enumerate(d.sents):
                if j>=start and j<=end:
                    sl += s.text
        try:
            if label in ins:
                ins[label] += summ(sl) 
            else:
                ins[label] = summ(sl)
        except Exception as e:
            print(e)
    return ins
            
    
################    
@app.route('/extract/<name>')
def extract(name):
    nlp = spacy.load("en_core_web_sm", disable=['tagger','ner','parser'])
    nlp.max_length=2e6
    nlp.add_pipe('sentencizer')
    
    fn = app.config["UPLOAD_FOLDER"]+'/'+name
    load(fn)
    fn = app.config["UPLOAD_FOLDER"]+'/text.txt'
    r = clean(fn)
    print('tokenization...')
    doc = nlp(r)
    return(jsonify(insight(doc)))
    

################
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('extract', name=filename))
    """
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
    """

if __name__=='__main__':
    #nlp = spacy.load("en_core_web_sm")
    app.run(debug=True, port=8055)
