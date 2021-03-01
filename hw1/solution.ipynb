{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to /home/coder/nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n",
      "[nltk_data] Downloading package stopwords to /home/coder/nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import os\n",
    "import re\n",
    "import gensim\n",
    "import logging\n",
    "import nltk.data \n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import urllib.request\n",
    "import zipfile\n",
    "import pandas as pd\n",
    "import string\n",
    "import pymorphy2\n",
    "import fasttext\n",
    "\n",
    "from sklearn.metrics import f1_score\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.manifold import TSNE\n",
    "from tqdm import tqdm_notebook\n",
    "from collections import Counter\n",
    "from bs4 import BeautifulSoup\n",
    "from nltk.corpus import stopwords\n",
    "from gensim.models import KeyedVectors\n",
    "from gensim.models import word2vec\n",
    "from nltk.tokenize import sent_tokenize, RegexpTokenizer\n",
    "nltk.download('punkt')\n",
    "nltk.download('stopwords')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "# Часть 1. Эксплоративный анализ\n",
    "\n",
    "##### 1. Найдите топ-1000 слов по частоте без учёта стоп-слов."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "Подсчитаем количество всех возможных слов в тексте"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c63329661c394620a34a87ab0fd6d3f2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=0, max=36225), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "dirname = 'hpac_lower_tokenized/hpac_source/'\n",
    "\n",
    "cnt = Counter()\n",
    "\n",
    "for filename in tqdm_notebook(os.listdir(dirname)):\n",
    "    with open(dirname + filename, 'r') as text:\n",
    "        cnt.update(text.readline().split(' '))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "удалим стоп-слова, пунктуацию и апострофы из счётчика"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {},
   "outputs": [],
   "source": [
    "stops = stopwords.words(\"english\") + list(string.punctuation) + \\\n",
    "    ['\\'\\'', '``', '\\'s', 'n\\'t', '\\'d', '--', '...', '\\'re', '\\'ll', '\\'ve', '\\'m']\n",
    "\n",
    "for key, value in cnt.items():\n",
    "    if key in stops:\n",
    "        cnt[key] = 0"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "топ-1000 слов по частоте:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('harry', 3991017),\n",
       " ('said', 2262072),\n",
       " ('would', 1903185),\n",
       " ('hermione', 1826879),\n",
       " ('could', 1687864),\n",
       " ('back', 1396452),\n",
       " ('draco', 1386180),\n",
       " ('one', 1376314),\n",
       " ('like', 1256561),\n",
       " ('know', 1179021),\n",
       " ('eyes', 1032498),\n",
       " ('time', 1005205),\n",
       " ('ron', 904175),\n",
       " ('looked', 893362),\n",
       " ('get', 846353),\n",
       " ('asked', 826344),\n",
       " ('well', 783881),\n",
       " ('even', 769710),\n",
       " ('around', 767192),\n",
       " ('see', 743647),\n",
       " ('head', 731227),\n",
       " ('going', 717904),\n",
       " ('think', 715577),\n",
       " ('still', 691923),\n",
       " ('go', 666027),\n",
       " ('severus', 654494),\n",
       " ('face', 652601),\n",
       " ('way', 652261),\n",
       " ('room', 646707),\n",
       " ('ginny', 637918),\n",
       " ('hand', 632499),\n",
       " ('sirius', 628666),\n",
       " ('something', 622341),\n",
       " ('want', 612850),\n",
       " ('thought', 607347),\n",
       " ('potter', 602535),\n",
       " ('right', 601611),\n",
       " ('snape', 597675),\n",
       " ('away', 579736),\n",
       " ('much', 577283),\n",
       " ('look', 565629),\n",
       " ('two', 562458),\n",
       " ('never', 557605),\n",
       " ('really', 525853),\n",
       " ('knew', 524579),\n",
       " ('first', 515131),\n",
       " ('let', 512802),\n",
       " ('made', 510792),\n",
       " ('good', 501137),\n",
       " ('malfoy', 483569),\n",
       " ('little', 480709),\n",
       " ('wand', 479615),\n",
       " ('felt', 475883),\n",
       " ('dumbledore', 472012),\n",
       " ('turned', 471850),\n",
       " ('james', 467757),\n",
       " ('come', 460185),\n",
       " ('got', 451705),\n",
       " ('make', 442598),\n",
       " ('took', 436058),\n",
       " ('remus', 432641),\n",
       " ('lily', 421348),\n",
       " ('though', 421212),\n",
       " ('sure', 413567),\n",
       " ('say', 411762),\n",
       " ('door', 407945),\n",
       " ('tell', 406042),\n",
       " ('take', 405349),\n",
       " ('us', 404001),\n",
       " ('looking', 400846),\n",
       " ('dark', 399376),\n",
       " ('voice', 396811),\n",
       " ('voldemort', 394760),\n",
       " ('last', 393860),\n",
       " ('long', 391363),\n",
       " ('told', 387717),\n",
       " ('need', 387573),\n",
       " ('left', 381585),\n",
       " ('yes', 379265),\n",
       " ('man', 375032),\n",
       " ('wanted', 368940),\n",
       " ('anything', 367090),\n",
       " ('next', 356988),\n",
       " ('oh', 351707),\n",
       " ('came', 343059),\n",
       " ('nodded', 341020),\n",
       " ('love', 339038),\n",
       " ('moment', 335972),\n",
       " ('people', 334474),\n",
       " ('saw', 332916),\n",
       " ('another', 332450),\n",
       " ('things', 328154),\n",
       " ('went', 325804),\n",
       " ('hands', 324495),\n",
       " ('ca', 322864),\n",
       " ('help', 322411),\n",
       " ('day', 319711),\n",
       " ('enough', 319304),\n",
       " ('death', 318723),\n",
       " ('smiled', 312156),\n",
       " ('professor', 310769),\n",
       " ('year', 308744),\n",
       " ('mind', 308557),\n",
       " ('nothing', 307577),\n",
       " ('found', 306303),\n",
       " ('ever', 304241),\n",
       " ('boy', 303284),\n",
       " ('hair', 300324),\n",
       " ('always', 300302),\n",
       " ('find', 297298),\n",
       " ('bit', 296269),\n",
       " ('seemed', 293951),\n",
       " ('behind', 293565),\n",
       " ('hogwarts', 291158),\n",
       " ('thing', 287130),\n",
       " ('bed', 286664),\n",
       " ('trying', 284142),\n",
       " ('feel', 281363),\n",
       " ('started', 281150),\n",
       " ('put', 280582),\n",
       " ('since', 277981),\n",
       " ('life', 275212),\n",
       " ('house', 274826),\n",
       " ('night', 274460),\n",
       " ('heard', 273306),\n",
       " ('black', 267141),\n",
       " ('without', 266548),\n",
       " ('smile', 266299),\n",
       " ('better', 264786),\n",
       " ('years', 264180),\n",
       " ('gave', 262484),\n",
       " ('magic', 261687),\n",
       " ('might', 261575),\n",
       " ('side', 259144),\n",
       " ('weasley', 258083),\n",
       " ('everyone', 257388),\n",
       " ('father', 254427),\n",
       " ('sat', 254183),\n",
       " ('began', 253548),\n",
       " ('someone', 253021),\n",
       " ('almost', 250652),\n",
       " ('walked', 250330),\n",
       " ('done', 247701),\n",
       " ('finally', 247446),\n",
       " ('already', 246454),\n",
       " ('-RRB-', 245754),\n",
       " ('tried', 245120),\n",
       " ('place', 244606),\n",
       " ('every', 243628),\n",
       " ('stood', 242592),\n",
       " ('everything', 240725),\n",
       " ('friends', 240538),\n",
       " ('-LRB-', 240302),\n",
       " ('three', 238326),\n",
       " ('lord', 238135),\n",
       " ('front', 238092),\n",
       " ('pulled', 238078),\n",
       " ('small', 237801),\n",
       " ('also', 237565),\n",
       " ('quickly', 237094),\n",
       " ('course', 236041),\n",
       " ('keep', 234298),\n",
       " ('girl', 232527),\n",
       " ('body', 232221),\n",
       " ('best', 230692),\n",
       " ('towards', 228033),\n",
       " ('else', 223261),\n",
       " ('arms', 222832),\n",
       " ('neville', 221019),\n",
       " ('table', 220543),\n",
       " ('give', 219404),\n",
       " ('mean', 219179),\n",
       " ('work', 218270),\n",
       " ('family', 217554),\n",
       " ('sorry', 216536),\n",
       " ('albus', 216310),\n",
       " ('please', 214754),\n",
       " ('end', 213973),\n",
       " ('many', 213747),\n",
       " ('world', 212826),\n",
       " ('school', 212144),\n",
       " ('mother', 212050),\n",
       " ('great', 210966),\n",
       " ('lucius', 210616),\n",
       " ('old', 210043),\n",
       " ('together', 209034),\n",
       " ('new', 207665),\n",
       " ('quite', 207365),\n",
       " ('stop', 206561),\n",
       " ('happened', 206065),\n",
       " ('leave', 205624),\n",
       " ('replied', 204729),\n",
       " ('maybe', 204598),\n",
       " ('mouth', 202974),\n",
       " ('open', 202701),\n",
       " ('yet', 201849),\n",
       " ('must', 201418),\n",
       " ('soon', 201416),\n",
       " ('later', 197969),\n",
       " ('floor', 196991),\n",
       " ('getting', 196666),\n",
       " ('able', 195090),\n",
       " ('words', 193878),\n",
       " ('arm', 192659),\n",
       " ('name', 192215),\n",
       " ('lips', 191042),\n",
       " ('friend', 190796),\n",
       " ('least', 190605),\n",
       " ('hard', 190254),\n",
       " ('feeling', 188443),\n",
       " ('actually', 188115),\n",
       " ('suddenly', 187196),\n",
       " ('anyone', 187149),\n",
       " ('seen', 186336),\n",
       " ('slowly', 186253),\n",
       " ('sighed', 184305),\n",
       " ('luna', 183037),\n",
       " ('talk', 182323),\n",
       " ('spell', 181268),\n",
       " ('shook', 179530),\n",
       " ('making', 178939),\n",
       " ('gone', 178283),\n",
       " ('rather', 177043),\n",
       " ('george', 176656),\n",
       " ('believe', 176515),\n",
       " ('probably', 175227),\n",
       " ('held', 174812),\n",
       " ('home', 174519),\n",
       " ('far', 173858),\n",
       " ('light', 173487),\n",
       " ('used', 172746),\n",
       " ('inside', 172635),\n",
       " ('granger', 171635),\n",
       " ('wo', 171528),\n",
       " ('second', 171259),\n",
       " ('opened', 170913),\n",
       " ('rest', 170642),\n",
       " ('taking', 169663),\n",
       " ('fred', 169059),\n",
       " ('stopped', 167142),\n",
       " ('hear', 166916),\n",
       " ('idea', 166282),\n",
       " ('whispered', 166020),\n",
       " ('rose', 165678),\n",
       " ('coming', 164395),\n",
       " ('across', 163973),\n",
       " ('thank', 163901),\n",
       " ('try', 163850),\n",
       " ('breath', 163822),\n",
       " ('care', 163799),\n",
       " ('slightly', 162679),\n",
       " ('tom', 162448),\n",
       " ('parents', 162122),\n",
       " ('yeah', 161439),\n",
       " ('students', 161156),\n",
       " ('called', 160680),\n",
       " ('needed', 160487),\n",
       " ('ask', 159910),\n",
       " ('laughed', 159458),\n",
       " ('close', 158955),\n",
       " ('may', 158649),\n",
       " ('watched', 158066),\n",
       " ('use', 157762),\n",
       " ('continued', 156894),\n",
       " ('slytherin', 156819),\n",
       " ('sitting', 156484),\n",
       " ('gryffindor', 156129),\n",
       " ('part', 155502),\n",
       " ('heart', 155416),\n",
       " ('morning', 155392),\n",
       " ('blood', 155071),\n",
       " ('feet', 153951),\n",
       " ('happy', 153924),\n",
       " ('lot', 153672),\n",
       " ('pain', 151341),\n",
       " ('book', 149318),\n",
       " ('muggle', 149191),\n",
       " ('remember', 149076),\n",
       " ('okay', 148983),\n",
       " ('hall', 148581),\n",
       " ('fact', 147786),\n",
       " ('past', 147293),\n",
       " ('stared', 146460),\n",
       " ('reached', 146026),\n",
       " ('mr.', 145917),\n",
       " ('however', 145398),\n",
       " ('fine', 145175),\n",
       " ('chapter', 145046),\n",
       " ('ministry', 144641),\n",
       " ('thinking', 144447),\n",
       " ('wrong', 143608),\n",
       " ('along', 143517),\n",
       " ('onto', 143483),\n",
       " ('hope', 142924),\n",
       " ('alone', 142520),\n",
       " ('shoulder', 141497),\n",
       " ('set', 140239),\n",
       " ('read', 140055),\n",
       " ('moved', 140028),\n",
       " ('wizard', 139546),\n",
       " ('bad', 139364),\n",
       " ('standing', 139268),\n",
       " ('either', 139165),\n",
       " ('red', 138709),\n",
       " ('chest', 138218),\n",
       " ('decided', 137221),\n",
       " ('fell', 137163),\n",
       " ('talking', 137063),\n",
       " ('dead', 136583),\n",
       " ('minutes', 135969),\n",
       " ('stay', 135622),\n",
       " ('ran', 135131),\n",
       " ('hurt', 134376),\n",
       " ('understand', 134210),\n",
       " ('young', 134160),\n",
       " ('potion', 134114),\n",
       " ('mcgonagall', 133828),\n",
       " ('fingers', 133732),\n",
       " ('instead', 132927),\n",
       " ('caught', 132705),\n",
       " ('person', 132504),\n",
       " ('kill', 132469),\n",
       " ('days', 131901),\n",
       " ('woman', 131869),\n",
       " ('miss', 131767),\n",
       " ('potions', 131664),\n",
       " ('point', 131470),\n",
       " ('others', 130994),\n",
       " ('air', 130910),\n",
       " ('tears', 130176),\n",
       " ('ready', 130001),\n",
       " ('kiss', 129938),\n",
       " ('ground', 129876),\n",
       " ('tonks', 129768),\n",
       " ('kept', 128768),\n",
       " ('matter', 128692),\n",
       " ('taken', 128052),\n",
       " ('son', 128041),\n",
       " ('times', 127819),\n",
       " ('wait', 127361),\n",
       " ('order', 127019),\n",
       " ('forward', 126684),\n",
       " ('deep', 126013),\n",
       " ('closed', 125853),\n",
       " ('noticed', 125310),\n",
       " ('eaters', 124788),\n",
       " ('curse', 124232),\n",
       " ('story', 124117),\n",
       " ('reason', 123295),\n",
       " ('lost', 122899),\n",
       " ('kind', 122789),\n",
       " ('raised', 122475),\n",
       " ('full', 121637),\n",
       " ('sound', 121616),\n",
       " ('sleep', 121238),\n",
       " ('magical', 121007),\n",
       " ('thanks', 120809),\n",
       " ('move', 120407),\n",
       " ('blaise', 119657),\n",
       " ('different', 118928),\n",
       " ('wall', 118604),\n",
       " ('whole', 118562),\n",
       " ('office', 117149),\n",
       " ('bellatrix', 116982),\n",
       " ('several', 116928),\n",
       " ('question', 115848),\n",
       " ('followed', 115344),\n",
       " ('start', 115300),\n",
       " ('half', 114544),\n",
       " ('turn', 114115),\n",
       " ('word', 113883),\n",
       " ('perhaps', 113591),\n",
       " ('anyway', 113542),\n",
       " ('large', 113442),\n",
       " ('outside', 112899),\n",
       " ('answer', 112853),\n",
       " ('holding', 112646),\n",
       " ('class', 112168),\n",
       " ('attention', 111910),\n",
       " ('answered', 111238),\n",
       " ('chair', 111197),\n",
       " ('hit', 111155),\n",
       " ('completely', 110870),\n",
       " ('lupin', 110827),\n",
       " ('exactly', 110638),\n",
       " ('quietly', 110189),\n",
       " ('scorpius', 110174),\n",
       " ('waiting', 110120),\n",
       " ('cold', 108680),\n",
       " ('immediately', 108597),\n",
       " ('mum', 108012),\n",
       " ('ago', 107921),\n",
       " ('minerva', 107912),\n",
       " ('spoke', 107613),\n",
       " ('robes', 107571),\n",
       " ('saying', 107564),\n",
       " ('eye', 107477),\n",
       " ('longer', 107251),\n",
       " ('today', 106822),\n",
       " ('leaving', 106679),\n",
       " ('watching', 106551),\n",
       " ('stepped', 105950),\n",
       " ('true', 105879),\n",
       " ('green', 105690),\n",
       " ('seeing', 105343),\n",
       " ('hold', 105239),\n",
       " ('staring', 104929),\n",
       " ('child', 104836),\n",
       " ('surprised', 104808),\n",
       " ('pansy', 104646),\n",
       " ('given', 104614),\n",
       " ('master', 104577),\n",
       " ('silence', 104415),\n",
       " ('dad', 104074),\n",
       " ('nearly', 103947),\n",
       " ('turning', 103820),\n",
       " ('softly', 103588),\n",
       " ('nice', 103395),\n",
       " ('neck', 103157),\n",
       " ('headmaster', 103023),\n",
       " ('finished', 102732),\n",
       " ('fire', 102540),\n",
       " ('meant', 102292),\n",
       " ('brought', 101996),\n",
       " ('closer', 101974),\n",
       " ('simply', 101814),\n",
       " ('witch', 101438),\n",
       " ('grabbed', 100639),\n",
       " ('four', 100436),\n",
       " ('show', 99832),\n",
       " ('grinned', 99585),\n",
       " ('hell', 99580),\n",
       " ('war', 99504),\n",
       " ('managed', 99334),\n",
       " ('realized', 99193),\n",
       " ('charm', 98993),\n",
       " ('says', 98968),\n",
       " ('call', 98927),\n",
       " ('change', 98885),\n",
       " ('bloody', 98879),\n",
       " ('looks', 98858),\n",
       " ('narcissa', 98747),\n",
       " ('top', 98439),\n",
       " ('running', 97882),\n",
       " ('children', 97702),\n",
       " ('quidditch', 97698),\n",
       " ('kissed', 97217),\n",
       " ('killed', 97102),\n",
       " ('chance', 97066),\n",
       " ('speak', 96696),\n",
       " ('common', 96446),\n",
       " ('brother', 96354),\n",
       " ('shut', 96085),\n",
       " ('skin', 95850),\n",
       " ('week', 95296),\n",
       " ('sent', 94967),\n",
       " ('alright', 94069),\n",
       " ('added', 93828),\n",
       " ('pointed', 93630),\n",
       " ('castle', 93484),\n",
       " ('met', 93285),\n",
       " ('rolled', 93176),\n",
       " ('known', 92836),\n",
       " ('expression', 92778),\n",
       " ('boys', 92590),\n",
       " ('molly', 92550),\n",
       " ('supposed', 92538),\n",
       " ('returned', 92417),\n",
       " ('walking', 92318),\n",
       " ('muttered', 92131),\n",
       " ('whatever', 92074),\n",
       " ('run', 91938),\n",
       " ('fight', 91913),\n",
       " ('knowing', 91859),\n",
       " ('hours', 91853),\n",
       " ('appeared', 91665),\n",
       " ('leaned', 91513),\n",
       " ('cast', 91440),\n",
       " ('girls', 91396),\n",
       " ('giving', 91014),\n",
       " ('less', 90615),\n",
       " ('happen', 90543),\n",
       " ('upon', 90424),\n",
       " ('books', 90316),\n",
       " ('thoughts', 89939),\n",
       " ('shrugged', 89892),\n",
       " ('pretty', 89735),\n",
       " ('water', 89504),\n",
       " ('sort', 89182),\n",
       " ('guess', 89007),\n",
       " ('spells', 88992),\n",
       " ('throat', 88934),\n",
       " ('gently', 88648),\n",
       " ('letter', 88515),\n",
       " ('worry', 88392),\n",
       " ('real', 88237),\n",
       " ('pushed', 88025),\n",
       " ('loved', 87699),\n",
       " ('big', 87508),\n",
       " ('stairs', 87356),\n",
       " ('hagrid', 87300),\n",
       " ('possible', 87201),\n",
       " ('surprise', 87000),\n",
       " ('couple', 86712),\n",
       " ('glanced', 86677),\n",
       " ('white', 86612),\n",
       " ('meet', 85952),\n",
       " ('seem', 85843),\n",
       " ('passed', 85799),\n",
       " ('fear', 85287),\n",
       " ('bill', 85233),\n",
       " ('five', 85142),\n",
       " ('beside', 85137),\n",
       " ('become', 84820),\n",
       " ('desk', 84626),\n",
       " ('watch', 84515),\n",
       " ('sit', 84150),\n",
       " ('smiling', 84105),\n",
       " ('stand', 84003),\n",
       " ('power', 83976),\n",
       " ('placed', 83913),\n",
       " ('dinner', 83810),\n",
       " ('although', 83364),\n",
       " ('sense', 83234),\n",
       " ('toward', 83220),\n",
       " ('corner', 83174),\n",
       " ('merlin', 83033),\n",
       " ('late', 82782),\n",
       " ('safe', 82731),\n",
       " ('die', 82209),\n",
       " ('sir', 82130),\n",
       " ('stupid', 81891),\n",
       " ('bring', 81841),\n",
       " ('walk', 81316),\n",
       " ('wizards', 81268),\n",
       " ('anymore', 80622),\n",
       " ('soft', 80549),\n",
       " ('kitchen', 80345),\n",
       " ('snapped', 80328),\n",
       " ('agreed', 80054),\n",
       " ('eater', 80033),\n",
       " ('wondered', 80032),\n",
       " ('baby', 79904),\n",
       " ('knows', 79813),\n",
       " ('wish', 79359),\n",
       " ('quiet', 79166),\n",
       " ('telling', 78989),\n",
       " ('break', 78666),\n",
       " ('changed', 78526),\n",
       " ('group', 78491),\n",
       " ('peter', 78303),\n",
       " ('laugh', 78237),\n",
       " ('window', 78227),\n",
       " ('pulling', 78113),\n",
       " ('moving', 78050),\n",
       " ('return', 77994),\n",
       " ('charlie', 77808),\n",
       " ('plan', 77771),\n",
       " ('angry', 77551),\n",
       " ('case', 77267),\n",
       " ('short', 77229),\n",
       " ('trust', 77087),\n",
       " ('explained', 77019),\n",
       " ('shot', 76855),\n",
       " ('free', 76785),\n",
       " ('dropped', 76771),\n",
       " ('shoulders', 76556),\n",
       " ('near', 76539),\n",
       " ('sister', 76285),\n",
       " ('gaze', 75954),\n",
       " ('barely', 75768),\n",
       " ('sight', 75763),\n",
       " ('reading', 75699),\n",
       " ('live', 75339),\n",
       " ('beautiful', 75018),\n",
       " ('especially', 74968),\n",
       " ('died', 74761),\n",
       " ('within', 74633),\n",
       " ('cut', 74495),\n",
       " ('figure', 74456),\n",
       " ('christmas', 74315),\n",
       " ('shaking', 74314),\n",
       " ('tonight', 74302),\n",
       " ('working', 74148),\n",
       " ('worried', 74130),\n",
       " ('spent', 74093),\n",
       " ('clear', 74076),\n",
       " ('months', 74035),\n",
       " ('seat', 73913),\n",
       " ('entire', 73856),\n",
       " ('blue', 73755),\n",
       " ('legs', 73651),\n",
       " ('shouted', 73628),\n",
       " ('dear', 73607),\n",
       " ('memory', 73521),\n",
       " ('frowned', 73386),\n",
       " ('living', 73099),\n",
       " ('stomach', 72749),\n",
       " ('mrs.', 72658),\n",
       " ('fall', 72638),\n",
       " ('conversation', 72610),\n",
       " ('straight', 72000),\n",
       " ('allowed', 71897),\n",
       " ('control', 71858),\n",
       " ('wizarding', 71735),\n",
       " ('stone', 71638),\n",
       " ('remembered', 71634),\n",
       " ('tone', 71590),\n",
       " ('afraid', 71545),\n",
       " ('fun', 71477),\n",
       " ('form', 71280),\n",
       " ('picked', 71070),\n",
       " ('anger', 70916),\n",
       " ('glad', 70883),\n",
       " ('broke', 70763),\n",
       " ('touch', 70659),\n",
       " ('entered', 70472),\n",
       " ('hour', 70469),\n",
       " ('silent', 70093),\n",
       " ('quick', 69822),\n",
       " ('grin', 69814),\n",
       " ('truth', 69808),\n",
       " ('cried', 69745),\n",
       " ('meeting', 69690),\n",
       " ('nose', 69415),\n",
       " ('laughing', 69370),\n",
       " ('hey', 69305),\n",
       " ('play', 69224),\n",
       " ('smirked', 69150),\n",
       " ('step', 69125),\n",
       " ('certain', 69019),\n",
       " ('important', 68812),\n",
       " ('became', 68581),\n",
       " ('weeks', 68382),\n",
       " ('filled', 68291),\n",
       " ('none', 68260),\n",
       " ('auror', 68203),\n",
       " ('warm', 68088),\n",
       " ('team', 68070),\n",
       " ('using', 67934),\n",
       " ('food', 67655),\n",
       " ('percy', 67634),\n",
       " ('older', 67404),\n",
       " ('empty', 67318),\n",
       " ('fleur', 67109),\n",
       " ('threw', 66777),\n",
       " ('hate', 66648),\n",
       " ('expected', 66504),\n",
       " ('cheek', 66372),\n",
       " ('yelled', 66147),\n",
       " ('worse', 66131),\n",
       " ('breakfast', 66109),\n",
       " ('problem', 65955),\n",
       " ('ones', 65556),\n",
       " ('clearly', 65543),\n",
       " ('teddy', 65477),\n",
       " ('cloak', 65420),\n",
       " ('memories', 65342),\n",
       " ('perfect', 65325),\n",
       " ('waited', 65166),\n",
       " ('obviously', 64740),\n",
       " ('shock', 64705),\n",
       " ('liked', 64422),\n",
       " ('arrived', 64387),\n",
       " ('alive', 64376),\n",
       " ('sometimes', 64321),\n",
       " ('certainly', 64312),\n",
       " ('riddle', 63899),\n",
       " ('wife', 63832),\n",
       " ('carefully', 63764),\n",
       " ('note', 63641),\n",
       " ('despite', 63506),\n",
       " ('summer', 63488),\n",
       " ('gotten', 63205),\n",
       " ('information', 63161),\n",
       " ('daphne', 63106),\n",
       " ('job', 63046),\n",
       " ('forced', 62920),\n",
       " ('madam', 62910),\n",
       " ('loud', 62885),\n",
       " ('except', 62845),\n",
       " ('ten', 62831),\n",
       " ('wrapped', 62659),\n",
       " ('tomorrow', 62533),\n",
       " ('clothes', 62472),\n",
       " ('glass', 62428),\n",
       " ('deal', 62357),\n",
       " ('library', 62319),\n",
       " ('strong', 62086),\n",
       " ('calm', 62050),\n",
       " ('strange', 62020),\n",
       " ('tea', 61872),\n",
       " ('worked', 61863),\n",
       " ('trouble', 61760),\n",
       " ('confused', 61683),\n",
       " ('battle', 61671),\n",
       " ('ear', 61510),\n",
       " ('evening', 61149),\n",
       " ('owl', 61070),\n",
       " ('mine', 60840),\n",
       " ('easy', 60732),\n",
       " ('normal', 60658),\n",
       " ('wide', 60578),\n",
       " ('minute', 60573),\n",
       " ('seems', 60541),\n",
       " ('parchment', 60514),\n",
       " ('lay', 60473),\n",
       " ('makes', 60446),\n",
       " ('paused', 60411),\n",
       " ('twins', 60355),\n",
       " ('apparently', 60346),\n",
       " ('daughter', 59782),\n",
       " ('middle', 59765),\n",
       " ('suppose', 59696),\n",
       " ('moments', 59655),\n",
       " ('tired', 59573),\n",
       " ('high', 59355),\n",
       " ('broom', 59220),\n",
       " ('notice', 59202),\n",
       " ('somehow', 59137),\n",
       " ('led', 59066),\n",
       " ('promise', 58613),\n",
       " ('usual', 58599),\n",
       " ('secret', 58598),\n",
       " ('lying', 58533),\n",
       " ('cup', 58425),\n",
       " ('broken', 58261),\n",
       " ('piece', 58077),\n",
       " ('chuckled', 58073),\n",
       " ('arthur', 58040),\n",
       " ('tongue', 57898),\n",
       " ('sounded', 57840),\n",
       " ('men', 57821),\n",
       " ('line', 57810),\n",
       " ('questions', 57763),\n",
       " ('pale', 57435),\n",
       " ('means', 57426),\n",
       " ('doubt', 57406),\n",
       " ('muggles', 57399),\n",
       " ('save', 57373),\n",
       " ('attack', 57336),\n",
       " ('wonder', 57288),\n",
       " ('helped', 57195),\n",
       " ('familiar', 57004),\n",
       " ('explain', 56881),\n",
       " ('finger', 56834),\n",
       " ('bella', 56833),\n",
       " ('aurors', 56687),\n",
       " ('gasped', 56670),\n",
       " ('uncle', 56583),\n",
       " ('game', 56564),\n",
       " ('hissed', 56533),\n",
       " ('jumped', 56526),\n",
       " ('minister', 56503),\n",
       " ('situation', 56461),\n",
       " ('charms', 56436),\n",
       " ('bright', 56427),\n",
       " ('pressed', 56422),\n",
       " ('hospital', 56182),\n",
       " ('mate', 56156),\n",
       " ('whether', 56106),\n",
       " ('position', 55998),\n",
       " ('lifted', 55991),\n",
       " ('glared', 55900),\n",
       " ('manor', 55785),\n",
       " ('moody', 55439),\n",
       " ('headed', 55277),\n",
       " ('seconds', 55161),\n",
       " ('shall', 55084),\n",
       " ('catch', 55077),\n",
       " ('wants', 55025),\n",
       " ('often', 54846),\n",
       " ('fast', 54677),\n",
       " ('asking', 54604),\n",
       " ('dobby', 54283),\n",
       " ('teeth', 54184),\n",
       " ('third', 54111),\n",
       " ('early', 54111),\n",
       " ('asleep', 54092),\n",
       " ('pomfrey', 53966),\n",
       " ('listen', 53890),\n",
       " ('wearing', 53733),\n",
       " ('neither', 53722),\n",
       " ('kingsley', 53360),\n",
       " ('bag', 53286),\n",
       " ('following', 53273),\n",
       " ('letting', 53266),\n",
       " ('sigh', 53249),\n",
       " ('besides', 53241),\n",
       " ('screamed', 53167),\n",
       " ('truly', 53108),\n",
       " ('forget', 53092),\n",
       " ('write', 53049),\n",
       " ('covered', 52980),\n",
       " ('pull', 52974),\n",
       " ('mark', 52960),\n",
       " ('follow', 52879),\n",
       " ('beginning', 52685),\n",
       " ('hoped', 52546),\n",
       " ('elf', 52503),\n",
       " ('keeping', 52499),\n",
       " ('shirt', 52488),\n",
       " ('fighting', 52471),\n",
       " ('student', 52344),\n",
       " ('flew', 52286),\n",
       " ('playing', 52275),\n",
       " ('wondering', 52114),\n",
       " ('likely', 52114),\n",
       " ('eyebrow', 52073),\n",
       " ('guys', 51990),\n",
       " ('direction', 51982),\n",
       " ('easily', 51920),\n",
       " ('usually', 51901),\n",
       " ('learn', 51790),\n",
       " ('hated', 51738),\n",
       " ('ended', 51693),\n",
       " ('crying', 51667),\n",
       " ('starting', 51627),\n",
       " ('lip', 51573),\n",
       " ('somewhere', 51519),\n",
       " ('enjoy', 51455),\n",
       " ('future', 51432),\n",
       " ('husband', 51430),\n",
       " ('edge', 51398),\n",
       " ('eat', 51393),\n",
       " ('flying', 51169),\n",
       " ('azkaban', 50788),\n",
       " ('ah', 50771),\n",
       " ('soul', 50711),\n",
       " ('offered', 50707),\n",
       " ('cheeks', 50538),\n",
       " ('hoping', 50401),\n",
       " ('cry', 50339),\n",
       " ('damn', 50263),\n",
       " ('falling', 50251),\n",
       " ('earlier', 50205),\n",
       " ('tightly', 50187),\n",
       " ('knees', 50119),\n",
       " ('send', 50111),\n",
       " ('shocked', 49939),\n",
       " ('powerful', 49917),\n",
       " ('train', 49838),\n",
       " ('al', 49837),\n",
       " ('serious', 49819),\n",
       " ('money', 49794),\n",
       " ('missed', 49788),\n",
       " ('pair', 49760),\n",
       " ('drink', 49731),\n",
       " ('disappeared', 49719),\n",
       " ('silver', 49685),\n",
       " ('choice', 49632),\n",
       " ('speaking', 49598),\n",
       " ('hot', 49579),\n",
       " ('pocket', 49504),\n",
       " ('hide', 49394),\n",
       " ('expect', 49204),\n",
       " ('age', 49195),\n",
       " ('brown', 49164),\n",
       " ('aware', 49152),\n",
       " ('ravenclaw', 49141),\n",
       " ('definitely', 49138),\n",
       " ('low', 48977),\n",
       " ('spot', 48881),\n",
       " ('eventually', 48733),\n",
       " ('wake', 48661),\n",
       " ('putting', 48563),\n",
       " ('hug', 48472),\n",
       " ('bedroom', 48312),\n",
       " ('forest', 48305),\n",
       " ('ears', 48214),\n",
       " ('killing', 48069),\n",
       " ('mad', 48059),\n",
       " ('aunt', 48036),\n",
       " ('single', 48031),\n",
       " ('simple', 47934),\n",
       " ('god', 47921),\n",
       " ('mirror', 47840),\n",
       " ('waved', 47778),\n",
       " ('scared', 47740),\n",
       " ('laughter', 47712),\n",
       " ('dress', 47694),\n",
       " ('six', 47614),\n",
       " ('present', 47610),\n",
       " ('continue', 47565),\n",
       " ('admit', 47557),\n",
       " ('sounds', 47310),\n",
       " ('join', 47251),\n",
       " ('exclaimed', 47220),\n",
       " ('a/n', 47194),\n",
       " ('handed', 47176),\n",
       " ('paper', 47147),\n",
       " ('feelings', 47101),\n",
       " ('blinked', 46975),\n",
       " ('corridor', 46932),\n",
       " ('breathing', 46776),\n",
       " ('slipped', 46766),\n",
       " ('check', 46702),\n",
       " ('showed', 46691),\n",
       " ('relief', 46589),\n",
       " ('forehead', 46558),\n",
       " ('force', 46362),\n",
       " ('sleeping', 46326),\n",
       " ('doors', 46277),\n",
       " ('smirk', 46270),\n",
       " ('younger', 46236),\n",
       " ('box', 46234),\n",
       " ('final', 46193),\n",
       " ('wands', 46174),\n",
       " ('news', 46156),\n",
       " ('murmured', 46087),\n",
       " ('writing', 46078),\n",
       " ('protect', 45920),\n",
       " ('causing', 45830),\n",
       " ('ok', 45665),\n",
       " ('indeed', 45612),\n",
       " ('bathroom', 45608),\n",
       " ('kissing', 45309),\n",
       " ('ball', 45098),\n",
       " ('steps', 45083),\n",
       " ('wards', 45070),\n",
       " ('dangerous', 44990),\n",
       " ('seamus', 44985),\n",
       " ('dean', 44825),\n",
       " ('lunch', 44798),\n",
       " ('glance', 44708),\n",
       " ('alley', 44596),\n",
       " ('obvious', 44591),\n",
       " ('entrance', 44509),\n",
       " ('sudden', 44468),\n",
       " ('stayed', 44416),\n",
       " ('business', 44396),\n",
       " ('apart', 44335),\n",
       " ('evil', 44326),\n",
       " ('married', 44276),\n",
       " ('cedric', 44232),\n",
       " ('busy', 44215),\n",
       " ('visit', 44188),\n",
       " ('touched', 44167),\n",
       " ('honestly', 44136),\n",
       " ('needs', 44131),\n",
       " ('difficult', 44128),\n",
       " ('ring', 44086),\n",
       " ('party', 44068),\n",
       " ('imagine', 43956),\n",
       " ('drew', 43939),\n",
       " ('fault', 43925),\n",
       " ('number', 43898),\n",
       " ('stuff', 43755),\n",
       " ('burst', 43710),\n",
       " ('fuck', 43705),\n",
       " ('dream', 43682),\n",
       " ('leaning', 43672),\n",
       " ('walls', 43659),\n",
       " ('aside', 43644),\n",
       " ('groaned', 43592),\n",
       " ('crowd', 43549),\n",
       " ('slytherins', 43434),\n",
       " ('interrupted', 43348),\n",
       " ('couch', 43152),\n",
       " ('silently', 43146),\n",
       " ('darkness', 43093),\n",
       " ('remained', 43072),\n",
       " ('learned', 43043),\n",
       " ('tower', 42998),\n",
       " ('hello', 42988),\n",
       " ('stated', 42839),\n",
       " ('foot', 42800),\n",
       " ('snake', 42721),\n",
       " ('dragon', 42667),\n",
       " ('odd', 42623),\n",
       " ('mr', 42609),\n",
       " ('teacher', 42550),\n",
       " ('ahead', 42548),\n",
       " ('growled', 42506),\n",
       " ('wanting', 42475),\n",
       " ('lavender', 42469),\n",
       " ('spend', 42463),\n",
       " ('allow', 42292),\n",
       " ('stuck', 42266),\n",
       " ('match', 42242),\n",
       " ('werewolf', 42186),\n",
       " ('lose', 42120),\n",
       " ('date', 42053),\n",
       " ('finish', 41995),\n",
       " ('points', 41973),\n",
       " ('human', 41967),\n",
       " ('grew', 41957),\n",
       " ('blonde', 41939),\n",
       " ('realised', 41916),\n",
       " ('act', 41841),\n",
       " ('hiding', 41837),\n",
       " ('relationship', 41817),\n",
       " ('response', 41707),\n",
       " ('regulus', 41665),\n",
       " ('caused', 41616),\n",
       " ('umbridge', 41606),\n",
       " ('unless', 41508),\n",
       " ('shop', 41435),\n",
       " ('tight', 41321),\n",
       " ('huge', 41273)]"
      ]
     },
     "execution_count": 103,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cnt.most_common(1000)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###### 2. Найдите топ-10 по частоте: имен, пар имя + фамилия, пар вида ''профессор'' + имя / фамилия.\n",
    "---\n",
    "Используем встроенный в nltk классификатор, который, в том числе, детектирует людей в тексте.\n",
    "\n",
    "Если в имени человека одно слово, будем считать, что это его имя. Если два слова, то это имя и фамилия.\n",
    "\n",
    "Для нахождения шаблонов вида professor + name, найдем в тексте все вхождения слова professor и проверим, что за ним стоит чьё-либо имя."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d969be3e0a254dccb537ca5c94d37709",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=0, max=36225), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "dirname = 'fanfiction_texts/'\n",
    "\n",
    "person_cnt = Counter()\n",
    "name_cnt = Counter()\n",
    "prof_cnt = Counter()\n",
    "   \n",
    "for filename in tqdm_notebook(os.listdir(dirname)):\n",
    "    with open(dirname + filename, 'r') as text:\n",
    "        text = text.readline()\n",
    "        tokens = nltk.word_tokenize(text)\n",
    "        pos = nltk.pos_tag(tokens)\n",
    "        sent = nltk.ne_chunk(pos, binary = False)\n",
    "        \n",
    "        for subtree in sent.subtrees(filter=lambda t: t.label() == 'PERSON'):\n",
    "            name = ''\n",
    "            name_tokens = []\n",
    "            for leaf in subtree.leaves():\n",
    "                if str(leaf[0]) != 'Mr.':\n",
    "                    name_tokens.append(str(leaf[0]))\n",
    "            name = ' '.join(name_tokens)\n",
    "            if len(name_tokens) == 0:\n",
    "                continue\n",
    "            name_cnt.update([name_tokens[0]])\n",
    "            if len(subtree.leaves()) > 1:\n",
    "                person_cnt.update([name])\n",
    "        \n",
    "        ind = 0\n",
    "        text_lower = text.lower()\n",
    "        while True:\n",
    "            next_ind = text_lower.find('professor', ind) + 1\n",
    "            if next_ind == 0:\n",
    "                break\n",
    "            text_splitted = text[ind:].split(' ')\n",
    "            if len(text_splitted) > 2 and ' '.join(text_splitted[1:3]) in person_cnt:\n",
    "                prof_cnt.update(['professor ' + ' '.join(text_splitted[1:3])])\n",
    "            elif len(text_splitted) > 1 and ' '.join(text_splitted[1]) in person_cnt:\n",
    "                prof_cnt.update(['professor ' + name])\n",
    "            ind = next_ind"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "топ-10 имен:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[('Harry', 735248), ('Draco', 301730), ('Hermione', 270292), ('Ron', 166044), ('Sirius', 127273), ('James', 114648), ('Ginny', 97489), ('Potter', 97043), ('Severus', 83886), ('Snape', 81294)]\n"
     ]
    }
   ],
   "source": [
    "print(name_cnt.most_common(10))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "топ-10 пар имя + фамилия:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[('Harry Potter', 32233), ('Draco Malfoy', 13566), ('Hermione Granger', 8027), ('James Potter', 6901), ('Severus Snape', 6528), ('Sirius Black', 6065), ('Miss Granger', 5799), ('Lucius Malfoy', 5008), ('Avada Kedavra', 4908), ('Albus Dumbledore', 4824)]\n"
     ]
    }
   ],
   "source": [
    "print(person_cnt.most_common(10))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "топ-10 пар вида \"профессор\" + имя/фамилия:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[('professor Harry Potter', 83), ('professor Severus Snape', 74), ('professor Minerva McGonagall', 46), ('professor Albus Dumbledore', 20), ('professor Professor McGonagall', 16), ('professor Severus Tobias', 15), ('professor Remus Lupin', 13), ('professor Hermione Granger', 10), ('professor Neville Longbottom', 9), ('professor Professor Snape', 8)]\n"
     ]
    }
   ],
   "source": [
    "print(prof_cnt.most_common(10))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "Получился неплохой и правдоподобный результат, единственный огрех - данный классификатор детектирует шаблон Professor + second name как имя и фамилию персонажа."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "# Часть 2. Модели представления слов\n",
    "\n",
    "### Обучите модель представления слов (word2vec, GloVe, fastText или любую другую) на материале корпуса HPAC.\n",
    "---\n",
    "Обучим word2vec"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6f29fba3009e495bbc085d160a9690c0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=0, max=36225), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "dirname = 'hpac_lower_tokenized/hpac_source/'\n",
    "\n",
    "model = None\n",
    "\n",
    "cnt = 0\n",
    "\n",
    "for filename in tqdm_notebook(os.listdir(dirname)):\n",
    "    with open(dirname + filename, 'r') as text:\n",
    "        cnt += 1\n",
    "        tokens = text.readline().split(' ')\n",
    "        if model is None:\n",
    "            model = word2vec.Word2Vec([tokens], workers=4, size=300, min_count=1, window=10, sample=1e-3)\n",
    "        else:\n",
    "            model.build_vocab([tokens], update=True)\n",
    "            model.train([tokens], total_examples=1, epochs=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "word_vectors = model.wv\n",
    "word_vectors.save(\"word2vec.wordvectors\")\n",
    "wv = KeyedVectors.load(\"word2vec.wordvectors\", mmap='r')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "###### 1. Продемонстрируйте, как работает поиск синонимов, ассоциаций, лишних слов в обученной модели."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Посмотрим на косинусное расстояние между похожими  по смыслу словами:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-0.059291188"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wv.similarity('harry', 'potter')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.53704333"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wv.similarity('griffindor', 'slytherin')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "Посмотрим на то, как модель отлавливает синонимичные слова:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('hermione', 0.3832247853279114),\n",
       " ('he', 0.3765811622142792),\n",
       " ('draco', 0.3759091794490814),\n",
       " ('vernon', 0.35086166858673096),\n",
       " ('ginny', 0.32585573196411133)]"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wv.most_similar(\"harry\", topn=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('griffindor', 0.7131483554840088),\n",
       " ('slytherin', 0.623772919178009),\n",
       " ('gryfindor', 0.6137665510177612),\n",
       " ('ravenclaw', 0.5765512585639954),\n",
       " ('hufflepuff', 0.532322883605957)]"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wv.most_similar(\"gryffindor\", topn=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('magics', 0.5212135910987854),\n",
       " ('rituals', 0.42723995447158813),\n",
       " ('powers', 0.42353391647338867),\n",
       " ('artifact', 0.41327735781669617),\n",
       " ('spellwork', 0.412217378616333)]"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wv.most_similar(\"magic\", topn=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('vehicle', 0.6776118874549866),\n",
       " ('truck', 0.6611902713775635),\n",
       " ('taxi', 0.6318948268890381),\n",
       " ('cab', 0.6017487645149231),\n",
       " ('driveway', 0.5986653566360474)]"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wv.most_similar(\"car\", topn=5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "И самое интересное: посмотрим на то, можно ли складывать и вычитать векторные представления слов и что из этого получается:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('women', 0.4324072599411011),\n",
       " ('figures', 0.35521000623703003),\n",
       " ('teenagers', 0.3481307029724121)]"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wv.most_similar(positive=[\"harry\", \"men\"], negative=[\"hermione\"], topn=3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('malfoy', 0.2778504490852356),\n",
       " ('beherzt', 0.27370816469192505),\n",
       " ('voldermort', 0.272976279258728)]"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wv.most_similar(positive=[\"potter\", \"bad\"], negative=[\"good\"], topn=3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('slytherin', 0.47085052728652954)]"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wv.most_similar(positive=[\"malfoy\", \"gryffindor\"], negative=[\"potter\"], topn=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "###### 2. Визуализируйте топ-1000 слов по частоте без учета стоп-слов (п. 1.1) с помощью TSNE или UMAP."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 3 µs, sys: 0 ns, total: 3 µs\n",
      "Wall time: 5.25 µs\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/coder/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:6: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).\n",
      "  \n"
     ]
    }
   ],
   "source": [
    "top_words = []\n",
    "\n",
    "for w in cnt.most_common(1000):\n",
    "    top_words.append(w[0])\n",
    "    \n",
    "top_words_vec = model[top_words]\n",
    "\n",
    "%time\n",
    "tsne = TSNE(n_components=2, random_state=0)\n",
    "top_words_tsne = tsne.fit_transform(top_words_vec)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "\n",
       "    <div class=\"bk-root\">\n",
       "        <a href=\"https://bokeh.pydata.org\" target=\"_blank\" class=\"bk-logo bk-logo-small bk-logo-notebook\"></a>\n",
       "        <span id=\"1395\">Loading BokehJS ...</span>\n",
       "    </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "(function(root) {\n",
       "  function now() {\n",
       "    return new Date();\n",
       "  }\n",
       "\n",
       "  var force = true;\n",
       "\n",
       "  if (typeof root._bokeh_onload_callbacks === \"undefined\" || force === true) {\n",
       "    root._bokeh_onload_callbacks = [];\n",
       "    root._bokeh_is_loading = undefined;\n",
       "  }\n",
       "\n",
       "  var JS_MIME_TYPE = 'application/javascript';\n",
       "  var HTML_MIME_TYPE = 'text/html';\n",
       "  var EXEC_MIME_TYPE = 'application/vnd.bokehjs_exec.v0+json';\n",
       "  var CLASS_NAME = 'output_bokeh rendered_html';\n",
       "\n",
       "  /**\n",
       "   * Render data to the DOM node\n",
       "   */\n",
       "  function render(props, node) {\n",
       "    var script = document.createElement(\"script\");\n",
       "    node.appendChild(script);\n",
       "  }\n",
       "\n",
       "  /**\n",
       "   * Handle when an output is cleared or removed\n",
       "   */\n",
       "  function handleClearOutput(event, handle) {\n",
       "    var cell = handle.cell;\n",
       "\n",
       "    var id = cell.output_area._bokeh_element_id;\n",
       "    var server_id = cell.output_area._bokeh_server_id;\n",
       "    // Clean up Bokeh references\n",
       "    if (id != null && id in Bokeh.index) {\n",
       "      Bokeh.index[id].model.document.clear();\n",
       "      delete Bokeh.index[id];\n",
       "    }\n",
       "\n",
       "    if (server_id !== undefined) {\n",
       "      // Clean up Bokeh references\n",
       "      var cmd = \"from bokeh.io.state import curstate; print(curstate().uuid_to_server['\" + server_id + \"'].get_sessions()[0].document.roots[0]._id)\";\n",
       "      cell.notebook.kernel.execute(cmd, {\n",
       "        iopub: {\n",
       "          output: function(msg) {\n",
       "            var id = msg.content.text.trim();\n",
       "            if (id in Bokeh.index) {\n",
       "              Bokeh.index[id].model.document.clear();\n",
       "              delete Bokeh.index[id];\n",
       "            }\n",
       "          }\n",
       "        }\n",
       "      });\n",
       "      // Destroy server and session\n",
       "      var cmd = \"import bokeh.io.notebook as ion; ion.destroy_server('\" + server_id + \"')\";\n",
       "      cell.notebook.kernel.execute(cmd);\n",
       "    }\n",
       "  }\n",
       "\n",
       "  /**\n",
       "   * Handle when a new output is added\n",
       "   */\n",
       "  function handleAddOutput(event, handle) {\n",
       "    var output_area = handle.output_area;\n",
       "    var output = handle.output;\n",
       "\n",
       "    // limit handleAddOutput to display_data with EXEC_MIME_TYPE content only\n",
       "    if ((output.output_type != \"display_data\") || (!output.data.hasOwnProperty(EXEC_MIME_TYPE))) {\n",
       "      return\n",
       "    }\n",
       "\n",
       "    var toinsert = output_area.element.find(\".\" + CLASS_NAME.split(' ')[0]);\n",
       "\n",
       "    if (output.metadata[EXEC_MIME_TYPE][\"id\"] !== undefined) {\n",
       "      toinsert[toinsert.length - 1].firstChild.textContent = output.data[JS_MIME_TYPE];\n",
       "      // store reference to embed id on output_area\n",
       "      output_area._bokeh_element_id = output.metadata[EXEC_MIME_TYPE][\"id\"];\n",
       "    }\n",
       "    if (output.metadata[EXEC_MIME_TYPE][\"server_id\"] !== undefined) {\n",
       "      var bk_div = document.createElement(\"div\");\n",
       "      bk_div.innerHTML = output.data[HTML_MIME_TYPE];\n",
       "      var script_attrs = bk_div.children[0].attributes;\n",
       "      for (var i = 0; i < script_attrs.length; i++) {\n",
       "        toinsert[toinsert.length - 1].firstChild.setAttribute(script_attrs[i].name, script_attrs[i].value);\n",
       "      }\n",
       "      // store reference to server id on output_area\n",
       "      output_area._bokeh_server_id = output.metadata[EXEC_MIME_TYPE][\"server_id\"];\n",
       "    }\n",
       "  }\n",
       "\n",
       "  function register_renderer(events, OutputArea) {\n",
       "\n",
       "    function append_mime(data, metadata, element) {\n",
       "      // create a DOM node to render to\n",
       "      var toinsert = this.create_output_subarea(\n",
       "        metadata,\n",
       "        CLASS_NAME,\n",
       "        EXEC_MIME_TYPE\n",
       "      );\n",
       "      this.keyboard_manager.register_events(toinsert);\n",
       "      // Render to node\n",
       "      var props = {data: data, metadata: metadata[EXEC_MIME_TYPE]};\n",
       "      render(props, toinsert[toinsert.length - 1]);\n",
       "      element.append(toinsert);\n",
       "      return toinsert\n",
       "    }\n",
       "\n",
       "    /* Handle when an output is cleared or removed */\n",
       "    events.on('clear_output.CodeCell', handleClearOutput);\n",
       "    events.on('delete.Cell', handleClearOutput);\n",
       "\n",
       "    /* Handle when a new output is added */\n",
       "    events.on('output_added.OutputArea', handleAddOutput);\n",
       "\n",
       "    /**\n",
       "     * Register the mime type and append_mime function with output_area\n",
       "     */\n",
       "    OutputArea.prototype.register_mime_type(EXEC_MIME_TYPE, append_mime, {\n",
       "      /* Is output safe? */\n",
       "      safe: true,\n",
       "      /* Index of renderer in `output_area.display_order` */\n",
       "      index: 0\n",
       "    });\n",
       "  }\n",
       "\n",
       "  // register the mime type if in Jupyter Notebook environment and previously unregistered\n",
       "  if (root.Jupyter !== undefined) {\n",
       "    var events = require('base/js/events');\n",
       "    var OutputArea = require('notebook/js/outputarea').OutputArea;\n",
       "\n",
       "    if (OutputArea.prototype.mime_types().indexOf(EXEC_MIME_TYPE) == -1) {\n",
       "      register_renderer(events, OutputArea);\n",
       "    }\n",
       "  }\n",
       "\n",
       "  \n",
       "  if (typeof (root._bokeh_timeout) === \"undefined\" || force === true) {\n",
       "    root._bokeh_timeout = Date.now() + 5000;\n",
       "    root._bokeh_failed_load = false;\n",
       "  }\n",
       "\n",
       "  var NB_LOAD_WARNING = {'data': {'text/html':\n",
       "     \"<div style='background-color: #fdd'>\\n\"+\n",
       "     \"<p>\\n\"+\n",
       "     \"BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \\n\"+\n",
       "     \"may be due to a slow or bad network connection. Possible fixes:\\n\"+\n",
       "     \"</p>\\n\"+\n",
       "     \"<ul>\\n\"+\n",
       "     \"<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\\n\"+\n",
       "     \"<li>use INLINE resources instead, as so:</li>\\n\"+\n",
       "     \"</ul>\\n\"+\n",
       "     \"<code>\\n\"+\n",
       "     \"from bokeh.resources import INLINE\\n\"+\n",
       "     \"output_notebook(resources=INLINE)\\n\"+\n",
       "     \"</code>\\n\"+\n",
       "     \"</div>\"}};\n",
       "\n",
       "  function display_loaded() {\n",
       "    var el = document.getElementById(\"1395\");\n",
       "    if (el != null) {\n",
       "      el.textContent = \"BokehJS is loading...\";\n",
       "    }\n",
       "    if (root.Bokeh !== undefined) {\n",
       "      if (el != null) {\n",
       "        el.textContent = \"BokehJS \" + root.Bokeh.version + \" successfully loaded.\";\n",
       "      }\n",
       "    } else if (Date.now() < root._bokeh_timeout) {\n",
       "      setTimeout(display_loaded, 100)\n",
       "    }\n",
       "  }\n",
       "\n",
       "\n",
       "  function run_callbacks() {\n",
       "    try {\n",
       "      root._bokeh_onload_callbacks.forEach(function(callback) {\n",
       "        if (callback != null)\n",
       "          callback();\n",
       "      });\n",
       "    } finally {\n",
       "      delete root._bokeh_onload_callbacks\n",
       "    }\n",
       "    console.debug(\"Bokeh: all callbacks have finished\");\n",
       "  }\n",
       "\n",
       "  function load_libs(css_urls, js_urls, callback) {\n",
       "    if (css_urls == null) css_urls = [];\n",
       "    if (js_urls == null) js_urls = [];\n",
       "\n",
       "    root._bokeh_onload_callbacks.push(callback);\n",
       "    if (root._bokeh_is_loading > 0) {\n",
       "      console.debug(\"Bokeh: BokehJS is being loaded, scheduling callback at\", now());\n",
       "      return null;\n",
       "    }\n",
       "    if (js_urls == null || js_urls.length === 0) {\n",
       "      run_callbacks();\n",
       "      return null;\n",
       "    }\n",
       "    console.debug(\"Bokeh: BokehJS not loaded, scheduling load and callback at\", now());\n",
       "    root._bokeh_is_loading = css_urls.length + js_urls.length;\n",
       "\n",
       "    function on_load() {\n",
       "      root._bokeh_is_loading--;\n",
       "      if (root._bokeh_is_loading === 0) {\n",
       "        console.debug(\"Bokeh: all BokehJS libraries/stylesheets loaded\");\n",
       "        run_callbacks()\n",
       "      }\n",
       "    }\n",
       "\n",
       "    function on_error() {\n",
       "      console.error(\"failed to load \" + url);\n",
       "    }\n",
       "\n",
       "    for (var i = 0; i < css_urls.length; i++) {\n",
       "      var url = css_urls[i];\n",
       "      const element = document.createElement(\"link\");\n",
       "      element.onload = on_load;\n",
       "      element.onerror = on_error;\n",
       "      element.rel = \"stylesheet\";\n",
       "      element.type = \"text/css\";\n",
       "      element.href = url;\n",
       "      console.debug(\"Bokeh: injecting link tag for BokehJS stylesheet: \", url);\n",
       "      document.body.appendChild(element);\n",
       "    }\n",
       "\n",
       "    for (var i = 0; i < js_urls.length; i++) {\n",
       "      var url = js_urls[i];\n",
       "      var element = document.createElement('script');\n",
       "      element.onload = on_load;\n",
       "      element.onerror = on_error;\n",
       "      element.async = false;\n",
       "      element.src = url;\n",
       "      console.debug(\"Bokeh: injecting script tag for BokehJS library: \", url);\n",
       "      document.head.appendChild(element);\n",
       "    }\n",
       "  };var element = document.getElementById(\"1395\");\n",
       "  if (element == null) {\n",
       "    console.error(\"Bokeh: ERROR: autoload.js configured with elementid '1395' but no matching script tag was found. \")\n",
       "    return false;\n",
       "  }\n",
       "\n",
       "  function inject_raw_css(css) {\n",
       "    const element = document.createElement(\"style\");\n",
       "    element.appendChild(document.createTextNode(css));\n",
       "    document.body.appendChild(element);\n",
       "  }\n",
       "\n",
       "  var js_urls = [\"https://cdn.pydata.org/bokeh/release/bokeh-1.3.4.min.js\", \"https://cdn.pydata.org/bokeh/release/bokeh-widgets-1.3.4.min.js\", \"https://cdn.pydata.org/bokeh/release/bokeh-tables-1.3.4.min.js\", \"https://cdn.pydata.org/bokeh/release/bokeh-gl-1.3.4.min.js\"];\n",
       "  var css_urls = [];\n",
       "\n",
       "  var inline_js = [\n",
       "    function(Bokeh) {\n",
       "      Bokeh.set_log_level(\"info\");\n",
       "    },\n",
       "    \n",
       "    function(Bokeh) {\n",
       "      \n",
       "    },\n",
       "    function(Bokeh) {} // ensure no trailing comma for IE\n",
       "  ];\n",
       "\n",
       "  function run_inline_js() {\n",
       "    \n",
       "    if ((root.Bokeh !== undefined) || (force === true)) {\n",
       "      for (var i = 0; i < inline_js.length; i++) {\n",
       "        inline_js[i].call(root, root.Bokeh);\n",
       "      }if (force === true) {\n",
       "        display_loaded();\n",
       "      }} else if (Date.now() < root._bokeh_timeout) {\n",
       "      setTimeout(run_inline_js, 100);\n",
       "    } else if (!root._bokeh_failed_load) {\n",
       "      console.log(\"Bokeh: BokehJS failed to load within specified timeout.\");\n",
       "      root._bokeh_failed_load = true;\n",
       "    } else if (force !== true) {\n",
       "      var cell = $(document.getElementById(\"1395\")).parents('.cell').data().cell;\n",
       "      cell.output_area.append_execute_result(NB_LOAD_WARNING)\n",
       "    }\n",
       "\n",
       "  }\n",
       "\n",
       "  if (root._bokeh_is_loading === 0) {\n",
       "    console.debug(\"Bokeh: BokehJS loaded, going straight to plotting\");\n",
       "    run_inline_js();\n",
       "  } else {\n",
       "    load_libs(css_urls, js_urls, function() {\n",
       "      console.debug(\"Bokeh: BokehJS plotting callback run at\", now());\n",
       "      run_inline_js();\n",
       "    });\n",
       "  }\n",
       "}(window));"
      ],
      "application/vnd.bokehjs_load.v0+json": "\n(function(root) {\n  function now() {\n    return new Date();\n  }\n\n  var force = true;\n\n  if (typeof root._bokeh_onload_callbacks === \"undefined\" || force === true) {\n    root._bokeh_onload_callbacks = [];\n    root._bokeh_is_loading = undefined;\n  }\n\n  \n\n  \n  if (typeof (root._bokeh_timeout) === \"undefined\" || force === true) {\n    root._bokeh_timeout = Date.now() + 5000;\n    root._bokeh_failed_load = false;\n  }\n\n  var NB_LOAD_WARNING = {'data': {'text/html':\n     \"<div style='background-color: #fdd'>\\n\"+\n     \"<p>\\n\"+\n     \"BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \\n\"+\n     \"may be due to a slow or bad network connection. Possible fixes:\\n\"+\n     \"</p>\\n\"+\n     \"<ul>\\n\"+\n     \"<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\\n\"+\n     \"<li>use INLINE resources instead, as so:</li>\\n\"+\n     \"</ul>\\n\"+\n     \"<code>\\n\"+\n     \"from bokeh.resources import INLINE\\n\"+\n     \"output_notebook(resources=INLINE)\\n\"+\n     \"</code>\\n\"+\n     \"</div>\"}};\n\n  function display_loaded() {\n    var el = document.getElementById(\"1395\");\n    if (el != null) {\n      el.textContent = \"BokehJS is loading...\";\n    }\n    if (root.Bokeh !== undefined) {\n      if (el != null) {\n        el.textContent = \"BokehJS \" + root.Bokeh.version + \" successfully loaded.\";\n      }\n    } else if (Date.now() < root._bokeh_timeout) {\n      setTimeout(display_loaded, 100)\n    }\n  }\n\n\n  function run_callbacks() {\n    try {\n      root._bokeh_onload_callbacks.forEach(function(callback) {\n        if (callback != null)\n          callback();\n      });\n    } finally {\n      delete root._bokeh_onload_callbacks\n    }\n    console.debug(\"Bokeh: all callbacks have finished\");\n  }\n\n  function load_libs(css_urls, js_urls, callback) {\n    if (css_urls == null) css_urls = [];\n    if (js_urls == null) js_urls = [];\n\n    root._bokeh_onload_callbacks.push(callback);\n    if (root._bokeh_is_loading > 0) {\n      console.debug(\"Bokeh: BokehJS is being loaded, scheduling callback at\", now());\n      return null;\n    }\n    if (js_urls == null || js_urls.length === 0) {\n      run_callbacks();\n      return null;\n    }\n    console.debug(\"Bokeh: BokehJS not loaded, scheduling load and callback at\", now());\n    root._bokeh_is_loading = css_urls.length + js_urls.length;\n\n    function on_load() {\n      root._bokeh_is_loading--;\n      if (root._bokeh_is_loading === 0) {\n        console.debug(\"Bokeh: all BokehJS libraries/stylesheets loaded\");\n        run_callbacks()\n      }\n    }\n\n    function on_error() {\n      console.error(\"failed to load \" + url);\n    }\n\n    for (var i = 0; i < css_urls.length; i++) {\n      var url = css_urls[i];\n      const element = document.createElement(\"link\");\n      element.onload = on_load;\n      element.onerror = on_error;\n      element.rel = \"stylesheet\";\n      element.type = \"text/css\";\n      element.href = url;\n      console.debug(\"Bokeh: injecting link tag for BokehJS stylesheet: \", url);\n      document.body.appendChild(element);\n    }\n\n    for (var i = 0; i < js_urls.length; i++) {\n      var url = js_urls[i];\n      var element = document.createElement('script');\n      element.onload = on_load;\n      element.onerror = on_error;\n      element.async = false;\n      element.src = url;\n      console.debug(\"Bokeh: injecting script tag for BokehJS library: \", url);\n      document.head.appendChild(element);\n    }\n  };var element = document.getElementById(\"1395\");\n  if (element == null) {\n    console.error(\"Bokeh: ERROR: autoload.js configured with elementid '1395' but no matching script tag was found. \")\n    return false;\n  }\n\n  function inject_raw_css(css) {\n    const element = document.createElement(\"style\");\n    element.appendChild(document.createTextNode(css));\n    document.body.appendChild(element);\n  }\n\n  var js_urls = [\"https://cdn.pydata.org/bokeh/release/bokeh-1.3.4.min.js\", \"https://cdn.pydata.org/bokeh/release/bokeh-widgets-1.3.4.min.js\", \"https://cdn.pydata.org/bokeh/release/bokeh-tables-1.3.4.min.js\", \"https://cdn.pydata.org/bokeh/release/bokeh-gl-1.3.4.min.js\"];\n  var css_urls = [];\n\n  var inline_js = [\n    function(Bokeh) {\n      Bokeh.set_log_level(\"info\");\n    },\n    \n    function(Bokeh) {\n      \n    },\n    function(Bokeh) {} // ensure no trailing comma for IE\n  ];\n\n  function run_inline_js() {\n    \n    if ((root.Bokeh !== undefined) || (force === true)) {\n      for (var i = 0; i < inline_js.length; i++) {\n        inline_js[i].call(root, root.Bokeh);\n      }if (force === true) {\n        display_loaded();\n      }} else if (Date.now() < root._bokeh_timeout) {\n      setTimeout(run_inline_js, 100);\n    } else if (!root._bokeh_failed_load) {\n      console.log(\"Bokeh: BokehJS failed to load within specified timeout.\");\n      root._bokeh_failed_load = true;\n    } else if (force !== true) {\n      var cell = $(document.getElementById(\"1395\")).parents('.cell').data().cell;\n      cell.output_area.append_execute_result(NB_LOAD_WARNING)\n    }\n\n  }\n\n  if (root._bokeh_is_loading === 0) {\n    console.debug(\"Bokeh: BokehJS loaded, going straight to plotting\");\n    run_inline_js();\n  } else {\n    load_libs(css_urls, js_urls, function() {\n      console.debug(\"Bokeh: BokehJS plotting callback run at\", now());\n      run_inline_js();\n    });\n  }\n}(window));"
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "\n",
       "\n",
       "\n",
       "\n",
       "\n",
       "\n",
       "  <div class=\"bk-root\" id=\"d7590962-e309-4256-a5be-8c9d2dc9e102\" data-root-id=\"1396\"></div>\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "(function(root) {\n",
       "  function embed_document(root) {\n",
       "    \n",
       "  var docs_json = {\"e6e3aea6-9874-45fc-b288-c5ac0881af19\":{\"roots\":{\"references\":[{\"attributes\":{\"below\":[{\"id\":\"1407\",\"type\":\"LinearAxis\"}],\"center\":[{\"id\":\"1411\",\"type\":\"Grid\"},{\"id\":\"1416\",\"type\":\"Grid\"},{\"id\":\"1432\",\"type\":\"LabelSet\"}],\"left\":[{\"id\":\"1412\",\"type\":\"LinearAxis\"}],\"renderers\":[{\"id\":\"1430\",\"type\":\"GlyphRenderer\"}],\"title\":{\"id\":\"1397\",\"type\":\"Title\"},\"toolbar\":{\"id\":\"1421\",\"type\":\"Toolbar\"},\"toolbar_location\":\"above\",\"x_range\":{\"id\":\"1399\",\"type\":\"DataRange1d\"},\"x_scale\":{\"id\":\"1403\",\"type\":\"LinearScale\"},\"y_range\":{\"id\":\"1401\",\"type\":\"DataRange1d\"},\"y_scale\":{\"id\":\"1405\",\"type\":\"LinearScale\"}},\"id\":\"1396\",\"subtype\":\"Figure\",\"type\":\"Plot\"},{\"attributes\":{\"callback\":null,\"data\":{\"names\":[\"harry\",\"said\",\"would\",\"hermione\",\"could\",\"back\",\"draco\",\"one\",\"like\",\"know\",\"eyes\",\"time\",\"ron\",\"looked\",\"get\",\"asked\",\"well\",\"even\",\"around\",\"see\",\"head\",\"going\",\"think\",\"still\",\"go\",\"severus\",\"face\",\"way\",\"room\",\"ginny\",\"hand\",\"sirius\",\"something\",\"want\",\"thought\",\"potter\",\"right\",\"snape\",\"away\",\"much\",\"look\",\"two\",\"never\",\"really\",\"knew\",\"first\",\"let\",\"made\",\"good\",\"malfoy\",\"little\",\"wand\",\"felt\",\"dumbledore\",\"turned\",\"james\",\"come\",\"got\",\"make\",\"took\",\"remus\",\"lily\",\"though\",\"sure\",\"say\",\"door\",\"tell\",\"take\",\"us\",\"looking\",\"dark\",\"voice\",\"voldemort\",\"last\",\"long\",\"told\",\"need\",\"left\",\"yes\",\"man\",\"wanted\",\"anything\",\"next\",\"oh\",\"came\",\"nodded\",\"love\",\"moment\",\"people\",\"saw\",\"another\",\"things\",\"went\",\"hands\",\"ca\",\"help\",\"day\",\"enough\",\"death\",\"smiled\",\"professor\",\"year\",\"mind\",\"nothing\",\"found\",\"ever\",\"boy\",\"hair\",\"always\",\"find\",\"bit\",\"seemed\",\"behind\",\"hogwarts\",\"thing\",\"bed\",\"trying\",\"feel\",\"started\",\"put\",\"since\",\"life\",\"house\",\"night\",\"heard\",\"black\",\"without\",\"smile\",\"better\",\"years\",\"gave\",\"magic\",\"might\",\"side\",\"weasley\",\"everyone\",\"father\",\"sat\",\"began\",\"someone\",\"almost\",\"walked\",\"done\",\"finally\",\"already\",\"-RRB-\",\"tried\",\"place\",\"every\",\"stood\",\"everything\",\"friends\",\"-LRB-\",\"three\",\"lord\",\"front\",\"pulled\",\"small\",\"also\",\"quickly\",\"course\",\"keep\",\"girl\",\"body\",\"best\",\"towards\",\"else\",\"arms\",\"neville\",\"table\",\"give\",\"mean\",\"work\",\"family\",\"sorry\",\"albus\",\"please\",\"end\",\"many\",\"world\",\"school\",\"mother\",\"great\",\"lucius\",\"old\",\"together\",\"new\",\"quite\",\"stop\",\"happened\",\"leave\",\"replied\",\"maybe\",\"mouth\",\"open\",\"yet\",\"must\",\"soon\",\"later\",\"floor\",\"getting\",\"able\",\"words\",\"arm\",\"name\",\"lips\",\"friend\",\"least\",\"hard\",\"feeling\",\"actually\",\"suddenly\",\"anyone\",\"seen\",\"slowly\",\"sighed\",\"luna\",\"talk\",\"spell\",\"shook\",\"making\",\"gone\",\"rather\",\"george\",\"believe\",\"probably\",\"held\",\"home\",\"far\",\"light\",\"used\",\"inside\",\"granger\",\"wo\",\"second\",\"opened\",\"rest\",\"taking\",\"fred\",\"stopped\",\"hear\",\"idea\",\"whispered\",\"rose\",\"coming\",\"across\",\"thank\",\"try\",\"breath\",\"care\",\"slightly\",\"tom\",\"parents\",\"yeah\",\"students\",\"called\",\"needed\",\"ask\",\"laughed\",\"close\",\"may\",\"watched\",\"use\",\"continued\",\"slytherin\",\"sitting\",\"gryffindor\",\"part\",\"heart\",\"morning\",\"blood\",\"feet\",\"happy\",\"lot\",\"pain\",\"book\",\"muggle\",\"remember\",\"okay\",\"hall\",\"fact\",\"past\",\"stared\",\"reached\",\"mr.\",\"however\",\"fine\",\"chapter\",\"ministry\",\"thinking\",\"wrong\",\"along\",\"onto\",\"hope\",\"alone\",\"shoulder\",\"set\",\"read\",\"moved\",\"wizard\",\"bad\",\"standing\",\"either\",\"red\",\"chest\",\"decided\",\"fell\",\"talking\",\"dead\",\"minutes\",\"stay\",\"ran\",\"hurt\",\"understand\",\"young\",\"potion\",\"mcgonagall\",\"fingers\",\"instead\",\"caught\",\"person\",\"kill\",\"days\",\"woman\",\"miss\",\"potions\",\"point\",\"others\",\"air\",\"tears\",\"ready\",\"kiss\",\"ground\",\"tonks\",\"kept\",\"matter\",\"taken\",\"son\",\"times\",\"wait\",\"order\",\"forward\",\"deep\",\"closed\",\"noticed\",\"eaters\",\"curse\",\"story\",\"reason\",\"lost\",\"kind\",\"raised\",\"full\",\"sound\",\"sleep\",\"magical\",\"thanks\",\"move\",\"blaise\",\"different\",\"wall\",\"whole\",\"office\",\"bellatrix\",\"several\",\"question\",\"followed\",\"start\",\"half\",\"turn\",\"word\",\"perhaps\",\"anyway\",\"large\",\"outside\",\"answer\",\"holding\",\"class\",\"attention\",\"answered\",\"chair\",\"hit\",\"completely\",\"lupin\",\"exactly\",\"quietly\",\"scorpius\",\"waiting\",\"cold\",\"immediately\",\"mum\",\"ago\",\"minerva\",\"spoke\",\"robes\",\"saying\",\"eye\",\"longer\",\"today\",\"leaving\",\"watching\",\"stepped\",\"true\",\"green\",\"seeing\",\"hold\",\"staring\",\"child\",\"surprised\",\"pansy\",\"given\",\"master\",\"silence\",\"dad\",\"nearly\",\"turning\",\"softly\",\"nice\",\"neck\",\"headmaster\",\"finished\",\"fire\",\"meant\",\"brought\",\"closer\",\"simply\",\"witch\",\"grabbed\",\"four\",\"show\",\"grinned\",\"hell\",\"war\",\"managed\",\"realized\",\"charm\",\"says\",\"call\",\"change\",\"bloody\",\"looks\",\"narcissa\",\"top\",\"running\",\"children\",\"quidditch\",\"kissed\",\"killed\",\"chance\",\"speak\",\"common\",\"brother\",\"shut\",\"skin\",\"week\",\"sent\",\"alright\",\"added\",\"pointed\",\"castle\",\"met\",\"rolled\",\"known\",\"expression\",\"boys\",\"molly\",\"supposed\",\"returned\",\"walking\",\"muttered\",\"whatever\",\"run\",\"fight\",\"knowing\",\"hours\",\"appeared\",\"leaned\",\"cast\",\"girls\",\"giving\",\"less\",\"happen\",\"upon\",\"books\",\"thoughts\",\"shrugged\",\"pretty\",\"water\",\"sort\",\"guess\",\"spells\",\"throat\",\"gently\",\"letter\",\"worry\",\"real\",\"pushed\",\"loved\",\"big\",\"stairs\",\"hagrid\",\"possible\",\"surprise\",\"couple\",\"glanced\",\"white\",\"meet\",\"seem\",\"passed\",\"fear\",\"bill\",\"five\",\"beside\",\"become\",\"desk\",\"watch\",\"sit\",\"smiling\",\"stand\",\"power\",\"placed\",\"dinner\",\"although\",\"sense\",\"toward\",\"corner\",\"merlin\",\"late\",\"safe\",\"die\",\"sir\",\"stupid\",\"bring\",\"walk\",\"wizards\",\"anymore\",\"soft\",\"kitchen\",\"snapped\",\"agreed\",\"eater\",\"wondered\",\"baby\",\"knows\",\"wish\",\"quiet\",\"telling\",\"break\",\"changed\",\"group\",\"peter\",\"laugh\",\"window\",\"pulling\",\"moving\",\"return\",\"charlie\",\"plan\",\"angry\",\"case\",\"short\",\"trust\",\"explained\",\"shot\",\"free\",\"dropped\",\"shoulders\",\"near\",\"sister\",\"gaze\",\"barely\",\"sight\",\"reading\",\"live\",\"beautiful\",\"especially\",\"died\",\"within\",\"cut\",\"figure\",\"christmas\",\"shaking\",\"tonight\",\"working\",\"worried\",\"spent\",\"clear\",\"months\",\"seat\",\"entire\",\"blue\",\"legs\",\"shouted\",\"dear\",\"memory\",\"frowned\",\"living\",\"stomach\",\"mrs.\",\"fall\",\"conversation\",\"straight\",\"allowed\",\"control\",\"wizarding\",\"stone\",\"remembered\",\"tone\",\"afraid\",\"fun\",\"form\",\"picked\",\"anger\",\"glad\",\"broke\",\"touch\",\"entered\",\"hour\",\"silent\",\"quick\",\"grin\",\"truth\",\"cried\",\"meeting\",\"nose\",\"laughing\",\"hey\",\"play\",\"smirked\",\"step\",\"certain\",\"important\",\"became\",\"weeks\",\"filled\",\"none\",\"auror\",\"warm\",\"team\",\"using\",\"food\",\"percy\",\"older\",\"empty\",\"fleur\",\"threw\",\"hate\",\"expected\",\"cheek\",\"yelled\",\"worse\",\"breakfast\",\"problem\",\"ones\",\"clearly\",\"teddy\",\"cloak\",\"memories\",\"perfect\",\"waited\",\"obviously\",\"shock\",\"liked\",\"arrived\",\"alive\",\"sometimes\",\"certainly\",\"riddle\",\"wife\",\"carefully\",\"note\",\"despite\",\"summer\",\"gotten\",\"information\",\"daphne\",\"job\",\"forced\",\"madam\",\"loud\",\"except\",\"ten\",\"wrapped\",\"tomorrow\",\"clothes\",\"glass\",\"deal\",\"library\",\"strong\",\"calm\",\"strange\",\"tea\",\"worked\",\"trouble\",\"confused\",\"battle\",\"ear\",\"evening\",\"owl\",\"mine\",\"easy\",\"normal\",\"wide\",\"minute\",\"seems\",\"parchment\",\"lay\",\"makes\",\"paused\",\"twins\",\"apparently\",\"daughter\",\"middle\",\"suppose\",\"moments\",\"tired\",\"high\",\"broom\",\"notice\",\"somehow\",\"led\",\"promise\",\"usual\",\"secret\",\"lying\",\"cup\",\"broken\",\"piece\",\"chuckled\",\"arthur\",\"tongue\",\"sounded\",\"men\",\"line\",\"questions\",\"pale\",\"means\",\"doubt\",\"muggles\",\"save\",\"attack\",\"wonder\",\"helped\",\"familiar\",\"explain\",\"finger\",\"bella\",\"aurors\",\"gasped\",\"uncle\",\"game\",\"hissed\",\"jumped\",\"minister\",\"situation\",\"charms\",\"bright\",\"pressed\",\"hospital\",\"mate\",\"whether\",\"position\",\"lifted\",\"glared\",\"manor\",\"moody\",\"headed\",\"seconds\",\"shall\",\"catch\",\"wants\",\"often\",\"fast\",\"asking\",\"dobby\",\"teeth\",\"third\",\"early\",\"asleep\",\"pomfrey\",\"listen\",\"wearing\",\"neither\",\"kingsley\",\"bag\",\"following\",\"letting\",\"sigh\",\"besides\",\"screamed\",\"truly\",\"forget\",\"write\",\"covered\",\"pull\",\"mark\",\"follow\",\"beginning\",\"hoped\",\"elf\",\"keeping\",\"shirt\",\"fighting\",\"student\",\"flew\",\"playing\",\"wondering\",\"likely\",\"eyebrow\",\"guys\",\"direction\",\"easily\",\"usually\",\"learn\",\"hated\",\"ended\",\"crying\",\"starting\",\"lip\",\"somewhere\",\"enjoy\",\"future\",\"husband\",\"edge\",\"eat\",\"flying\",\"azkaban\",\"ah\",\"soul\",\"offered\",\"cheeks\",\"hoping\",\"cry\",\"damn\",\"falling\",\"earlier\",\"tightly\",\"knees\",\"send\",\"shocked\",\"powerful\",\"train\",\"al\",\"serious\",\"money\",\"missed\",\"pair\",\"drink\",\"disappeared\",\"silver\",\"choice\",\"speaking\",\"hot\",\"pocket\",\"hide\",\"expect\",\"age\",\"brown\",\"aware\",\"ravenclaw\",\"definitely\",\"low\",\"spot\",\"eventually\",\"wake\",\"putting\",\"hug\",\"bedroom\",\"forest\",\"ears\",\"killing\",\"mad\",\"aunt\",\"single\",\"simple\",\"god\",\"mirror\",\"waved\",\"scared\",\"laughter\",\"dress\",\"six\",\"present\",\"continue\",\"admit\",\"sounds\",\"join\",\"exclaimed\",\"a/n\",\"handed\",\"paper\",\"feelings\",\"blinked\",\"corridor\",\"breathing\",\"slipped\",\"check\",\"showed\",\"relief\",\"forehead\",\"force\",\"sleeping\",\"doors\",\"smirk\",\"younger\",\"box\",\"final\",\"wands\",\"news\",\"murmured\",\"writing\",\"protect\",\"causing\",\"ok\",\"indeed\",\"bathroom\",\"kissing\",\"ball\",\"steps\",\"wards\",\"dangerous\",\"seamus\",\"dean\",\"lunch\",\"glance\",\"alley\",\"obvious\",\"entrance\",\"sudden\",\"stayed\",\"business\",\"apart\",\"evil\",\"married\",\"cedric\",\"busy\",\"visit\",\"touched\",\"honestly\",\"needs\",\"difficult\",\"ring\",\"party\",\"imagine\",\"drew\",\"fault\",\"number\",\"stuff\",\"burst\",\"fuck\",\"dream\",\"leaning\",\"walls\",\"aside\",\"groaned\",\"crowd\",\"slytherins\",\"interrupted\",\"couch\",\"silently\",\"darkness\",\"remained\",\"learned\",\"tower\",\"hello\",\"stated\",\"foot\",\"snake\",\"dragon\",\"odd\",\"mr\",\"teacher\",\"ahead\",\"growled\",\"wanting\",\"lavender\",\"spend\",\"allow\",\"stuck\",\"match\",\"werewolf\",\"lose\",\"date\",\"finish\",\"points\",\"human\",\"grew\",\"blonde\",\"realised\",\"act\",\"hiding\",\"relationship\",\"response\",\"regulus\",\"caused\",\"umbridge\",\"unless\",\"shop\",\"tight\",\"huge\"],\"x1\":{\"__ndarray__\":\"60Q5QUUJD8KQYo7BSjhSQfriisG8DF5BB9xQQX1uQUHGFynAzNecwWZTgUHr8whCtytrQUFkUsKQRhTCNj4Vwg4YN0EruOQ/j85iQV4bzsF7HphBPoS/wX5xmcFYFaJAeeQbwp4wmEGNcXVBuHSuP9iWMULfa2RBImaqQfiaiEGFJMBBOm9VwZM/k8FjgJZBYyEFwQqAnEGofqlBNMseQbC/U8LwwOtBdvrkQGA8zT8AHanBHnWUQamigcGJO+vBefAIwe6lj0FaBSJBk/noQUJpncFLPqNBs5Q/wviggUGo0R3CU0kVwgOF8sExVirCrOaCQckue0H7RkBAeq7swF13rsFbpEhC2H7BwdZ0JsKLXFhBoZdTwtgHvUFEo0/AZZS1QfXLj0ECAR5B0J25wbfgScFA3/nBY5XnwPGK80GftqzBJXTCQe9CkEFIevHAhD4kwozKAMKYLG/BgkoMQhJ/DUJ2oNnBJbdoQRyyJEJ25yjCFAmsQegrYsHy58nBGQ4OQsEkYMB3eDdBLfXnwctTt0HlxhZCfKCFQWNmtkFa1uHBtFXtQFQb6kFAAHVBZfuVQD4W2sE9kjVB4H2BwcBtjUHRZBtCxkwDQrlUrUEolMXBUjefwSIl+sGdakzC35qJQAzy8kFLJSZCeGMQQtsO48EtXURBtbULQZK5+8AftAu/Ow0mQs5JFcJg4MtArdSPwfpWm0FPOJtBRvvUQYvCBELrNy7C0N78wRiR2EHXgQ9BYug1wgtVoMFBDQFBsVTEQGVU0EAj4MjB4ODgQalSdUHHsirCFBnCQeyH7EFu+01AqCntQdGF50HQu6lB7l5AwlugEUHTAW1AEZIJQZusgUHYVizC+EnqQfU/f0Efeac+GJiVQTiOrsCOE6JBzGVtQd7mpEEVShDCChWTwdH1e8HA5v1BveYswW8skkGSJM3AuLCuQXT/2UEciuxBq6AbQvGGB0IjffNA6dLOQQkn9kAXRFPAqpJAQWuKSb/CytfBXledwQ2H8cGPeg/CyWoTwLDeHkG4pZbA6Pm+QPqilMGddvBAvHOIQanCkUGFRxPCumvYwR9750FeRKZBLX79QadcGkF5IOdBRWfWv3rdHsD2NJ3BNVXcP7huKkGiU9NBeTK0waPKGkFnw+/B6sdgQc8xxcFOOEhCoMfFwTKM7cESJZrBh445vy7vqkFLz67B2aphv9OQP8L84hRCwFYoQR+J+kA/4t7BluFxQfIsi0HpMmHBcHanQVRin8AVIcZB9dQpwu0CrEHk4vLBoIDJwWfc+b6qTBDCSq9sQYJdJMKbbnpBq6ZEwcUMMcHeWiFBxtObwamfTUErlqdBHhEIQt41C8HmZRVC7Bn3wWnnrMHcSZ/BPyb3wa7ZFEA0HJDBeoBlwV7H5MEEPADCbD8BQpBaLsJkGQFCDnm+QQzkZEEIWxJCzGJIQVVorUGL+hLB0Co8QZikd0AiZzBCaDYFQmr3q8E6xe/A5f41QnId7z/gc2xBuqlXwgvpMMLC3dJBDhdnQIZe3sD330rBeAkqQuQulMGCXhPBWVJ5QWZDgkFL3HjBtxn8QCdbjUG2BlDCw2iJwbqvN8LvHP1BvYAIwRXzLMI0en5AjOgpQRiXc0HN/rzB/ntmwhT3lsFlmC/BWC4zQt+eJcL+EkLCXXZLwUtapsGnIrFA2EjmQeeZoUGYOrRBCdgrQQATAcKkfwNCx0jbwT/NK0K0f/JBsMKJwB1W5kH0QSm+NGkOQokBN0FHyqLBKoa9wRVVMcGZ+JFB5XO0QfSSKsJm/eC/bP4mwhL6BULSTSpChIsXwjJAt0EwZo9B4K0BQWrtw8AF98/B5Z4yQq81REIjSlvBzOpDQA90gsFQTXpByIRgwnhRiEC913c/U3Y8P90BKkHPbETBaBYwwlVHgkEz3LXAxQF6QSK2wUG+ciZCvufVQeTi4EGZXNtBG4gEwl6r2cE4ACxBw1BCwvEW5EERNyfAmGRuPZn7BEHErWlBzRGRwe7qPsLrB/BBJ0ARQCjBEcKIOLpBxmjxwc2muUB4V7NBJ3yAQE7mJkGSbYBB0joywczljkBO4RdBF1cFQuJ8hkGj3qtB0pUZwrj1DEK+uajBvGyAQWDXR8DQoUPAlzP5we4CYcEG/jzCp5oswBQ8HUEkd1zB6S45wlSgWMJ6xgVCJTokwZROiEHpGxDCZAW/QSn88UCS9QVC2AYOQUcsP8IELSVB39MLwcQ+ZUEjV6lBuaf3wXT8N0EVgp/BZlggwhwQjEF6MuVAj8T0QbsXVcLHA/JBex/OwfC27sFAlJ7A9QydwAl8zcFbVcjBAtxLQjiaE8KsgPDBzvaowTQ0BsH3yVLCl5LPQf8Xk0EzUkLCJsAIQhOGAEL17ETBPmBTwTfMG8Dv5sLB8Gh6wcv0EUI0nsTAXPttQegcEkL/FQzCFAviwKcrCcIzSFDCAWsmQkUxzMGwn87AYM+uwZC7lsB3XBFCOgvFQdgxwMHadgDCfxc6wuI1DMLHFbVBPKZAwrHzDMETAKzBlsUuQmT/C8JJcCzCwsHuwUNAEkLX1xTCPEryv6m8sMEkpUJB26kvQnHoiEE74f/BHEh8wNiaOkEe6nhBDW+IwUBqSUKKi1JBieocQWTiIEKp6VbBQroZwCg9QsLK5oTBmAPzQKM+PUIXNo9BciUAQNcPwUDdLeFBkkNQwnqkL0HJRtvBfshiwUExB8L9xr9AHlW0QaW38EFtF5BB5p4MwpI9qEHXClDBVmgnwrFN4cElDibCm0m8QP8KT8KDq1hA1/VhQCXahcCZgZVBfRliQeFRyMAzbPHAiPcjwSl24MG+ucLAhMGMvw33HsL2pDvCGs0DQlasvUAXNzBBJlIwQogjEsJgWAnCuagyQuQstsHozgNC60mfwcGnacEWlMVAAk20wZmgT8JY+KjBnjcbQnCigUH8aMDBX1tGQn+LQML5DzXC0iH9wRedsEHGu5U/Daz1wFbx2j9yeg5Bdmy/wSmJC8KMhVPCYuZoQQJ3WMIdWYxBnQJLQeVTEULR2YVBhWTpQC0OdUEHCIbByAHvwZyf3MByCGg/hxhZwXHZbkH2EijCwCviwZjYA0I0S8XBkcRhwNa0gcFxlknBBs8PwhicpMDvlilCAhe9QV3xwUHMuRtBc1+QQWdYDcInEgTBLH+BQfQs9MH6zn3B6m9tQQrlz0G/+mnC8lNKPtB/X0HdJePB4j2vQLdSKkE9PcdA/K/RwdN9UcCkEjLBW1VUvhELIkERQhvC0KupQHpGJ8Fc61PCyfF1wbAsBsLryxFCnrbTQK+6+EAhZvTAsbBxQMiuCcKNWaa/kzSJQT1ozsHBjw7BS/ncwI1288Ex2g9BQzTQwBd+TsAs4w3CfBEsQhQqjEBJaKJBVmsgQhPdXkDJHgpCyzLkwYJb5D9DqrVBH83MQJ58lL/M17hB495OwlEegsE0WqvB8KR9QctHDcKY3cG+izhKQD94nL7ZzAlCRQSlQFn4tUFXqA1CHV2DQabA2cB8VzLBtaVkQAKNv0C7J4HByMkDwtoBLMGwpc8/H+Sev6MDrEEVNgZCZhgaQTRMIkKvI6VAKdoJQlHVE8KBKbtBz7eKQTyTlz+ETuHBdC62Qfz6PkE/vtPAdinxQWa6RcIMR4TAVqsPQi/460CIFJO/mBArQtgtqD/s+adAYQmewOCSukAEoYbB0UoFP/w9FcE2tJ/AroxRQdGNEEIYhxtCWd63QPU6OcApW9a/zMIbQX4cDUJfO3zBSjYtQjGsNcKYkuPBqjP0wYQat0EZYoRAUQYKQnwGq0Hz44bBqdsxQplpFsGFlThB1UGjQRRuqsF9OcNAXx8ywof4WcE8XSQ/LeIdwSygM8L0sQZBpCJVwuv6ukF6lPnBCl/BQeJ5vkE8nRq+Gk0MQkHOGULnLN9B24QoQStIl8HQIbW/Y7QIQg340cFqdivB1UCwwVwQCsKbo6vAOQ24wRXlukEPGNVBNSweQoKsBMIJewtCyMEFQruoDMKE/k3CFGQqQlLOxj/7mk1CkzEMQeR4NsIF0ZdBHnXfQWSOXkBDncVBXVtfwtaBVMJ8HhJCzurFQV3zMcIgYjNCFIWQwVvg/8GMbp3BOlggPl9+y0Cp65vB8K3KQW1ew0HWq5pB38nowOA1CEBgb6dBKNvQwSEwpkC5mpxB20bGQQuOOUI4qgLCcKCBwbrp5MGAxbQ+uPYIwruh7kB9pqTBBRCCwVy5XUDo1j7C+7/nQVCN+sEGV9XBpti5wU0n0EFFGCrCa00LQg2iDMH5n/FB7dtOwiro3MAev7PBv14gwI4BtcFzeZrAH96iQXIdu0D2oYA/uO+1wbHkg8Eq3BfC7oewwSUK1MHoOwNBjS3ZQbNEc8H4Jx5BYagIQmr7nEHu6o0/fEtMwgli1EEJXuvAVi0RQX4mDML+DnBBbUm7wcO5tMGzaALBBRhnwuAmfUElfnLAeT+NQanlC8LzXx3BL2w5PmNuh0Ao+oBBri3ZwNPPfUAH1YjBMm8SQtnz4kCaigrCX/InQZOdib6cV6PB7KlDQLMFPULcYxXCamNnwTnzGULoahFBSNu7wQ1WAELYTrm/+dg1QRTX3kEgyOVAWoMYwiRDS8KwnBzBOvsyQt5qKkIEe05BAE0+wVRb2cAv7wpCrKNPQbVBwrzGjcjA9P6GQW0vTcJjgC3Bh/PHwXESCULH3vRBhWF7QAqC28G4NG3B6w4lP7a64MF3DhLCt8K3wMdkDMLSFi1CzUOZQXdv6sERXDtCm4YWQbd+RsI5ZG/BqCTdwXwS8kD+zYBBglFTQECVhD9e9EhC8KjqwL8BxkC4KzZCc9SJQX0l6kEMabxBLnMPwmjafsHc4s/BmmP4waQv8MBFrEBA++A1QuKyQ8ETyiZAVXAQQc2AUULLF1i/xWp7QcnieEET1mhAA5RVwtJaK0KWAIjAZvX4QLmZ+kD9cyfCHfYBQFk2r0GKarlBkEaIwNIhY0EhHADBD4nIv4H1esFpioQ+VmmcwV2+PMAu2SRBPUEnQKevtcHpTDvCW3QSQrOvyUESBI5AiEdXwuAj2MAJLV5BIDoswqTmcUGmFqJBd70AwkpKHEJPRRZCFmojwtPDrUH2xilBJk/WQHEGKMLmmrXBY/cjQpzLEcHTeAzCJYatQXtBqkESgKFBHyiYwNm40kHYpu5BetxIQaPJCsIRsLTBKJ1cQbvaD8KoU+3BRTpIwj4SB0J9K8xBvmeCwUwmjj5KNd3B8tEEQlkQzEEKmQ/CXbpKQaldyMGAsDPBcFcTwloODj8XuY1AfpaPQWxb+MGwJqVBWscYQDwdGkIMRnPAthgEQQ==\",\"dtype\":\"float32\",\"shape\":[1000]},\"x2\":{\"__ndarray__\":\"SWDeP2ZBhkFhc9HBQvKDPyLRy8EWOEfA8ekBQFtAKz2AiIjAtuLDQAPyjkG6eq3B8bb0PyTHWcEKchw/pwppQWcwzT//fL3A4+uswK/tbEAXXvs/XeHXweCqtUDUiCnAdne1P1D/gEAj4qNBAr00QBw6+L4elao/20bZQTsB4EBnFa/ByDixQNOgcMBQdRtBm4zfP2O0qECWyxXBXA4zwbDefcF/fgbCYobbwDyHa8AWnWDBj/uAwQM0RcI+hAzCvJQywakEG0GSHdjAKgHNQQGBW0HB0pdAbE6gwGxfyEBUzGI/b2jrv9B9D8IGmbHA8rrYQFZ/r0DxPfTAF8TJwJLN0D698tlAd0EtQBRvrsBoO5pAVdBbwTb0oj4bbe5BB4kYPh9+f8HzglzBXw9HwFl/rkA/y4LAL10mQVpv5UCFvKnBypavwc+1TsF4TShBtc6cvkDCe0EKExxBi9DIwUj6sUDq0BrBHL9RwT9/gsEUUTi/CFHfQTNi08GNL8VAlbGjwcTtkcFV99XBc+OcQduldEBXZqzBOLX9wdqntcGjAN3A6drowKmC7EDPBSdBhcqrwNJcmkDJxdTAvYSzwUN3McGnJqC/I6tNPrz4FkIECLjBwkNEQUOLL8G1a+5AhDMJwQFUhkEjuLS/2+mewbfaLMEQrotBAjVXwFFFFkKIdLPBtvSpwVLp6kDSGmzBWGvXwU6mNMCP0EBBFkeiwTmMiUEuoGzB2fI2wWKFqsG3Nl2/J4ePvgxlCMGcgIw/98ErwKfWLMELmbPB6l0AQv5DXsGUvkLB4aWhwQndUkES2wnAtTwIwrfFCMD+m07Bu2SlQMQLlcFshzzA9DKEQAzvyMFLp8PBnLz+QLnV3UG3MoHBg6LXwInJgsEbpOJBmd3xPlgBIELVtaNAgVfOQBRZtMDDS4ZBe/XhwO7FmEBybspAB7l1wFUWEMIGR4dBmVCiv2cOjkHtXLDB6snGP16PBsKmb9HBC/kAwAVMksEOcDO/KX/8wD43M8CpUIFBaYLJvwK6zkFb/QXCwsibwBvW3ME4MkhAMHacwSAPC0JMPtM9TkzGwScEesGpBNZBWXGaQTcqzEHMq1JBBvc9PhIBdcEMw1hBI9UqwH8iKEA+6qjBJgIXwVzKrkD9Ha5BcWbHPvvm6D4l2z3BzUqCQWa9DMKQoiPBLbWlwdsWiEGuyMlAwfEjwHAQIkHcdIS/a9IywcsEhUEfkpPBUgEPwZD58b+8tdPBf3eGwToqB8KPJknBlzO5wC/li0HtyB/BD66LQKPEAUHpqZlBcuJrQJb1Rj95RqLASsu0QU12RkDNoWw/rNYBQYncTkCHa6Y/mWqPQTB8sUDNS7NAylgTQC/up8FYs7k/TeSjQQ2BWcH6Wd7B1LMPwqJpj8HHxTHBPe+wwD+xbMHI5bPA3h/wwMM48EFLR5XBInDMQEGYtUGD+grBb7znwH7zMEHDsZtBrDaKQICNokBJGWBA2nRcwPDOg7+WZITBqB88wf6OkUCYW6dBvNjWwBb+HUDpoI1BVzDhQCUwQMAh7pw/hQxbwM5fwMDoruxAFNsVwZS4yUHBoQlBV4yVQcSsL0C0hONANFg3wfKxRMHVLCfBnN2bQVhu3UGO5ZLBMcy+P1C8D8A/7XnBtd22wR0koMH9Z/o9tN1fP2y7v0By4gPC8qMpwaY2ykC47+FBz6N/wFYnpUCHWKY/kqUFQQM9pcFb2wJBYLs0QYR7KcExiT8/g8GqQEd3NEE8FsdBpOHgwQ8U9UFRFApCA3Y2Qew7wcGV5CZBgIbYwOQzbEE9OZLBTg3MwHdN8kCwSQnBR0BKwRimCsLtICXB2pRFQRPsRMGNwY1BL/x6wcm5KME+vtPBWgHSQPWpm8EAGBjCXIXewU1THsLD0rRBT0sNQILZrT8j4g7BkAoQQuhIdcEfWAlAwASAQClADsL8btXAJCvsvzNnKMBB9y+/fFlswPs2d8EKadK/9GUYwJLnmcEShQPBgcsMQMSsI0FHuYY/D0INQcKgekH7+RlCJWdvwapjiL/ANjNB/tJyv7BwBEFh1YVAfl3zwT8lykDCrWFAArqrQY4+m8GJqKRAymWBQelA/0EhdkG/GBGGQZryrcG7BA5AMHhywH8oD8J9lClA4d4HwaNXk0GLNQbBnH0wQWEbQ8E8Wz9BT4XUv0XpkD+63MtAwrpaQBqk4cHwGqxBuEZyv5sxqMC1KQ1BRegcwbFeykGyl4JAEVr/wLQGcUAOqo3BILXOvz1IDME4fGLAx1QPQUwhuUDWkAfCUmEjQOemnUFKQ6ZBti6NQRqMr8G+ZjfByho/wdsFikGXdQ5AMoAOwsgBpEH/JXbB7m0SQL5Me8C2sB++/WUxQaIgScFUzPRBANxLwVanpEC5hV0+DxbVQQozgEGJdgTCfovSQemgq8HHriZASw9OQNDWfEEisgfBvhwswOdL7MAOvQzCSNEZwVA/A0L+CwtBC0qDQdgR1MFDL9jAN1tQviY0lUH2l6bBL8Oev+RkgkENw1/B4kKtwXwav8Dxg/1AIySVwcOrCUHnw+VAqvSswfmdoz+aq7rBBA6aQT/LAsIiNo5B99tOwJr5pkCtjtXBzTKsQD6EL8GJod5BakobQUyHyUGTkvC/I1EPwTbLhECS0XTBhsuhwS/LQMCNG0pAqpWRwXH6UUHeUgfCygxPwVadhkGFHDVAFY6rwXFlbMBSQStBhQBvQXOjDsKVe0HBYyCAwYyOHkJxAg3C5FNmwVolnkEH50jBsQN4wZ914kAy8QhCdCjqwFQJXL+6rdjA6v9EQXt7xEH4mbfBLciLwWNsBkFNdyhBIuN3wMTsCMCBCqC/90HiQAJjDME2h5LBBegIP4ioq0E4q1tB3r9FQdqHCUKx7kZBeRJfwbjJAEE0ndfBVuszwBvydcCWfg7CWwYDwdh4/0BGi81BH+YDQc1GtUB2gRJA1SbFwHEReEG2Xy1BHxIkwFsNDj66tX/BCYDgQEB6a0FIy84/VxKtwYP1nUCT2M9BpKAXwV1ffkEXWJNBU7G8v6sKYUELlpVBcywOQRpZXcGBhMHAO3hLwV1sIMGmeZBBlbGrQPtjlMGTNoJBJiElQHGyycAy+zXA5yelwdHQpcBipKrBYbgbQk35dcELi5RBcr/pQVWav0GJJj9Bv7UIwgM8i0EmDc9BofPjQXcdnUFAZ5s/ug57QadWBUG6jsXBs4aCwRt7HsJJds1AuwJJwf/l7kFj4pLA+TGlvw1HF8B8Kx/B0BspQbzp6cB+sU7AO4spvQL19sB2p7vBjbrbwXTRokAU8RVC+bo8PyT8x0Ff34FBd+mzQR3ZwkFmR/RA92EownBNmkETTwFC/jvVwBeEP8Gq9XjB1RGqwa+iqcERor/BdKIfQbky0UD/RjvBZheSwQi2DkKXrIFBWtoHwiTUw0Fvc2NB2wWDQMFxEUGQTYjBiPq6QR5lvkHmD7fB+1ILQj6XkkADnf4/A+9twFoPO0FP2QRCYaAHwj3BWMFqcPPBNTCHwP4OUEGQz3HBoC/ywBsKgcEiXXjATWDCwA1itz+MUHpBZlLWQO1nyUFPj9nAgs+bwZGXFsA9yOTBCN0jP/qBrUHPhMXBqE61QJWfmcEGVY3B5K8Qws6sC0G3qUVATCv8QS2wGkJ+li9Ah2BjPsmoWMGxdNLBEdIjwQBqFUJlotbA5WqSQFbwCsCsDI1Ba+jBQRSBmsGxnMpBG3QHQTzgW8HPkyHB4/+owTjhxsG8yrDBrRjCQetleMGODAzC+giKQQ2VkEF7kIDAfONqQVhMT8G7CK1A/Rm+wep0i8BhoILB7cYGQkLBFkGwa9E+K//uv4cFB0GUVxTByOOUwUo8f8ECXx1Cx3SFwAgIuMBU+p1BhDeCQf9p7kHMYxrCYwwHQTYrQ8EorNjAisZ8Qbyfk8FVXA5B/WecQNV0EkGtumVB264HQoJOFEFSCDPB9FBXQAjw6EGzh3hAlJkIQV+/ukEvWLFBRuNCwbjVqkFGJShA8dXhQPu/nz8NwDbBKKCHQex+0kC7bOC+XsNQQX2GNMH8HhZCey/NQFuDJcE2aBrA0HkgQYo9l7601LrB2O/owXxtr0DblrLBB5v3wDUIfUDuJyE/mKTUQNgI1EH+doTBx067wV3D6MGGhQdBvSKNPhdmSsEG1MHBFZUYQXShpEGrv+m/qzFFwvTpu0FWb5rA8ojAQXysosBP34JAvl6KQcn7tsGjN9RATo0PwE3UHb8rqebBjTaFwYkF20ClJ8bB31YAQvDlgkEHfj1AkvupP2RaKMI80AhCFQXOwCiZnUHoDF9BXr3FwP2sG0CJ5P7AcRTJwPOcfsFTXSXBJivJQTMi5cFR78dBqKq5wWO2S0GgKgzBtVx9QemHYcBQURBCX8NQPdySOj4jTydBLvKpwK7nMkFgpaVBqzaDwSx2ykEnqqRBxDK1P1inkMESkvJA688CQkexKUCfGOq/W31WwRIRikGHlGlAwijQPucNu0Hvt1jBTXwFwXzMGULtN7TALkiMQYZ+zkAETsC/pbfTQFA9pkEoSNbBZFSrQCZAeT/kLZxBAq9DwQjmy8CnYsDALKmJwakfBELE9H0/kSZ2QMqD9EAmi/VBNoeDPjwTUMBC7hxBLEZMwd0dKMARXppBC69HwUkENMEsL8RBBC1CQXHF8MAJyojA7v3NQZ4L90FW1gvC39ogQOXPMcDaLRtAYMsYwsaHC0ACmMFBP9JoQZ68MkEZ4cFBCBABwhwVg0FF9UTAgnnbP3hJo0DY/JDATRcKwTKzSEE5zLlBrj+DQFvU6MFTitNAgQwVQvjjB8IU86RBoV95wU8MzUGfk+LBsUCVQRQyjkHFpRZBptP4wRezgEDMC6nATkpjP7HR9EGIaelBmDQBQvq9JcF9vU7Bl3BTv7xJib9dSwpC4hiHwYg78cAisNnAbgS5wU4B8UDioarB4ESXQSEKGcHfYci/x+rVwXcp0kAJDrHBJJWiQfmyLL4jcGrAEZOywU73YMFOg1VB1VHpQbM5q0BAiZhAXmOTQY03G8HWvbBBJ6NVv3U8qkFrKwzCIT3/QE7JEEKzGxbB/ROzQWycAsG6ir5AVJSPQYXXF0J5FOpA+AiAQVTVrcFWFtPAW+5uwIy+A0HrKYVB2jS1QZEXNb8GZW1BJ6Ugwe3epkHfpiFAlfmGwG1aqkEdQbfBbU5tv4s1pcEeO8TBwlw4QTwyQcHeDUDA0b4mwVwoUUF6pIXAIVVuwFUkecDh627BUPdrQfRjNcHAlTVBByHZwZKddkFtsF1BS+vkQEbE+MEIfeZAhTkbwbG1isCfr/FAy6idwQ==\",\"dtype\":\"float32\",\"shape\":[1000]}},\"selected\":{\"id\":\"1468\",\"type\":\"Selection\"},\"selection_policy\":{\"id\":\"1467\",\"type\":\"UnionRenderers\"}},\"id\":\"1426\",\"type\":\"ColumnDataSource\"},{\"attributes\":{},\"id\":\"1467\",\"type\":\"UnionRenderers\"},{\"attributes\":{\"fill_alpha\":{\"value\":0.1},\"fill_color\":{\"value\":\"#1f77b4\"},\"line_alpha\":{\"value\":0.1},\"line_color\":{\"value\":\"#1f77b4\"},\"size\":{\"units\":\"screen\",\"value\":8},\"x\":{\"field\":\"x1\"},\"y\":{\"field\":\"x2\"}},\"id\":\"1429\",\"type\":\"Scatter\"},{\"attributes\":{},\"id\":\"1466\",\"type\":\"BasicTickFormatter\"},{\"attributes\":{},\"id\":\"1468\",\"type\":\"Selection\"},{\"attributes\":{},\"id\":\"1419\",\"type\":\"ResetTool\"},{\"attributes\":{},\"id\":\"1403\",\"type\":\"LinearScale\"},{\"attributes\":{},\"id\":\"1417\",\"type\":\"PanTool\"},{\"attributes\":{},\"id\":\"1418\",\"type\":\"WheelZoomTool\"},{\"attributes\":{},\"id\":\"1405\",\"type\":\"LinearScale\"},{\"attributes\":{\"ticker\":{\"id\":\"1408\",\"type\":\"BasicTicker\"}},\"id\":\"1411\",\"type\":\"Grid\"},{\"attributes\":{\"fill_color\":{\"value\":\"#1f77b4\"},\"line_color\":{\"value\":\"#1f77b4\"},\"size\":{\"units\":\"screen\",\"value\":8},\"x\":{\"field\":\"x1\"},\"y\":{\"field\":\"x2\"}},\"id\":\"1428\",\"type\":\"Scatter\"},{\"attributes\":{\"formatter\":{\"id\":\"1464\",\"type\":\"BasicTickFormatter\"},\"ticker\":{\"id\":\"1408\",\"type\":\"BasicTicker\"}},\"id\":\"1407\",\"type\":\"LinearAxis\"},{\"attributes\":{},\"id\":\"1408\",\"type\":\"BasicTicker\"},{\"attributes\":{\"callback\":null},\"id\":\"1399\",\"type\":\"DataRange1d\"},{\"attributes\":{\"dimension\":1,\"ticker\":{\"id\":\"1413\",\"type\":\"BasicTicker\"}},\"id\":\"1416\",\"type\":\"Grid\"},{\"attributes\":{\"text\":\"word2vec T-SNE (HPAC model, top1000 words)\"},\"id\":\"1397\",\"type\":\"Title\"},{\"attributes\":{\"formatter\":{\"id\":\"1466\",\"type\":\"BasicTickFormatter\"},\"ticker\":{\"id\":\"1413\",\"type\":\"BasicTicker\"}},\"id\":\"1412\",\"type\":\"LinearAxis\"},{\"attributes\":{\"source\":{\"id\":\"1426\",\"type\":\"ColumnDataSource\"}},\"id\":\"1431\",\"type\":\"CDSView\"},{\"attributes\":{\"callback\":null},\"id\":\"1401\",\"type\":\"DataRange1d\"},{\"attributes\":{\"data_source\":{\"id\":\"1426\",\"type\":\"ColumnDataSource\"},\"glyph\":{\"id\":\"1428\",\"type\":\"Scatter\"},\"hover_glyph\":null,\"muted_glyph\":null,\"nonselection_glyph\":{\"id\":\"1429\",\"type\":\"Scatter\"},\"selection_glyph\":null,\"view\":{\"id\":\"1431\",\"type\":\"CDSView\"}},\"id\":\"1430\",\"type\":\"GlyphRenderer\"},{\"attributes\":{},\"id\":\"1420\",\"type\":\"SaveTool\"},{\"attributes\":{\"source\":{\"id\":\"1426\",\"type\":\"ColumnDataSource\"},\"text\":{\"field\":\"names\"},\"text_align\":\"center\",\"text_color\":{\"value\":\"#555555\"},\"text_font_size\":{\"value\":\"8pt\"},\"x\":{\"field\":\"x1\"},\"y\":{\"field\":\"x2\"},\"y_offset\":{\"value\":6}},\"id\":\"1432\",\"type\":\"LabelSet\"},{\"attributes\":{},\"id\":\"1464\",\"type\":\"BasicTickFormatter\"},{\"attributes\":{},\"id\":\"1413\",\"type\":\"BasicTicker\"},{\"attributes\":{\"active_drag\":\"auto\",\"active_inspect\":\"auto\",\"active_multi\":null,\"active_scroll\":\"auto\",\"active_tap\":\"auto\",\"tools\":[{\"id\":\"1417\",\"type\":\"PanTool\"},{\"id\":\"1418\",\"type\":\"WheelZoomTool\"},{\"id\":\"1419\",\"type\":\"ResetTool\"},{\"id\":\"1420\",\"type\":\"SaveTool\"}]},\"id\":\"1421\",\"type\":\"Toolbar\"}],\"root_ids\":[\"1396\"]},\"title\":\"Bokeh Application\",\"version\":\"1.3.4\"}};\n",
       "  var render_items = [{\"docid\":\"e6e3aea6-9874-45fc-b288-c5ac0881af19\",\"roots\":{\"1396\":\"d7590962-e309-4256-a5be-8c9d2dc9e102\"}}];\n",
       "  root.Bokeh.embed.embed_items_notebook(docs_json, render_items);\n",
       "\n",
       "  }\n",
       "  if (root.Bokeh !== undefined) {\n",
       "    embed_document(root);\n",
       "  } else {\n",
       "    var attempts = 0;\n",
       "    var timer = setInterval(function(root) {\n",
       "      if (root.Bokeh !== undefined) {\n",
       "        embed_document(root);\n",
       "        clearInterval(timer);\n",
       "      }\n",
       "      attempts++;\n",
       "      if (attempts > 100) {\n",
       "        console.log(\"Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing\");\n",
       "        clearInterval(timer);\n",
       "      }\n",
       "    }, 10, root)\n",
       "  }\n",
       "})(window);"
      ],
      "application/vnd.bokehjs_exec.v0+json": ""
     },
     "metadata": {
      "application/vnd.bokehjs_exec.v0+json": {
       "id": "1396"
      }
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from bokeh.models import ColumnDataSource, LabelSet\n",
    "from bokeh.plotting import figure, show, output_file\n",
    "from bokeh.io import output_notebook\n",
    "\n",
    "output_notebook()\n",
    "\n",
    "p = figure(tools=\"pan,wheel_zoom,reset,save\",\n",
    "           toolbar_location=\"above\",\n",
    "           title=\"word2vec T-SNE (HPAC model, top1000 words)\")\n",
    "\n",
    "source = ColumnDataSource(data=dict(x1=top_words_tsne[:,0],\n",
    "                                    x2=top_words_tsne[:,1],\n",
    "                                    names=top_words))\n",
    "\n",
    "p.scatter(x=\"x1\", y=\"x2\", size=8, source=source)\n",
    "\n",
    "labels = LabelSet(x=\"x1\", y=\"x2\", text=\"names\", y_offset=6,\n",
    "                  text_font_size=\"8pt\", text_color=\"#555555\",\n",
    "                  source=source, text_align='center')\n",
    "p.add_layout(labels)\n",
    "\n",
    "show(p)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "# Часть 3. Классификация текстов\n",
    "---\n",
    "Предобработаем данные: токенизируем и заменим метки классов на индексы, каждому токену сопоставим свой индекс.\n",
    "Также добавим два токена для паддинга и неизвестных слов."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data = pd.read_csv('hpac_splits/hpac_corpus/hpac_training_128.tsv', sep = '\\t', header = None)\n",
    "valid_data = pd.read_csv('hpac_splits/hpac_corpus/hpac_dev_128.tsv', sep = '\\t', header = None)\n",
    "test_data = pd.read_csv('hpac_splits/hpac_corpus/hpac_test_128.tsv', sep = '\\t', header = None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "tokenizer = nltk.tokenize.WordPunctTokenizer()\n",
    "\n",
    "for i in range(1,3):\n",
    "    train_data[i] = train_data[i].apply(lambda x: \" \".join(tokenizer.tokenize(str(x).lower())))\n",
    "    valid_data[i] = valid_data[i].apply(lambda x: \" \".join(tokenizer.tokenize(str(x).lower())))\n",
    "    test_data[i] = test_data[i].apply(lambda x: \" \".join(tokenizer.tokenize(str(x).lower())))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "classes = train_data[1].unique()\n",
    "\n",
    "classes_to_idx = {word: i for i, word in enumerate(classes)}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = np.array(train_data[2])\n",
    "X_valid = np.array(valid_data[2])\n",
    "X_test = np.array(test_data[2])\n",
    "\n",
    "y_train = np.array([classes_to_idx[word] for word in train_data[1]])\n",
    "y_valid = np.array([classes_to_idx[word] for word in valid_data[1]])\n",
    "y_test = np.array([classes_to_idx[word] for word in test_data[1]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "from collections import Counter\n",
    "\n",
    "counter = Counter()\n",
    "\n",
    "for line in X_train:\n",
    "    counter.update(line.split(\" \"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "vocab = [key for key, item in counter.items()]\n",
    "\n",
    "UNK, PAD = \"UNK\", \"PAD\"\n",
    "vocab = [UNK, PAD] + vocab\n",
    "\n",
    "token_to_id = {t: i for i, t in enumerate(vocab)}\n",
    "\n",
    "UNK_IX, PAD_IX = map(token_to_id.get, [UNK, PAD])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "###### 1. Используйте fastText в качестве baseline-классификатора."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c697a85601f945faba95108bf3cd90d6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=0, max=36225), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "dirname = 'hpac_lower_tokenized/hpac_source/'\n",
    "\n",
    "for filename in tqdm_notebook(os.listdir(dirname)):\n",
    "    with open(dirname + filename, 'r') as text:\n",
    "        with open('hpac_lower_tokenized.txt', 'a') as f:\n",
    "            f.write(text.readline())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "ft_model = fasttext.train_supervised('hpac_lower_tokenized.txt', minn=3, maxn=4, dim=300, thread=2, verbose=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "###### 2. Используйте сверточные сети в качестве более продвинутого классификатора. Поэкспериментируйте с количеством и размерностью фильтров, используйте разные размеры окон, попробуйте использовать $k$-max pooling. \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "maxL_train = max([len(text.split(' ')) for text in X_train])\n",
    "maxL_valid = max([len(text.split(' ')) for text in X_valid])\n",
    "maxL_test = max([len(text.split(' ')) for text in X_test])\n",
    "\n",
    "maxL = max([maxL_train, maxL_valid, maxL_test])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "def word2vec_embedding(text, embeddings, length):\n",
    "    \"\"\"\n",
    "    implement a function that converts preprocessed comment to a sum of token vectors\n",
    "    \"\"\"\n",
    "    embedding_dim = embeddings.vectors.shape[1]\n",
    "    features = []\n",
    "    \n",
    "    for word in text.split():\n",
    "        if word in embeddings:\n",
    "            features.append(embeddings[word])\n",
    "    \n",
    "    while len(features) < length:\n",
    "        features.append(np.zeros(features[0].shape[0]))\n",
    "    \n",
    "    return np.asarray(features).T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "wv = KeyedVectors.load(\"word2vec.wordvectors\", mmap='r')\n",
    "\n",
    "wv_matrix = np.array([wv[w] if w in wv else np.zeros(300) for w in vocab])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(49837, 300)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wv_matrix.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "class ConvNet(nn.Module):\n",
    "    def __init__(self, n_tokens, num_classes, wv, device, emb_dim=300):\n",
    "        super().__init__()\n",
    "        self.device = device\n",
    "        self.n_tokens = n_tokens\n",
    "        self.wv = wv\n",
    "        \n",
    "        self.feature_extractor = nn.Sequential(\n",
    "            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(emb_dim, 3), padding=(0, 1)),\n",
    "            nn.ReLU(),\n",
    "            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),\n",
    "            nn.Flatten()\n",
    "        )\n",
    "        self.final_predictor = nn.Sequential(\n",
    "            nn.Linear(n_tokens // 2, num_classes),\n",
    "            nn.Softmax()\n",
    "        )\n",
    "        \n",
    "    def forward(self, batch):\n",
    "        batch = np.stack([word2vec_embedding(text, self.wv, self.n_tokens) for text in batch])\n",
    "        batch = torch.tensor(batch, device=self.device, dtype=torch.float32).unsqueeze(1)\n",
    "        \n",
    "        features = self.feature_extractor(batch).squeeze()\n",
    "        return self.final_predictor(features)\n",
    "    \n",
    "    def predict(self, data):\n",
    "        answer = []\n",
    "        for item in data:\n",
    "            pred = torch.argmax(self.forward([item]))\n",
    "            answer.append(pred.detach().cpu().numpy())\n",
    "        return answer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "BATCH_SIZE = 512\n",
    "EPOCHS = 150\n",
    "DEVICE = torch.device('cuda')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "def iterate_minibatches(data, batch_size=256, shuffle=True):\n",
    "    \"\"\" iterates minibatches of data in random order \"\"\"\n",
    "    indices = np.arange(len(data[0]))\n",
    "    if shuffle:\n",
    "        indices = np.random.permutation(indices)\n",
    "\n",
    "    for start in range(0, len(indices), batch_size):\n",
    "        batch = [data[0][indices[start : start + batch_size]], data[1][indices[start : start + batch_size]]]\n",
    "        yield batch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "def print_metrics(model, data, batch_size=BATCH_SIZE, name=\"\", device=torch.device('cuda')):\n",
    "    loss = accuracy = num_samples = 0.0\n",
    "    model.eval()\n",
    "    with torch.no_grad():\n",
    "        for batch in iterate_minibatches(data, batch_size=batch_size, shuffle=False):\n",
    "            batch_pred = model(batch[0])\n",
    "            y = torch.tensor(batch[1], dtype=torch.int64, device=DEVICE)\n",
    "            loss += criterion(batch_pred.float(), y)\n",
    "            accuracy += torch.mean((torch.argmax(batch_pred, axis=-1).float() == y.float()).float())\n",
    "            num_samples += 1\n",
    "            \n",
    "    loss = loss.detach().cpu().numpy() / num_samples\n",
    "    accuracy = accuracy / num_samples\n",
    "    print(\"%s val results:\" % (name or \"\"))\n",
    "    print(\"loss: %.5f\" % loss)\n",
    "    print(\"accuracy: %.5f\" % accuracy.detach().cpu().numpy())\n",
    "    return loss, accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0ac44720f5f340ce8ff4af0713800c56",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=0, max=200), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f1f46c7617e345e7ab7b93098418854c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 0\n",
      "train results:\n",
      "loss:  4.360982767740885\n",
      "accuracy: 0.1020432710647583\n",
      " val results:\n",
      "loss: 4.34126\n",
      "accuracy: 0.11928\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9caea18cecb547de82674af377e9b9a5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 1\n",
      "train results:\n",
      "loss:  4.335551961263021\n",
      "accuracy: 0.12528045177459718\n",
      " val results:\n",
      "loss: 4.34340\n",
      "accuracy: 0.11357\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fdf88654cd7545d2937d4258925712a7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 2\n",
      "train results:\n",
      "loss:  4.332478841145833\n",
      "accuracy: 0.1289212703704834\n",
      " val results:\n",
      "loss: 4.33915\n",
      "accuracy: 0.12490\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cadac1a9d5604747851c095579338f7d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 3\n",
      "train results:\n",
      "loss:  4.329773457845052\n",
      "accuracy: 0.1341922124226888\n",
      " val results:\n",
      "loss: 4.33985\n",
      "accuracy: 0.12028\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f8abb0d2943442d9b658ce5a87ac0667",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 4\n",
      "train results:\n",
      "loss:  4.327152506510417\n",
      "accuracy: 0.13613406817118326\n",
      " val results:\n",
      "loss: 4.33919\n",
      "accuracy: 0.12308\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e70beb5cb2ac4a64ae3059301e45019a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 5\n",
      "train results:\n",
      "loss:  4.324077860514323\n",
      "accuracy: 0.14133238792419434\n",
      " val results:\n",
      "loss: 4.33522\n",
      "accuracy: 0.12672\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d9ba6e4bb5b84387b650dd215c76fd35",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 6\n",
      "train results:\n",
      "loss:  4.321332295735677\n",
      "accuracy: 0.14387645721435546\n",
      " val results:\n",
      "loss: 4.33388\n",
      "accuracy: 0.12889\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "004e82be3cfc4d86b9d24410ffdc997f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 7\n",
      "train results:\n",
      "loss:  4.317196146647135\n",
      "accuracy: 0.14855895042419434\n"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-133-cfdda6158b57>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     25\u001b[0m     \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"loss: \"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mepoch_loss\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdetach\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcpu\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnumpy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m/\u001b[0m \u001b[0miterations\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     26\u001b[0m     \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"accuracy:\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mepoch_accuracy\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdetach\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcpu\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnumpy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m/\u001b[0m \u001b[0miterations\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 27\u001b[0;31m     \u001b[0mprint_metrics\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mX_valid\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_valid\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m<ipython-input-99-b52a3a7855c4>\u001b[0m in \u001b[0;36mprint_metrics\u001b[0;34m(model, data, batch_size, name, device)\u001b[0m\n\u001b[1;32m      4\u001b[0m     \u001b[0;32mwith\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mno_grad\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0mbatch\u001b[0m \u001b[0;32min\u001b[0m \u001b[0miterate_minibatches\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch_size\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mbatch_size\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mshuffle\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mFalse\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 6\u001b[0;31m             \u001b[0mbatch_pred\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbatch\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      7\u001b[0m             \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtensor\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbatch\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mint64\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdevice\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mDEVICE\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      8\u001b[0m             \u001b[0mloss\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0mcriterion\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbatch_pred\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/lib/python3.7/site-packages/torch/nn/modules/module.py\u001b[0m in \u001b[0;36m_call_impl\u001b[0;34m(self, *input, **kwargs)\u001b[0m\n\u001b[1;32m    725\u001b[0m             \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_slow_forward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    726\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 727\u001b[0;31m             \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mforward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    728\u001b[0m         for hook in itertools.chain(\n\u001b[1;32m    729\u001b[0m                 \u001b[0m_global_forward_hooks\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m<ipython-input-131-3540f513a853>\u001b[0m in \u001b[0;36mforward\u001b[0;34m(self, batch)\u001b[0m\n\u001b[1;32m     19\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mforward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     20\u001b[0m         \u001b[0mbatch\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstack\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mword2vec_embedding\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtext\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwv\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mn_tokens\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mtext\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mbatch\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 21\u001b[0;31m         \u001b[0mbatch\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtensor\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbatch\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdevice\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdevice\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat32\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0munsqueeze\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     22\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     23\u001b[0m         \u001b[0mfeatures\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfeature_extractor\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbatch\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msqueeze\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "model = ConvNet(n_tokens=maxL, num_classes=len(classes), wv=wv, device=DEVICE).to(DEVICE)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n",
    "\n",
    "for epoch in tqdm_notebook(range(EPOCHS)):\n",
    "    \n",
    "    model.train()\n",
    "    epoch_loss = 0\n",
    "    epoch_accuracy = 0\n",
    "    iterations = 0\n",
    "    for i, batch in tqdm_notebook(enumerate(iterate_minibatches([X_train, y_train], batch_size=BATCH_SIZE))):\n",
    "        pred = model(batch[0])\n",
    "        y = torch.tensor(batch[1], dtype=torch.int64, device=DEVICE)\n",
    "        loss = criterion(pred, y)\n",
    "        accuracy = torch.mean((torch.argmax(pred, axis=-1).float() == y.float()).float())\n",
    "        optimizer.zero_grad()\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        epoch_loss += loss\n",
    "        epoch_accuracy += accuracy\n",
    "        iterations += 1\n",
    "    \n",
    "    print(f\"epoch: {epoch}\")\n",
    "    print(\"train results:\")\n",
    "    print(\"loss: \", epoch_loss.detach().cpu().numpy() / iterations)\n",
    "    print(\"accuracy:\", epoch_accuracy.detach().cpu().numpy() / iterations)    \n",
    "    print_metrics(model, [X_valid, y_valid])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "metadata": {},
   "outputs": [],
   "source": [
    "predicted_classes = model.predict(X_train)\n",
    "predicted_classes_valid = model.predict(X_valid)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.005714932378504552"
      ]
     },
     "execution_count": 135,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f1_score(y_train, predicted_classes, average=\"macro\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "metadata": {},
   "outputs": [],
   "source": [
    "class ConvNetv2(nn.Module):\n",
    "    def __init__(self, n_tokens, num_classes, n_filters, wv, device, emb_dim=300):\n",
    "        super().__init__()\n",
    "        self.device = device\n",
    "        self.n_tokens = n_tokens\n",
    "        self.wv = wv\n",
    "        \n",
    "        self.conv0 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(emb_dim, 3), padding=(0, 1))\n",
    "        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(emb_dim, 5), padding=(0, 2))\n",
    "        self.conv2 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(emb_dim, 7), padding=(0, 3))\n",
    "        \n",
    "        self.dropout = torch.nn.Dropout(0.2)\n",
    "        \n",
    "        self.final_predictor = nn.Sequential(\n",
    "            nn.Linear(3 * n_filters * n_tokens // 2, num_classes),\n",
    "            nn.Softmax()\n",
    "        )\n",
    "        \n",
    "    def forward(self, batch):\n",
    "        batch = np.stack([word2vec_embedding(text, self.wv, self.n_tokens) for text in batch])\n",
    "        batch = torch.tensor(batch, device=self.device, dtype=torch.float32).unsqueeze(1)\n",
    "        \n",
    "        conved0 = F.relu(self.conv0(batch))\n",
    "        conved1 = F.relu(self.conv1(batch))\n",
    "        conved2 = F.relu(self.conv2(batch))\n",
    "        \n",
    "        pooled0 = F.max_pool2d(conved0, kernel_size=(1, 2), stride=(1, 2)).squeeze(1)\n",
    "        pooled1 = F.max_pool2d(conved0, kernel_size=(1, 2), stride=(1, 2)).squeeze(1)\n",
    "        pooled2 = F.max_pool2d(conved0, kernel_size=(1, 2), stride=(1, 2)).squeeze(1)\n",
    "        \n",
    "        cat = self.dropout(torch.cat((pooled0, pooled1, pooled2), dim=1)).flatten(1)\n",
    "        \n",
    "        return self.final_predictor(cat)\n",
    "    \n",
    "    def predict(self, data):\n",
    "        answer = []\n",
    "        for item in data:\n",
    "            pred = torch.argmax(self.forward([item]))\n",
    "            answer.append(pred.detach().cpu().numpy())\n",
    "        return answer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f26e55e395094e0287ce9176932649f0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=0, max=50), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d8703538012d471f8bf7a7d7ff0ca1a7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "\n"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-156-48d614128289>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     10\u001b[0m     \u001b[0miterations\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     11\u001b[0m     \u001b[0;32mfor\u001b[0m \u001b[0mi\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mtqdm_notebook\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0menumerate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0miterate_minibatches\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mX_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch_size\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mBATCH_SIZE\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 12\u001b[0;31m         \u001b[0mpred\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbatch\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     13\u001b[0m         \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtensor\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbatch\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mint64\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdevice\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mDEVICE\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     14\u001b[0m         \u001b[0mloss\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcriterion\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpred\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/lib/python3.7/site-packages/torch/nn/modules/module.py\u001b[0m in \u001b[0;36m_call_impl\u001b[0;34m(self, *input, **kwargs)\u001b[0m\n\u001b[1;32m    725\u001b[0m             \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_slow_forward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    726\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 727\u001b[0;31m             \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mforward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    728\u001b[0m         for hook in itertools.chain(\n\u001b[1;32m    729\u001b[0m                 \u001b[0m_global_forward_hooks\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m<ipython-input-155-fcb5b3de1210>\u001b[0m in \u001b[0;36mforward\u001b[0;34m(self, batch)\u001b[0m\n\u001b[1;32m     18\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     19\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mforward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 20\u001b[0;31m         \u001b[0mbatch\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstack\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mword2vec_embedding\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtext\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwv\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mn_tokens\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mtext\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mbatch\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     21\u001b[0m         \u001b[0mbatch\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtensor\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbatch\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdevice\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdevice\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat32\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0munsqueeze\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     22\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m<ipython-input-155-fcb5b3de1210>\u001b[0m in \u001b[0;36m<listcomp>\u001b[0;34m(.0)\u001b[0m\n\u001b[1;32m     18\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     19\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mforward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 20\u001b[0;31m         \u001b[0mbatch\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstack\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mword2vec_embedding\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtext\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwv\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mn_tokens\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mtext\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mbatch\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     21\u001b[0m         \u001b[0mbatch\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtensor\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbatch\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdevice\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdevice\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat32\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0munsqueeze\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     22\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m<ipython-input-32-0d6eda3fa12c>\u001b[0m in \u001b[0;36mword2vec_embedding\u001b[0;34m(text, embeddings, length)\u001b[0m\n\u001b[1;32m     13\u001b[0m         \u001b[0mfeatures\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mzeros\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfeatures\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     14\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 15\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masarray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfeatures\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mT\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m~/anaconda3/lib/python3.7/site-packages/numpy/core/_asarray.py\u001b[0m in \u001b[0;36masarray\u001b[0;34m(a, dtype, order)\u001b[0m\n\u001b[1;32m     83\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     84\u001b[0m     \"\"\"\n\u001b[0;32m---> 85\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0marray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0ma\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcopy\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mFalse\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0morder\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0morder\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     86\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     87\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "model = ConvNetv2(n_tokens=maxL, num_classes=len(classes), n_filters=100, wv=wv, device=DEVICE).to(DEVICE)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n",
    "\n",
    "for epoch in tqdm_notebook(range(EPOCHS)):\n",
    "    \n",
    "    model.train()\n",
    "    epoch_loss = 0\n",
    "    epoch_accuracy = 0\n",
    "    iterations = 0\n",
    "    for i, batch in tqdm_notebook(enumerate(iterate_minibatches([X_train, y_train], batch_size=BATCH_SIZE))):\n",
    "        pred = model(batch[0])\n",
    "        y = torch.tensor(batch[1], dtype=torch.int64, device=DEVICE)\n",
    "        loss = criterion(pred, y)\n",
    "        accuracy = torch.mean((torch.argmax(pred, axis=-1).float() == y.float()).float())\n",
    "        optimizer.zero_grad()\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        epoch_loss += loss\n",
    "        epoch_accuracy += accuracy\n",
    "        iterations += 1\n",
    "    \n",
    "    print(f\"epoch: {epoch}\")\n",
    "    print(\"train results:\")\n",
    "    print(\"loss: \", epoch_loss.detach().cpu().numpy() / iterations)\n",
    "    print(\"accuracy:\", epoch_accuracy.detach().cpu().numpy() / iterations)    \n",
    "    print_metrics(model, [X_valid, y_valid])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'model' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-14-88eddbcb209c>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mpredicted_classes\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpredict\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX_train\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0mpredicted_classes_valid\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpredict\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX_valid\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'model' is not defined"
     ]
    }
   ],
   "source": [
    "predicted_classes = model.predict(X_train)\n",
    "predicted_classes_valid = model.predict(X_valid)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "def as_matrix(data, max_len):\n",
    "    \"\"\" Convert a list of tokens into a matrix with padding \"\"\"\n",
    "    data = list(map(str.split, data))\n",
    "    \n",
    "    matrix = np.full((len(data), max_len), np.int32(PAD_IX))\n",
    "    for i, seq in enumerate(data):\n",
    "        row_ix = [token_to_id.get(word, UNK_IX) for word in seq[:max_len]]\n",
    "        matrix[i, :len(row_ix)] = row_ix\n",
    "    \n",
    "    return matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [],
   "source": [
    "class ConvNetv3(nn.Module):\n",
    "    def __init__(self, n_tokens, num_classes, n_filters, wv, device, emb_dim=300):\n",
    "        super().__init__()\n",
    "        self.device = device\n",
    "        self.n_tokens = n_tokens\n",
    "        self.wv = torch.tensor(wv, dtype=torch.float32, device=device)\n",
    "        \n",
    "        self.embedder = nn.Embedding.from_pretrained(self.wv, freeze=False)\n",
    "#         self.embedder = nn.Embedding(len(vocab), embedding_dim=emb_dim)\n",
    "        \n",
    "        self.conv0 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(3, emb_dim), padding=(1, 0))\n",
    "        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(5, emb_dim), padding=(2, 0))\n",
    "        self.conv2 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(7, emb_dim), padding=(3, 0))\n",
    "        \n",
    "        self.dropout = torch.nn.Dropout(0.3)\n",
    "        \n",
    "        self.final_predictor = nn.Sequential(\n",
    "            nn.Linear(3 * n_filters * n_tokens // 2, 3 * n_tokens // 2),\n",
    "            nn.Tanh(),\n",
    "            nn.Linear(3 * n_tokens // 2, num_classes),\n",
    "        )\n",
    "        \n",
    "    def forward(self, batch):\n",
    "        \n",
    "        batch = as_matrix(batch, self.n_tokens)\n",
    "        batch = torch.tensor(batch, device=self.device, dtype=torch.int64).unsqueeze(1)\n",
    "        \n",
    "        embedded = self.embedder(batch)\n",
    "        \n",
    "        conved0 = F.relu(self.conv0(embedded))\n",
    "        conved1 = F.relu(self.conv1(embedded))\n",
    "        conved2 = F.relu(self.conv2(embedded))\n",
    "        \n",
    "        pooled0 = F.max_pool2d(conved0, kernel_size=(2, 1), stride=(2, 1)).squeeze(-1)\n",
    "        pooled1 = F.max_pool2d(conved1, kernel_size=(2, 1), stride=(2, 1)).squeeze(-1)\n",
    "        pooled2 = F.max_pool2d(conved2, kernel_size=(2, 1), stride=(2, 1)).squeeze(-1)\n",
    "        \n",
    "        cat = self.dropout(torch.cat((pooled0, pooled1, pooled2), dim=1)).flatten(1)\n",
    "        \n",
    "        return self.final_predictor(cat)\n",
    "    \n",
    "    def predict(self, data):\n",
    "        answer = []\n",
    "        for item in data:\n",
    "            pred = torch.argmax(self.forward([item]))\n",
    "            answer.append(pred.detach().cpu().numpy())\n",
    "        return answer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cff7ecedd10140dc93d4c1a36c2b6102",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=0), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f8370f21291c487299f2df4427f5a208",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 0\n",
      "train results:\n",
      "loss:  3.493702443440755\n",
      "accuracy: 0.11831180254618327\n",
      " val results:\n",
      "loss: 3.49027\n",
      "accuracy: 0.12177\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f7feba1bdaf34625b43a7cfb3dc5e7c3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 1\n",
      "train results:\n",
      "loss:  3.4641273498535154\n",
      "accuracy: 0.12269506454467774\n",
      " val results:\n",
      "loss: 3.48311\n",
      "accuracy: 0.11386\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d6ae8bc0f79649009509449e6d5e0f21",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 2\n",
      "train results:\n",
      "loss:  3.4669583638509116\n",
      "accuracy: 0.12212289969126383\n",
      " val results:\n",
      "loss: 3.49556\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d7205cad62274374b0b07a49097e11d1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 3\n",
      "train results:\n",
      "loss:  3.465723673502604\n",
      "accuracy: 0.12238832314809163\n",
      " val results:\n",
      "loss: 3.49184\n",
      "accuracy: 0.11422\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fa6ce2eac7d4441db161ae128ac28ac1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 4\n",
      "train results:\n",
      "loss:  3.4681734720865887\n",
      "accuracy: 0.1213090976079305\n",
      " val results:\n",
      "loss: 3.48413\n",
      "accuracy: 0.11342\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3dca75301af04b949bd13054e1d7656d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 5\n",
      "train results:\n",
      "loss:  3.462624104817708\n",
      "accuracy: 0.1239896297454834\n",
      " val results:\n",
      "loss: 3.51266\n",
      "accuracy: 0.12179\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "51ad51b2f8f84eb092ae879d9a0083ec",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 6\n",
      "train results:\n",
      "loss:  3.462508138020833\n",
      "accuracy: 0.12443159421284994\n",
      " val results:\n",
      "loss: 3.48185\n",
      "accuracy: 0.11414\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6cb95c5ba9db44b48d07b21035392bf5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 7\n",
      "train results:\n",
      "loss:  3.362731679280599\n",
      "accuracy: 0.14597355524698893\n",
      " val results:\n",
      "loss: 3.30554\n",
      "accuracy: 0.16211\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "be6d3cf8cef94309805ce9dcfcb8afb2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 8\n",
      "train results:\n",
      "loss:  3.249993387858073\n",
      "accuracy: 0.16698843638102215\n",
      " val results:\n",
      "loss: 3.27749\n",
      "accuracy: 0.15922\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4251e171c385473d95bc8a75b713b70c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 9\n",
      "train results:\n",
      "loss:  3.213726043701172\n",
      "accuracy: 0.16951371828715006\n",
      " val results:\n",
      "loss: 3.26078\n",
      "accuracy: 0.16706\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "306c925a0c94484fa5b4d7fe5b10c64e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 10\n",
      "train results:\n",
      "loss:  3.183105723063151\n",
      "accuracy: 0.17731119791666666\n",
      " val results:\n",
      "loss: 3.22997\n",
      "accuracy: 0.16172\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5c398e60803a4416ae1ae3f11b680866",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 11\n",
      "train results:\n",
      "loss:  3.1540738423665364\n",
      "accuracy: 0.18106220563252767\n",
      " val results:\n",
      "loss: 3.20264\n",
      "accuracy: 0.17075\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "872f095efc12481c8cf1a26d93b707a5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 12\n",
      "train results:\n",
      "loss:  3.1305582682291666\n",
      "accuracy: 0.18302408854166666\n",
      " val results:\n",
      "loss: 3.19534\n",
      "accuracy: 0.16749\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "80a8e7ea1f0245ebbf841da8a59cd820",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 13\n",
      "train results:\n",
      "loss:  3.114148712158203\n",
      "accuracy: 0.188290007909139\n",
      " val results:\n",
      "loss: 3.17237\n",
      "accuracy: 0.17954\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d4390cdaf62e4a0ea407c6f385209ec5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 14\n",
      "train results:\n",
      "loss:  3.0942120869954426\n",
      "accuracy: 0.19020307858784993\n",
      " val results:\n",
      "loss: 3.17275\n",
      "accuracy: 0.18334\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "21d7895a727e4a6dbbf09f39bce1f238",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 15\n",
      "train results:\n",
      "loss:  3.084136962890625\n",
      "accuracy: 0.19360225995381672\n",
      " val results:\n",
      "loss: 3.17184\n",
      "accuracy: 0.17910\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "088b1674511645079d6c26c5ca8a71c7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 16\n",
      "train results:\n",
      "loss:  3.072154998779297\n",
      "accuracy: 0.1973294734954834\n",
      " val results:\n",
      "loss: 3.16003\n",
      "accuracy: 0.18434\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "12be1d8d7e504e9ab485bbb4014906e7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 17\n",
      "train results:\n",
      "loss:  3.0694620768229166\n",
      "accuracy: 0.1961788813273112\n",
      " val results:\n",
      "loss: 3.14617\n",
      "accuracy: 0.19091\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e02f47cabb474bbdb15f681bf0eccfee",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 18\n",
      "train results:\n",
      "loss:  3.059326934814453\n",
      "accuracy: 0.19344326655069988\n",
      " val results:\n",
      "loss: 3.16512\n",
      "accuracy: 0.17909\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "571178e1dad04da58e55df61f00c1e21",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 19\n",
      "train results:\n",
      "loss:  3.0590794881184897\n",
      "accuracy: 0.19844001134236652\n",
      " val results:\n",
      "loss: 3.13762\n",
      "accuracy: 0.18887\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "81c70f7a760c4c55b24b985d0a5006b7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 20\n",
      "train results:\n",
      "loss:  3.030565134684245\n",
      "accuracy: 0.20482522646586102\n",
      " val results:\n",
      "loss: 3.14215\n",
      "accuracy: 0.19014\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "961e35d398384b7dbf9e5a836da3a801",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 21\n",
      "train results:\n",
      "loss:  3.0230138142903646\n",
      "accuracy: 0.20716396967569986\n",
      " val results:\n",
      "loss: 3.13957\n",
      "accuracy: 0.18251\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9272aeba659a4af793c54ac2baa03409",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 22\n",
      "train results:\n",
      "loss:  3.0105328877766926\n",
      "accuracy: 0.2090670108795166\n",
      " val results:\n",
      "loss: 3.13187\n",
      "accuracy: 0.19228\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "af01e3f2cadc4ad3b5d50fe485786b08",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 23\n",
      "train results:\n",
      "loss:  3.000629425048828\n",
      "accuracy: 0.20850610733032227\n",
      " val results:\n",
      "loss: 3.13754\n",
      "accuracy: 0.18844\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "771e6e0fc1f743b492b24dc3d9baa3e4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 24\n",
      "train results:\n",
      "loss:  2.9966649373372394\n",
      "accuracy: 0.21056941350301106\n",
      " val results:\n",
      "loss: 3.13787\n",
      "accuracy: 0.19129\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "432bb649148640119479f04c658b7263",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 25\n",
      "train results:\n",
      "loss:  2.987615203857422\n",
      "accuracy: 0.21238231658935547\n",
      " val results:\n",
      "loss: 3.11141\n",
      "accuracy: 0.19614\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "496e76c5450e470db97b160e6804c9d0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 26\n",
      "train results:\n",
      "loss:  2.9666709899902344\n",
      "accuracy: 0.2188176155090332\n",
      " val results:\n",
      "loss: 3.11237\n",
      "accuracy: 0.19367\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3e377957f9754731b921caaaeb4b6142",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 27\n",
      "train results:\n",
      "loss:  2.9640037536621096\n",
      "accuracy: 0.21957756678263346\n",
      " val results:\n",
      "loss: 3.09993\n",
      "accuracy: 0.19990\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "643155cf81324f878a89481f27a026f2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 28\n",
      "train results:\n",
      "loss:  2.9517585754394533\n",
      "accuracy: 0.22017102241516112\n",
      " val results:\n",
      "loss: 3.11321\n",
      "accuracy: 0.20229\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c2e36d27ce96416ca8e21e914619885a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 29\n",
      "train results:\n",
      "loss:  2.9498026529947916\n",
      "accuracy: 0.22240584691365559\n",
      " val results:\n",
      "loss: 3.11032\n",
      "accuracy: 0.20161\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "447e7499e0c142859be65f918c34e0eb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 30\n",
      "train results:\n",
      "loss:  2.933446248372396\n",
      "accuracy: 0.2276980717976888\n",
      " val results:\n",
      "loss: 3.09562\n",
      "accuracy: 0.21007\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ff9610c0800d4e27b5862815ea844855",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 31\n",
      "train results:\n",
      "loss:  2.9201090494791666\n",
      "accuracy: 0.2285369078318278\n",
      " val results:\n",
      "loss: 3.08360\n",
      "accuracy: 0.21043\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "334da6a9730b4c1d90d019c73b04c1f3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 32\n",
      "train results:\n",
      "loss:  2.9103609720865884\n",
      "accuracy: 0.2319974422454834\n",
      " val results:\n",
      "loss: 3.07368\n",
      "accuracy: 0.20908\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b6df796cf3264afd85c34c5d7412a14b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 33\n",
      "train results:\n",
      "loss:  2.8895975748697915\n",
      "accuracy: 0.23531025250752766\n",
      " val results:\n",
      "loss: 3.05195\n",
      "accuracy: 0.21223\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d1c6e053a5dd454d8552a63b326cd9ea",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 34\n",
      "train results:\n",
      "loss:  2.8677518208821615\n",
      "accuracy: 0.2400666077931722\n",
      " val results:\n",
      "loss: 3.06219\n",
      "accuracy: 0.21064\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0bbafb017817462a97885a94c5c98033",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 35\n",
      "train results:\n",
      "loss:  2.868766276041667\n",
      "accuracy: 0.24135241508483887\n",
      " val results:\n",
      "loss: 3.05359\n",
      "accuracy: 0.22074\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6c599a62c5c44d38a24a773057ca0a7d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 36\n",
      "train results:\n",
      "loss:  2.845923105875651\n",
      "accuracy: 0.24585086504618328\n",
      " val results:\n",
      "loss: 3.04111\n",
      "accuracy: 0.21429\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cb6dc61ab03545afb14e3a93121e4847",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 37\n",
      "train results:\n",
      "loss:  2.8344128926595054\n",
      "accuracy: 0.2524201234181722\n",
      " val results:\n",
      "loss: 3.03909\n",
      "accuracy: 0.20956\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f0aae4ea0695465c8f00342225bfd8f7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 38\n",
      "train results:\n",
      "loss:  2.824876658121745\n",
      "accuracy: 0.2518254280090332\n",
      " val results:\n",
      "loss: 3.04609\n",
      "accuracy: 0.21787\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c24e11dd9cda4d47bb482abe8af55af1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 39\n",
      "train results:\n",
      "loss:  2.8212259928385417\n",
      "accuracy: 0.2501590092976888\n",
      " val results:\n",
      "loss: 3.04709\n",
      "accuracy: 0.22682\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "34e824a43afb49b185aa778aa3503733",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 40\n",
      "train results:\n",
      "loss:  2.799211120605469\n",
      "accuracy: 0.2577761967976888\n",
      " val results:\n",
      "loss: 3.03126\n",
      "accuracy: 0.22386\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "94d249aa34fe4ee3943609fc4d591109",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 41\n",
      "train results:\n",
      "loss:  2.788800048828125\n",
      "accuracy: 0.2653733412424723\n",
      " val results:\n",
      "loss: 3.03487\n",
      "accuracy: 0.22972\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d87779b5d72446708cb1787b5c534c06",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 42\n",
      "train results:\n",
      "loss:  2.765203603108724\n",
      "accuracy: 0.2695675532023112\n",
      " val results:\n",
      "loss: 3.03603\n",
      "accuracy: 0.21772\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "35eb184642e84f4981093c5c3d0a82c3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 43\n",
      "train results:\n",
      "loss:  2.7425188700358074\n",
      "accuracy: 0.27424128850301105\n",
      " val results:\n",
      "loss: 3.02527\n",
      "accuracy: 0.24168\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "71b8755337d8473a94d2e64f1ecdd6d5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 44\n",
      "train results:\n",
      "loss:  2.7388931274414063\n",
      "accuracy: 0.2774739583333333\n",
      " val results:\n",
      "loss: 3.01543\n",
      "accuracy: 0.23678\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6c0d789109f944ef93a15e05d01f24fe",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 45\n",
      "train results:\n",
      "loss:  2.7142766316731772\n",
      "accuracy: 0.28386042912801107\n",
      " val results:\n",
      "loss: 3.03357\n",
      "accuracy: 0.23838\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e5dca50655ed4ada85d036df4b064c98",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 46\n",
      "train results:\n",
      "loss:  2.707768758138021\n",
      "accuracy: 0.2859963417053223\n",
      " val results:\n",
      "loss: 3.00783\n",
      "accuracy: 0.23938\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "abdc756a74de4ceab1446bbec612eb78",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 47\n",
      "train results:\n",
      "loss:  2.6930501302083334\n",
      "accuracy: 0.29140373865763347\n",
      " val results:\n",
      "loss: 3.01018\n",
      "accuracy: 0.24266\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3947a58de7624a7298e2ba85b4069afb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 48\n",
      "train results:\n",
      "loss:  2.676660664876302\n",
      "accuracy: 0.2944098154703776\n",
      " val results:\n",
      "loss: 3.00705\n",
      "accuracy: 0.24220\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2f118bf353104c0d82e273dd8fa68649",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 49\n",
      "train results:\n",
      "loss:  2.666015116373698\n",
      "accuracy: 0.2982622146606445\n",
      " val results:\n",
      "loss: 3.00417\n",
      "accuracy: 0.24302\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f59f483db9e340879218209d0d02aed5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 50\n",
      "train results:\n",
      "loss:  2.6388951619466146\n",
      "accuracy: 0.304802672068278\n",
      " val results:\n",
      "loss: 2.99485\n",
      "accuracy: 0.25115\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "dea21fb473f544bfb8682e16975cc707",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 51\n",
      "train results:\n",
      "loss:  2.6233375549316404\n",
      "accuracy: 0.3112905502319336\n",
      " val results:\n",
      "loss: 2.99125\n",
      "accuracy: 0.25543\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "bc8f4e43eeb946f98dc01293a4e3a5be",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 52\n",
      "train results:\n",
      "loss:  2.6166417439778646\n",
      "accuracy: 0.3122570991516113\n",
      " val results:\n",
      "loss: 2.98901\n",
      "accuracy: 0.25069\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3fd4d674aafb44fc86287326d422b424",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 53\n",
      "train results:\n",
      "loss:  2.5985448201497396\n",
      "accuracy: 0.31947991053263347\n",
      " val results:\n",
      "loss: 2.99265\n",
      "accuracy: 0.26206\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5622df94b5b94bd9bfe4fd93cc47e6d4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 54\n",
      "train results:\n",
      "loss:  2.582250467936198\n",
      "accuracy: 0.3248121897379557\n",
      " val results:\n",
      "loss: 3.00127\n",
      "accuracy: 0.25494\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d43c8ce6d9cd411387548680ef362880",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 55\n",
      "train results:\n",
      "loss:  2.5612442016601564\n",
      "accuracy: 0.3294734001159668\n",
      " val results:\n",
      "loss: 2.98913\n",
      "accuracy: 0.25723\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "79a8c07c28de49858d9a80c52c2b73a8",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 56\n",
      "train results:\n",
      "loss:  2.5377354939778645\n",
      "accuracy: 0.3332093874613444\n",
      " val results:\n",
      "loss: 2.99601\n",
      "accuracy: 0.25857\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f54be540aa0840d4802fb1db9e2dee77",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 57\n",
      "train results:\n",
      "loss:  2.5223012288411457\n",
      "accuracy: 0.33978115717569984\n",
      " val results:\n",
      "loss: 2.99953\n",
      "accuracy: 0.25697\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "222b1e2e71e04ff7b23f6ce382ba4913",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 58\n",
      "train results:\n",
      "loss:  2.5020210266113283\n",
      "accuracy: 0.342951234181722\n",
      " val results:\n",
      "loss: 2.97635\n",
      "accuracy: 0.25963\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ba53bdb6afa24c11b3b0dc9326680fe2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 59\n",
      "train results:\n",
      "loss:  2.4893391927083335\n",
      "accuracy: 0.34685996373494465\n",
      " val results:\n",
      "loss: 2.97648\n",
      "accuracy: 0.26447\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d0f8948c2ab14b7e845f996b959ea922",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 60\n",
      "train results:\n",
      "loss:  2.4757708231608073\n",
      "accuracy: 0.35226236979166664\n",
      " val results:\n",
      "loss: 2.99541\n",
      "accuracy: 0.26600\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "411dea754c374cf3b72fa97816c3829b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 61\n",
      "train results:\n",
      "loss:  2.442059326171875\n",
      "accuracy: 0.35767854054768883\n",
      " val results:\n",
      "loss: 2.98924\n",
      "accuracy: 0.26344\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "be0305457ca345959064125f399c30e1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 62\n",
      "train results:\n",
      "loss:  2.4183527628580728\n",
      "accuracy: 0.3635028521219889\n",
      " val results:\n",
      "loss: 3.00692\n",
      "accuracy: 0.27045\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "974ef4d136104d3a8d4d37e3ddc2f914",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 63\n",
      "train results:\n",
      "loss:  2.410317230224609\n",
      "accuracy: 0.367121156056722\n",
      " val results:\n",
      "loss: 2.98379\n",
      "accuracy: 0.26596\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "92d1a814136244858835685859cd3f5c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 64\n",
      "train results:\n",
      "loss:  2.3998878479003904\n",
      "accuracy: 0.3694210688273112\n",
      " val results:\n",
      "loss: 3.01790\n",
      "accuracy: 0.26706\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "501645bff2364a0da6606c8af7f4151a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 65\n",
      "train results:\n",
      "loss:  2.3807782491048175\n",
      "accuracy: 0.37557843526204426\n",
      " val results:\n",
      "loss: 2.99803\n",
      "accuracy: 0.27030\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f35c12a7e1c74967867f88c1bf12a48b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 66\n",
      "train results:\n",
      "loss:  2.3577227274576824\n",
      "accuracy: 0.3780385971069336\n",
      " val results:\n",
      "loss: 3.00328\n",
      "accuracy: 0.26936\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b34287e592284c08bf99d1222ae284f9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 67\n",
      "train results:\n",
      "loss:  2.3459042867024738\n",
      "accuracy: 0.38139397303263345\n",
      " val results:\n",
      "loss: 3.00423\n",
      "accuracy: 0.27299\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5bd97d07e8ad48999db7782043974ca6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 68\n",
      "train results:\n",
      "loss:  2.3289871215820312\n",
      "accuracy: 0.3862116813659668\n",
      " val results:\n",
      "loss: 3.01788\n",
      "accuracy: 0.27292\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "bb1555770b924d5d988eac2bff815746",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 69\n",
      "train results:\n",
      "loss:  2.302038065592448\n",
      "accuracy: 0.3942833582560221\n",
      " val results:\n",
      "loss: 3.04536\n",
      "accuracy: 0.26847\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6f25670bdb0041468794c8dbdb6633ea",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 70\n",
      "train results:\n",
      "loss:  2.2725791931152344\n",
      "accuracy: 0.4031350135803223\n",
      " val results:\n",
      "loss: 3.02605\n",
      "accuracy: 0.27549\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3fa2446b66ce455ea2387e85003d1271",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 71\n",
      "train results:\n",
      "loss:  2.255554962158203\n",
      "accuracy: 0.4057830174763997\n",
      " val results:\n",
      "loss: 3.02138\n",
      "accuracy: 0.27258\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d16fd3b8ec4f47b292d9dab1321cc4d2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 72\n",
      "train results:\n",
      "loss:  2.2449005126953123\n",
      "accuracy: 0.40862255096435546\n",
      " val results:\n",
      "loss: 3.04485\n",
      "accuracy: 0.27650\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7a128cc6f349407fa30aa698c707d71e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 73\n",
      "train results:\n",
      "loss:  2.222159576416016\n",
      "accuracy: 0.4143842697143555\n",
      " val results:\n",
      "loss: 3.04438\n",
      "accuracy: 0.26500\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e826b50a9e454cf693481e9af85e7c35",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 74\n",
      "train results:\n",
      "loss:  2.2061060587565104\n",
      "accuracy: 0.41709108352661134\n",
      " val results:\n",
      "loss: 3.05346\n",
      "accuracy: 0.27523\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3651aeffb7d64bec926e063cd3ee8c5e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 75\n",
      "train results:\n",
      "loss:  2.1715899149576825\n",
      "accuracy: 0.42506634394327797\n",
      " val results:\n",
      "loss: 3.05390\n",
      "accuracy: 0.26832\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "be9af51e9f7b4489a78445b5a26df7a2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 76\n",
      "train results:\n",
      "loss:  2.1513071695963544\n",
      "accuracy: 0.43146158854166666\n",
      " val results:\n",
      "loss: 3.07391\n",
      "accuracy: 0.27327\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1fc535e438a44845a30b7b117a88e828",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 77\n",
      "train results:\n",
      "loss:  2.1389498392740887\n",
      "accuracy: 0.43303661346435546\n",
      " val results:\n",
      "loss: 3.06463\n",
      "accuracy: 0.26873\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "67b0de92d29f41ad82355eebb818e634",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 78\n",
      "train results:\n",
      "loss:  2.1271827697753904\n",
      "accuracy: 0.4357234001159668\n",
      " val results:\n",
      "loss: 3.08968\n",
      "accuracy: 0.26196\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "dc1febebecdd4d128c2cc300e7056191",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 79\n",
      "train results:\n",
      "loss:  2.1097180684407553\n",
      "accuracy: 0.4393705050150553\n",
      " val results:\n",
      "loss: 3.08810\n",
      "accuracy: 0.26993\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "eabac8eb81b74d9c9cac289ff85404dd",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 80\n",
      "train results:\n",
      "loss:  2.078189468383789\n",
      "accuracy: 0.4473758061726888\n",
      " val results:\n",
      "loss: 3.09347\n",
      "accuracy: 0.27334\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ee63e8d7fc794a20964a2786dd34bdbe",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 81\n",
      "train results:\n",
      "loss:  2.056459681193034\n",
      "accuracy: 0.45435822804768883\n",
      " val results:\n",
      "loss: 3.09879\n",
      "accuracy: 0.27327\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c3662233134a48dabc3d6cbd8d293d64",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 82\n",
      "train results:\n",
      "loss:  2.024919637044271\n",
      "accuracy: 0.46166114807128905\n",
      " val results:\n",
      "loss: 3.14223\n",
      "accuracy: 0.26274\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b6de763adc45454783c6a5699ce9dad8",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 83\n",
      "train results:\n",
      "loss:  2.0057573954264325\n",
      "accuracy: 0.46554864247639977\n",
      " val results:\n",
      "loss: 3.13494\n",
      "accuracy: 0.27551\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cabe0fb5d306494db3faed7f64010d7b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 84\n",
      "train results:\n",
      "loss:  1.9884610493977866\n",
      "accuracy: 0.4697566032409668\n",
      " val results:\n",
      "loss: 3.16744\n",
      "accuracy: 0.27687\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1e379dae929547e893429544bb584e90",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-60-2c05781fcee9>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     11\u001b[0m     \u001b[0;32mfor\u001b[0m \u001b[0mi\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mtqdm_notebook\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0menumerate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0miterate_minibatches\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mX_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch_size\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mBATCH_SIZE\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     12\u001b[0m         \u001b[0mpred\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbatch\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 13\u001b[0;31m         \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtensor\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbatch\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mint64\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdevice\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mDEVICE\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     14\u001b[0m         \u001b[0mloss\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcriterion\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpred\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     15\u001b[0m         \u001b[0maccuracy\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmean\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0margmax\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpred\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "model = ConvNetv3(n_tokens=maxL, num_classes=len(classes), n_filters=100, wv=wv_matrix, device=DEVICE).to(DEVICE)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n",
    "\n",
    "for epoch in tqdm_notebook(range(EPOCHS)):\n",
    "    \n",
    "    model.train()\n",
    "    epoch_loss = 0\n",
    "    epoch_accuracy = 0\n",
    "    iterations = 0\n",
    "    for i, batch in tqdm_notebook(enumerate(iterate_minibatches([X_train, y_train], batch_size=BATCH_SIZE))):\n",
    "        pred = model(batch[0])\n",
    "        y = torch.tensor(batch[1], dtype=torch.int64, device=DEVICE)\n",
    "        loss = criterion(pred, y)\n",
    "        accuracy = torch.mean((torch.argmax(pred, axis=-1).float() == y.float()).float())\n",
    "        optimizer.zero_grad()\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        epoch_loss += loss\n",
    "        epoch_accuracy += accuracy\n",
    "        iterations += 1\n",
    "    \n",
    "    print(f\"epoch: {epoch}\")\n",
    "    print(\"train results:\")\n",
    "    print(\"loss: \", epoch_loss.detach().cpu().numpy() / iterations)\n",
    "    print(\"accuracy:\", epoch_accuracy.detach().cpu().numpy() / iterations)    \n",
    "    print_metrics(model, [X_valid, y_valid])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 1min 24s, sys: 254 ms, total: 1min 24s\n",
      "Wall time: 1min 24s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "predicted_classes = model.predict(X_train)\n",
    "predicted_classes_valid = model.predict(X_valid)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.16240582671642415"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f1_score(y_train, predicted_classes, average=\"macro\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.06130408044958813"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f1_score(y_valid, predicted_classes_valid, average=\"macro\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "class ConvNetv4(nn.Module):\n",
    "    def __init__(self, n_tokens, num_classes, n_filters, wv, device, emb_dim=300):\n",
    "        super().__init__()\n",
    "        self.device = device\n",
    "        self.n_tokens = n_tokens\n",
    "        self.wv = torch.tensor(wv, dtype=torch.float32, device=device)\n",
    "        \n",
    "        self.embedder = nn.Embedding.from_pretrained(self.wv, freeze=False)\n",
    "#         self.embedder = nn.Embedding(len(vocab), embedding_dim=emb_dim)\n",
    "        \n",
    "        self.conv0 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(3, emb_dim), padding=(1, 0))\n",
    "        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(5, emb_dim), padding=(2, 0))\n",
    "        self.conv2 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(7, emb_dim), padding=(3, 0))\n",
    "        self.conv3 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(9, emb_dim), padding=(4, 0))\n",
    "        self.conv4 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(11, emb_dim), padding=(5, 0))\n",
    "        \n",
    "        \n",
    "        self.dropout = torch.nn.Dropout(0.3)\n",
    "        \n",
    "        self.final_predictor = nn.Sequential(\n",
    "            nn.Linear(5 * n_filters * n_tokens // 2, 5 * n_tokens // 2),\n",
    "            nn.Tanh(),\n",
    "            nn.Linear(5 * n_tokens // 2, num_classes),\n",
    "        )\n",
    "        \n",
    "    def forward(self, batch):\n",
    "        \n",
    "        batch = as_matrix(batch, self.n_tokens)\n",
    "        batch = torch.tensor(batch, device=self.device, dtype=torch.int64).unsqueeze(1)\n",
    "        \n",
    "        embedded = self.embedder(batch)\n",
    "        \n",
    "        conved0 = F.relu(self.conv0(embedded))\n",
    "        conved1 = F.relu(self.conv1(embedded))\n",
    "        conved2 = F.relu(self.conv2(embedded))\n",
    "        conved3 = F.relu(self.conv3(embedded))\n",
    "        conved4 = F.relu(self.conv4(embedded))\n",
    "        \n",
    "        \n",
    "        pooled0 = F.max_pool2d(conved0, kernel_size=(2, 1), stride=(2, 1)).squeeze(-1)\n",
    "        pooled1 = F.max_pool2d(conved1, kernel_size=(2, 1), stride=(2, 1)).squeeze(-1)\n",
    "        pooled2 = F.max_pool2d(conved2, kernel_size=(2, 1), stride=(2, 1)).squeeze(-1)\n",
    "        pooled3 = F.max_pool2d(conved3, kernel_size=(2, 1), stride=(2, 1)).squeeze(-1)\n",
    "        pooled4 = F.max_pool2d(conved4, kernel_size=(2, 1), stride=(2, 1)).squeeze(-1)\n",
    "        \n",
    "        cat = self.dropout(torch.cat((pooled0, pooled1, pooled2, pooled3, pooled4), dim=1)).flatten(1)\n",
    "        \n",
    "        return self.final_predictor(cat)\n",
    "    \n",
    "    def predict(self, data):\n",
    "        answer = []\n",
    "        for item in data:\n",
    "            pred = torch.argmax(self.forward([item]))\n",
    "            answer.append(pred.detach().cpu().numpy())\n",
    "        return answer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "923332039f114242a8bd52854a9148ab",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=0), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3174b95af1854538af7a2a469e259a17",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 0\n",
      "train results:\n",
      "loss:  2.844646199544271\n",
      "accuracy: 0.2513834635416667\n",
      " val results:\n",
      "loss: 3.06187\n",
      "accuracy: 0.21354\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f70aebfb1f26472abd1eb37f8b8e617a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 1\n",
      "train results:\n",
      "loss:  2.8257741292317706\n",
      "accuracy: 0.2551056702931722\n",
      " val results:\n",
      "loss: 3.05577\n",
      "accuracy: 0.22173\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5330f1bd7f584f2893ae4a700e2b5ceb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 2\n",
      "train results:\n",
      "loss:  2.8231300354003905\n",
      "accuracy: 0.25529847145080564\n",
      " val results:\n",
      "loss: 3.11012\n",
      "accuracy: 0.20843\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0afa465b4e3547928547a8e37248bdb1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 3\n",
      "train results:\n",
      "loss:  2.8209938049316405\n",
      "accuracy: 0.25699369112650555\n",
      " val results:\n",
      "loss: 3.07800\n",
      "accuracy: 0.21831\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "efffc759da7d4847a231981e3c672f8a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 4\n",
      "train results:\n",
      "loss:  2.8115750630696614\n",
      "accuracy: 0.2598269780476888\n",
      " val results:\n",
      "loss: 3.05310\n",
      "accuracy: 0.22325\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "225c6a55734743cea58e423b98bbe372",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 5\n",
      "train results:\n",
      "loss:  2.802898152669271\n",
      "accuracy: 0.2604705015818278\n",
      " val results:\n",
      "loss: 3.07463\n",
      "accuracy: 0.22437\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "061671d8949e4a5fa01bf3cb812f619e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 6\n",
      "train results:\n",
      "loss:  2.7977149963378904\n",
      "accuracy: 0.2667868614196777\n",
      " val results:\n",
      "loss: 3.06098\n",
      "accuracy: 0.22347\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "464691033cc44512a10e88c990344133",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 7\n",
      "train results:\n",
      "loss:  2.7950177510579426\n",
      "accuracy: 0.2648712952931722\n",
      " val results:\n",
      "loss: 3.05398\n",
      "accuracy: 0.22708\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "bcacecaf56ee4979bab29372f6ede13e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 8\n",
      "train results:\n",
      "loss:  2.7824620564778644\n",
      "accuracy: 0.26781975428263344\n",
      " val results:\n",
      "loss: 3.07356\n",
      "accuracy: 0.21918\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "aed2f1cd55f74088b8f8f52ee501886d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 9\n",
      "train results:\n",
      "loss:  2.771570587158203\n",
      "accuracy: 0.27312450408935546\n",
      " val results:\n",
      "loss: 3.05368\n",
      "accuracy: 0.23248\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4743f2b6f4d147baad178285207107ea",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 10\n",
      "train results:\n",
      "loss:  2.777374013264974\n",
      "accuracy: 0.2701009114583333\n",
      " val results:\n",
      "loss: 3.01850\n",
      "accuracy: 0.23383\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "08e7ed2112c44514b7a51e700027c9c8",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 11\n",
      "train results:\n",
      "loss:  2.766382344563802\n",
      "accuracy: 0.27588392893473307\n",
      " val results:\n",
      "loss: 3.03973\n",
      "accuracy: 0.23131\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ff8eb67eb5944473a28ebbb4bdeea5b2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 12\n",
      "train results:\n",
      "loss:  2.752320098876953\n",
      "accuracy: 0.2772085189819336\n",
      " val results:\n",
      "loss: 3.04840\n",
      "accuracy: 0.23439\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "01563fe5e6e94fe2aaf15afda5cc3601",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 13\n",
      "train results:\n",
      "loss:  2.7507601420084637\n",
      "accuracy: 0.2771033604939779\n",
      " val results:\n",
      "loss: 3.05569\n",
      "accuracy: 0.22698\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0812fd894cfa4bcfa6f5f49d4160c66f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 14\n",
      "train results:\n",
      "loss:  2.736506144205729\n",
      "accuracy: 0.28375399907430016\n",
      " val results:\n",
      "loss: 3.04709\n",
      "accuracy: 0.22664\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7baefcdd9cf7468089805736a22932ad",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 15\n",
      "train results:\n",
      "loss:  2.7212150573730467\n",
      "accuracy: 0.2854817708333333\n",
      " val results:\n",
      "loss: 3.03739\n",
      "accuracy: 0.22760\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "70a19ac0d8dd4a339988f6bf6ad89549",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 16\n",
      "train results:\n",
      "loss:  2.714629364013672\n",
      "accuracy: 0.2890086491902669\n",
      " val results:\n",
      "loss: 3.04439\n",
      "accuracy: 0.22022\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a247130f9a6d4f2f9ad0706a1c22d3ac",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 17\n",
      "train results:\n",
      "loss:  2.7074918111165363\n",
      "accuracy: 0.2849233627319336\n",
      " val results:\n",
      "loss: 3.05233\n",
      "accuracy: 0.23604\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "80c9247ec9fa474986da8730b4a97104",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 18\n",
      "train results:\n",
      "loss:  2.7014976501464845\n",
      "accuracy: 0.2900616010030111\n",
      " val results:\n",
      "loss: 3.03056\n",
      "accuracy: 0.24499\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "95e084a0e4b84445a22ed624f5775548",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 19\n",
      "train results:\n",
      "loss:  2.6877197265625\n",
      "accuracy: 0.29539511998494467\n",
      " val results:\n",
      "loss: 3.04784\n",
      "accuracy: 0.23332\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ec3c6d694a1d4f609b2690afbbcb1a4f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 20\n",
      "train results:\n",
      "loss:  2.6790306091308596\n",
      "accuracy: 0.2983924229939779\n",
      " val results:\n",
      "loss: 3.02822\n",
      "accuracy: 0.24358\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fd111a67ffed491393d10a2af07132df",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 21\n",
      "train results:\n",
      "loss:  2.6719703674316406\n",
      "accuracy: 0.29778019587198895\n",
      " val results:\n",
      "loss: 3.02758\n",
      "accuracy: 0.23210\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8c7a092eb32c4ef1993c300aea8290e2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 22\n",
      "train results:\n",
      "loss:  2.6708717346191406\n",
      "accuracy: 0.29889322916666666\n",
      " val results:\n",
      "loss: 3.03740\n",
      "accuracy: 0.22918\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cdc9d59621f74e0ca9fd12d761ddc0e6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 23\n",
      "train results:\n",
      "loss:  2.6686429341634113\n",
      "accuracy: 0.2972718874613444\n",
      " val results:\n",
      "loss: 3.02948\n",
      "accuracy: 0.23637\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b067738b2d2640cda7d8f43b480a96fa",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 24\n",
      "train results:\n",
      "loss:  2.6427792867024738\n",
      "accuracy: 0.30475009282430016\n",
      " val results:\n",
      "loss: 3.02832\n",
      "accuracy: 0.23570\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "22b04adc277c4b8e948df1a390b22c79",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 25\n",
      "train results:\n",
      "loss:  2.636304982503255\n",
      "accuracy: 0.3075646082560221\n",
      " val results:\n",
      "loss: 3.02163\n",
      "accuracy: 0.23600\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8de9d858311a439ab5abbef56efd39b4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 26\n",
      "train results:\n",
      "loss:  2.636047108968099\n",
      "accuracy: 0.30972429911295574\n",
      " val results:\n",
      "loss: 3.02803\n",
      "accuracy: 0.24783\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "348875eb2ee1481597d848eb57b363c5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 27\n",
      "train results:\n",
      "loss:  2.626629130045573\n",
      "accuracy: 0.31270907719930013\n",
      " val results:\n",
      "loss: 3.02199\n",
      "accuracy: 0.25175\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b091f187d68647fca3c7199181cf3b8b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 28\n",
      "train results:\n",
      "loss:  2.60869878133138\n",
      "accuracy: 0.3135016123453776\n",
      " val results:\n",
      "loss: 3.02547\n",
      "accuracy: 0.24483\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "13af77bfa15c46e0907e1e6e009fa42c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-25-19ebf66a36e8>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     15\u001b[0m         \u001b[0maccuracy\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmean\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0margmax\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpred\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     16\u001b[0m         \u001b[0moptimizer\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mzero_grad\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 17\u001b[0;31m         \u001b[0mloss\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbackward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     18\u001b[0m         \u001b[0moptimizer\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstep\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     19\u001b[0m         \u001b[0mepoch_loss\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0mloss\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/lib/python3.7/site-packages/torch/tensor.py\u001b[0m in \u001b[0;36mbackward\u001b[0;34m(self, gradient, retain_graph, create_graph)\u001b[0m\n\u001b[1;32m    219\u001b[0m                 \u001b[0mretain_graph\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mretain_graph\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    220\u001b[0m                 create_graph=create_graph)\n\u001b[0;32m--> 221\u001b[0;31m         \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mautograd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbackward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgradient\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mretain_graph\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcreate_graph\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    222\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    223\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mregister_hook\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mhook\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/lib/python3.7/site-packages/torch/autograd/__init__.py\u001b[0m in \u001b[0;36mbackward\u001b[0;34m(tensors, grad_tensors, retain_graph, create_graph, grad_variables)\u001b[0m\n\u001b[1;32m    130\u001b[0m     Variable._execution_engine.run_backward(\n\u001b[1;32m    131\u001b[0m         \u001b[0mtensors\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgrad_tensors_\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mretain_graph\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcreate_graph\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 132\u001b[0;31m         allow_unreachable=True)  # allow_unreachable flag\n\u001b[0m\u001b[1;32m    133\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    134\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "model = ConvNetv4(n_tokens=maxL, num_classes=len(classes), n_filters=100, wv=wv_matrix, device=DEVICE).to(DEVICE)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n",
    "\n",
    "for epoch in tqdm_notebook(range(EPOCHS)):\n",
    "    \n",
    "    model.train()\n",
    "    epoch_loss = 0\n",
    "    epoch_accuracy = 0\n",
    "    iterations = 0\n",
    "    for i, batch in tqdm_notebook(enumerate(iterate_minibatches([X_train, y_train], batch_size=BATCH_SIZE))):\n",
    "        pred = model(batch[0])\n",
    "        y = torch.tensor(batch[1], dtype=torch.int64, device=DEVICE)\n",
    "        loss = criterion(pred, y)\n",
    "        accuracy = torch.mean((torch.argmax(pred, axis=-1).float() == y.float()).float())\n",
    "        optimizer.zero_grad()\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        epoch_loss += loss\n",
    "        epoch_accuracy += accuracy\n",
    "        iterations += 1\n",
    "    \n",
    "    print(f\"epoch: {epoch}\")\n",
    "    print(\"train results:\")\n",
    "    print(\"loss: \", epoch_loss.detach().cpu().numpy() / iterations)\n",
    "    print(\"accuracy:\", epoch_accuracy.detach().cpu().numpy() / iterations)    \n",
    "    print_metrics(model, [X_valid, y_valid])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 3min 11s, sys: 183 ms, total: 3min 11s\n",
      "Wall time: 3min 14s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "predicted_classes = model.predict(X_train)\n",
    "predicted_classes_valid = model.predict(X_valid)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.0691466993197574"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f1_score(y_train, predicted_classes, average=\"macro\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.04881859623126991"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f1_score(y_valid, predicted_classes_valid, average=\"macro\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "class ConvNetv5(nn.Module):\n",
    "    def __init__(self, n_tokens, num_classes, n_filters, wv, device, emb_dim=300):\n",
    "        super().__init__()\n",
    "        self.device = device\n",
    "        self.n_tokens = n_tokens\n",
    "        self.wv = torch.tensor(wv, dtype=torch.float32, device=device)\n",
    "        \n",
    "        self.embedder = nn.Embedding.from_pretrained(self.wv, freeze=False)\n",
    "#         self.embedder = nn.Embedding(len(vocab), embedding_dim=emb_dim)\n",
    "        \n",
    "        self.conv00 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(3, emb_dim), padding=(1, 0))\n",
    "        self.conv01 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(5, emb_dim), padding=(2, 0))\n",
    "        self.conv02 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(7, emb_dim), padding=(3, 0))\n",
    "        self.conv03 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(9, emb_dim), padding=(4, 0))\n",
    "        self.conv04 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(11, emb_dim), padding=(5, 0))\n",
    "        \n",
    "        self.dropout0 = torch.nn.Dropout(0.3)\n",
    "        \n",
    "        self.conv10 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(3, 5 * n_filters), padding=(1, 0))\n",
    "        self.conv11 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(5, 5 * n_filters), padding=(2, 0))\n",
    "        self.conv12 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(7, 5 * n_filters), padding=(3, 0))\n",
    "        \n",
    "        self.dropout1 = torch.nn.Dropout(0.3)\n",
    "        \n",
    "        self.final_predictor = nn.Sequential(\n",
    "            nn.Linear(3 * n_filters * (n_tokens // 4), 3 * (n_tokens // 4)),\n",
    "            nn.Tanh(),\n",
    "            nn.Linear(3 * (n_tokens // 4), num_classes),\n",
    "        )\n",
    "        \n",
    "    def forward(self, batch):\n",
    "        \n",
    "        batch = as_matrix(batch, self.n_tokens)\n",
    "        batch = torch.tensor(batch, device=self.device, dtype=torch.int64).unsqueeze(1)\n",
    "        \n",
    "        embedded = self.embedder(batch)\n",
    "        \n",
    "        conved00 = F.relu(self.conv00(embedded))\n",
    "        conved01 = F.relu(self.conv01(embedded))\n",
    "        conved02 = F.relu(self.conv02(embedded))\n",
    "        conved03 = F.relu(self.conv03(embedded))\n",
    "        conved04 = F.relu(self.conv04(embedded))\n",
    "        \n",
    "        pooled00 = F.max_pool2d(conved00, kernel_size=(2, 1), stride=(2, 1)).squeeze(-1)\n",
    "        pooled01 = F.max_pool2d(conved01, kernel_size=(2, 1), stride=(2, 1)).squeeze(-1)\n",
    "        pooled02 = F.max_pool2d(conved02, kernel_size=(2, 1), stride=(2, 1)).squeeze(-1)\n",
    "        pooled03 = F.max_pool2d(conved03, kernel_size=(2, 1), stride=(2, 1)).squeeze(-1)\n",
    "        pooled04 = F.max_pool2d(conved04, kernel_size=(2, 1), stride=(2, 1)).squeeze(-1)\n",
    "        \n",
    "        cat = self.dropout0(torch.cat((pooled00, pooled01, pooled02, pooled03, pooled04), dim=1)).transpose(1, 2)\n",
    "        cat = cat.unsqueeze(1)\n",
    "        \n",
    "        conved10 = F.relu(self.conv10(cat))\n",
    "        conved11 = F.relu(self.conv11(cat))\n",
    "        conved12 = F.relu(self.conv12(cat))\n",
    "        \n",
    "        pooled10 = F.max_pool2d(conved10, kernel_size=(2, 1), stride=(2, 1)).squeeze(-1)\n",
    "        pooled11 = F.max_pool2d(conved11, kernel_size=(2, 1), stride=(2, 1)).squeeze(-1)\n",
    "        pooled12 = F.max_pool2d(conved12, kernel_size=(2, 1), stride=(2, 1)).squeeze(-1)\n",
    "        \n",
    "        cat = self.dropout1(torch.cat((pooled10, pooled11, pooled12), dim=1)).flatten(1)\n",
    "        \n",
    "        return self.final_predictor(cat)\n",
    "    \n",
    "    def predict(self, data):\n",
    "        answer = []\n",
    "        for item in data:\n",
    "            pred = torch.argmax(self.forward([item]))\n",
    "            answer.append(pred.detach().cpu().numpy())\n",
    "        return answer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b2c8703b455441cb9c74184abf924ceb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=0, max=150), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8000bb827721422ea3d4ce218003ceb8",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 0\n",
      "train results:\n",
      "loss:  3.477552286783854\n",
      "accuracy: 0.1213529109954834\n",
      " val results:\n",
      "loss: 3.47504\n",
      "accuracy: 0.11373\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cfaa99a9dbe04a2287bacada1420adaa",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 1\n",
      "train results:\n",
      "loss:  3.4526046752929687\n",
      "accuracy: 0.12243839899698893\n",
      " val results:\n",
      "loss: 3.48237\n",
      "accuracy: 0.12177\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d62ab136e0c149b3966a64f2591ecac8",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 2\n",
      "train results:\n",
      "loss:  3.4543256123860675\n",
      "accuracy: 0.12124399344126384\n",
      " val results:\n",
      "loss: 3.47342\n",
      "accuracy: 0.11373\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d65f4801814b47ea9399047cf450adfd",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 3\n",
      "train results:\n",
      "loss:  3.4540672302246094\n",
      "accuracy: 0.1209134578704834\n",
      " val results:\n",
      "loss: 3.47750\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3cb8b1ba14274c84b79a9793f1f10d19",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 4\n",
      "train results:\n",
      "loss:  3.4544746398925783\n",
      "accuracy: 0.12269256114959717\n",
      " val results:\n",
      "loss: 3.47501\n",
      "accuracy: 0.11307\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7514c601ae214b0c9e626601bcdafd27",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 5\n",
      "train results:\n",
      "loss:  3.453150685628255\n",
      "accuracy: 0.12109500567118327\n",
      " val results:\n",
      "loss: 3.47895\n",
      "accuracy: 0.11420\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a57a783924a340cf89670ebcccef6e16",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 6\n",
      "train results:\n",
      "loss:  3.453974405924479\n",
      "accuracy: 0.12264623641967773\n",
      " val results:\n",
      "loss: 3.47945\n",
      "accuracy: 0.11399\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c5109fde128647fd959d4f7f6e457aef",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 7\n",
      "train results:\n",
      "loss:  3.4548749287923175\n",
      "accuracy: 0.1220903476079305\n",
      " val results:\n",
      "loss: 3.47993\n",
      "accuracy: 0.11399\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "bac0e47ea549440da053c30879f4e597",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 8\n",
      "train results:\n",
      "loss:  3.454284413655599\n",
      "accuracy: 0.1212740421295166\n",
      " val results:\n",
      "loss: 3.48013\n",
      "accuracy: 0.11373\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ce55893186674d1dbddd614d01c6c035",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 9\n",
      "train results:\n",
      "loss:  3.4551073710123696\n",
      "accuracy: 0.12164212862650553\n",
      " val results:\n",
      "loss: 3.47994\n",
      "accuracy: 0.11373\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c1c7a84c8f634074b8bfa07e68c9f9d4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 10\n",
      "train results:\n",
      "loss:  3.4568099975585938\n",
      "accuracy: 0.12127904891967774\n",
      " val results:\n",
      "loss: 3.48157\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8ea1dbada9ae4930b667c086c5a42502",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 11\n",
      "train results:\n",
      "loss:  3.4579683939615884\n",
      "accuracy: 0.12063051064809163\n",
      " val results:\n",
      "loss: 3.47809\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3f136ddb2a5945f8b86a24aa601e2034",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 12\n",
      "train results:\n",
      "loss:  3.450653076171875\n",
      "accuracy: 0.12203400135040283\n",
      " val results:\n",
      "loss: 3.47495\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a061929694254daa895363b71afe4f8b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 13\n",
      "train results:\n",
      "loss:  3.452806345621745\n",
      "accuracy: 0.1208420991897583\n",
      " val results:\n",
      "loss: 3.47459\n",
      "accuracy: 0.11305\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "87005fb236f9424c954acd008b73b03c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 14\n",
      "train results:\n",
      "loss:  3.4544662475585937\n",
      "accuracy: 0.12085336844126383\n",
      " val results:\n",
      "loss: 3.47484\n",
      "accuracy: 0.11346\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d87a37d0963f4c9d8131b5c6940a4373",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 15\n",
      "train results:\n",
      "loss:  3.453424580891927\n",
      "accuracy: 0.1230055570602417\n",
      " val results:\n",
      "loss: 3.48487\n",
      "accuracy: 0.11399\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1ce12bd9ce6b4820ad3a29847cb45b97",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 16\n",
      "train results:\n",
      "loss:  3.452980550130208\n",
      "accuracy: 0.12365535100301107\n",
      " val results:\n",
      "loss: 3.47764\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b4485abe35694e788567d53b7dbf996e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 17\n",
      "train results:\n",
      "loss:  3.4526796976725262\n",
      "accuracy: 0.12265499432881673\n",
      " val results:\n",
      "loss: 3.47857\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1570268824574adb918c414cac238dd5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 18\n",
      "train results:\n",
      "loss:  3.452684783935547\n",
      "accuracy: 0.12314327557881673\n",
      " val results:\n",
      "loss: 3.48283\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "09823ff74d1e424e97a636b27746af68",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 19\n",
      "train results:\n",
      "loss:  3.45146484375\n",
      "accuracy: 0.12242212295532226\n",
      " val results:\n",
      "loss: 3.47414\n",
      "accuracy: 0.11305\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5cfd734beb6049299e356d741f68d442",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 20\n",
      "train results:\n",
      "loss:  3.44993896484375\n",
      "accuracy: 0.12121394475301107\n",
      " val results:\n",
      "loss: 3.47506\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "aa79cf393b0548a682892c97d7dc389d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 21\n",
      "train results:\n",
      "loss:  3.451739756266276\n",
      "accuracy: 0.12270131905873617\n",
      " val results:\n",
      "loss: 3.48042\n",
      "accuracy: 0.12151\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8b9b8267f0884f13bae7d68d1826c57e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 22\n",
      "train results:\n",
      "loss:  3.4530487060546875\n",
      "accuracy: 0.12361027399698893\n",
      " val results:\n",
      "loss: 3.47585\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b2780ee7893d45c78ca2494bf16647b1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 23\n",
      "train results:\n",
      "loss:  3.4530062357584637\n",
      "accuracy: 0.12272761662801107\n",
      " val results:\n",
      "loss: 3.47431\n",
      "accuracy: 0.12190\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b2d830608d5843ebb898e5dd4a206d45",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 24\n",
      "train results:\n",
      "loss:  3.451496124267578\n",
      "accuracy: 0.1228365421295166\n",
      " val results:\n",
      "loss: 3.47608\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f023256fb472416494adfbb017316076",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 25\n",
      "train results:\n",
      "loss:  3.4542755126953124\n",
      "accuracy: 0.12340118885040283\n",
      " val results:\n",
      "loss: 3.47514\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7486756218d54e36a71077bb4bdf023a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 26\n",
      "train results:\n",
      "loss:  3.455554707845052\n",
      "accuracy: 0.12230193614959717\n",
      " val results:\n",
      "loss: 3.48226\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "62dc16bd2b374bf5a75c591e2a8649dc",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 27\n",
      "train results:\n",
      "loss:  3.453899383544922\n",
      "accuracy: 0.12270382245381674\n",
      " val results:\n",
      "loss: 3.47749\n",
      "accuracy: 0.11425\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "67ea9951cd1f4e6087e6736d7ca905fd",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 28\n",
      "train results:\n",
      "loss:  3.4499267578125\n",
      "accuracy: 0.12303435802459717\n",
      " val results:\n",
      "loss: 3.47607\n",
      "accuracy: 0.11370\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "500b84297dc345a5addbedd2d761829c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 29\n",
      "train results:\n",
      "loss:  3.453449503580729\n",
      "accuracy: 0.12099985281626384\n",
      " val results:\n",
      "loss: 3.47805\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3d2d86f4161841849610c5a137504b3a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 30\n",
      "train results:\n",
      "loss:  3.4524159749348957\n",
      "accuracy: 0.1210161288579305\n",
      " val results:\n",
      "loss: 3.47912\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5b9d61d70c3f449891cc25d8488bcb59",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 31\n",
      "train results:\n",
      "loss:  3.4556594848632813\n",
      "accuracy: 0.12319711844126384\n",
      " val results:\n",
      "loss: 3.47937\n",
      "accuracy: 0.11370\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b5fb1c81d8414105bf6799985925a86c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 32\n",
      "train results:\n",
      "loss:  3.453688557942708\n",
      "accuracy: 0.12274889945983887\n",
      " val results:\n",
      "loss: 3.47950\n",
      "accuracy: 0.11409\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1b16570afd6247f69be3a55ceccbbe6b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 33\n",
      "train results:\n",
      "loss:  3.4541216532389325\n",
      "accuracy: 0.12229066689809164\n",
      " val results:\n",
      "loss: 3.47324\n",
      "accuracy: 0.11383\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "79ef0dc06bc14758a185eb04553c4c51",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 34\n",
      "train results:\n",
      "loss:  3.4517443339029947\n",
      "accuracy: 0.12205278078715007\n",
      " val results:\n",
      "loss: 3.47704\n",
      "accuracy: 0.11370\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "105c96b37dc7461389d1ec4af2c0c7dc",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 35\n",
      "train results:\n",
      "loss:  3.4524820963541667\n",
      "accuracy: 0.12384690443674723\n",
      " val results:\n",
      "loss: 3.48567\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9fbbc99b41e74ac69e89ebadb22f1784",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 36\n",
      "train results:\n",
      "loss:  3.45453364054362\n",
      "accuracy: 0.12230443954467773\n",
      " val results:\n",
      "loss: 3.48188\n",
      "accuracy: 0.11412\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0b12bbb321e443d08cda9c4ce7607c4c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 37\n",
      "train results:\n",
      "loss:  3.453175099690755\n",
      "accuracy: 0.1213654359181722\n",
      " val results:\n",
      "loss: 3.47004\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e7ea2178aa3c4493bc43504edf76642c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 38\n",
      "train results:\n",
      "loss:  3.4524126688639325\n",
      "accuracy: 0.12189753850301106\n",
      " val results:\n",
      "loss: 3.47658\n",
      "accuracy: 0.12177\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b455216c2a27498cb440f79b6c6bdc58",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 39\n",
      "train results:\n",
      "loss:  3.451831817626953\n",
      "accuracy: 0.12326096693674723\n",
      " val results:\n",
      "loss: 3.48027\n",
      "accuracy: 0.11399\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "68e0cce9666b4fa4a344022bc9353e70",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 40\n",
      "train results:\n",
      "loss:  3.455181630452474\n",
      "accuracy: 0.11994691689809163\n",
      " val results:\n",
      "loss: 3.47478\n",
      "accuracy: 0.11370\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "80345bb149db469f926b4dfab25e1178",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 41\n",
      "train results:\n",
      "loss:  3.454461669921875\n",
      "accuracy: 0.12295047442118327\n",
      " val results:\n",
      "loss: 3.47987\n",
      "accuracy: 0.11438\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ccfc263982df4dce8d9f95af92654884",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 42\n",
      "train results:\n",
      "loss:  3.452653249104818\n",
      "accuracy: 0.12231069405873617\n",
      " val results:\n",
      "loss: 3.48862\n",
      "accuracy: 0.11318\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2c8bf6bc2ae84779ac25056b216b7740",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 43\n",
      "train results:\n",
      "loss:  3.4499921162923175\n",
      "accuracy: 0.12332857449849446\n",
      " val results:\n",
      "loss: 3.48343\n",
      "accuracy: 0.11399\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "da40ca9b10964ad580d8b8ed029285bf",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 44\n",
      "train results:\n",
      "loss:  3.452323150634766\n",
      "accuracy: 0.12191381454467773\n",
      " val results:\n",
      "loss: 3.48112\n",
      "accuracy: 0.11242\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "912f4941ee434ca4b2e1f155e6d0a78f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 45\n",
      "train results:\n",
      "loss:  3.4521893819173175\n",
      "accuracy: 0.12203025023142497\n",
      " val results:\n",
      "loss: 3.47798\n",
      "accuracy: 0.11266\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b6c30b5304414076a29fb088bcafc6ed",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 46\n",
      "train results:\n",
      "loss:  3.4530288696289064\n",
      "accuracy: 0.12337114016215006\n",
      " val results:\n",
      "loss: 3.47613\n",
      "accuracy: 0.11357\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "777326ec67a548ffbe2739a3a4f24bb6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 47\n",
      "train results:\n",
      "loss:  3.4550318400065105\n",
      "accuracy: 0.12167843182881673\n",
      " val results:\n",
      "loss: 3.47692\n",
      "accuracy: 0.12177\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9664f470e0aa46c78f9cb057d31fb260",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 48\n",
      "train results:\n",
      "loss:  3.453382364908854\n",
      "accuracy: 0.12314077218373616\n",
      " val results:\n",
      "loss: 3.47669\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b995e2e1a57848c195b5ed8b9cf1895e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 49\n",
      "train results:\n",
      "loss:  3.4493418375651044\n",
      "accuracy: 0.1224759578704834\n",
      " val results:\n",
      "loss: 3.48036\n",
      "accuracy: 0.11399\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7f3ab4e124d4430c9678db2b5f5490bb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 50\n",
      "train results:\n",
      "loss:  3.4536158243815103\n",
      "accuracy: 0.1218462069829305\n",
      " val results:\n",
      "loss: 3.47484\n",
      "accuracy: 0.12203\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b0516c4cb151429abc3237822404a4ac",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 51\n",
      "train results:\n",
      "loss:  3.4532374064127604\n",
      "accuracy: 0.12335737546284993\n",
      " val results:\n",
      "loss: 3.47187\n",
      "accuracy: 0.11318\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f0584224b0f24e3dba85c86e9365f93e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 52\n",
      "train results:\n",
      "loss:  3.450805409749349\n",
      "accuracy: 0.1239608367284139\n",
      " val results:\n",
      "loss: 3.47962\n",
      "accuracy: 0.11370\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "762d9979c3674aa082011ef02522ca07",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 53\n",
      "train results:\n",
      "loss:  3.452136993408203\n",
      "accuracy: 0.12220678329467774\n",
      " val results:\n",
      "loss: 3.47881\n",
      "accuracy: 0.12255\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6dee3d979c984dc78390dba4e1c97312",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 54\n",
      "train results:\n",
      "loss:  3.4535593668619793\n",
      "accuracy: 0.12284029324849446\n",
      " val results:\n",
      "loss: 3.48419\n",
      "accuracy: 0.11370\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "46e181e4eaba420b9013a2e4ff5fbef2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 55\n",
      "train results:\n",
      "loss:  3.4528704325358075\n",
      "accuracy: 0.12115509510040283\n",
      " val results:\n",
      "loss: 3.47832\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f105c27a928a4bf685dfdd9a66db1d25",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 56\n",
      "train results:\n",
      "loss:  3.4522061665852863\n",
      "accuracy: 0.12159330050150553\n",
      " val results:\n",
      "loss: 3.47056\n",
      "accuracy: 0.11331\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8521580aa61f4827a35de91d10ea2014",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 57\n",
      "train results:\n",
      "loss:  3.4541539510091144\n",
      "accuracy: 0.12169220447540283\n",
      " val results:\n",
      "loss: 3.47804\n",
      "accuracy: 0.11344\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "32e03cd52b6946069b2fb90db74764ad",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 58\n",
      "train results:\n",
      "loss:  3.45324223836263\n",
      "accuracy: 0.12258989016215006\n",
      " val results:\n",
      "loss: 3.47758\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2d27b9a7eb2d49a6b05b764102bd817f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 59\n",
      "train results:\n",
      "loss:  3.4524383544921875\n",
      "accuracy: 0.12121394475301107\n",
      " val results:\n",
      "loss: 3.47056\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3fad902743694937b3be22df993d7ad3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 60\n",
      "train results:\n",
      "loss:  3.4519798278808596\n",
      "accuracy: 0.12161834239959717\n",
      " val results:\n",
      "loss: 3.46841\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "bcd0eeaa34d3422f9a1aca56b0323a97",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 61\n",
      "train results:\n",
      "loss:  3.45111083984375\n",
      "accuracy: 0.1210787296295166\n",
      " val results:\n",
      "loss: 3.47936\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "aea550fa65084f039c5503803b3ab0b3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 62\n",
      "train results:\n",
      "loss:  3.453047180175781\n",
      "accuracy: 0.12033503850301107\n",
      " val results:\n",
      "loss: 3.47983\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "eda97bdc4a1841b48828adbecb1ba989",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 63\n",
      "train results:\n",
      "loss:  3.4513231913248696\n",
      "accuracy: 0.12080203692118327\n",
      " val results:\n",
      "loss: 3.47407\n",
      "accuracy: 0.11344\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9ec0ad3b90da4ee7af7e524596b772d3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 64\n",
      "train results:\n",
      "loss:  3.450434366861979\n",
      "accuracy: 0.12268754641215006\n",
      " val results:\n",
      "loss: 3.47200\n",
      "accuracy: 0.11344\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f874ca0d296a424ba132debc8a906beb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 65\n",
      "train results:\n",
      "loss:  3.4495257059733073\n",
      "accuracy: 0.1218812624613444\n",
      " val results:\n",
      "loss: 3.47598\n",
      "accuracy: 0.11882\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "14afeb9a397b4fbc957b19108a507d4c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 66\n",
      "train results:\n",
      "loss:  3.4486597696940104\n",
      "accuracy: 0.1225873867670695\n",
      " val results:\n",
      "loss: 3.47817\n",
      "accuracy: 0.11360\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e9423da1716840b18e0e34f1eb5a467a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 67\n",
      "train results:\n",
      "loss:  3.4523475646972654\n",
      "accuracy: 0.12242712974548339\n",
      " val results:\n",
      "loss: 3.47785\n",
      "accuracy: 0.11399\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8cd0812180594f1b96f4d2062de1d5b5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 68\n",
      "train results:\n",
      "loss:  3.454003651936849\n",
      "accuracy: 0.1223557710647583\n",
      " val results:\n",
      "loss: 3.47143\n",
      "accuracy: 0.11373\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b4de64d7431c4ce7a355d5f182f70efc",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 69\n",
      "train results:\n",
      "loss:  3.453749084472656\n",
      "accuracy: 0.12015600204467773\n",
      " val results:\n",
      "loss: 3.47136\n",
      "accuracy: 0.11373\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "59310bd1ce8a4fd3b1e02863079d2af6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 70\n",
      "train results:\n",
      "loss:  3.451074981689453\n",
      "accuracy: 0.12161332766215006\n",
      " val results:\n",
      "loss: 3.47899\n",
      "accuracy: 0.12151\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "16ce14a7a1cd4cc3b2dbc352e475b853",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 71\n",
      "train results:\n",
      "loss:  3.4519579569498697\n",
      "accuracy: 0.12269506454467774\n",
      " val results:\n",
      "loss: 3.47855\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1777e3b19f1b4e85aed7b370b4839972",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 72\n",
      "train results:\n",
      "loss:  3.4518526713053386\n",
      "accuracy: 0.12185121377309163\n",
      " val results:\n",
      "loss: 3.47205\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b0c545893bbc4ba786ab312fc0422461",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 73\n",
      "train results:\n",
      "loss:  3.4537760416666665\n",
      "accuracy: 0.12175105412801107\n",
      " val results:\n",
      "loss: 3.47196\n",
      "accuracy: 0.12164\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "13451447eee049499924c6dede82f9e4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-22-2fba0c81269b>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     11\u001b[0m     \u001b[0;32mfor\u001b[0m \u001b[0mi\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mtqdm_notebook\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0menumerate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0miterate_minibatches\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mX_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch_size\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mBATCH_SIZE\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     12\u001b[0m         \u001b[0mpred\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbatch\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 13\u001b[0;31m         \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtensor\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbatch\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mint64\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdevice\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mDEVICE\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     14\u001b[0m         \u001b[0mloss\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcriterion\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpred\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     15\u001b[0m         \u001b[0maccuracy\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmean\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0margmax\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpred\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "model = ConvNetv5(n_tokens=maxL, num_classes=len(classes), n_filters=100, wv=wv_matrix, device=DEVICE).to(DEVICE)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n",
    "\n",
    "for epoch in tqdm_notebook(range(EPOCHS)):\n",
    "    \n",
    "    model.train()\n",
    "    epoch_loss = 0\n",
    "    epoch_accuracy = 0\n",
    "    iterations = 0\n",
    "    for i, batch in tqdm_notebook(enumerate(iterate_minibatches([X_train, y_train], batch_size=BATCH_SIZE))):\n",
    "        pred = model(batch[0])\n",
    "        y = torch.tensor(batch[1], dtype=torch.int64, device=DEVICE)\n",
    "        loss = criterion(pred, y)\n",
    "        accuracy = torch.mean((torch.argmax(pred, axis=-1).float() == y.float()).float())\n",
    "        optimizer.zero_grad()\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        epoch_loss += loss\n",
    "        epoch_accuracy += accuracy\n",
    "        iterations += 1\n",
    "    \n",
    "    print(f\"epoch: {epoch}\")\n",
    "    print(\"train results:\")\n",
    "    print(\"loss: \", epoch_loss.detach().cpu().numpy() / iterations)\n",
    "    print(\"accuracy:\", epoch_accuracy.detach().cpu().numpy() / iterations)    \n",
    "    print_metrics(model, [X_valid, y_valid])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "class ConvNetv6(nn.Module):\n",
    "    def __init__(self, n_tokens, num_classes, n_filters, wv, device, emb_dim=300):\n",
    "        super().__init__()\n",
    "        \n",
    "        self.device = device\n",
    "        self.n_tokens = n_tokens\n",
    "        self.wv = torch.tensor(wv, dtype=torch.float32)\n",
    "        \n",
    "        self.embedder = nn.Embedding.from_pretrained(self.wv, freeze=False)\n",
    "        self.encoder = nn.Sequential(\n",
    "            nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding=1),\n",
    "            nn.Dropout(p=0.25),\n",
    "            nn.ReLU(),\n",
    "            nn.Conv1d(emb_dim, emb_dim, kernel_size=5, padding=2),\n",
    "            nn.Dropout(p=0.25),\n",
    "            nn.ReLU(),\n",
    "            nn.MaxPool1d(kernel_size=2, stride=2)\n",
    "        )\n",
    "        self.final_predictor = nn.Sequential(\n",
    "            nn.Linear(emb_dim * n_tokens // 2, emb_dim),\n",
    "            nn.Tanh(),\n",
    "            nn.Linear(emb_dim, 128),\n",
    "            nn.Tanh(),\n",
    "            nn.Linear(128, len(classes))\n",
    "        )\n",
    "    def forward(self, batch):\n",
    "        batch = as_matrix(batch, self.n_tokens)\n",
    "        batch = torch.tensor(batch, device=self.device, dtype=torch.int64)\n",
    "        \n",
    "        embedded = self.embedder(batch).transpose(1,2)\n",
    "        \n",
    "        features = self.encoder(embedded).flatten(1)\n",
    "        return self.final_predictor(features)\n",
    "    def predict(self, data):\n",
    "        answer = []\n",
    "        for item in data:\n",
    "            pred = torch.argmax(self.forward([item]))\n",
    "            answer.append(pred.detach().cpu().numpy())\n",
    "        return answer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "256cdf35c8e849f89e5107871fe817fe",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=0), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7198d09a1a9b434a92ec89e5e223dce1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 0\n",
      "train results:\n",
      "loss:  3.481501007080078\n",
      "accuracy: 0.1191957155863444\n",
      " val results:\n",
      "loss: 3.47387\n",
      "accuracy: 0.12177\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a44c688e55504eedbc10128fd5360948",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 1\n",
      "train results:\n",
      "loss:  3.447528076171875\n",
      "accuracy: 0.12216922442118326\n",
      " val results:\n",
      "loss: 3.47150\n",
      "accuracy: 0.11373\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "88123a0cecad4ccfbb3203e10026d946",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 2\n",
      "train results:\n",
      "loss:  3.448070780436198\n",
      "accuracy: 0.12159079710642497\n",
      " val results:\n",
      "loss: 3.46986\n",
      "accuracy: 0.12177\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "75c0559507604a008fdd274ff7d6aa02",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 3\n",
      "train results:\n",
      "loss:  3.4500890096028645\n",
      "accuracy: 0.1232271671295166\n",
      " val results:\n",
      "loss: 3.47134\n",
      "accuracy: 0.12177\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cae169bc374e4999b5da1aa686c3e0c5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 4\n",
      "train results:\n",
      "loss:  3.448822021484375\n",
      "accuracy: 0.12385441462198893\n",
      " val results:\n",
      "loss: 3.47436\n",
      "accuracy: 0.12177\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fbf8439764ba44af88afb9b4a7a1cce2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 5\n",
      "train results:\n",
      "loss:  3.4458824157714845\n",
      "accuracy: 0.1221617062886556\n",
      " val results:\n",
      "loss: 3.47212\n",
      "accuracy: 0.12177\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d70c017197c54dd1ae23bdddcd6476ae",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 6\n",
      "train results:\n",
      "loss:  3.4478937784830728\n",
      "accuracy: 0.11974409421284994\n",
      " val results:\n",
      "loss: 3.47149\n",
      "accuracy: 0.11373\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0a27d9ee7a1e41f181ed1ac499bec7e2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 7\n",
      "train results:\n",
      "loss:  3.446660359700521\n",
      "accuracy: 0.1225385586420695\n",
      " val results:\n",
      "loss: 3.46954\n",
      "accuracy: 0.12177\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b1719ff6db7e45f9b42f0e2c2a8ed895",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 8\n",
      "train results:\n",
      "loss:  3.4473388671875\n",
      "accuracy: 0.12267628510793051\n",
      " val results:\n",
      "loss: 3.46993\n",
      "accuracy: 0.12177\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d511f96700dd47dc94349e7f73c2ef4e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 9\n",
      "train results:\n",
      "loss:  3.446807098388672\n",
      "accuracy: 0.11999324162801107\n",
      " val results:\n",
      "loss: 3.47323\n",
      "accuracy: 0.12177\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9a2cfa8e78d8477abee4c227958b20ea",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 10\n",
      "train results:\n",
      "loss:  3.4469950358072916\n",
      "accuracy: 0.12001827557881674\n",
      " val results:\n",
      "loss: 3.47427\n",
      "accuracy: 0.12177\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e327a77b9f5b48b8b1b5c71832214987",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "epoch: 11\n",
      "train results:\n",
      "loss:  3.447948710123698\n",
      "accuracy: 0.12153194745381674\n",
      " val results:\n",
      "loss: 3.46866\n",
      "accuracy: 0.12177\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e7da84a2c2fe4fb8a170d3e0673b95e9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-58-7961710e579c>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     11\u001b[0m     \u001b[0;32mfor\u001b[0m \u001b[0mi\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mtqdm_notebook\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0menumerate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0miterate_minibatches\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mX_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch_size\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mBATCH_SIZE\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     12\u001b[0m         \u001b[0mpred\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbatch\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 13\u001b[0;31m         \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtensor\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbatch\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mint64\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdevice\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mDEVICE\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     14\u001b[0m         \u001b[0mloss\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcriterion\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpred\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     15\u001b[0m         \u001b[0maccuracy\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmean\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0margmax\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpred\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "model = ConvNetv6(n_tokens=maxL, num_classes=len(classes), n_filters=100, wv=wv_matrix, device=DEVICE).to(DEVICE)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n",
    "\n",
    "for epoch in tqdm_notebook(range(EPOCHS)):\n",
    "    \n",
    "    model.train()\n",
    "    epoch_loss = 0\n",
    "    epoch_accuracy = 0\n",
    "    iterations = 0\n",
    "    for i, batch in tqdm_notebook(enumerate(iterate_minibatches([X_train, y_train], batch_size=BATCH_SIZE))):\n",
    "        pred = model(batch[0])\n",
    "        y = torch.tensor(batch[1], dtype=torch.int64, device=DEVICE)\n",
    "        loss = criterion(pred, y)\n",
    "        accuracy = torch.mean((torch.argmax(pred, axis=-1).float() == y.float()).float())\n",
    "        optimizer.zero_grad()\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        epoch_loss += loss\n",
    "        epoch_accuracy += accuracy\n",
    "        iterations += 1\n",
    "    \n",
    "    print(f\"epoch: {epoch}\")\n",
    "    print(\"train results:\")\n",
    "    print(\"loss: \", epoch_loss.detach().cpu().numpy() / iterations)\n",
    "    print(\"accuracy:\", epoch_accuracy.detach().cpu().numpy() / iterations)    \n",
    "    print_metrics(model, [X_valid, y_valid])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "predicted_classes = model.predict(X_train)\n",
    "predicted_classes_valid = model.predict(X_valid)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/coder/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.\n",
      "  'precision', 'predicted', average, warn_for)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0.10825636838926078"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f1_score(y_train, predicted_classes, average=\"macro\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.07291165282491346"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f1_score(y_valid, predicted_classes_valid, average=\"macro\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
