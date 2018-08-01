# encoding=utf-8
import json
import itertools
import numpy as np
from kw import hmm
import traceback
import re
from kw.utils import logger
import random
from collections import Counter

with open('./char2prop.json', 'r', encoding='UTF-8') as load_f:
    CHAR2PROP = json.load(load_f)

STATE_DICT = {"".join(k): i for i, k in enumerate(itertools.product("BKEMN", "BMES"))}
MARK_SET = [STATE_DICT[s] for s in STATE_DICT if s[0] == "K"]
NUM_RE = "[\d〇一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾佰仟萬]+"
TRAIN_TEST_RATE = 0.90
CAND_MIN_FREQ = 10000
RANDOM_CHARS = [char for char in CHAR2PROP if CHAR2PROP[char]["freq"] > CAND_MIN_FREQ]


AMBIGUOUS_PREFIX = ["zh", "ch", "sh", "z", "c", "s", "l", "n", "r", "f", "h"]
AMBIGUOUS_POSTFIX = ["uang", "iang", "ing", "eng", "ang", "uan", "ian", "in", "en", "an"]
AMBIGUOUS_ALT = {
    "z": ["zh"], "zh": ["z"],
    "c": ["ch"], "ch": ["c"],
    "s": ["sh"], "sh": ["s"],
    "l": ["n", "r"], "n": ["l", "r"], "r": ["n", "l"],
    "f": ["h"], "h": ["f"],
    "an": ["ang"], "ang": ["an"],
    "en": ["eng"], "eng": ["en"],
    "in": ["ing"], "ing": ["in"],
    "ian": ["iang"], "iang": ["ian"],
    "uan": ["uang"], "uang": ["uan"]
}

def get_pinyin2chrs():
    pinyin2chrs = {}
    for char in CHAR2PROP:
        if CHAR2PROP[char]["freq"] > CAND_MIN_FREQ:
            pinyins = CHAR2PROP[char]["pinyin"]
            for pinyin in pinyins:
                if pinyin not in pinyin2chrs:
                    pinyin2chrs[pinyin] = []
                pinyin2chrs[pinyin].append(char)
    return pinyin2chrs

PINYIN2CHRS = get_pinyin2chrs()



def train_model(marked_sentence, method, param):
    sentence_array = [np.array([[CHAR2PROP[c]["id"]] for c in case["sentence"]]) for case in marked_sentence]
    state_array = [np.array([STATE_DICT[c] for c in case["mark"]]) for case in marked_sentence]
    hmm_model = hmm.DiscreteHMM(len(STATE_DICT), len(CHAR2PROP), param["iteration"])
    hmm_model.train_batch(sentence_array, state_array)
    model = {
        "method": "HMM",
        "model": hmm_model,
        "char2prop": CHAR2PROP,
        "state_dict": STATE_DICT,
        "mark_set": MARK_SET,
        "inv_state_dict": {v: k for k, v in STATE_DICT.items()}
    }
    return model


def keep_char(c):
    if c not in CHAR2PROP:  # 去掉不认识的字符
        return False
    if CHAR2PROP[c]["is_punc"] == 1:  # 不保留标点
        return False
    if CHAR2PROP[c]["is_common"] == 0:  # 不常见字变成简体字
        return CHAR2PROP[c]["to_simple"]
    if CHAR2PROP[c]["is_en"] == 1:  # 小写字符变大写
        return c.upper()
    return c


def clean_sentence(sentence):
    sentence = [keep_char(s) for s in sentence]
    return "".join([s for s in sentence if s])


def clean_sentence_list(sentence_list):
    sentence_list = [clean_sentence(s) for s in sentence_list]
    sentence_list = [s for s in sentence_list if s]
    return sentence_list


def num_re_kw(kw):
    return re.sub(NUM_RE, NUM_RE, kw)


def sort_autore_kw_list(kw_list):
    sorted_kw = []
    for kw in sorted(kw_list, key=lambda x: len(x), reverse=True):
        kw = num_re_kw(kw)
        if kw not in sorted_kw:
            sorted_kw.append(kw)
    return sorted_kw


def sort_kw_list(kw_list):
    return sorted(list(set(kw_list)), key=lambda x: len(x), reverse=True)


def phrase_mark(phrase, p_type):
    if len(phrase) <= 1:
        return [p_type + "S"] * len(phrase)
    else:
        return [p_type + "B"] + [p_type + "M"] * (len(phrase) - 2) + [p_type + "E"]


def marked_seq(sentence, kw_list, non_kw_list):
    marks = sentence
    for kw in kw_list:
        # mark keywords with "#", later will be replaced with "K"
        if NUM_RE in kw:
            marks = re.sub(kw, lambda x: "#" * len(x.group()), marks)
        else:
            marks = marks.replace(kw, "#" * len(kw))

    for non_kw in non_kw_list:
        # mark keywords blacklist with "$", later will be replaced with "S"
        if NUM_RE in kw:
            marks = re.sub(non_kw, lambda x: "$" * len(x.group()), marks)
        else:
            marks = marks.replace(kw, "#" * len(kw))

    marks = "".join(["X" if (s != "#" and s != "$") else s for s in list(marks)])
    marks = marks.replace("#", "K")
    marks = marks.replace("$", "S")

    if "K" not in marks:
        # mark non-keyword sentence rest elements with "N"
        marks = marks.replace("X", "N")
    else:
        # mark pre keywords part of sentence
        begin_k = marks.find("K")
        begin_s = marks.find("S")
        if begin_s == -1:
            begin_i = begin_k
        else:
            begin_i = min(begin_k, begin_s)
        marks = "B" * begin_i + marks[begin_i:]
        # mark post keywords part of sentence
        end_i = max(marks.rfind("K"), marks.rfind("S"))
        marks = marks[:end_i + 1] + "E" * (len(marks) - (end_i + 1))
        # mark rest part of sentence
        marks = marks.replace("X", "M")

    # split marks with same character
    marks = ("".join([s if s == marks[i + 1] else s + "_" for i, s in enumerate(marks[:-1])]) + marks[-1]).split("_")
    final_marks = []
    for m in marks:
        final_marks += phrase_mark(m, m[0])
    return final_marks

def ambiguous_pinyin(pinyin):
    pinyin_cand = [pinyin]
    for pre_fix in AMBIGUOUS_PREFIX:
        if pinyin[:len(pre_fix)] == pre_fix:
            for alt in AMBIGUOUS_ALT[pre_fix]:
                pinyin_cand.append(alt + pinyin[len(pre_fix):])
            break

    for post_fix in AMBIGUOUS_POSTFIX:
        if pinyin[len(pinyin) - len(post_fix):] == post_fix:
            for alt in AMBIGUOUS_ALT[post_fix]:
                pinyin_cand.append(pinyin[:len(pinyin) - len(post_fix)] + alt)
            break
    return pinyin_cand



def mutate_kw(sentence, mark, mutation_rate, use_a_pinyin):
    # only mutate [K]eyword part
    def pinyin_mutation(char):
        if CHAR2PROP[char]["is_en"]:
            return char
        cand_pinyin = []
        for p in CHAR2PROP[char]["pinyin"]:
            cand_pinyin += ambiguous_pinyin(p)
        cand_w = [char]
        for pinyin in cand_pinyin:
            cand_w += PINYIN2CHRS[pinyin] if pinyin in PINYIN2CHRS else []
        return random.choice(cand_w)

    def random_mutation(c):
        return random.choice(RANDOM_CHARS)

    phrase_list = []
    for c, m in zip(sentence, mark):
        if use_a_pinyin is True:
            phrase_list += pinyin_mutation(c) if m[0] == "K" and random.random() < mutation_rate else c
        else:
            phrase_list += random_mutation(c) if m[0] == "K" and random.random() < mutation_rate else c
    return "".join(phrase_list)


def auto_mark_sequence(sentence_list, keyword_list, nonkeyword_list, mutation_rate, use_a_pinyin):
    content = []
    logger.info("auto_mark_sequence: len(sentence_list) = %d" % len(sentence_list))
    for i, sentence in enumerate(sentence_list):
        mark = marked_seq(sentence, keyword_list, nonkeyword_list)
        sentence = mutate_kw(sentence, mark, mutation_rate, use_a_pinyin)
        content.append({"sentence": sentence,
                        "mark": mark})
        if i > 0 and i % 5000 == 0:
            logger.info("finished %d / %d" % (i, len(sentence_list)))
    return content

def test_model(model, marked_sentence):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for case in marked_sentence:
        true_kws = get_all_keywrods(case["sentence"], case["mark"])
        _, keywords, _ = decode(case["sentence"], model)
        predict_kw = max(keywords, key=lambda x: len(x)) if keywords else ""
        if predict_kw in true_kws:
            tp += 1
        elif len(predict_kw) == 0 and true_kws == []:
            tn += 1
        elif len(predict_kw) == 0 and true_kws != []:
            fn += 1
        elif predict_kw not in true_kws:
            fp += 1
        else:
            logger.error("this should not happen")

    percision = tp * 1.0 / (tp + fp + 0.01)
    recall = tp * 1.0 / (tp + fn + 0.01)
    accuracy = (tp + tn) * 1.0 / (tp + tn + fn + fp)
    return percision, recall, accuracy

def get_all_keywrods(sentence, mark):
    keywords = [[]]
    for i, w in enumerate(sentence):
        if mark[i] == "KS":
            keywords.append([w])
        elif mark[i] == "KB":
            keywords.append([w])
        elif mark[i] == "KM" or mark[i] == "KE":
            keywords[-1].append(w)
        else:
            keywords.append([])
    keywords = ["".join(k) for k in keywords if k]
    return keywords


def decode(sentence, model):
    sentence = clean_sentence(sentence)
    if model["method"] == "HMM":
        sentence_array = np.array([[CHAR2PROP[c]["id"] if c in CHAR2PROP else 0] for c in sentence.strip()])
        num_mark = model["model"].decode(sentence_array)
        mark = [model["inv_state_dict"][m] for m in num_mark]

    keywords = get_all_keywrods(sentence, mark)
    return sentence, keywords, mark



def kw_stat(model, sentence_list):
    kw_counter = Counter()
    for sentence in sentence_list:
        _, keywords, _ = decode(sentence, model)
        kw_counter.update(keywords)
    return kw_counter


def train(sentence, keywords, nonkeyword):
    try:
        train_material = {"sentence": ["你好请问活期盈产品是不是能随时将资金转出的呢1",
                                       "活期盈赎回有限额吗"],
                          "keyword": ["活期盈", "安赢A"],
                          "nonkeyword": ["客服", "人工"],
                          "train_param": {
                              "iteration": 10,
                              "auto_re_number": True,
                              "use_only_positive_case": True,
                              "train_set_mutaion_rate": 0.5,
                              "ambiguous_pinyin_mutation": True,
                          },
                          "method": "HMM"

                          }

        if sentence:
            train_material['sentence'] = sentence

        if keywords:
            train_material['keyword'] = keywords

        if nonkeyword:
            train_material['nonkeyword'] = nonkeyword

        print('sentence Len:{}, keyword len:{}, nonkeyword len{}'.format(len(sentence), len(keywords), len(nonkeyword)))

        sentence = clean_sentence_list(train_material["sentence"])
        param = train_material["train_param"]
        kw_list = clean_sentence_list(train_material["keyword"])
        non_kw_list = clean_sentence_list(train_material["nonkeyword"])
        method = train_material["method"]

        # if len(sentence) < 10 or len(kw_list) < 1:
        #     error_msg = "trian need at least 10 valid sentence and 1 keyword"
        #     update_status(app_id, model_id, "error", error_msg)
        #     return

        if method not in ["CRF", "HMM"]:
            error_msg = "unknown method: %s" % method
            return

        logger.info("sentence_count: %d, keyword_count: %d, keyword_blacklist_count: %d" %
                    (len(sentence), len(kw_list), len(non_kw_list)))

        # prepare trainning set and testing set.
        test_count = int((1 - TRAIN_TEST_RATE) * len(sentence))
        sorted_kw_list = sort_kw_list(
            kw_list)  # sort_autore_kw_list(kw_list) if param["auto_re_number"] is True else sort_kw_list(kw_list)
        sorted_non_kw_list = sort_autore_kw_list(non_kw_list) if param["auto_re_number"] is True else sort_kw_list(
            non_kw_list)

        train_set = auto_mark_sequence(sentence,
                                       sorted_kw_list,
                                       sorted_non_kw_list,
                                       param["train_set_mutaion_rate"],
                                       param["ambiguous_pinyin_mutation"])

        if param["use_only_positive_case"] is True:
            train_set = [s for s in train_set if ("KB" in s["mark"] or "KS" in s["mark"])]

        full_mutate_test_set = auto_mark_sequence(random.sample(sentence, test_count),
                                                  sorted_kw_list,
                                                  sorted_non_kw_list,
                                                  1.0,
                                                  param["ambiguous_pinyin_mutation"])
        half_mutate_test_set = auto_mark_sequence(random.sample(sentence, test_count),
                                                  sorted_kw_list,
                                                  sorted_non_kw_list,
                                                  0.5,
                                                  param["ambiguous_pinyin_mutation"])
        none_mutate_test_set = auto_mark_sequence(random.sample(sentence, test_count),
                                                  sorted_kw_list,
                                                  sorted_non_kw_list,
                                                  0,
                                                  param["ambiguous_pinyin_mutation"])

        logger.info("prepare train_set done, len(train_set) = %d" % len(train_set))

        # train_model
        model = train_model(train_set, method, param)

        # test_model
        _fp, _fr, _fa = test_model(model, full_mutate_test_set)
        _hp, _hr, _ha = test_model(model, half_mutate_test_set)
        _np, _nr, _na = test_model(model, none_mutate_test_set)

        # false_positive_keyword_enrich
        kw_counter = kw_stat(model, sentence)
        fp_enrich = [(k, v) for k, v in kw_counter.most_common() if k not in set(kw_list)]

        # test_result
        test_result = {
            "test_sentence_total": sum([len(test_set)
                                        for test_set in
                                        [full_mutate_test_set, none_mutate_test_set, half_mutate_test_set]]),
            "test_sentence_positive": sum([len([s for s in test_set if "K" in "".join(s["mark"])])
                                           for test_set in
                                           [full_mutate_test_set, none_mutate_test_set, half_mutate_test_set]]),
            "keywords_full_mutation_percision": _fp,
            "keywords_full_mutation_recall": _fr,
            "keywords_full_mutation_accuracy": _fa,
            "keywords_half_mutation_percision": _hp,
            "keywords_half_mutation_recall": _hr,
            "keywords_half_mutation_accuracy": _ha,
            "keywords_none_mutation_percision": _np,
            "keywords_none_mutation_recall": _nr,
            "keywords_none_mutation_accuracy": _na,
        }
        test_result["test_sentence_negative"] = test_result["test_sentence_total"] - test_result[
            "test_sentence_positive"]
        logger.info("train succeed, half_accuracy: %.2f" % test_result["keywords_half_mutation_accuracy"])

        # write_model
        write_model(model, fp_enrich, test_result)
        print(test_result)
        logger.info("Success!")

    except Exception as e:
        logger.warning("error @ train_celery, %s" % (traceback.format_exc()))

def write_model(app_id, model_id, model, fp_enrich, train_result):
    # model_meta = DB_APPID2MODEL_META.find_one({"$and": [{"app_id": app_id}, {"model_id": model_id}]})
    #
    # DB_FS.delete(ObjectId(model_meta["_material_id"]))
    # _model_id = DB_FS.put(pickle.dumps(model))
    #
    # model_meta.update({
    #     "status": "ready",
    #     "train_result": train_result,
    #     "_fp_enrich": fp_enrich,
    #     "_model_id": _model_id
    # })
    # DB_APPID2MODEL_META.update_one({"$and": [{"app_id": app_id}, {"model_id": model_id}]}, {"$set": model_meta})
    return

if __name__ == '__main__':
    import sys
    args = sys.argv
    sentencePath = args[1]
    keywordPath = args[2]

    print(args)

    with open(sentencePath, 'r', encoding='UTF-8') as sentenceFile:
        sentences = sentenceFile.readlines()

    with open(keywordPath, 'r', encoding='UTF-8') as keywordFile:
        keywords = keywordFile.readlines()

    train(sentences, keywords, [])