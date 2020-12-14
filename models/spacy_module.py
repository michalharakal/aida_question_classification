import spacy

en_nlp = spacy.load("en_core_web_md")


def process_question(question, en_nlp=en_nlp):
    en_doc = en_nlp(question)
    sent_list = list(en_doc.sents)
    sent = sent_list[0]
    wh_bi_gram = []
    root_token = ""
    wh_pos = ""
    wh_nbor_pos = ""
    wh_word = ""
    for token in sent:
        if token.tag_ == "WDT" or token.tag_ == "WP" or token.tag_ == "WP$" or token.tag_ == "WRB":
            wh_pos = token.tag_
            wh_word = token.text
            wh_bi_gram.append(token.text)
            wh_bi_gram.append(str(en_doc[token.i + 1]))
            wh_nbor_pos = en_doc[token.i + 1].tag_
        if token.dep_ == "ROOT":
            root_token = token.tag_

    # return f'{wh_pos}|{wh_word}|{wh_bi_gram}|{wh_nbor_pos}|{root_token}'
    return f'{wh_pos}|{wh_word}|{wh_nbor_pos}|{root_token}'



def main():

    question01 = 'How did serfdom develop in and then leave Russia ?'
    qclass01 = 'DESC'

    process_question(question01)


if __name__ == '__main__':
    main()
