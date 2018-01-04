import pprint
import pickle
import os
from collections import defaultdict

from Assignment2.utils import find_files


import RAKE
import pytextrank
import json
import networkx as nx
import pylab as plt

cwd = os.path.abspath(os.path.curdir)
directory = os.path.join(cwd, "agglomerated")#"FilteredCompleteData"

text_keyword_count_DC = defaultdict(int)
text_keyword_count_Marvel = defaultdict(int)
title_keyword_count_DC = defaultdict(int)
title_keyword_count_Marvel = defaultdict(int)

Rake = RAKE.Rake(RAKE.SmartStopList())
pprint.pprint(Rake.run("the cat is stinky"))

for version in ["Paperback",
                "Hardcover",
                "Kindle",
                "Omnibus",
                "Collector or Limited Edition"]:
    for publisher in ["DC",
                      "Marvel"
                      ]:
        version_text_keyword_count = defaultdict(int)
        version_title_keyword_count = defaultdict(int)
        reviews_by_ASIN = defaultdict(list)
        filename = "{publisher}_{version}_UserItemReviews.duplicatesRemoved".format(version=version, publisher=publisher)
        final_filename = os.path.join(directory, filename)
        books = pickle.load(open(final_filename, 'rb'))
        fake_json_graph_dicts = [] #stage 1 output
        for book in books:
            texts_pooled = []
            titles_pooled = []
            fake_json = [{'id': user, 'text': review_text}for user, asin, title, review_text, timestamp in books[book]]
            print(book)
            # pprint.pprint(books[book][0])
            for graf in pytextrank.parse_doc(fake_json):
                graph_dict = graf._asdict()
                fake_json_graph_dicts.append([graph_dict])
        stage_1_filename = "{publisher}_{version}_textRank.grafDict.".format(version=version, publisher=publisher)
        stage_1_out = os.path.join(directory, "TextRankStages", stage_1_filename)
        pickle.dump(fake_json_graph_dicts, open(stage_1_out, 'wb'))
        graph, ranks = pytextrank.text_rank(fake_json_graph_dicts)
        pytextrank.render_ranks(graph, ranks)
        rl_fake_json = [] #stage 2 output
        for rl in pytextrank.normalize_key_phrases(fake_json_graph_dicts, ranks, stopwords=RAKE.SmartStopList()):
            print(pytextrank.pretty_print(rl))
            rl_fake_json.append([rl._asdict()])

        stage_2_filename = "{publisher}_{version}_textRank.normalizedKeyPhrases.".format(version=version, publisher=publisher)
        stage_2_out = os.path.join(directory, "TextRankStages", stage_2_filename)
        pickle.dump(fake_json_graph_dicts, open(stage_2_out, 'wb'))
        
        kernel = pytextrank.rank_kernel(rl_fake_json)
        sentences_fake_json = [] #stage 3 output
        for s in pytextrank.top_sentences(kernel, fake_json_graph_dicts):
            sentence_dict = s._asdict()
            sentences_fake_json.append([sentence_dict])
            print(pytextrank.pretty_print(sentence_dict))

        stage_3_filename = "{publisher}_{version}_textRank.topSentences.".format(version=version,
                                                                                         publisher=publisher)
        stage_3_out = os.path.join(directory, "TextRankStages", stage_3_filename)
        pickle.dump(fake_json_graph_dicts, open(stage_3_out, 'wb'))

        phrases = ", ".join(set([p for p in pytextrank.limit_keyphrases(rl_fake_json, phrase_limit=12)]))
        sent_iter = sorted(pytextrank.limit_sentences(sentences_fake_json, word_limit=150), key=lambda x: x[1])
        s = []

        for sent_text, idx in sent_iter:
            s.append(pytextrank.make_sentence(sent_text))

        graf_text = " ".join(s)
        print("**excerpts:** %s\n\n**keywords:** %s" % (graf_text, phrases,))

        stage_4_filename = "{publisher}_{version}_textRank.summaryText.".format(version=version,
                                                                                         publisher=publisher)
        stage_4_out = os.path.join(directory, "TextRankStages", stage_4_filename)
        pickle.dump(fake_json_graph_dicts, open(stage_4_out, 'wb'))


        nx.draw(graph, with_labels=True)
        plt.show()
        input("exit?")
        exit(10)
        # pytextrank.json_iter()
        # graph, ranks =pytextrank.text_rank()

#             for user, asin, title, review_text, timestamp in books[book]:
#                 # title = review[2]
#                 # review_text = review[3]
#                 # texts_pooled.append(review_text)
#                 # titles_pooled.append(title)
#                 # RAKE_result_title = Rake.run(titles_pooled)
#                 # RAKE_result_text = Rake.run(texts_pooled)
#                 RAKE_result_title = Rake.run(title)
#                 RAKE_result_text = Rake.run(review_text)
#                 # textrank_words = keywords(review_text)
#                 # pprint.pprint(textrank_words)
#
#                 for keyword, score in RAKE_result_title:
#                     # keyword = result[0]
#                     # score = result[1]
#                     if publisher == "Marvel":
#                         title_keyword_count_Marvel[keyword] += score
#                     else:
#                         title_keyword_count_DC[keyword] += score
#                     version_title_keyword_count[keyword] += score
#                 for keyword, score in RAKE_result_text:
#                     # if score > 1:
#                     #     print(keyword + str(score))
#                     # keyword = result[0]
#                     # score = result[1]
#                     # if keyword ==
#                     if publisher == "Marvel":
#                         text_keyword_count_Marvel[keyword] = +score
#                     else:
#                         text_keyword_count_DC[keyword] = +score
#                     version_text_keyword_count[keyword] = +score
#         ordered_text_keywords = sorted(version_text_keyword_count, key=version_text_keyword_count.get, reverse=True)
#         ordered_text_keywords = [(key, version_text_keyword_count[key]) for key in ordered_text_keywords]
#         ordered_title_keywords = sorted(version_title_keyword_count, key=version_title_keyword_count.get, reverse=True)
#         ordered_title_keywords = [(key, version_title_keyword_count[key]) for key in ordered_title_keywords]
#
#         pprint.pprint(ordered_text_keywords[:10])
#         print()
#         print(ordered_text_keywords[:10])
#         pprint.pprint(ordered_title_keywords[:10])
#         # with open(os.path.join(directory, "{publisher}_{version}_RAKEd_title.txt".format(version=version, publisher=publisher)), 'w+') as file:
#             # file.write(pprint.pformat(ordered_title_keywords).encode('utf-8'))
#             # pprint.pprint(ordered_title_keywords, stream=file)
#
#         # with open(os.path.join(directory, "{publisher}_{version}_RAKEd_text.txt".format(version=version, publisher=publisher)), 'w+') as file:
#             # file.write(pprint.pformat(ordered_text_keywords).encode('utf-8').decode('utf-8'))
#             # pprint.pprint(ordered_text_keywords, stream=file)
#
#
#
#         books["RAKE: Title"] = version_title_keyword_count
#         books["RAKE: Text"] = version_text_keyword_count
#         out_filename = "{publisher}_{version}_UserItemReviews.RAKEd".format(version=version, publisher=publisher)
#         out_final_filename = os.path.join(directory, out_filename)
#         pickle.dump(books, open(out_final_filename, 'wb'))
#
# out_filename = "DC_UserItemReviews_Titles.RAKEd"
# out_final_filename = os.path.join(directory, out_filename)
# pickle.dump(title_keyword_count_DC, open(out_filename, 'wb'))
#
# out_filename = "DC_UserItemReviews_Text.RAKEd"
# out_final_filename = os.path.join(directory, out_filename)
# pickle.dump(text_keyword_count_DC, open(out_filename, 'wb'))
#
# out_filename = "Marvel_UserItemReviews_Titles.RAKEd"
# out_final_filename = os.path.join(directory, out_filename)
# pickle.dump(title_keyword_count_Marvel, open(out_filename, 'wb'))
#
# out_filename = "Marvel_UserItemReviews_Text.RAKEd"
# out_final_filename = os.path.join(directory, out_filename)
# pickle.dump(text_keyword_count_Marvel, open(out_filename, 'wb'))
# #
# #
# DC_ordered_text_keywords = sorted(text_keyword_count_DC    , key=text_keyword_count_DC .get, reverse=True)
# DC_ordered_text_keywords = [(key, DC_ordered_text_keywords[key]) for key in DC_ordered_text_keywords]
#
# DC_ordered_title_keywords = sorted(title_keyword_count_DC, key=text_keyword_count_DC .get, reverse=True)
# DC_ordered_title_keywords = [(key, DC_ordered_title_keywords[key]) for key in DC_ordered_title_keywords]
#
# #
# Marvel_ordered_text_keywords = sorted(text_keyword_count_Marvel, key=text_keyword_count_Marvel .get, reverse=True)
# Marvel_ordered_text_keywords = [(key, Marvel_ordered_text_keywords[key]) for key in Marvel_ordered_text_keywords]
#
# Marvel_ordered_title_keywords = sorted(title_keyword_count_Marvel, key=text_keyword_count_Marvel .get, reverse=True)
# Marvel_ordered_title_keywords = [(key, Marvel_ordered_title_keywords[key]) for key in Marvel_ordered_title_keywords]
#
# #
#
#
# # for publisher in [
# #     "Marvel",
# #                   "DC"]:
# #     if publisher == "Marvel":
# #         text_keywords = Marvel_ordered_text_keywords
# #         title_keywords = Marvel_ordered_title_keywords
# #     else:
# #         text_keywords = DC_ordered_text_keywords
# #         title_keywords = DC_ordered_title_keywords
# #     with open(os.path.join(directory, "{publisher}_ALL_RAKEd_title.txt".format(publisher=publisher)), 'w+') as file:
# #         file.write(str(pprint.pformat(title_keywords).decode('utf-8')))
# #
# #         # pprint.pprint(title_keywords, stream=file)
# #     with open(os.path.join(directory, "{publisher}_ALL_RAKEd_text.txt".format(publisher=publisher)), 'w+') as file:
# #         file.write(str(pprint.pformat(text_keywords).decode('utf-8')))
# #
# #         # pprint.pprint(text_keywords, stream=file)
    