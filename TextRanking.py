import pprint
import pickle
import os
from collections import defaultdict
from multiprocessing import  Pool, freeze_support
from utils import find_files

import RAKE
import pytextrank
import json
import networkx as nx
import pylab as plt

cwd = os.path.abspath(os.path.curdir)
# "FilteredCompleteData"
directory = os.path.join(cwd, "FilteredCompleteData")


def stage_1_multiprocess(args):
    publisher = args[0]
    version = args[1]
    book = args[2]
    reviews = args[3]
    out_file_name =  "{publisher}_{version}_{asin}.Stage1".format(version=version, asin=book,
                                                                  publisher=publisher)
    stage_1_directory = os.path.join(directory, "Stage1Results")

    save_directory = os.path.join(stage_1_directory, out_file_name)
    if os.path.isfile(save_directory):  # already there
        return None
    print(book)
    fake_json = [{'id': user, 'text': review_text} for user, asin, title, review_text, timestamp in reviews]
    # pprint.pprint(books[book][0])
    fake_json_graph_dicts = []
    for graf in pytextrank.parse_doc(fake_json):
        graph_dict = graf._asdict()
        fake_json_graph_dicts.append([graph_dict])
    pickle.dump(fake_json_graph_dicts, open(save_directory, 'wb'))

def stage_2_multiprocess(args):
    fake_json_graph_dicts = args[0]
    ranks = args[1]
    thread_num = args[2]
    rl_fake_json = []
    for rl in pytextrank.normalize_key_phrases(fake_json_graph_dicts, ranks, stopwords=RAKE.SmartStopList()):
        print(pytextrank.pretty_print(rl))
        rl_fake_json.append([rl._asdict()])
    stage_2_filename = "{publisher}_{version}_textRank_{thread_num}_rl.Stage2.".format(version=version, thread_num=thread_num,
                                                                                     publisher=publisher)
    stage_2_out = os.path.join(directory, "Stage2Results", stage_2_filename)
    pickle.dump(rl_fake_json, open(stage_2_out, 'wb'))
if __name__ == '__main__':
    freeze_support()
    text_keyword_count_DC = defaultdict(int)
    text_keyword_count_Marvel = defaultdict(int)
    title_keyword_count_DC = defaultdict(int)
    title_keyword_count_Marvel = defaultdict(int)


    for version in ["Paperback",
                    "Hardcover",
                    "Kindle",
                    "Omnibus",
                    "Collector or Limited Edition"]:
        for publisher in [
            # "DC",
                           "Marvel"
                          ]:



            version_text_keyword_count = defaultdict(int)
            version_title_keyword_count = defaultdict(int)
            reviews_by_ASIN = defaultdict(list)
            filename = "{publisher}_{version}_UserItemReviews.final".format(version=version,
                                                                            publisher=publisher)
            final_filename = os.path.join(directory, filename)
            books = pickle.load(open(final_filename, 'rb'))

            #stage 1: concatenate reviews from all books into one graph
            stage_1_directory = os.path.join(directory, "Stage1Results")
            fake_json_graph_dicts = []  # stage 1 output
            # stage_1_files = find_files(stage_1_directory, "*.Stage1")
            # relevant_files = [file for files in stage_1_files if not(publisher in file and version in file)]
            # stage_1_args = [[publisher, version, book, books[book]] for book in books if ]
            # processes = Pool(processes=64)
            # processes.map_async(stage_1_multiprocess, stage_1_args)
            # processes.close()
            # processes.join()
            # stage_1_multiprocess(stage_1_args[1])
            # for book in books:
            #     texts_pooled = []
            #     titles_pooled = []

            # fake_json = [{'id': user, 'text': review_text} for user, asin, title, review_text, timestamp in books[book]]
            # print(book)
            # # pprint.pprint(books[book][0])
            # for graf in pytextrank.parse_doc(fake_json):
            #     graph_dict = graf._asdict()
            #     fake_json_graph_dicts.append([graph_dict])



            stage_1_filename = "{publisher}_{version}_textRank.grafDict".format(version=version, publisher=publisher)
            stage_1_out = os.path.join(stage_1_directory, "agglomerated", stage_1_filename)
            if not os.path.isfile(stage_1_out):
                stage_1_files = find_files(stage_1_directory, "*.Stage1")
                relevant_files = [file for file in stage_1_files if (publisher in file and version in file)]
                stage_1_args = []
                for book in books:
                    book_done = False
                    for file in relevant_files:
                        if book in file:
                            book_done = True
                    if not(book_done):
                        stage_1_args.append([publisher, version, book, books[book]])
                # stage_1_args = [[publisher, version, book, books[book]] for book in books]
                processes = Pool(processes=16)
                processes.map_async(stage_1_multiprocess, stage_1_args)
                processes.close()
                processes.join()
                for file in find_files(stage_1_directory, "*.Stage1"):
                    if publisher in file and version in file:  # ex. DC & paperback
                        with open(file, 'rb') as the_file:
                            try:

                                fake_json_graph_dicts.extend(pickle.load(the_file))
                            except (OSError):
                                print(str(file))
                if len(fake_json_graph_dicts) > 0:
                    pickle.dump(fake_json_graph_dicts, open(stage_1_out, 'wb'))
                else:
                    print("error in writing stage1 graphdict")
            else:
                fake_json_graph_dicts = pickle.load(open(stage_1_out, 'rb'))
            stage_1_a_filename = "{publisher}_{version}_textRank.graph".format(version=version, publisher=publisher)
            stage_1_a_out = os.path.join(stage_1_directory, "agglomerated", stage_1_a_filename)

            stage_1_b_filename = "{publisher}_{version}_textRank.rank".format(version=version, publisher=publisher)
            stage_1_b_out = os.path.join(stage_1_directory, "agglomerated", stage_1_b_filename)
            if not(os.path.isfile(stage_1_a_out) and os.path.isfile(stage_1_b_out)):
                graph, ranks = pytextrank.text_rank(fake_json_graph_dicts)
                pickle.dump(graph, open(stage_1_a_out, 'wb'))
                pickle.dump(ranks, open(stage_1_b_out, 'wb'))
            else:
                graph = pickle.load( open(stage_1_a_out, 'rb'))
                ranks = pickle.load( open(stage_1_b_out, 'rb'))

            # pytextrank.render_ranks(graph, ranks)

            #stage 2: normalize key phrases

            stage_2_directory = os.path.join(directory, "Stage2Results")
            stage_2_files = find_files(stage_2_directory, "*.Stage2")
            stage_2_filename = "{publisher}_{version}_textRank.normalizedKeyPhrases.".format(version=version,
                                                                                             publisher=publisher)
            stage_2_out = os.path.join(directory, "Stage2Results", "agglomerated", stage_2_filename)
            rl_fake_json = []  # stage 2 output
            if not os.path.isfile(stage_2_out):
                counter = 0
                for rl in pytextrank.normalize_key_phrases(fake_json_graph_dicts, ranks,
                                                           stopwords=RAKE.SmartStopList()):
                    # print(pytextrank.pretty_print(rl))
                    rl_fake_json.append([rl._asdict()])
                    stage_2_rl_filename = "___{publisher}_{version}_textRank_{thread_num}_rl.Stage2.".format(version=version,
                                                                                                          thread_num=counter,
                                                                                                          publisher=publisher)
                    stage_2_rl_out = os.path.join(directory, "Stage2Results", stage_2_rl_filename)
                    pickle.dump([rl._asdict()], open(stage_2_rl_out, 'wb'))
                    counter += 1
                # stage_2_args = [[fake_json_graph_dict, ranks, i]
                #                 for i, fake_json_graph_dict in
                #                 zip([k for k in range(0, len(fake_json_graph_dicts))],
                #                     fake_json_graph_dicts)]
                # for arg in stage_2_args:
                #     stage_2_multiprocess(arg)
                # processes = Pool(processes=64)
                # processes.map_async(stage_2_multiprocess, stage_2_args)
                # processes.close()
                # processes.join()
                if len(rl_fake_json) == 0:
                    for file in find_files(stage_2_directory, "*.Stage2"):
                        if publisher in file and version in file:  # ex. DC & paperback
                            rl_fake_json.extend(pickle.load(open(file, 'rb')))
                else:
                    pass
                pickle.dump(rl_fake_json, open(stage_2_out, 'wb'))

            else:
                rl_fake_json = pickle.load(open(stage_2_out, 'rb'))





            #stage 3: get top sentences ("Calculate a significance weight for each sentence,
            # using MinHash to approximate a Jaccard distance from key phrases determined by TextRank")
            # -https://github.com/ceteri/pytextrank/blob/master/example.ipynb

            kernel = pytextrank.rank_kernel(rl_fake_json)
            sentences_fake_json = []  # stage 3 output
            for s in pytextrank.top_sentences(kernel, fake_json_graph_dicts):
                sentence_dict = s._asdict()
                sentences_fake_json.append([sentence_dict])
                print(pytextrank.pretty_print(sentence_dict))

            stage_3_filename = "{publisher}_{version}_textRank.topSentences.".format(version=version,
                                                                                     publisher=publisher)
            stage_3_out = os.path.join(directory, "Stage3Results", "agglomerated", stage_3_filename)
            pickle.dump(fake_json_graph_dicts, open(stage_3_out, 'wb'))

            #stage 4: generate a summary of the entire set of books
            phrases = ", ".join(set([p for p in pytextrank.limit_keyphrases(rl_fake_json, phrase_limit=12)]))
            sent_iter = sorted(pytextrank.limit_sentences(sentences_fake_json, word_limit=150), key=lambda x: x[1])
            s = []

            for sent_text, idx in sent_iter:
                s.append(pytextrank.make_sentence(sent_text))

            graf_text = " ".join(s)
            print("**excerpts:** %s\n\n**keywords:** %s" % (graf_text, phrases,))

            stage_4_filename = "{publisher}_{version}_textRank.summaryText.".format(version=version,
                                                                                    publisher=publisher)
            stage_4_out = os.path.join(directory, "Stage4Results", "agglomerated", stage_4_filename)
            pickle.dump(fake_json_graph_dicts, open(stage_4_out, 'wb'))


            #stage 5: show graph
            # nx.draw(graph, with_labels=True)
            # plt.show()


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
