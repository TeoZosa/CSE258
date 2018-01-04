import pprint
import pickle
from collections import defaultdict
import os

from Assignment2.utils import find_files

cwd = os.path.abspath(os.path.curdir)
directory = os.path.join(cwd, "multiprocessingASINs")

asins_separate = list(find_files(directory, "*.asin"))
for version in ["Paperback",#
                "Hardcover",
                "Kindle",
                "Omnibus", "Collector or Limited Edition"
                ]:
    for publisher in [
        # "DC",
                      "Marvel"
                      ]:
        reviews_by_ASIN = defaultdict(list)
        filename = "{publisher}_{version}_UserItemReviewsALL_agglomerated.p".format(version=version, publisher=publisher)
        original_filename = "{publisher}_{version}_UserItemReviewsALL.p".format(version=version, publisher=publisher)

        final_filename = os.path.join(directory, filename)
        original_filename = os.path.join(directory, original_filename)
        print(original_filename)
        print(final_filename)
        if os.path.isfile(original_filename):  # already there
            reviews_by_ASIN = pickle.load(open(original_filename, 'rb'))
        for file in asins_separate:
            if publisher in file and version in file:  # ex. DC & paperback
                current_asin_reviews = pickle.load(open(file, 'rb'))
                for key in current_asin_reviews:
                    if key in reviews_by_ASIN:

                        if len(reviews_by_ASIN[key]) > 0 and len(reviews_by_ASIN[key]) != len(current_asin_reviews[key]):
                            print(len(reviews_by_ASIN[key]))
                            print(len(current_asin_reviews[key]))
                            input("Continue?")
                    reviews_by_ASIN[key] = current_asin_reviews[key]
        pickle.dump(reviews_by_ASIN, open(final_filename, 'wb'))
        # ASINS = pickle.load(open("{publisher}Asins_{version}.p".format(publisher=publisher, version=version), "rb"))
