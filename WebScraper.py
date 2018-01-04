import urllib.request
import requests
import re
from bs4 import BeautifulSoup
import pprint
import pickle
from time import sleep
import datetime
import calendar
from collections import defaultdict
import os.path
import os
from os import walk
from fnmatch import filter as fnmatch_filter
from sys import stderr
import random
from multiprocessing import Pool, freeze_support
from fake_useragent import UserAgent
def find_files(file_path, extension):
    """'
     Recursively find files at path with extension; pulled from StackOverflow
    ''"""#

    for root, dirs, files in walk(file_path):
        for file in fnmatch_filter(files, extension):
            yield os.path.join(root, file)
def multiprocessing_reviews(args):
    # reviews_by_ASIN = args[0]
    asin = args[1]
    dict_with_potential_duplicates = args[4]
    publisher = args[5]
    version = args[6]

    cwd = os.path.abspath(os.path.curdir)
    reviews_by = defaultdict(list)
    # print(cwd)
    filename = "{publisher}_{version}_UserItemReviewsALL_{asin}.asin".format(version=version, publisher=publisher, asin=asin)
    directory = os.path.join(cwd,  "multiprocessingASINs")
    final_filename = os.path.join(directory, filename)
    if os.path.isfile(final_filename):#already there
        return None, None
    else:
        print(filename + " doesn't exist")
    user_re = re.compile(r'customer_review-([\d|A-Z]+)')
    rating_re = re.compile(r'(\d\.\d|\d) out of 5 stars')
    specificASIN_re = re.compile(r'product-reviews/([\d|A-Z]+)')

    total_reviews = 0
    total_reviews_asin = None
    skip_to_next_ASIN = False
    # if asin not in reviews_by_ASIN:
    #     reviews_by[asin] = []
    #     start = 1
    # else:
    #     original_reviews = reviews_by_ASIN[asin]
    #     start = int(len(original_reviews) / 10) + 1
    #     extra_reviews = int(len(original_reviews) % 10)
    #     reviews_by[asin] = reviews_by_ASIN[asin][:len(original_reviews) - extra_reviews]



        # print(len(reviews_by_ASIN[asin]))

    for i in range(1, 2000):  # 2000 => 100 data points
        #             asin = "0785149325"
        if i > 1:
            url = "https://www.amazon.com/product-reviews/{asin}/\
                    ref=cm_cr_arp_d_paging_btm_{page_num}\
                    ?ie=UTF8&reviewerType=all_reviews&pageSize=10&?formatType=all_formats&pageNumber={page_num}".format(
                page_num=i, asin=asin)
        else:
            url = "https://www.amazon.com/product-reviews/{asin}/\
                    ref=cm_cr_arp_d_viewopt_fmt\
                    ?ie=UTF8&reviewerType=all_reviews&formatType=all_formats&pageNumber=1".format(page_num=i, asin=asin)

        relevant_results_area = []
        readablePage = None
        times_reading_page = 0
        ua = UserAgent()
        while len(relevant_results_area) == 0 :
            sleep(random.randint(3, 5) + random.random())  # don't activate the bot flag by being too fast
            os_num = random.randint(4, 12)
            ver_num = random.randint(1, 12)
            chrome_ver = random.randint(1, 95)
            safari_ver = random.randint(1, 36)
            appleWebKit_ver = random.randint(1, 36)
            user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_{os}_{ver}) AppleWebKit/537.{appleWebKit_ver} (KHTML, like Gecko) Chrome/41.0.2280.{chrome_ver} Safari/537.{safari_ver}'.format(
                    os=os_num, ver=ver_num, chrome_ver=chrome_ver, safari_ver=safari_ver, appleWebKit_ver=appleWebKit_ver)
            # if times_reading_page > 1000:
            #     user_agent = ua.random
            headers = {

                'User-Agent': user_agent }

            with requests.get(url, headers=headers) as req:
                the_page = req.content
            readablePage = BeautifulSoup(the_page, "html.parser")
            if total_reviews_asin is None:
                review_count_area = readablePage.find_all(class_="a-size-medium totalReviewCount")[0]
                total_reviews_asin = int(review_count_area.text.strip().replace(",", ""))
                # print()
                if total_reviews_asin == 0:
                    print(filename + " has no reviews")
                    # pprint.pprint(readablePage)
                    # sleep(600)
                    return None, None
            relevant_results_area = readablePage.findAll("div", id="cm_cr-review_list")
            times_reading_page += 1
            #             pprint.pprint(relevant_results_area)

        # if times_reading_page >= 1000:
        #     pprint.pprint(readablePage)
        #     print("too much time reloading; restarting")
        #     finished_checking = False
        #     return reviews_by, finished_checking



        id_area = relevant_results_area[0].find_all(class_="a-section review")
        # print(len(id_area))

        for item in id_area:
            user_id = user_re.findall(str(item))[0]

            rating = item.find_all(class_="a-icon-alt")
            rating = float(rating_re.findall(rating[0].text.strip())[0])

            review_title = item.find_all(
                class_="a-size-base a-link-normal review-title a-color-base a-text-bold")
            review_title = review_title[0].text.strip()

            review_text = item.find_all(class_="a-size-base review-text")
            review = review_text[0].text.strip()

            ASIN_area = item.find_all(class_="a-size-mini a-link-normal a-color-secondary")

            reviewASIN = specificASIN_re.findall(
                str(ASIN_area))  # since pooled reviews include other ASINS e.g. paperback, single issues, etc.
            if len(reviewASIN) > 0:
                reviewASIN = reviewASIN[0]
            else:  # oddly, some reviews don't have asins associated but are verified?
                reviewASIN = asin
            time_area = item.find_all(class_="a-size-base a-color-secondary review-date")
            time = time_area[0].text.strip().replace("on ", "")
            time = datetime.datetime.strptime(time, '%B %d, %Y')
            timestamp = calendar.timegm(time.utctimetuple())
            if dict_with_potential_duplicates is None or i > 1:
                reviews_by[asin].append([user_id, reviewASIN, rating, review_title, review, timestamp])
            else:
                for umbrella_ASIN in dict_with_potential_duplicates:
                    _user = dict_with_potential_duplicates[umbrella_ASIN][0]
                    _ASIN = dict_with_potential_duplicates[umbrella_ASIN][1]
                    if _user == user_id and _ASIN == reviewASIN:
                        # duplicate reviews, skip to next ASIN
                        skip_to_next_ASIN = True
                        break
                if skip_to_next_ASIN:
                    break
                reviews_by[asin].append([user_id, reviewASIN, rating, review_title, review, timestamp])

                #                 user_product_reviews.append([asin, user_id, reviewASIN, review_title, review, timestamp])
                # original asin first (one used to index in the first place and for which the reviewASIN
                # is a subset)
        if skip_to_next_ASIN:
            break
        else:
            num_reviews_on_page = len(reviews_by[asin]) - ((i - 1) * 10)
            print("page " + str(i) + " for ASIN " + asin + " has " + str(num_reviews_on_page) + " reviews")
            if num_reviews_on_page < 10:  # no more reviews, stop early
                break
    if not skip_to_next_ASIN:
        pickle.dump(reviews_by, open(final_filename, "wb"))

        print("ASIN " + asin + " has " + str(len(reviews_by[asin])) + " reviews")
        print("total reviews should be: " + str(total_reviews_asin))
        total_reviews += len(reviews_by[asin])

        #         pprint.pprint(user_product_reviews)


def get_reviews(ASINs, publisher=None, version=None, dict_with_potential_duplicates=None, times_checked=1):

    user_re = re.compile(r'customer_review-([\d|A-Z]+)')
    rating_re = re.compile(r'(\d\.\d|\d) out of 5 stars')
    specificASIN_re = re.compile(r'product-reviews/([\d|A-Z]+)')

    reviews_by_ASIN = defaultdict(list)
    filename = "{publisher}_{version}_UserItemReviewsALL.p".format(version=version, publisher=publisher)
    cwd = os.path.abspath(os.path.curdir)
    final_filename = os.path.join(cwd, filename)
    directory = os.path.join(cwd, "multiprocessingASINs")
    if os.path.isfile(final_filename):
        reviews_by_ASIN = pickle.load(open(filename, 'rb'))
    num_done = 0
    filtered_args = [[rank, asin, rating, num_reviews, dict_with_potential_duplicates, publisher, version] for rank, asin, rating, num_reviews in ASINs if asin not in reviews_by_ASIN or len(reviews_by_ASIN[asin]) ==0]
    processes = Pool(processes=128)
    processes.map_async(multiprocessing_reviews, filtered_args)
    processes.close()
    processes.join()
    # for arg in filtered_args:
    #     multiprocessing_reviews(arg)
    # for file in find_files(directory, '*.asin'):
    #     if publisher in file and version in file: #ex. DC & paperback
    #         with pickle.load(open(file, 'rb')) as current_asin_reviews:
    #             for key in current_asin_reviews:
    #                 reviews_by_ASIN[key] = current_asin_reviews[key]
    # pickle.dump(reviews_by_ASIN, open(final_filename, 'wb'))
    print("{num_done} out of {num_books} complete".format(num_done=len(filtered_args), num_books=len(ASINs)))

    print(publisher + " has " + str("lots") + " reviews for different versions of " + version + " books")
    finished_checking = True
    return reviews_by_ASIN, finished_checking

if __name__ == '__main__':
    freeze_support()

    DC_user_item_pairs = defaultdict(lambda: defaultdict(list))
    Marvel_user_item_pairs = defaultdict(lambda: defaultdict(list))
    num_dc = 0
    num_marvel = 0
 #    for version in ["Paperback"]:
 #        for publisher in [#"DC",
 #                          "Marvel"]:
 #            finished_checking = False
 #            ASINS = pickle.load(open("{publisher}Asins_{version}.p".format(publisher=publisher, version=version), "rb"))
 #            times_checked = 1
 #            while (not finished_checking):
 #                reviews, finished_checking = get_reviews(ASINS, publisher=publisher, version=version, dict_with_potential_duplicates=None, times_checked=times_checked)
 #                times_checked +=1
 #            if publisher == "Marvel":
 #                num_marvel += len(reviews)
 #                Marvel_user_item_pairs[version] = reviews
 #            else:
 #                num_dc += len(reviews)
 #                DC_user_item_pairs[version] = reviews
 # #
 #
 #    exit(5)
    cwd = os.path.abspath(os.path.curdir)
    directory = os.path.join(cwd)
    DC_paperback_fname = os.path.join(directory, "DC_Paperback_UserItemReviewsALL.p")
    Marvel_paperback_fname = os.path.join(directory, "Marvel_Paperback_UserItemReviewsALL.p")

    DC_user_item_pairs["Paperback"] = pickle.load(open(DC_paperback_fname, 'rb'))
    Marvel_user_item_pairs["Paperback"] = pickle.load(open(Marvel_paperback_fname, 'rb'))

    print(cwd)
    for version in [#"Hardcover",
                    # "Kindle",
                    # "Omnibus",
                    "Collector or Limited Edition"]:
        for publisher in [#"DC",
                           "Marvel"
                          ]:
            ASINS = pickle.load(open("{publisher}Asins_{version}.p".format(publisher=publisher, version=version), "rb"))
            finished_checking = False
            if publisher == "Marvel":
                times_checked = 1
                while (not finished_checking):
                    reviews, finished_checking = get_reviews(ASINS, publisher=publisher, version=version,
                                                             dict_with_potential_duplicates=Marvel_user_item_pairs["Paperback"], times_checked=times_checked)
                    times_checked +=1
                num_marvel += len(reviews)
                Marvel_user_item_pairs[version] = reviews
            else:
                times_checked = 1
                while (not finished_checking):
                    reviews, finished_checking = get_reviews(ASINS, publisher=publisher, version=version,
                                                             dict_with_potential_duplicates=DC_user_item_pairs["Paperback"], times_checked=times_checked)
                    times_checked += 1

                num_dc += len(reviews)
                DC_user_item_pairs[version] = reviews

    # ASINS = pickle.load(open( "marvelAsins_{}.p".format(version), "rb" ) )
    #     reviews = get_reviews(ASINS, Marvel_user_item_pairs)
    #     num_marvel += len(reviews)
    #     Marvel_user_item_pairs[version] = reviews


    #     ASINS = pickle.load(open( "DCAsins_{}.p".format(version), "rb" ) )
    #     reviews = get_reviews(ASINS, DC_user_item_pairs)
    #     num_dc += len(reviews)
    #     DC_user_item_pairs[version] = reviews

    #     pickle.dump(DC_user_item_pairs, open( "DC_{}_UserItemReviews.p".format(version), "wb" ) )
    #     pickle.dump(Marvel_user_item_pairs, open( "Marvel_{}_UserItemReviews.p".format(version), "wb" ) )


    #
    # pickle.dump(DC_user_item_pairs, open("DC_ALL_UserItemReviews.p", "wb"))
    # pickle.dump(Marvel_user_item_pairs, open("Marvel_ALL_UserItemReviews.p", "wb"))

    # print(str(num_dc) + " DC Ratings")
    # print(str(num_marvel) + " Marvel Ratings")
    # print(str(num_dc + num_marvel) + "Total Ratings")
#only retain the review for the specific ASINs ; all others are not related
