import pickle
import os
from collections import defaultdict


filtered_marvel_names =  [
 "Ender's Game: Speaker for the Dead",
 "Ender's Game Graphic Novel",
 "Dark Tower: The Gunslinger - The Little Sisters of Eluria",
 "Dark Tower: The Gunslinger: The Man in Black",
 "Dark Tower: The Gunslinger- The Battle of Tull",
 "Dark Tower - the Gunslinger: The Way Station",
 "Stephen King's Dark Tower: The Gunslinger - Last Shots",
 "Pride and Prejudice (Marvel Illustrated)",
 "Oz: The Wonderful Wizard of Oz"
 "Dark Tower: The Gunslinger Born"
 ]
filtered_marvel_ASINs = {
"Ender's Game: Speaker for the Dead":
 [
 "0785135863", #hardcover
    "B00PSN1FZS" #kindle
    ],
"Ender's Game Graphic Novel":
[
    "078518533X"
],

"Dark Tower: The Gunslinger - The Little Sisters of Eluria":
[
"0785149325", #paperback
"0785149317",#hardcover
 "B00AWR00HE"#kindle
],
"Dark Tower: The Gunslinger: The Man in Black":
[
# ,
    "0785149384", #paperback
 "0785149376", #hardcover
    "B00AWR00I8" #kindle
],
"Dark Tower: The Gunslinger- The Battle of Tull":
[
 #  ,
    "0785149341", #paperback
 "0785149333", #hardcover
"B00AWR00HY" #Kindle
],
"Dark Tower - the Gunslinger: The Way Station":
[
 #  ,
"0785149368", #paperback
 "078514935X", #hardcover
    "B00AWR015U" #kindle
],

"Stephen King's Dark Tower: The Gunslinger - Last Shots":
[

    "0785149414",#paperback
],

"Pride and Prejudice (Marvel Illustrated)":
[
"0785139168", #paperback
 "078513915X", #hardcover
    "B00CKWNMM4" #kindle
],

"Oz: The Wonderful Wizard of Oz":
[
 "0785145907", #paperback
    "0785154477" #hardcover
],

"Civil War":
[
 "0785145907", #paperback
    "0785154477" #hardcover
],

"Dark Tower: The Gunslinger Born":
[
 #
 "0785121447", #hardcover/single issue?
    "B00ZO97QQI", #kindle
    "1302906577" #paperback/trade?
],

}


def input_unique_reviews(version=None, master_book_list = None, books=None):

    for book_reviews in books:
        new_review = books[book_reviews][0]
        new_review_user = new_review[0]
        new_review_asin = new_review[1]
        in_master_list = False
        for key in master_book_list:
            for existing_book in master_book_list[key]:
                existing_review = master_book_list[key][existing_book][0]
                existing_review_user = existing_review[0]
                existing_review_asin = existing_review[1]
                if new_review_user == existing_review_user and new_review_asin == existing_review_asin:
                    in_master_list = True
                    break
            if in_master_list:
                break


        if (not in_master_list):
            master_book_list[version][book_reviews] = books[book_reviews]

def filter_by_specific_ASIN(master_book_list = None, filter_dict=None):
    num_filtered = 0
    for key in master_book_list:
        for existing_book in master_book_list[key]:
            printed_yet = False
            for book_to_filter in filter_dict:
                book_asins = filter_dict[book_to_filter]
                if len(master_book_list[key][existing_book]) > 800 and not printed_yet:
                    print(existing_book)
                    print("review amounts = " + str(len(master_book_list[key][existing_book])))
                    printed_yet = True
                if existing_book in book_asins:
                    #filter out non-related books
                    filtered_reviews = []
                    print(existing_book)
                    original_num = len(master_book_list[key][existing_book])
                    print("original # reviews: " + str(original_num))
                    for review in master_book_list[key][existing_book]:
                        specific_review_asin = review[1]
                        if specific_review_asin in book_asins:
                            filtered_reviews.append(review)
                    master_book_list[key][existing_book] = filtered_reviews
                    new_num = len(master_book_list[key][existing_book])
                    print("filtered # reviews: " + str(new_num))
                    print()
                    num_filtered += original_num - new_num
    print(num_filtered)




DC_user_item_pairs = defaultdict(lambda: defaultdict(list))
Marvel_user_item_pairs = defaultdict(lambda: defaultdict(list))
num_dc = 0
num_marvel = 0
cwd = os.path.abspath(os.path.curdir)
directory = os.path.join(cwd, "agglomerated")
DC_paperback_fname = os.path.join(directory, "DC_Paperback_UserItemReviewsALL_agglomerated.ratings")
Marvel_paperback_fname = os.path.join(directory, "Marvel_Paperback_UserItemReviewsALL_agglomerated.ratings")



DC_user_item_pairs["Paperback"] = pickle.load(open(DC_paperback_fname, 'rb'))
Marvel_user_item_pairs["Paperback"] = pickle.load(open(Marvel_paperback_fname, 'rb'))
del Marvel_user_item_pairs["Paperback"]['0785122370'] # remove duplicate Civil War entry
#
# #load reviews and deduplicate
# for version in ["Hardcover",
#                 "Kindle",
#                 "Omnibus", "Collector or Limited Edition"]:
#     for publisher in [
#                         # "DC",
#                      "Marvel"
#                       ]:
#         books = pickle.load(open(os.path.join(directory,"{publisher}_{version}_UserItemReviewsALL_agglomerated.p".format(publisher=publisher, version=version)), "rb"))
#         if publisher == "Marvel":
#             input_unique_reviews(version, master_book_list=Marvel_user_item_pairs, books=books)
#         else:
#             input_unique_reviews(version, master_book_list=DC_user_item_pairs, books=books)
#
#
# #checkpoint the deduplicated reviews
#
# for version in ["Paperback",
#                 "Hardcover",
#                 "Kindle",
#                 "Omnibus", "Collector or Limited Edition"]:
#     for publisher in [
#         # "DC",
#                       "Marvel"
#                       ]:
#         filename = "{publisher}_{version}_UserItemReviews.duplicatesRemoved".format(publisher=publisher, version=version)
#
#         if publisher == "DC":
#             item_to_save = DC_user_item_pairs[version]
#         else:
#             item_to_save = Marvel_user_item_pairs[version]
#         pickle.dump(item_to_save, open(os.path.join(directory,filename), 'wb'))
# exit(5)
#remove the pooled reviews that are not at all related to the specific item being reviewed
filter_by_specific_ASIN(master_book_list = Marvel_user_item_pairs, filter_dict=filtered_marvel_ASINs)

#checkpoint the filtered, deduplicated reviews
for version in ["Paperback",
                # "Hardcover",
                # "Kindle",
                # "Omnibus", "Collector or Limited Edition"
                ]:
    for publisher in ["Marvel"
                      ]:
        filename = "{publisher}_{version}_UserItemReviews.duplicatesAndUnrelatedRemoved".format(publisher=publisher, version=version)


        item_to_save = Marvel_user_item_pairs[version]
        pickle.dump(item_to_save, open(filename, 'wb'))
