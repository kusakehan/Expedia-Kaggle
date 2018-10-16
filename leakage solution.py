# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:47:08 2017

@author: wenja
"""
import numpy as np
import pandas as pd
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict

def prepare_arrays_match():
    print('Preparing arrays...')
    f = open("train_n.csv","r")
    f.readline()
    
    best_hotels_od_ulc = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest = defaultdict(lambda: defaultdict(int))
    popular_hotel_cluster = defaultdict(int)
    total = 0
    count_empty = 0
    
    #calc counts
    while 1:
        line = f.readline().strip()
        total += 1

        if total % 4000000 == 0:
            print('Read {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        user_location_city = arr[4]
        orig_destination_distance = arr[5]
        srch_destination_id = arr[13]
        is_booking = int(arr[15])
        hotel_country = arr[17]
        hotel_market = arr[18]
        hotel_cluster = arr[19]

        append_1 = 3 + 17*is_booking

        if user_location_city != '' and orig_destination_distance != '':
            best_hotels_od_ulc[(user_location_city, orig_destination_distance)][hotel_cluster] += append_1

        if srch_destination_id != '' and hotel_country != '' and hotel_market != '':
            best_hotels_search_dest[(srch_destination_id, hotel_country, hotel_market)][hotel_cluster] += append_1
        else:
            count_empty += 1

        popular_hotel_cluster[hotel_cluster] += append_1

    f.close()
    print('Empty: ', count_empty)
    return best_hotels_od_ulc, best_hotels_search_dest, popular_hotel_cluster


def gen_submission(best_hotels_search_dest, best_hotels_od_ulc, popular_hotel_cluster):
    print('Generate submission...')
    path = 'match_pred.csv'
    out = open(path, "w")
    f = open("test_n.csv", "r")
    f.readline()
    total = 0
    out.write("id,hotel_cluster\n")
    topclasters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))

    while 1:
        line = f.readline().strip()
        total += 1

        if total % 3000000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        id = arr[0]
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        srch_destination_id = arr[14]
        hotel_country = arr[17]
        hotel_market = arr[18]

        out.write(str(id) + ',')
        filled = []

        s1 = (user_location_city, orig_destination_distance)
        if s1 in best_hotels_od_ulc:
            d = best_hotels_od_ulc[s1]
            topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])

        s2 = (srch_destination_id, hotel_country, hotel_market)
        if s2 in best_hotels_search_dest:
            d = best_hotels_search_dest[s2]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(' ' + topitems[i][0])
                filled.append(topitems[i][0])

        for i in range(len(topclasters)):
            if topclasters[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            out.write(' ' + topclasters[i][0])
            filled.append(topclasters[i][0])

        out.write("\n")
    out.close()
    print('Completed!')
    
best_hotels_od_ulc, best_hotels_search_dest, popular_hotel_cluster = prepare_arrays_match()
gen_submission(best_hotels_search_dest, best_hotels_od_ulc, popular_hotel_cluster)


#get 0.37905 test score
#training score

def gen_trainpred(best_hotels_search_dest, best_hotels_od_ulc, popular_hotel_cluster):
    print('Generate train error...')
    path = 'train_pred.csv'
    out = open(path, "w")
    f = open("train_n.csv", "r")
    f.readline()
    total = 0
    out.write("id,pred1,pred2,pred3,pred4,pred5\n")
    topclasters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))

    while 1:
        line = f.readline().strip()
        total += 1

        if total % 3000000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break
        id = total-1
        arr = line.split(",")
        user_location_city = arr[4]
        orig_destination_distance = arr[5]
        srch_destination_id = arr[13]
        hotel_country = arr[17]
        hotel_market = arr[18]
        

        out.write(str(id))
        filled = []

        s1 = (user_location_city, orig_destination_distance)
        if s1 in best_hotels_od_ulc:
            d = best_hotels_od_ulc[s1]
            topitems = nlargest(5, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(',' + str(topitems[i][0]))
                filled.append(topitems[i][0])

        s2 = (srch_destination_id, hotel_country, hotel_market)
        if s2 in best_hotels_search_dest:
            d = best_hotels_search_dest[s2]
            topitems = nlargest(5, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break
                out.write(',' + str(topitems[i][0]))
                filled.append(topitems[i][0])

        for i in range(len(topclasters)):
            if topclasters[i][0] in filled:
                continue
            if len(filled) == 5:
                break
            out.write(',' + str(topclasters[i][0]))
            filled.append(topclasters[i][0])

        out.write("\n")
    out.close()
    print('Completed!')

gen_trainpred(best_hotels_search_dest, best_hotels_od_ulc, popular_hotel_cluster)

train_pred = pd.read_csv("train_pred.csv")
predicted = train_pred.drop('id', axis=1)
actual = train.hotel_cluster

def mapk(actual, predicted):
    score = 0.0
    for i in range(5):
        score += np.sum(actual==predicted.iloc[:,i])/(i+1)
    score /= actual.shape[0]
    return score

# get a train score 0.74
