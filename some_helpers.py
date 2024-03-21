import polars as pl
from typing import Dict, List, Tuple
import pandas as pd
pd.options.display.max_columns = None

country_city_words_dict = {
    'israel': {
        'jerusalem': ['the', 'boy', 'is', 'running'],
        'haifa': ['ice', 'the', 'cream'],
        'ramat gan': ['cold', 'ice'],
        'givat shemuel': ['tea']
    },
    'england': {
        'liverpool': ['running', 'on', 'the', 'ice'],
        'london': ['cold', 'water']
    }
}

parts = pd.DataFrame(
   data=[
        ['ice cream', 'ice', 1, 0],
        ['ice cream', 'cream', 0, 1]
    ],
    columns=['full', 'part', 'ind_first_part', 'ind_second_part']
)

def construct_hashtable(country_city_words_dict: Dict[str, Dict[str, List[str]]]):
    result = {}
    for country, country_data in country_city_words_dict.items():
        for city, city_data in country_data.items():
            for word in city_data:
                if word not in result:
                    result[word] = {}
                if country not in result[word]:
                    result[word][country] = []
                if city not in result[word][country]:
                    result[word][country].append(city)
    return result


def namsim(lst1: List[str], lst2: List[str]):
    result = []
    for e1 in lst1:
        for e2 in lst2:
            if e2 in e1:
                result.append((e1, e2, 1.0))
    return result

def transform_matches_to_dict(matches: List[Tuple[str, str, float]]):
    result = {}
    for e1, e2, s in matches:
        if e1 not in result:
            result[e1] = []
        result[e1].append((e2, s))
    return result

def find_matches(parts_pd: pd.DataFrame, full_part_col: str, candidates: List[str]):
    all_full_parts = parts_pd[full_part_col].tolist()
    matches = namsim(all_full_parts, candidates)
    matches_dict = transform_matches_to_dict(matches)

    return matches_dict

def summarize_matches(parts_pd: pd.DataFrame, namsim_matches_dict: Dict[str, List[Tuple[str, float]]],
                      words_hashtable: Dict[str, Dict[str, List[str]]],
                      country_city_words_dict: Dict[str, Dict[str, List[str]]]):
    summarized_data = []
    for _, row in parts_pd.iterrows():
        full_part = row['full']
        if full_part in namsim_matches_dict:
            ind_first = row['ind_first_part']
            ind_second = row['ind_second_part']
            curr_part = row['part']
            for match_word, score in namsim_matches_dict[full_part]:
                for country, country_cities in words_hashtable[match_word].items():
                    for city in country_cities:
                        city_words = country_city_words_dict[country][city]
                        curr_part_matches = namsim([curr_part], city_words)
                        for curr_part_match in curr_part_matches:
                            appended_city_match = [full_part, curr_part, country, city, ind_first, ind_second,
                                                   curr_part_match[1], curr_part_match[2]]
                            summarized_data.append(appended_city_match)
    summarized_data_pd = pd.DataFrame(data=summarized_data,
                                      columns=['full_part', 'part', 'country', 'city', 'ind_first_part',
                                               'ind_second_part', 'match_word', 'score'])
    return summarized_data_pd

words_hashtable = construct_hashtable(country_city_words_dict)
namsim_matches_dict = find_matches(parts_pd=parts, full_part_col='full', candidates=[word for word in words_hashtable.keys()])
summarize_matches_pd = summarize_matches(parts_pd=parts, namsim_matches_dict=namsim_matches_dict, words_hashtable=words_hashtable,
                        country_city_words_dict=country_city_words_dict)

print(pl.DataFrame(summarize_matches_pd))

