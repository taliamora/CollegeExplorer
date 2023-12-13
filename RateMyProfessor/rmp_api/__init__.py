"""
ratemyprofessor

RateMyProfessor API: University Reviews
Scrapes student reviews and ratings for Universities (NOT PROFESSORS)

"""
import requests
import re
import json
import base64
import os
import pandas as pd

from RateMyProfessor.rmp_api.university import University


with open(os.path.join(os.path.dirname(__file__), "json/header.json"), 'r') as f:
    headers = json.load(f)


def get_uni_by_name(school_name: str, most_reviews = False):
    """
    Gets a School with the specified name.

    This only returns 1 school name, so make sure that the name is specific.
    For instance, searching "Ohio State" will return 6 schools,
    but only the first one will return by calling this method.

    :param school_name: The school's name.
    :param most_reviews: choose the university with the most amount of reviews (when multiple come up)
    :return: The school that match the school name. If no schools are found, this will return None.
    """
    schools = get_unis_by_name(school_name)
    max_school = None
    
    if most_reviews == True:
        for school in schools:
            if max_school is None or max_school.num_ratings < school.num_ratings:
                max_school = school
        return max_school
    
    else:
        if schools:
            return schools[0]
        else:
            return None


def get_unis_by_name(school_name: str):
    """
    Gets a list of Schools with the specified name.

    This only returns up to 20 schools, so make sure that the name is specific.
    For instance, searching "University" will return more than 20 schools, but only the first 20 will be returned.

    :param school_name: The school's name.
    :return: List of schools that match the school name. If no schools are found, this will return an empty list.
    """
    school_name.replace(' ', '+')
    url = "https://www.ratemyprofessors.com/search/schools?q=%s" % school_name
    page = requests.get(url)
    data = re.findall(r'"legacyId":(\d+)', page.text)
    school_list = []

    for school_data in data:
        try:
            school_list.append(University(int(school_data)))
        except ValueError:
            pass
    
    return school_list



# Get university reviews to a csv
def get_schools_reviews(school, output="lists"):
    comments = []
    ratings = []
    if type(school) == str:
         school = get_uni_by_name(school, most_reviews=True)
    reviews = school.get_reviews()
    for post in reviews:
        txt = post.comment
        score = post.rating
        if txt == 'Not Specified.':
            pass
        else:
            comments.append(txt)
            ratings.append(score)
    if output == "dataframe":
        df = pd.DataFrame(list(zip(ratings, comments)),columns=["Ratings", "Reviews"])
        return df
    elif output == "lists":
        return comments, ratings