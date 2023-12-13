import re
import requests
import json
import base64
import os
import datetime
from bs4 import BeautifulSoup  
from functools import total_ordering
import numpy as np

current_path = os.path.dirname(__file__)

with open(os.path.join(current_path, "json/header.json"), 'r') as f:
    headers = json.load(f)
    
with open(os.path.join(current_path, "json/universityquery.json"), 'r') as f:
    university_query = json.load(f)

with open(os.path.join(current_path, "json/schoolquery.json"), 'r') as f:
    school_query = json.load(f)    

@total_ordering
class University:
    """Represents a school."""

    def __init__(self, school_id: int):
        """
        Initializes a school to the school id.

        :param school_id: The school's id.
        """

        self.id = school_id
        self.name = self._get_name()
        self._get_review_info(school_id)
        

    def _get_name(self):
        url = "https://www.ratemyprofessors.com/campusRatings.jsp?sid=%s" % self.id
        page = requests.get(url)
        school_names = re.findall(r'"legacyId":%s,"name":"(.*?)"' % self.id, page.text)
        if school_names:
            school_name = str(school_names[0])
        else:
            raise ValueError('Invalid school id or bad request.')

        return school_name
    
    

    def _get_review_info(self, school_id: int):
        headers["Referer"] = "https://www.ratemyprofessors.com/campusRatings.jsp?sid=%s" % school_id
        school_query["variables"]["id"] = base64.b64encode(("School-%s" % school_id)
                                                          .encode('ascii')).decode('ascii')
        data = requests.post(url="https://www.ratemyprofessors.com/graphql", json=school_query, headers=headers)

        if data is None or json.loads(data.text)["data"]["node"] is None:
            raise ValueError("University not found with that id or bad request.")

        school_data = json.loads(data.text)["data"]["node"]
        dept_data = school_data["departments"]


        self.departments = []
        for d in dept_data:
            self.departments.append(Department(school=self, name=d["name"]))

    
        self.num_ratings = school_data["numRatings"]
    
        
    
    def get_reviews(self):
            """
            Returns a list of ratings for the university.

            :return: A list of ratings for the university.
            """
            if self.num_ratings == 0:
                return []

            headers["Referer"] = "https://www.ratemyprofessors.com/campusRatings.jsp?sid=%s" % self.id
            university_query["variables"]["id"] = base64.b64encode(("School-%s" % self.id).encode('ascii')).decode('ascii')
            university_query["variables"]["count"] = self.num_ratings
            
            data = requests.post(url="https://www.ratemyprofessors.com/graphql", json=university_query, headers=headers)
    
            if data is None or json.loads(data.text)["data"]["node"]["ratings"]["edges"] is None:
                return []

            ratings_data = json.loads(data.text)["data"]["node"]["ratings"]["edges"]
            reviews = []

            for rating_data in ratings_data:
                rating = rating_data["node"]

                date = datetime.datetime.strptime(rating["date"][0:19], '%Y-%m-%d %H:%M:%S')
                scores = [rating["facilitiesRating"], rating["foodRating"],
                                       rating["happinessRating"], rating["socialRating"],
                                       rating["safetyRating"], rating["internetRating"], 
                                       rating["locationRating"], rating["opportunitiesRating"], 
                                       rating["reputationRating"], rating["clubsRating"]]

                        
                total_rating = round(np.mean(list(map(int, scores))),1)

                reviews.append(Review(rating=total_rating, comment=rating["comment"], date=date,
                                      facilities=rating["facilitiesRating"], food=rating["foodRating"],
                                      happiness=rating["happinessRating"], social=rating["socialRating"],
                                      safety=rating["safetyRating"], internet=rating["internetRating"], 
                                      location=rating["locationRating"], opportunities=rating["opportunitiesRating"], 
                                      reputation=rating["reputationRating"], clubs=rating["clubsRating"]))


            return reviews


    # What to return when class object is called/printed
    def __repr__(self): 
        return self.name

    # What to return when 2 school objects are compared (school1 < school2)
    # Who has more ratings?
    # USEFULL: Can be used to sort schools by number of ratings!
    def __lt__(self, other):
            return self.num_ratings < other.num_ratings
        
    # Do two schools have the same name, departments and school
    def __eq__(self, other):
            return (self.name, self.departments) == (other.name, other.departments)
        
    
class Department:
    """Represents a department at the university."""

    def __init__(self, school: University, name: str):
        """
        Initializes a course.

        :param professor: The professor who teaches the course.
        :param count: The number of ratings for the course.
        :param name: The name of the course.
        """
        self.school = school
        self.name = name



        
@total_ordering
class Review:
    """Represents a university review."""

    def __init__(self,  rating: float, comment: str, date: datetime,
                 facilities: int, food: int, happiness: int, social: int,
                 safety: int, internet: int, location: int, opportunities: int, 
                 reputation: int, clubs: int):
    
        """
        Initializes a review.
        """

        self.rating = rating
        self.date = date
        self.comment = BeautifulSoup(comment, "lxml").text
        self.facilities = facilities
        self.food = food
        self.happiness = happiness
        self.social = social
        self.safety = safety
        self.internet = internet
        self.location = location
        self.opportunities = opportunities
        self.reputation = reputation
        self.clubs = clubs
        
    # Order reviews by how recent they are
    def __lt__(self, other):
        return self.date > other.date


#["facilitiesRating","foodRating","happinessRating","socialRating", "safetyRating", "internetRating","locationRating","opportunitiesRating","reputationRating","clubsRating"]        
