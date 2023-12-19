# CollegeExplorer
Streamlit web app dashboard for exploring colleges based on preferences. Filter for colleges that meet your input preferences and then read summarized reviews from students at the colleges of interest

COLLEGE EXPLORER WEB APP: https://collegeexplorer.streamlit.app/

### How To Use College Explorer Dashboard:
1. Select personal preferences from side menu bar in order to filter universities based on Location, Institution, Cost, Academics, Mission, etc.
2. Press the "GET COLLEGES" button to produce a list of reccomended schools (sourced from CollegeScorecard database)
3. Select a few colleges to find out more about what studentsa have to say
4. Press the "Lock In Selections" button to finalize the list of schools you are interested in
   _To change your slections after they are locked in, simply press the "Clear Selected" button_
6. Press the "Get Student Review Summaries" to generate summaries detailing the main Pros and Cons as described by students (based on reviews sourced from RateMyProfessor.com)
   _NOTE: this can take a while as school summaries that do not already exist in the app database, will be fetched and generated in real time_

## Project Background
In the increasingly competitive era of higher education It is easy to get trapped by the pressure of prestigious, institutions and rigorous academia. Amidst a competitive academic landscape many young adults carry the misconception that big-name, ivy league universities are the best option. However, in reality the most known school is not always the best choice on both a personal as well as an academic level. Instead, individuals should seek schools that align well with their personal interests. This project aims to encourage individuals to evaluate preferences and priorities to narrow their search and set their sites on achievable, personalized applications. The College Explorer, web app, combines meta-like preferences, such as location, tuition cost, and institution-type (sourced from the US Department of Education's ColleegScorecard database), with first hand student reviews sourced from ratemyprofessor.com. The app leverages sentiment analysis text classification combined with GPT summarization to explore what students have to say about colleges of interest. In this way, individuals can make informed decisions in their application process. 
To explore the system in greater detail, check out the full project exploration at: https://editor.zyro.com/AE0566Z4o2cny4kg/preview

## System Overview
![IMG_0380](https://github.com/taliamora/CollegeExplorer/assets/97256085/ca13ee4b-9694-46d0-8538-37bec97ca7a4)


## Data Pipeline
![IMG_0381](https://github.com/taliamora/CollegeExplorer/assets/97256085/e348ffe9-3297-4a38-91c8-063ed6baf669)


_NOTE: The Sentiment Analysis LSTM was trained using FastText embeddings, however the vector file is substantially large and therefore was excluded from this project repo. The original vectors can be downloaded directly here:
https://fasttext.cc/docs/en/english-vectors.html_
