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

The College Explorer web app uses both data warehouse storage and real-time ETL throughout the system. First, User input is gathered via the Preferences dashboard on the web app's user interface. This user input is used to query school names from the US Department of Education CollegeScorecard database through an established GET API. The query then returns schools that fit within the user's given preferences. Again, the user provides input by selecting schools from the list which they wish to know more about.

At this point, there are two possible pathways. First, the data warehouse (BigQuery) is checked for the requested summaries. If the school's reviews have already been fetched and summarized, then the data is fetched directly from the cloud storage. However, if the data doesn't exist yet, the data is generated on the spot through the following ETL process:

1. Student reviews are retrieved from the school's evaluation page on Rate My Professor
2. The reviews are classified as either "positive" or "negative" by a trained sentiment analyses LSTM
3. The classified reviews are saved to a reviews data table in the data warehouse
4. OpenAI API's GPT-3.5 Turbo is used to generate summaries of each review grouping
5. The generated summaries are saved to a Summaries data table in the warehouse (along with a unique ID that links the summaries to the source reviews in the Reviews table)
6. Now, the web app attempts to fetch the reviews from the warehouse once again
7. Finally, the summaries are displayed to the user outlining the main pros and cons about the college

The following (low fidelity) diagram shows the pipeline:

![IMG_0381](https://github.com/taliamora/CollegeExplorer/assets/97256085/e348ffe9-3297-4a38-91c8-063ed6baf669)


_NOTE: The Sentiment Analysis LSTM was trained using FastText embeddings, however the vector file is substantially large and therefore was excluded from this project repo. The original vectors can be downloaded directly here:
https://fasttext.cc/docs/en/english-vectors.html_
