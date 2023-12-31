import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import requests
import json
import os
from openai import OpenAI
from google.cloud import bigquery
from google.oauth2 import service_account
from DB.gcp_warehouse import fetch_summaries

# Initialize session state variables
if 'message' not in st.session_state:
    st.session_state.message = "No Recommendations Yet"

if 'edited_df' not in st.session_state:
    st.session_state.edited_df = pd.DataFrame()


if 'selected_list' not in st.session_state:
    st.session_state.selected_list = []

if 'sort' not in st.session_state:
    st.session_state.sort = "asc"


if 'should_summarize' not in st.session_state:
    st.session_state.should_summarize = False

if "edflg" not in st.session_state:
    st.session_state.edflg = False

if "cost_flag" not in st.session_state:
    st.session_state.cost_flag = False


# Load dataset
pref = pd.read_csv("preference_key.csv")

### GET SECRET ENVIRONMENT VARIABLES
KEY = st.secrets['SCORECARD_API_KEY']

# Connect to OpenAI
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
OpenAI_client = OpenAI()

# Connect to BigQuery Warehouse
credentials = service_account.Credentials.from_service_account_info(st.secrets['SERVICE_ACCOUNT_KEY'])
GCP_client = bigquery.Client(credentials=credentials)


# Streamlit title
#def main_page():
# Create three columns
top_col1, top_col2, top_col3 = st.columns([0.25,3,0.25])
with top_col2:
    st.title('📚 COLLEGE EXPLORER 📚')
st.header("", divider='rainbow')

st.markdown('''
        Set your preferences in the side bar to the left to get reccomended colleges. 
            ***Note: The selected preferences are exclusive, so only schools that fit all given preferences will be reccomended.***
        ''') 

# User preferences for college filtering
st.sidebar.header('Personal Preferences', divider="red")
st.sidebar.write("Select from as few or as many of the preferences below (Note: you can add as many or few selections as you would like)! If you don't feel strongly about any categories just leave them blank")  # Display text

### ------- LOCATION
st.sidebar.header('Location', divider="grey")
st.sidebar.write("This section pertains to location, surrounding & geographical preferences")



# State
select_states = st.sidebar.multiselect('Prefered State(s)', pref[pref['VAR']=="ST_FIPS"].LABEL)

# Region
select_region = st.sidebar.multiselect('Setting/Degree of Urbanization', pref[pref['VAR']=="LOCALE"].LABEL)



### ------- SCHOOL
st.sidebar.header('Institution Characteristics', divider="grey")
st.sidebar.write("This section contains charactersistics of colleges & their programs")

# Profit/Public/Private
multiselect_control = st.sidebar.multiselect('Institution Type', pref[pref['VAR']=="CONTROL"].LABEL, disabled = st.session_state.cost_flag)


# Admission
if st.sidebar.toggle('Filter based on Acceptance Rate', value=False):
    Admission = st.sidebar.slider('Admission Rate', 0, 100, 0, help="Select a MINIMUM rate of acceptance")
    st.sidebar.write(f'***College accepts at least **{Admission}%** of applicants***')
    rate = Admission/100
    admission = f"{rate}.."
else:
    admission = None



# Degrees
select_degree = st.sidebar.multiselect('Highest Degree Type Offered', pref[pref['VAR']=="HIGHDEG"].LABEL)

def change_control():
    st.session_state.cost_flag = not st.session_state.cost_flag

# Cost
if st.sidebar.toggle('Filter based on Tuition Cost', value=False, on_change=change_control):
    radio_control = st.sidebar.radio('Institution Type', pref[pref['VAR']=="CONTROL"].LABEL, disabled= not st.session_state.cost_flag)

    Cost = st.sidebar.slider(
        'Specify a maximum acceptable tuition cost (in Thousands of Dollars)',
        0, 100, 50, help="Values based on approximate cost of tuition & fees for 1 academic year (assuming out of state tuition)")
    st.sidebar.write(f'Annual Cost of Tuition < ${Cost}K')
    cost = 1000*Cost
    if radio_control=="Public":
        cost_pub = f"..{cost}"
    else:
        cost_pub = None
    if radio_control == "Private nonprofit" or radio_control == "Private for-profit":
        cost_priv = f"..{cost}"
    else:
        cost_priv = None

    select_control = [radio_control]
else:
    cost_priv = None
    cost_pub = None
    select_control = multiselect_control

    

# SAT
if st.sidebar.toggle('Filter based on SAT Score', value=False):
    SAT = st.sidebar.slider(
        'Average SAT Score of Admitted Students',
        400, 1600, help="Average total SAT score (made up of two 200-800 scored sections)")
    sat = f"..{SAT}"
else:
    sat = None

### ------- DEMOGRAPHIC
st.sidebar.header('Demographic & Mission', divider="grey")

# Religion
st.sidebar.write('Are you interested in schools that have a particular religeous affiliation?')
if st.sidebar.checkbox('Yes'):
    select_religion = st.sidebar.multiselect('Specify Religion (optional)', pref[pref['VAR']=="RELAFFIL"].LABEL)
else:
    select_religion = None


# Gender
st.sidebar.write("\nAre you looking for a Gender-Specific Institution?")
men_toggle = st.sidebar.toggle("Men Only", value=False)
women_toggle = st.sidebar.toggle("Women Only", value=False)

# Minority Serving
st.sidebar.write("Are you looking for a college that serves a particular minority group (MSI)?")
st.sidebar.markdown(":red[NOTE: Select at most 1 of the following categories at a time to help increase search robustness]")
black_serving = st.sidebar.toggle("Historically Black College & University", value=False)
ALHI_serving = st.sidebar.toggle("Alaska Native & Native Hawaiian Serving Institutions", value=False)
tribal_serving = st.sidebar.toggle("Tribal College & University", value=False)
AANHPI_serving = st.sidebar.toggle("AANAPISI", value=False, help="Asian American and Native American Pacific Islander-Serving Institutions")
hispanic_serving = st.sidebar.toggle("HSI", value=False, help= "Hispanic-Serving Institutions")
native_serving = st.sidebar.toggle("Native American Non-Tribal Institutions", value=False)


#### GET RECCOMENDATIONS
# Build callback function (called when button is clicked)
def get_Recommendations():
    st.session_state.message = "Recommendations Gotten"
    preferences = {"school.degrees_awarded.highest":select_degree, "school.ownership":select_control, "school.state_fips":select_states, "school.locale":select_region,
    "school.minority_serving.historically_black":black_serving, "school.minority_serving.annh":ALHI_serving, "school.minority_serving.tribal":tribal_serving,
     "school.minority_serving.aanipi":AANHPI_serving, "school.minority_serving.hispanic":hispanic_serving, "school.minority_serving.nant":native_serving, "school.men_only":men_toggle,
     "school.women_only":women_toggle, "school.religious_affiliation":select_religion, "admissions.admission_rate.overall":admission, "admissions.sat_scores.average.overall":sat,
     "cost.avg_net_price.public":cost_pub, "cost.avg_net_price.private":cost_priv}
    
    QUERY = "school.operating=1&"
    for parameter, values in preferences.items():
        param = str(parameter)
        if values == None or values == [] or values == False:
            pass
        else:
            if type(values) == list:
                param_df = pref[pref["QUERY_PARAM"]==param]
                dict = pd.Series(param_df.VALUE.values, index=param_df.LABEL).to_dict()
                query_vals = ",".join(map(str,[dict[x] for x in preferences[parameter]]))
            elif type(values) == str:
                param = param+"__range"
                query_vals = values
            elif values == True:
                query_vals = "1"
            QUERY += param+"="+query_vals+"&"
    SORT = st.session_state.sort
    CollegeScorecardAPI = f"https://api.data.gov/ed/collegescorecard/v1/schools.json?{QUERY}fields=school.name&per_page={NUM}&sort=school.name:{SORT}&api_key={KEY}"
    #CollegeScorecardAPI
    cs_data = requests.get(CollegeScorecardAPI)
    results = json.loads(cs_data.text)["results"]
    schools = []
    for school in results:
        schools.append(school['school.name'])

    st.session_state.edited_df = pd.DataFrame(data=list(zip(schools, [False]*len(schools))), columns=["School", "Select"])


def btn_callbk():
    st.session_state.edflg = True
    edited_df = st.session_state.edited_df
    st.session_state.selected_list = list(edited_df[edited_df[edited_df.columns[1]]== True][edited_df.columns[0]])
    #st.session_state.selected_list



def set_sort():
    if order == "Ascending":
        st.session_state.sort = "asc"
    elif order == "Descending":
        st.session_state.sort = "desc"

col1, col2, col3 = st.columns([0.33, 0.25, 0.33])
with col1:
    st.button("**Get Colleges**", type="primary", on_click=get_Recommendations)

### NUMBER OF RESULTS TO FETCH & SORTING ORDER
with col2:
    NUM = st.number_input("Maximum Results Returned:", value=20, placeholder="Type a number...")

with col3:
    order = st.radio("Sort Results:", 
    ["Ascending", "Descending"],
    captions = ["A→Z", "Z→A"],
    on_change = set_sort)


def build_query_output():  
    st.session_state.edited_df = st.data_editor(
        st.session_state.edited_df,
        key = "school_select",
        column_config={
            "School": st.column_config.Column(
                "Recommended Schools",
                required = True,
                disabled = True,
            ),
            "Select": st.column_config.CheckboxColumn(
                "Explore 🔎",
                help="Which schools would you like to hear more about? Select schools to generate student review summaries!",
                default= False,
                disabled=st.session_state.edflg,
                #(False if sum(st.session_state.edited_df[st.session_state.edited_df.columns[1]]) < 5 else True),
            ),  
        },
        disabled = ["Recommended Schools"],
        hide_index=True,
    )
    

    if st.button(label="Locked In" if  st.session_state.edflg else "Lock In Selections", type="primary",
                on_click=btn_callbk):
        st.session_state.disabled=True

if st.session_state.message == "Recommendations Gotten":
    build_query_output()


    def reset_selection():
        edited_df = st.session_state.edited_df
        edited_df["Explore 🔎"] = False
        if len(edited_df.columns) > 2:
            edited_df = edited_df.drop(edited_df.columns[1], axis=1)
        st.session_state.edited_df = edited_df
        st.session_state.edflg = False
        
    
  
    st.button("***Clear Selected***", type="secondary", on_click=reset_selection)
    st.markdown("##### **Next, choose a few of the resulting schools to find out a bit more! Select 1-5 schools to read about the PRO's and CON's according to the student body. Reviews are collected from RateMyProfessor.com, sectioned into positive and negative groupings and then summarized**")

   

    if st.session_state.selected_list != []:

        if st.button("**Get Student Review Summaries**", type="primary"):
            st.session_state.should_summarize = True

        if st.session_state.should_summarize:
            st.session_state.should_summarize = False  # Reset the flag immediately
            schools = st.session_state.selected_list
            # Initialize progress bar
            progress_bar = st.progress(0)
            total = len(schools)
            for i, school in enumerate(schools):
                st.text(f"Fetching Reviews for {school}...")
                message, pos_sum, neg_sum = fetch_summaries(school, OpenAI_client, GCP_client)
                if not message:
                    # Display Summaries
                    st.divider()
                    st.markdown(f"## ***:blue[{school}]***")
                    st.markdown(f'''
                        *:green[Positive Aspects:]* 
                                {pos_sum}]


                        *:red[Negative Aspects:]*
                                {neg_sum}]
                        ''')
                else:
                    st.text(message)
                
                # Update the progress bar
                progress = int((i + 1) / total * 100)
                progress_bar.progress(progress)
            # Complete the progress bar
            st.markdown("#### *Complete!*")
            progress_bar.progress(100)


#https://api.data.gov/ed/collegescorecard/v1/schools.json?school.operating=1&cost.avg_net_price.public__range=..80000&cost.avg_net_price.private__range=..80000&fields=school.name&api_key=kiAstPJVUqbuRNL9czkNqb18NWY0Rb3e7zK6slpq