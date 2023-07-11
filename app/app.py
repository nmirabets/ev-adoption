import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy import special
from scipy.optimize import least_squares

def main():

    # ---- Load Data ----
    # IEA Global EV Data Explorer API url
    api_url = 'https://api.iea.org/evs/?parameter=EV%20sales&mode=Cars&category=Historical&csv=true'
    # Download data from IEA website
    ev_sales_df = pd.read_csv(api_url)
    
    # Add a title and optional description
    st.title("EV Adoption Tracker")
    st.markdown(
    '''
    Electric Vehicles (EVs) have been around for a couple decades, but its adoption has really picked up in the last 3 years. In this project we will analyze the increasing demand of EVs and who's leading it, figure out in which stage of adoption we are (according to Everett M. Rogger's [*Diffusion of Innovation Theory*](https://en.wikipedia.org/wiki/Diffusion_of_innovations)) and try to estimate how long will it take for the all car transportation to become electric by finding the EV adoption curve.
    ''')
    if ev_sales_df is not None:

        # Display a seaborn plot
        st.subheader("Seaborn Plot")



if __name__ == "__main__":
    main()
