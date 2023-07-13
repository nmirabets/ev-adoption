import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy import special
from scipy.optimize import least_squares

import functions

def main():



    # Set the page configuration
    st.set_page_config(
    page_title="EV Adoption Tracker",
    page_icon="random",
    layout="centered",
    initial_sidebar_state="expanded")

    # Load datasets
    ev_sales_df, car_sales_df = functions.load_data()

    # Merge & transform datasets
    df = functions.merge_and_transform_data(ev_sales_df, car_sales_df)
    
    # Add a title and optional description
    st.title("EV Adoption Tracker")
    st.markdown(
    '''
    Electric Vehicles (EVs) have been around for a couple decades, but its adoption has really picked up in the last 3 years. In this project we will analyze the increasing demand of EVs and who's leading it, figure out in which stage of adoption we are (according to Everett M. Rogger's [*Diffusion of Innovation Theory*](https://en.wikipedia.org/wiki/Diffusion_of_innovations)) and try to estimate how long will it take for the all car transportation to become electric by finding the EV adoption curve.
    ''')
    st.divider() # Add a section divider
    if df is not None:
        # ----------------------- DASHBOARD -----------------------
        
        # -> GET SECTION DATA
        
        # find current adoption & delta
        current_year, current_ev_sales_share, delta = functions.get_current_year_cdf_and_delta(df)

        # find current adoption segment
        current_adoption_segment = functions.get_population_segment(current_ev_sales_share)

        # find mu & sigma for normal curve
        mu, sd = functions.find_normal_curve(df)

        # --------> HEADER

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Mean (μ)', str(mu.round(1)), delta=None, delta_color="normal", help=None, label_visibility="visible")
        with col2:
            # CDF METRIC 
            st.metric('EV Sales Share', str((current_ev_sales_share).round(2))+'%', 
                      delta=str(((current_ev_sales_share-delta)).round(2))+'% (2021)', 
                      delta_color="normal", help=None, label_visibility="visible")
        with col3:
            st.metric('Standard deviation (σ)', str(sd.round(1))+' years', delta=None, delta_color="normal", help=None, label_visibility="visible")

        # --------> NORMAL CURVE PLOT
        # plot curve
        fig = functions.create_adoption_curve(mu, sd, current_ev_sales_share/100)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        # --------> Message boards
        msg1="We are currently in the **{} stage** with **{}%** of car sales being electric.".format(
            current_adoption_segment,((current_ev_sales_share)).round(2))
        st.success(msg1, icon="🔥")

                # --------> Message boards
        month_name, year = functions.get_month_and_year(functions.get_year_from_cdf(mu, sd, 0.16))
        msg4="We will enter the **Early Majority** stage in {}, {}".format(month_name, year)
        st.info(msg4, icon="🚀")

        col1, col2 = st.columns(2)
        
        with col1:
            # --------> Adoption message
            interval = (functions.get_year_from_cdf(mu, sd, 0.5) - current_year).round(1)
            msg2="**{} years left** for **50%** of sales to be electric".format(interval)
            st.info(msg2, icon="⌛")
        with col2:
            # --------> Adoption message
            interval = (functions.get_year_from_cdf(mu, sd, 0.95) - current_year).round(1)
            msg3="**{} years left** for **95%** of sales to be electric".format(interval)
            st.info(msg3, icon="🌍")


        # --------> 
            
        # get x and y values
        world_df = df[df['region'] == 'World']    
        
    
        # ---- SECTION: EV Car Sales Share Trend Line Fit ----
        st.divider() # Add a section divider
        st.subheader("Historical Sales") # Display subheader
    
        # Plot stacked bar chart EV vs ICE
        fig = functions.plot_world_sales(df)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
        # Plot world sales 
        fig = functions.get_world_sales_share_fig(df)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
        # ---- SECTION: EV Car Sales Share Trend Line Fit ----
        st.divider() # Add a section divider
        st.subheader("EV Sales Country Rank") # Display subheader
    
        # Display table and pie chart of leading countries by sales
        sales_rank_df, fig = functions.create_country_sales_rank(df, current_year)
        st.dataframe(sales_rank_df, hide_index=True)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        
        # Display table of leading countries by sales growth
        country_sales_growth_rank_df = functions.create_country_sales_growth_rank_df(df, current_year)
        st.dataframe(country_sales_growth_rank_df, hide_index=True)
    
    
        # ---- SECTION: EV Car Sales Share Trend Line Fit ----
        st.divider() # Add a section divider
        st.subheader("EV Sales Country Rank") # Display subheader
    
    
        selected_category = st.selectbox('Select a region', df['region'].unique(),index=len(df['region'].unique())-1)
        
        filtered_data = df[df['region'] == selected_category]
    
        fig, r_sq_df = functions.create_plot_and_fit(filtered_data)
    
        highest_fit = r_sq_df.loc[r_sq_df["R-Squared"].idxmax(), "Fit"]
    
        highest_r_squared = str(r_sq_df.loc[r_sq_df["R-Squared"].idxmax(), "R-Squared"].round(3))
        
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
        st.info('The best fit for *' + selected_category + '* is **' + highest_fit + '** with an R-Squared of **' + highest_r_squared + '**')
    
        st.dataframe(r_sq_df, hide_index=True)#, use_container_width=True)

if __name__ == "__main__":
    main()
