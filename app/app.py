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
    
    with st.container():
        # Add a title and optional description
        st.title("EV Adoption Tracker")
        st.markdown(
        '''
        Electric Vehicles (EVs) have been around for a couple decades, but its adoption has really picked up in the last 3 years. In this project we will analyze the increasing demand of EVs and who's leading it, figure out in which stage of adoption we are (according to Everett M. Rogger's [*Diffusion of Innovation Theory*](https://en.wikipedia.org/wiki/Diffusion_of_innovations)) and try to estimate how long will it take for the all car transportation to become electric by finding the EV adoption curve.
        ''')
        st.divider() # Add a section divider
        if df is not None:
    
            # Filter df to get current_ev_sales_share
            current_ev_sales_share = df[df['region'] == 'World']['ev_sales_share'].tail(1).iloc[0]
            # Get current adoption segment
            current_adoption_segment = functions.get_population_segment(current_ev_sales_share)
    
            # Display current sales_share & segment
            st.write('Answer to question #1: We are currently in the **{} stage** with **{}%** of car sales being electric.'.format(
                current_adoption_segment,current_ev_sales_share.round(2)))
    
            world_df = df[df['region'] == 'World']
    
            with st.container():

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
                sales_rank_df, fig = functions.create_country_sales_rank(df)
                st.dataframe(sales_rank_df, hide_index=True)
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                
                # Display table of leading countries by sales growth
                country_sales_growth_rank_df = functions.create_country_sales_growth_rank_df(df)
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
