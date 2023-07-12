import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy import special
from scipy.optimize import least_squares

def load_data() -> tuple:
    '''
    '''
    # Global car sales extracted manually from Internation Energy Agency (IEA) website
    car_sales = {'year':[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
             'value': [66700000,71200000,73800000,77500000,80000000,81800000,86400000,86300000,85600000,81300000,71800000,74900000,74800000]}

    # Convert dictionary into Pandas Dataframe
    car_sales_df = pd.DataFrame(car_sales)
    

    # IEA Global EV Data Explorer API url
    api_url = 'https://api.iea.org/evs/?parameter=EV%20sales&mode=Cars&category=Historical&csv=true'
    
    # Download data from IEA website
    ev_sales_df = pd.read_csv(api_url)

    # Filter BEV to discard PHEV
    ev_sales_df = ev_sales_df[ev_sales_df['powertrain'] == 'BEV']

    # Drop all columns with only 1 value
    ev_sales_df.drop(['category','parameter','mode','powertrain','unit'], axis=1, inplace=True)

    ev_sales_df = ev_sales_df.sort_values(['region', 'year']).reset_index(drop=True)


    return ev_sales_df, car_sales_df


def merge_and_transform_data(ev_sales_df: pd.DataFrame, car_sales_df: pd.DataFrame) -> pd.DataFrame:
    '''
    '''
    # Create copies of dfs
    ev_sales_df1 = ev_sales_df.copy()
    car_sales_df1 = car_sales_df.copy()
    
    # Rename the column 'value' to 'ev_sales'
    ev_sales_df1.rename( columns = {'value':'ev_sales'}, inplace = True)
    
    # Add calculated column: ev_sales_share = ev_sales / car_sales.value
    ev_sales_df1 = ev_sales_df1.join(car_sales_df1.set_index('year'), on='year'
                                    ).rename(columns={'value':'global_car_sales'}) # Join the global car_sales data
    # Add calculated column: ice_sales = global_car_sales - ev_sales
    ev_sales_df1['ice_sales'] = ev_sales_df1['global_car_sales'] - ev_sales_df1['ev_sales']
    # Add calculated column: ev_sales_share = (ev_sales / global_car_sales) * 100
    ev_sales_df1['ev_sales_share'] = ((ev_sales_df1['ev_sales'] / ev_sales_df1['global_car_sales']) * 100).round(5)
    
    merged_df = pd.DataFrame()
    
    for country, group in ev_sales_df1.groupby('region'):
        # Add calculated column: 'ev_sales_share_change' = 'ev_sales_share'(current year) - 'ev_sales_share'(previous year)
        group['ev_sales_share_change'] = group['ev_sales_share'] - group['ev_sales_share'].shift(+1)
        group['ev_sales_share_change'].fillna(0, inplace=True)
        # Add calculated column: 'ev_sales_share_growth'
        group['ev_sales_share_growth'] = (((group['ev_sales_share'] - group['ev_sales_share'].shift(+1)) / group['ev_sales_share'].shift(+1)) * 100).round(1)
        group['ev_sales_share_growth'].fillna(0, inplace=True)
        # Add calculated column: 'ev_sales_growth'
        group['ev_sales_growth'] = (((group['ev_sales'] - group['ev_sales'].shift(+1))/group['ev_sales'].shift(+1)) * 100).round(1)
        group['ev_sales_growth'].fillna(0, inplace=True)
        # Concatenate the group with the merged dataframe
        merged_df = pd.concat([merged_df, group])

    return merged_df


def get_population_segment(accum_market_share: float) -> str:
    '''
    '''
    if accum_market_share < 2.5:
        return 'Innovators'
    elif accum_market_share < 16:
        return 'Early Adopters'
    elif accum_market_share < 50:
        return 'Early Majority'
    elif accum_market_share < 84:
        return 'Late Majority'
    elif accum_market_share <= 100:
        return 'Laggards'


def get_world_sales_share_fig(df: pd.DataFrame):
    '''
    '''
    world_df = df[df['region'] == 'World']
    
    def exponential_func(x, a, b, c):
        return a * np.exp(b * (x - 2010)) + c
    
    # World EV Sales
    fig = px.bar(world_df, x='year', y='ev_sales_share', title='World EV Sales')
    
    fig.update_layout(
        xaxis_title='',
        yaxis_title='World car sales share (%)',
        xaxis={'type': 'category'},
        showlegend=False
    )
    
    # Add exponential line
    x_values = world_df['year']
    a, b, c = 0.01396663, 0.54538719, 0.11010673
    y_exp = exponential_func(x_values, a, b, c)
    
    fig.add_trace(go.Scatter(x=x_values, y=y_exp, mode='lines', name='Exponential Fit'))
    
    return fig


def plot_world_sales(df: pd.DataFrame):
    '''
    '''
    # Filter & rename dataframe
    world_df = df[df['region'] == 'World'].rename(columns={'ev_sales': 'Electric', 'ice_sales': 'Combustion'})
    
    # Create the stacked bar chart
    fig = px.bar(world_df, x='year', y=['Electric', 'Combustion'], title='Global Car Sales',
                color_discrete_sequence=['#90EE90', '#ADD8E6'])
    
    # Update the layout
    fig.update_layout(
        barmode='stack',
        xaxis_title=None,
        yaxis_title='Sales',
        xaxis={'type': 'category'},
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.2,
            xanchor='center',
            x=0.5
        ))
    
    return fig


def create_plot_and_fit(df: pd.DataFrame) -> tuple:
    '''
    '''

    # Assign x and y data
    x_data = df['year']
    y_data = df['ev_sales_share']
    
    # Define the R-squared function
    def r_squared(y, y_pred):
        residual = np.sum((y - y_pred) ** 2)
        total = np.sum((y - np.mean(y)) ** 2)
        r_sq = 1 - (residual / total)
        return r_sq
    
    # Fit a linear model to the data
    def linear_func(x, a, b):
        return a * x + b
    
    linear_params, _ = curve_fit(linear_func, x_data, y_data)
    
    # Fit a quadratic model to the data
    def quadratic_func(x, a, b, c):
        return a * x**2 + b * x + c
    
    quadratic_guesses = [1, 1, 1]  # Initial guesses for the quadratic fit
    quadratic_params, _ = curve_fit(quadratic_func, x_data, y_data, p0=quadratic_guesses)
    
    # Fit an exponential model to the data
    def exponential_func(x, a, b, c):
        return a * np.exp(b * (x - 2010)) + c
    
    exponential_guesses = [1, 1, 1]  # Initial guesses for the exponential fit
    exponential_params, _ = curve_fit(exponential_func, x_data, y_data, p0=exponential_guesses)
    
    # Calculate R-squared values
    r_squared_values = {
        'Fit': ['Linear', 'Quadratic', 'Exponential'],
        'R-Squared': [r_squared(y_data, linear_func(x_data, *linear_params)), 
                      r_squared(y_data, quadratic_func(x_data, *quadratic_params)), 
                      r_squared(y_data, exponential_func(x_data, *exponential_params))]}
    
    # Create a DataFrame with R-squared values
    r_squared_values_df = pd.DataFrame(r_squared_values)
    
    # Visualize the data and fits using Plotly
    fig = go.Figure()
    
    # Original data points
    fig.add_trace(go.Bar(x=x_data, y=y_data, name='Electric Vehicles', marker_color='#90EE90'))
    
    # Linear fit
    linear_fit = linear_func(x_data, *linear_params)
    fig.add_trace(go.Scatter(x=x_data, y=linear_fit, mode='lines', name='Linear Fit', line=dict(color='#000080', dash='dashdot')))
    
    # Quadratic fit
    quadratic_fit = quadratic_func(x_data, *quadratic_params)
    fig.add_trace(go.Scatter(x=x_data, y=quadratic_fit, mode='lines', name='Quadratic Fit', line=dict(color='#FFA500', dash='dash')))
    
    # Exponential fit
    exponential_fit = exponential_func(x_data, *exponential_params)
    fig.add_trace(go.Scatter(x=x_data, y=exponential_fit, mode='lines', name='Exponential Fit', line=dict(color='#FF0000', width=2)))
    
    fig.update_layout(
        title="EV Sales Share Trend Line Fit",
        xaxis_title="",
        yaxis_title="Sales Share (%)",
        xaxis={'type': 'category'},
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.2,
            xanchor='center',
            x=0.5
        ),
    )

    return fig, r_squared_values_df


def create_country_sales_rank(df: pd.DataFrame) -> tuple:
    '''
    '''

    # List regions that are not countries to filter them out
    exclude_regions = ['World','Europe','EU27','Other Europe','Rest of the world']
    
    # Filter df to get rank of countries
    sales_country_rank = df[(df['year']==2022) & (~df['region'].isin(exclude_regions))][['region','ev_sales','ev_sales_share']].sort_values('ev_sales_share', ascending = False)
    
    # Create rank column
    sales_country_rank['Rank'] = sales_country_rank.reset_index().index + 1
    
    # Get EV global sales for 2022
    current_world_sales = df[df['region'] == 'World']['ev_sales'].tail(1).iloc[0]
    
    # Create calculated column with share of ev cars sold over total ev sales by country
    sales_country_rank['ev_sales_country_share'] = ((sales_country_rank['ev_sales'] / current_world_sales) * 100).round(1)

    sales_country_rank.rename(columns={'region':'Country','ev_sales':'Sales','ev_sales_country_share':'Sales share (%)'}, inplace=True)

    sales_country_rank = sales_country_rank.head(10)

    sales_country_rank.loc[sales_country_rank['Sales share (%)'] < 2, 'Country'] = 'All other countries' # Represent only large countries

    fig = px.pie(sales_country_rank, values='Sales share (%)', names='Country', title='Sales Share by Country')
    
    return sales_country_rank[['Rank', 'Country','Sales share (%)', 'Sales']].head(10).style.hide(axis="index"), fig


def create_country_sales_growth_rank_df(df: pd.DataFrame) -> pd.DataFrame:
    '''
    '''
    ranking_column = 'ev_sales_share_growth'

    # List regions that are not countries to filter them out
    exclude_regions = ['World','Europe','EU27','Other Europe','Rest of the world']
    
    # Filter df to get rank of countries
    sales_growth_country_rank = df[(df['year']==2022) & (~df['region'].isin(exclude_regions))].sort_values(ranking_column, ascending = False)
    
    # Create rank column
    sales_growth_country_rank['Rank'] = sales_growth_country_rank.reset_index().index + 1

    sales_growth_country_rank.rename(columns={'region':'Country','ev_sales':'Sales','ev_sales_share_growth':'Sales share growth (%)'}, inplace=True)

    return sales_growth_country_rank[['Rank', 'Country', 'Sales share growth (%)', 'Sales']].head(10).style.hide(axis="index")







