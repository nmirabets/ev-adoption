import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy import special
from scipy.optimize import least_squares

def load_data() -> tuple:
    '''
    Load and preprocess car sales data from different sources.

    Returns:
        tuple: A tuple containing two Pandas DataFrames - ev_sales_df and car_sales_df.

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
    Merge and transform the EV sales and car sales data.

    Args:
        ev_sales_df (pd.DataFrame): DataFrame containing the EV sales data.
        car_sales_df (pd.DataFrame): DataFrame containing the car sales data.

    Returns:
        pd.DataFrame: Merged and transformed DataFrame.

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
    This function takes the market share and returns the population segment
    according to the Diffusion of Innovations Theory.

    Input:
    accum_market_share: market share as percent i.e 15 for 15%

    Output:
    population:_segment_name: segment name i.e Innovators
    
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


def plot_ev_sales_share(df: pd.DataFrame):
    '''
    Generate a Plotly figure for world EV sales share.

    Args:
        df (pd.DataFrame): DataFrame containing the data.

    Returns:
        go.Figure: Plotly figure.

    '''
    world_df = df[df['region'] == 'World']
    
    def exponential_func(x, a, b, c):
        return a * np.exp(b * (x - 2010)) + c
    
    # World EV Sales
    fig = px.bar(world_df, x='year', y='ev_sales_share', title='World EV Market Share')
    
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Market Share (%)',
        xaxis={'type': 'category'},
        showlegend=False
    )
    
    # Add exponential line
    x_values = world_df['year']
    a, b, c = 0.01396663, 0.54538719, 0.11010673
    y_exp = exponential_func(x_values, a, b, c)
    
    fig.add_trace(go.Scatter(x=x_values, y=y_exp, mode='lines', name='Exponential Fit'))
    
    return fig


def plot_stacked_car_sales(df: pd.DataFrame):
    '''
    Create a stacked bar chart showing global car sales split by combustion/EV over time.

    Args:
        df (pd.DataFrame): DataFrame containing the data.

    Returns:
        fig: Plotly figure object representing the stacked bar chart.

    '''
    # Filter & rename dataframe
    world_df = df[df['region'] == 'World'].rename(columns={'ev_sales': 'Electric', 'ice_sales': 'Combustion'})
    
    # Create the stacked bar chart
    fig = px.bar(world_df, x='year', y=['Electric', 'Combustion'], title='World Car Sales',
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


def plot_and_fit_market_share(df: pd.DataFrame) -> tuple:
    '''
    Create a plot and fit different models to the data.

    Args:
        df (pd.DataFrame): DataFrame containing the data with columns 'year' and 'ev_sales_share'.

    Returns:
        tuple: A tuple containing the Plotly figure object and a DataFrame with R-squared values.

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

    # Set layout configuration
    fig.update_layout(
        title="EV Market Share Trend Line Fit",
        xaxis_title="",
        yaxis_title="Market Share (%)",
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


def create_country_sales_rank(df: pd.DataFrame, current_year: int) -> tuple:
    '''
    This function creates a ranked table sorted by 'ev_sales' and 
    a fig to plot a pie chart from a df.

    Input:
    df: pandas DataFrame

    Output:
    tuple: ranked_df, fig
    
    '''

    # List regions that are not countries to filter them out
    exclude_regions = ['World','Europe','EU27','Other Europe']
    
    # Filter df to get rank of countries
    rank_df = df[(df['year']==current_year) & (~df['region'].isin(exclude_regions))][['region','ev_sales']].sort_values('ev_sales', ascending = False)
    
    # Create rank column
    rank_df['Rank'] = rank_df.reset_index().index + 1

    # Rename columns to show in table or plot
    rank_df.rename(columns={'region':'Country','ev_sales':'Sales','ev_sales_country_share':'Sales share (%)'}, inplace=True)

    # Filter out countries with sales < 80000 to represent only large countries
    rank_df.loc[rank_df['Sales'] < 80000, 'Country'] = 'All other countries' 

    # Create pie chart fig
    fig = px.pie(rank_df, values='Sales', names='Country', title='Sales by Country')
    
    return rank_df[['Rank', 'Country', 'Sales']].head(10), fig


def create_country_sales_growth_rank_df(df: pd.DataFrame, current_year: int) -> pd.DataFrame:
    '''
    This function creates a ranked table sorted by 'ev_sales_share_growth' and from a df.

    Input:
    df: pandas DataFrame

    Output:
    df: pandas DataFrame
    
    '''
    # Define the ranking column
    ranking_column = 'ev_sales_share_growth'
    
    # List regions that are not countries to filter them out
    exclude_regions = ['World','Europe','EU27','Other Europe','Rest of the world']
    
    # Filter df to get rank of countries with at least sales > 10000
    rank_df = df[(df['year']== current_year) & (~df['region'].isin(exclude_regions)) & (df['ev_sales']> 10000)].sort_values(ranking_column, ascending = False)
    
    # Create rank column
    rank_df['Rank'] = rank_df.reset_index().index + 1

    # Rename columns to show in table
    rank_df.rename(columns={'region':'Country','ev_sales':'Sales','ev_sales_share_growth':'Sales share growth (%)'}, inplace=True)

    return rank_df[['Rank', 'Country', 'Sales share growth (%)', 'Sales']].head(10)


def find_normal_curve(df: pd.DataFrame) -> tuple:
    '''
    Find the parameters (mean and standard deviation) of a normal distribution curve that best fits the given data points.

    Args:
        x_values (pd.Series): Series of x values.
        y_values (pd.Series): Series of y values.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the fitted normal curve.

    '''
    
    # Filter world data
    world_df = df[df['region'] == 'World']

    # Get x, y values
    x_values = world_df['year']
    y_values = world_df['ev_sales_share'] / 100

    # Define the system of equations
    def equations(p):
        mu, sd = p
        return y_values - 0.5 * (1 + special.erf((x_values - mu) / (sd * np.sqrt(2))))
    
    # Initial guess for the mean and standard deviation
    mu_init, sd_init = 2028, 2
    
    # Solve the system of equations
    solution = least_squares(equations, (mu_init, sd_init))
    
    mu, sd = solution.x
    
    return mu, sd


def plot_adoption_curve(mu: float, sd: float, current_cdf: float):
    '''
    Create an adoption curve plot showing the cumulative distribution function (CDF) and probability density function (PDF) of a normal distribution.

    Args:
        mu (float): Mean of the normal distribution.
        sd (float): Standard deviation of the normal distribution.
        current_adoption (float): Current adoption rate (between 0 and 1).

    Returns:
        go.Figure: Plotly figure object representing the adoption curve.

    '''

    # Generate the x values with a wider range
    x = np.linspace(mu - 3 * sd, mu + 3 * sd, 1000)
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Calculate the CDF using the same x values
    cdf_y = norm.cdf(x, mu, sd)
    
    # Plot the CDF
    fig.add_trace(go.Scatter(x=x, y=cdf_y, mode='lines', name='CDF', line=dict(color='orange'), yaxis='y2'),secondary_y=True)
    
    # Plot the normal distribution using a separate y-axis
    fig.add_trace(go.Scatter(x=x, y=norm.pdf(x, mu, sd), mode='lines', name='PDF', line=dict(color='darkblue'), yaxis='y2'),secondary_y=False)
    
    # Shade the left tail up to current_adoption
    x_fill = np.linspace(mu - 3 * sd, norm.ppf(current_cdf, mu, sd), 100)
    fig.add_trace(go.Scatter(x=x_fill, y=norm.pdf(x_fill, mu, sd), fill='tozeroy', fillcolor='#98FB98', opacity=0, name='Current PDF', yaxis='y2'),secondary_y=False)
    
    # Add a vertical line on the current_adoption
    x_line = norm.ppf(current_cdf, mu, sd)
    vertical_line2 = go.Scatter(x=[x_line, x_line], y=[0, norm.pdf(x_line, mu, sd)], mode='lines', name='Vertical Line 2', line=dict(color='darkblue', width=2), yaxis='y2')
    fig.add_trace(vertical_line2, secondary_y=False)
    
    # Add annotation with the current adoption
    fig.add_annotation(x=x_line-0.8, y=norm.pdf(x_line, mu, sd)+0.0115, text='<b>Current CDF</b>', showarrow=False, yshift=10, font=dict(size=12, color='darkgreen'), yref='y1')
    fig.add_annotation(x=x_line-0.8, y=norm.pdf(x_line, mu, sd)+0.006, text='<b>'+str((current_cdf*100).round(2))+'%</b>', showarrow=False, yshift=10, font=dict(size=12, color='darkgreen'), yref='y1')
    
    # Segment CDF boundaries
    segments = [2.5, 16, 50, 84]
    
    for segment in segments:
        # Add a vertical line for every segment
        x_line = norm.ppf(segment/100, mu, sd)
        vertical_line2 = go.Scatter(x=[x_line, x_line], y=[0, norm.pdf(x_line, mu, sd)], mode='lines', name='Vertical Line 2', line=dict(color='darkblue', width=2), yaxis='y2')
        fig.add_trace(vertical_line2, secondary_y=False)
    
    annotations = {'Innovators':[-1.7, 0.14, ''],
                   '':[-1.95, 0.14, '2.5%'],
                   'Early Adopters':[-1, 0.5, '16%'],
                   'Early Majority':[0, 0.78, '50%'],
                   'Late Majority':[1, 0.5, '84%'],
                   'Laggards':[2, 0.025, ''],}
    
    for key, value in annotations.items():
        fig.add_annotation(x=mu+(value[0]*sd)-2, y=0.01, text='<b>'+key+'</b>', showarrow=False, yshift=10, font=dict(size=12), yref='y2')
        fig.add_annotation(x=mu+(value[0]*sd), y=value[1], text=value[2], showarrow=False, yshift=10, font=dict(size=12), yref='y2')
    
    # Customize the layout
    fig.update_layout(
        xaxis=dict(showgrid=True, zeroline=False, dtick=2),  # Show gridlines, zeroline, and set dtick to 1 for 1 unit interval in x-axis
        yaxis=dict(showgrid=False, zeroline=True, range=[0, 0.13], side='left', showticklabels=False ),
        yaxis2=dict(showgrid=True, zeroline=False, range=[0, 1], side='right', overlaying='y', ticktext=['0%', '25%', '50%', '75%', '100%'], tickvals=[0, 0.25, 0.5, 0.75, 1]),# 0.195]),  # Create a secondary y-axis without tick labels
        showlegend=False,
        margin=dict(l=2, r=2, t=2, b=20)
    )
    return fig


def get_current_year_cdf_and_delta(df) -> tuple:
    '''
    Get the current cumulative distribution function (CDF) and its delta from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.

    Returns:
        tuple: A tuple containing the current year, current CDF, and delta.

    '''
    
    # Get current CDF
    current_cdf = (df[df['region'] == 'World']['ev_sales_share'].tail(2).iloc[1]).round(3)
    
    # Get current CDF delta
    delta = df[df['region'] == 'World']['ev_sales_share'].tail(2).iloc[0].round(3)
    
    # Get current year
    current_year = df[df['region'] == 'World']['year'].tail(1).iloc[0]

    return current_year, current_cdf, delta


def get_year_from_cdf(mu: float, sd: float, cdf: float):
    '''
    Get the year corresponding to a given cumulative distribution function (CDF) value of a normal distribution.

    Args:
        mu (float): Mean of the normal distribution.
        sd (float): Standard deviation of the normal distribution.
        cdf (float): Cumulative distribution function (CDF) value.

    Returns:
        float: The year corresponding to the given CDF value.

    '''
    rv = norm(loc=mu, scale=sd)
    return rv.ppf(cdf)


def get_month_and_year(decimal_year):
    '''
    Convert a decimal year to the corresponding month and year.

    Args:
        decimal_year (float): Decimal representation of a year.

    Returns:
        tuple: A tuple containing the month name (str) and the year (int).

    '''

    # Extract the integer part of the decimal year
    year = int(decimal_year)

    # Calculate the fractional month
    fractional_month = (decimal_year - year) * 12

    # Extract the integer part of the fractional month
    month = int(fractional_month)

    # Get the month name corresponding to the calculated month
    month_name = datetime.date(1900, month, 1).strftime('%B')

    # Return the month name and year as a tuple
    return month_name, year
