
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.options.mode.chained_assignment = None  # default='warn'

# initialize cwd
PROJECT_ROOT = os.getcwd()


def data_comprehension(df):
    """
    Calculate which teams have certain players that meet the conditions:
        - went to duke and drafted in or before 2000
        - have a first name that begins with D and were drafted in an even year draft

    :param:
    df: DataFrame
        df containing draft data since 1989 wth each picks team and selection along with their career stats

    :return:
    None
    """
    # get only players who went to Duke and were drafted in or after 2000
    duke_df = df[(df['college'] == 'Duke') & (df['year'] >= 2000)]

    # group by the team drafted and then sort by the number of players drafted by the team
    duke_df = duke_df.groupby('team').count().sort_values(['college', 'team'], ascending=[False, True])
    duke_df = duke_df['college'].rename('Duke Picks from 2000 Onwards')
    print("Duke players drafted by team:\n", duke_df)

    # get players whose name starts with d and were drafted in an even draft
    evenD_df = df[(df['player'].str.startswith('D')) & (df['year'] % 2 == 0)]

    # group by the team drafted and then sort by the number of players drafted by the team
    evenD_df = evenD_df.groupby('team').count().sort_values(['college', 'team'], ascending=[False, True])
    evenD_df = evenD_df['college'].rename('Players Drafted in an Even Year Whose Name Starts with \"D\"')
    print("\nPlayers starting with D in an even numbered draft by team:\n", evenD_df)


def yearly_draft_analysis(df):
    """
    Find the relationship between a team's first round pick in one year and how it changes in each following year

    :param:
    df: DataFrame
        df containing draft data since 1989 wth each picks team and selection along with their career stats

    :return:
    None
    """
    # select only first round picks and get all the teams who have drafted
    df = df[(df['overall_pick'].between(1, 30))]
    teams = df['team'].unique()

    # use list comprehension to get the pearson correlation and p-value between the year and pick number for every team
    corrs = [pearsonr((df[df['team'] == team])['year'], (df[df['team'] == team])['rank'])[0] for team in teams]
    pvals = [pearsonr((df[df['team'] == team])['year'], (df[df['team'] == team])['rank'])[1] for team in teams]

    # print the average correlation and p-value for all teams
    print("\nAverage correlation between year and first round pick number:", np.mean(corrs))
    print("Average p-value:", np.mean(pvals))

    # get the values for the atlanta hawks specifically and print them
    hawks_corr = pearsonr((df[df['team'] == 'ATL'])['year'], (df[df['team'] == 'ATL'])['rank'])[0]
    hawks_pval = pearsonr((df[df['team'] == 'ATL'])['year'], (df[df['team'] == 'ATL'])['rank'])[1]
    print(f'\nFor the Atlanta Hawks specifically, the correlation is {hawks_corr} and the p-value is {hawks_pval}')


def evaluate_players(df):
    """
    For every statistical column get the percentiles of a player within that column. Then take the averages
    of all their percentiles to get an overall evaluation of the player

    :param:
    df: DataFrame
        df containing draft data since 1989 wth each picks team and selection along with their career stats

    :return:
    df: DataFrame
        same df as above with an additional column of the player's evaluation
    """
    # get only the numeric statistical columns and make a copy of the dataframe to run calcs on
    cols = ['years_active', 'games', 'minutes_played', 'points', 'total_rebounds', 'assists',
            'field_goal_percentage', '3_point_percentage', 'free_throw_percentage', 'average_minutes_played',
            'points_per_game', 'average_total_rebounds', 'average_assists', 'win_shares', 'win_shares_per_48_minutes',
            'box_plus_minus', 'value_over_replacement']
    temp_df = df

    # get the percentile ranks for different columns (place NaN values at lowest percentile)
    for col in cols:
        temp_df[col] = temp_df[col].rank(pct=True, na_option='top')

    # create column with the calculated value of each player drafted and then save it to original df
    temp_df['avg_pct'] = temp_df[cols].mean(axis=1)
    df['player_val'] = temp_df['avg_pct']

    return df


def value_draft_pos(df):
    """
    Create a dictionary of draft slots and the expected value of the slot

    :param:
    df: DataFrame
        df containing draft data since 1989 wth each picks team and selection along with their career stats

    :return:
    df: DataFrame
        same df with the addition of each player's evaluation, the expected value of their slot, and the difference
    """
    # get the evaluations of each player stored in the dataframe
    df = evaluate_players(df)

    # get the average value of all players in each slot
    slot_values = df.groupby('overall_pick')['player_val'].mean()
    print("Average Draft Value by Pick:\n", slot_values)

    # create a column with the expected value of the draft slot
    df['expected_val'] = df['overall_pick'].apply(lambda x: slot_values.iloc[x - 1])

    # create a column with the difference between the actual value of the player and their expected value
    df['val_diff'] = df['player_val'] - df['expected_val']

    return df


def plot_bar(df, x, y, xlabel, ylabel, title, rotation=90, atl_color=False):
    """
    Plot a bar chart based on certain specifications

    :param:
    df: DataFrame
        df containing the data from which to plot
    x: str
        string of the column name for the x values
    y: str
        string of the column name for the y values
    xlabel: str
        string of the x-axis label
    ylabel: str
        string of the y-axis label
    title: str
        string of the title label
    rotation : int, default=90
        degrees to rotate the x-axis labels
    atl_color: bool; default=False
        denotes if the graph should color based on "ATL" acronym

    :return:
    None
    """
    # color atlanta's column in graph if requested
    if atl_color:

        # get all the teams and create a list of colors for them
        vals = list(df[x])
        colors = ["crimson" if i == 'ATL' else "silver" for i in vals]

        # plot the bar chart
        plt.bar(df[x], df[y], color=colors)

    else:
        plt.bar(df[x], df[y])

    # graph organization
    plt.xticks(rotation=rotation)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def team_evaluation(df):
    """
    Evaluate NBA teams based on their drafting ability and college teams based on their preparation ability
    (based on my player evaluation method).

    :param:
    df: DataFrame
        df containing draft data since 1989 wth each picks team and selection along with their career stats with
        the addition of each player's evaluation, the expected value of their slot, and the difference

    :return:
    None
    """
    # get the average value discrepancy for every team and order from best to worst
    nba_df = df.groupby('team')['val_diff'].mean().sort_values(ascending=False).reset_index()

    # plot the nba drafting data
    plot_bar(nba_df, 'team', 'val_diff',
             'NBA Team',
             'AVG Difference in Value',
             'AVG Difference in Draft Pick Value by Team (since 1989)',
             atl_color=True)

    # get the average and summed value discrepancy for every college team and order from best to worst
    college_df_avg = df.groupby('college')['val_diff'].mean().sort_values(ascending=False).reset_index()
    college_df_sum = df.groupby('college')['val_diff'].sum().sort_values(ascending=False).reset_index()

    # plot the top 15 schools by preparing ability
    plot_college_df_avg = college_df_avg.head(15)
    plot_college_df_sum = college_df_sum.head(15)

    # plot the college drafting data
    plot_bar(plot_college_df_avg, 'college', 'val_diff',
             'College Team',
             'AVG Difference in Value',
             'AVG Difference in Draft Pick Value by College Team (since 1989)')

    # plot the college drafting data
    plot_bar(plot_college_df_sum, 'college', 'val_diff',
             'College Team',
             'Total Difference in Value',
             'Total Difference in Draft Pick Value by College Team (since 1989)',
             rotation=55)


def Main():

    # read in the draft data
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'nbaplayersdraft.csv'))

    # complete data comprehension and trend tasks
    data_comprehension(df)
    yearly_draft_analysis(df)

    # evaluate each draft slot
    df = value_draft_pos(df)
    print("Top 20 Players by Evaluation:\n", df.sort_values('player_val', ascending=False).head(20))

    # evaluate team and college ability picking and preparing players respectively
    team_evaluation(df)


Main()


"""
Used Resources:
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    - https://campus.datacamp.com/courses/visualizing-time-series-data-in-python/work-with-multiple-time-series?ex=9
    - https://pandas.pydata.org/docs/reference/api/pandas.Series.rank.html
    - https://dataindependent.com/pandas/pandas-rank-rank-your-data-pd-df-rank/
    - https://stackoverflow.com/questions/33750326/compute-row-average-in-pandas
    - https://sparkbyexamples.com/pandas/pandas-add-column-based-on-another-column/
    - https://www.stechies.com/indexerror-single-positional-indexer-outofbounds-error/#:~:text=What%20is%20this%20%E2%80%9CIndexerror%3A%20single,the%20scope%20of%20the%20index.
    - https://stackoverflow.com/questions/42739327/iloc-giving-indexerror-single-positional-indexer-is-out-of-bounds
    - https://stackoverflow.com/questions/20084487/use-index-in-pandas-to-plot-data
    - https://www.tutorialspoint.com/how-to-change-the-color-of-a-single-bar-if-a-condition-is-true-matplotlib
    - https://matplotlib.org/stable/gallery/color/named_colors.html
    - 
"""
