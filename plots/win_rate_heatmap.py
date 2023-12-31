import pandas as pd
import plotly.graph_objects as go


def win_rate_heatmap(df):
    """
    Generates and displays a heatmap of the win rate of trades by month and year.
    
    Parameters:
    df (DataFrame): A Pandas DataFrame containing the trading data. 
                    The DataFrame should have columns 'Date' and 'trade_res', 
                    where 'Date' is the date of the trade and 'trade_res' is 
                    the result of the trade (profit or loss).
    
    The function adds new columns to the DataFrame for the win status of each trade, 
    the year, and the month extracted from the 'Date' column. It then aggregates 
    this data by year and month, calculates the win rate, and displays a heatmap 
    using Plotly, where each cell shows the win rate for a particular month and year, 
    along with the number of wins over total trades.
    
    Returns:
    None. The function is used for plotting the heatmap.
    """

    # Find which rows correspond to winning trades
    df['win'] = df['trade_res'] > 0

    # Convert the Date to a datetime object, to extract the month and year
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Group over the year and month to aggregate statistics
    df = (
        df
        .groupby(['Year', 'Month'])
        .agg({'win': 'sum', 'trade_res': 'count'})
        .rename(columns={'trade_res': 'n_trades'})
        .reset_index()
    )

    df['win_rate'] = 100*(df['win']/df['n_trades'])

    # Form the heatmap text (to be used in each cell) - note that plotly
    # needs the new line separator in HTML format
    df['heatmap_text'] = (
        df['win_rate'].astype(int).astype(str)
        + '%<br>'
        + df['win'].astype(str)
        + '/'
        + df['n_trades'].astype(str)
    )

    # Get the pivot tables for both the heatmap data, and the text
    heatmap_data = df.pivot_table(
        index='Year', 
        columns='Month', 
        values='win_rate',
    )

    heatmap_text = df.pivot_table(
        index='Year', 
        columns='Month', 
        values='heatmap_text',
        aggfunc='first',
    )

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=heatmap_data.columns.values,
            y=heatmap_data.index.values,
            z=heatmap_data.values,
            text=heatmap_text.values,
            texttemplate='%{text}',
            textfont={'size': 12},
            colorbar={'title': 'Win Rate (%)'},
        )
    )

    fig.update_layout(
        title='Monthly Win Rate Heatmap',
        xaxis={
            'title': 'Month',
            'tickmode': 'array',
            'tickvals': list(range(1, 13)),
            'ticktext': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        },
        yaxis={
            'title': 'Year',
            'tickvals': heatmap_data.index.values,
        },
    )

    fig.show()
    
    return