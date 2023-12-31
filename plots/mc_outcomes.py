import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_mc_outcome(trades, n_sims, n_samples, title, resample=False):
    """
    Performs Monte Carlo simulations on trade data to assess the performance and risk of a trading strategy.

    Parameters:
    trades (array-like): An array of trade results (profits or losses).
    n_sims (int): Number of Monte Carlo simulations to run.
    n_samples (int): Number of samples to draw in each simulation.
    title (str): Title for the resulting plot.
    resample (bool): Whether to resample the trades with replacement (True for bootstrapping).

    This function simulates different trading scenarios by randomly sampling from the provided trades data.
    It then calculates cumulative returns, median returns, upper and lower quantiles, and maximum drawdowns.
    These statistics are visualized using Plotly, showing the median equity curve, the range between the 90th and 10th quantiles,
    and a histogram of the maximum drawdowns.

    Returns:
    None. The function plots the results and prints key statistics.
    """
    
    # Initialize an array to store results from each simulation
    mc_res = []
    
    n_trades = trades.shape[0]
    
    if n_trades < n_samples:
        n_samples = len(trades)
    
    # Perform the MC simulations
    for _ in range(n_sims):
    
        random_trades = np.random.choice(
            a=trades,
            size=n_samples,
            replace=resample,
        )
    
        # Calculate cumulative returns based on the sampled trades and append to the array
        mc_res.append(np.cumprod(1+random_trades))
    
    # Convert the list of arrays into a 2D NumPy array for easy analysis
    mc_res = np.array(mc_res)
    
    # Calculate statistics over each column, this gets the mean and standard deviation for that point in time
    median_returns = np.median(mc_res, axis=0)
    upper_quantile = np.quantile(mc_res, 0.9, axis=0)
    lower_quantile = np.quantile(mc_res, 0.1, axis=0)
    
    # Get the max drawdown per equity curve
    peak_values = np.maximum.accumulate(mc_res, axis=1)
    drawdowns = peak_values/mc_res - 1
    max_drawdowns = -np.max(drawdowns, axis=1)
    
    # Create a Plotly figure with subplots
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.15)
    
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_samples), 
            y=median_returns, 
            mode='lines', 
            name='Median Equity Curve',
        ), 
        row=1, 
        col=1,
    )
    
    # Add the shaded area for the quantiles
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_samples),
            y=upper_quantile,
            mode='lines', 
            fill='tonexty', 
            line={'color':'gray'},
            name='90/10 quantiles',
        ), 
        row=1, 
        col=1,
    )
    
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_samples),
            y=lower_quantile,
            mode='lines', 
            fill='tonexty', 
            line={'color':'gray'},
            showlegend=False,
        ), 
        row=1, 
        col=1,
    )
    
    fig.add_trace(
        go.Histogram(
            x=max_drawdowns, 
            showlegend=False,
        ), 
        row=2,
        col=1,
    )
    
    # Update subplot layout and labels
    fig.update_xaxes(title_text='Trade Index', row=1, col=1)
    fig.update_xaxes(title_text='Drawdown', row=2, col=1)
    fig.update_yaxes(title_text='Cumulative Returns', row=1, col=1)
    fig.update_yaxes(title_text='Number of Sims', row=2, col=1)
    
    # Update subplot titles and overall title
    fig.update_layout(title_text=title, legend={'orientation': 'h', 'x': 0, 'y': -0.1})
    
    # Show the Plotly figure
    fig.show()
    
    # Print statistics
    print(f"Median Returns: {median_returns[-1]:.2f}")
    print(f"90% Quantile: {upper_quantile[-1]:.2f}")
    print(f"10% Quantile: {lower_quantile[-1]:.2f}")
    print(f"Maximum Drawdown (worst case): {np.min(max_drawdowns):.2f}")
    
    return