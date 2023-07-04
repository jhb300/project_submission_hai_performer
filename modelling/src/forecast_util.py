import matplotlib.pyplot as plt


def plot_forecast(tss: list, forecasts: list, path: str = None) -> None:
    """
    Plots the training target timeseries and the predicted forecast(s).
    If forecasts is an array of multiple forecast windows, these
    will be placed accordingly.

    Parameters
    ----------
    tss
        Yields the corresponding ground truth series.
    forecasts
        Yields the forecast timeseries.


    Returns: None
    """
    # Plot TTS
    for ts in tss:
        plt.plot(ts.to_timestamp(), color="black")
    # Plot forecast(s)
    for forecast in forecasts:
        forecast.plot()
    plt.legend(["True values"], loc="upper left", fontsize="large")
    # Save figure
    if path:
        plt.savefig(path)
