import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import numpy as np
from lifelines.statistics import logrank_test
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def log_rank_analysis(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    survival_time_column: str = "EFSTM",
    survival_event_column: str = "EFSSTAT"
) -> float:
    """
    Perform a log-rank test between two survival cohorts.

    Parameters
    ----------
    df1 : pd.DataFrame
        First cohort dataframe.

    df2 : pd.DataFrame
        Second cohort dataframe.

    survival_time_column : str, default="EFSTM"
        Survival outcome time column name.
    
    survival_event_column : str, default="EFSSTAT"
        Survival outcome event column name

    Returns
    -------
    float
        Log-rank test p-value.
    """
    
    results = logrank_test(
        df1[survival_time_column],
        df2[survival_time_column],
        event_observed_A=df1[survival_event_column],
        event_observed_B=df2[survival_event_column],
    )

    return float(results.p_value)



def kaplan_meier_curve( df1: pd.DataFrame,
    df2: pd.DataFrame,
    survival_time_column: str = "EFSTM",
    survival_event_column: str = "EFSSTAT",
    df1_plot_label: str = "Original",
    df2_plot_label: str = "Synthetic",
    display_logrank_pvalue: bool = False,
    df1_plot_color: str = "turquoise",
    df2_plot_color: str = "tomato",
    ax: Axes = None,
    ci_alpha: float =0.1, 
    linewidth: float =2 , 
    add_risk_counts: bool = False) :
    
    kmf_1 = KaplanMeierFitter()
    kmf_2 = KaplanMeierFitter()

    kmf_1.fit(
        durations=df1[survival_time_column],
        event_observed=df1[survival_event_column],
        label=df1_plot_label
            )
    kmf_2.fit(
        durations=df2[survival_time_column],
        event_observed=df2[survival_event_column],
        label=df2_plot_label
    )


    if ax is None:
        _, ax = plt.subplots()

    kmf_1.plot_survival_function(ci_show=True, ax=ax, c=df1_plot_color, linewidth=linewidth, ci_alpha=ci_alpha)
    kmf_2.plot_survival_function(ci_show=True, ax=ax, c=df2_plot_color, linewidth=linewidth, ci_alpha=ci_alpha)
    
    if add_risk_counts:
        add_at_risk_counts(kmf_1,kmf_2, ax=ax)
        
    ax.set_xlabel("Time in Months")
    ax.set_ylabel("Survival probability")
    legend=ax.legend(frameon=False)
    if display_logrank_pvalue:
        
        pvalue=log_rank_analysis(
        df1,
        df2,
        survival_time_column=survival_time_column,
        survival_event_column=survival_event_column)
        
        bbox = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
        ax.text(
        0.98, 0.80,
        f"Log-rank $\\it{{p}}$ value = {pvalue}",
        transform=ax.transAxes,
        ha="right",
        fontsize=10
        )
    return ax


def short_sightedness_score(df1: pd.DataFrame,
    df2: pd.DataFrame,
    survival_time_column: str = "EFSTM",
    survival_event_column: str = "EFSSTAT"):
    
    """
    Compute short-sightedness between real and synthetic survival data.

    Original Norcliffe et al. style definition:
        short_sightedness = (T_real - T_syn) / T_real

    where T_real is the maximum KM time horizon in the real data
    and T_syn is the maximum KM time horizon in the synthetic data.

    Parameters
    ----------
    df1 : pd.DataFrame
        Dataframe with survival time and survival event values

    df2 : pd.DataFrame
        Dataframe with survival time and survival event values

    survival_time_column : str, default="EFSTM"
        Survival outcome time column name.
    
    survival_event_column : str, default="EFSTM"
        Survival outcome event column name
        
    clip : bool
        If True, clip at 0 so that longer synthetic horizons do not
        produce negative values.

    Returns
    -------
    float
        Short-sightedness score.
        0   = no short-sightedness
        1   = extreme short-sightedness
    """
    # time to event variables
    time_real=df1[survival_time_column]
    event_real=df1[survival_event_column]
    time_syn=df2[survival_time_column]
    event_syn=df2[survival_event_column]

    #initialize KM graphs
    kmf_real = KaplanMeierFitter()
    kmf_syn = KaplanMeierFitter()

    kmf_real.fit(time_real, event_observed=event_real)
    kmf_syn.fit(time_syn, event_observed=event_syn)

    # KM support / horizon
    T_real = kmf_real.survival_function_.index.max()
    T_syn = kmf_syn.survival_function_.index.max()

    if T_real <= 0:
        raise ValueError("T_real must be > 0")

    score = (T_real - T_syn) / max(T_real, T_syn)

    return float(score)


def km_divergence(df1: pd.DataFrame,
    df2: pd.DataFrame,
    survival_time_column: str = "EFSTM",
    survival_event_column: str = "EFSSTAT"):
   
    """
    Compute KM divergence between real and synthetic survival data.
    
    Parameters
    ----------
    df1 : pd.DataFrame
        Dataframe with survival time and survival event values

    df2 : pd.DataFrame
        Dataframe with survival time and survival event values

    survival_time_column : str, default="EFSTM"
        Survival outcome time column name.
    
    survival_event_column : str, default="EFSSTAT"
        Survival outcome event column name

    Returns
    -------
    float
        KM divergence score (lower = better)
    """
    
    
    # time to event variables
    time_real=df1[survival_time_column]
    event_real=df1[survival_event_column]
    time_syn=df2[survival_time_column]
    event_syn=df2[survival_event_column]

    # Km curves
    kmf_real = KaplanMeierFitter()
    kmf_syn = KaplanMeierFitter()

    # Fit KM curves
    kmf_real.fit(time_real, event_observed=event_real)
    kmf_syn.fit(time_syn, event_observed=event_syn)

    # Create a common time grid (union of event times)
    times = np.unique(
        np.concatenate([kmf_real.survival_function_.index.values,
                        kmf_syn.survival_function_.index.values])
    )

    # Interpolate survival probabilities onto common grid
    S_real = kmf_real.predict(times)
    S_syn = kmf_syn.predict(times)

    # Compute absolute difference
    diff = np.abs(S_real - S_syn)

    # Numerical integration (trapezoidal rule)
    km_div = np.trapz(diff, times) / (times.max() - times.min())

    return km_div


def optimism_score(df1: pd.DataFrame,
    df2: pd.DataFrame,
    survival_time_column: str = "EFSTM",
    survival_event_column: str = "EFSSTAT", 
    normalize : bool =True):
    """
    Compute optimism between real and synthetic survival data.

    Parameters
    ----------
    df1 : pd.DataFrame
        Dataframe with survival time and survival event values

    df2 : pd.DataFrame
        Dataframe with survival time and survival event values

    survival_time_column : str, default="EFSTM"
        Survival outcome time column name.
    
    survival_event_column : str, default="EFSSTAT"
        Survival outcome event column name
   
    normalize : bool
        Whether to normalize by time range
    Returns
    -------
    float
        Optimism score (lower = better)
    """
    
    # time to event variables
    time_real=df1[survival_time_column]
    event_real=df1[survival_event_column]
    time_syn=df2[survival_time_column]
    event_syn=df2[survival_event_column]

    #initialize KM graphs
    kmf_real = KaplanMeierFitter()
    kmf_syn = KaplanMeierFitter()

    # Fit KM curves
    kmf_real.fit(time_real, event_observed=event_real)
    kmf_syn.fit(time_syn, event_observed=event_syn)

    # Common time grid
    times = np.unique(
        np.concatenate([
            kmf_real.survival_function_.index.values,
            kmf_syn.survival_function_.index.values
        ])
    )

    # Interpolated survival probabilities
    S_real = kmf_real.predict(times)
    S_syn = kmf_syn.predict(times)

    # SIGNED difference (important!)
    diff = S_syn - S_real

    # Integrate
    optimism = np.trapz(diff, times)

    # Optional normalization
    if normalize and (times.max() > times.min()):
        optimism /= (times.max() - times.min())

    return optimism
    
