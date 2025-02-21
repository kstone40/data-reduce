"""Plotting utilities for reduction algorithms"""
import plotly.graph_objects as go
from plotly.graph_objects import Figure
import numpy as np


def show_reduction(x: np.ndarray, x_culled: np.ndarray, id: list[int] = [0, 1]) -> Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x[:, id[0]], y=x[:, id[1]], mode='lines', name='Original Data'))
    fig.add_trace(go.Scatter(x=x_culled[:, id[0]], y=x_culled[:, id[1]], mode='lines+markers', name='Reduced Data', opacity=0.7))
    fig.update_xaxes(title_text='X')
    fig.update_yaxes(title_text='Y')
    fig.update_layout(height=600, width=800, template='simple_white')
    return fig
