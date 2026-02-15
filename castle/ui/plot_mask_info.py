"""Mask information plotting and visualization utilities."""

import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np


class Plotter:
    """Static plotting utilities for ROI mask kinematics (position, speed, area).

    All methods are ``@staticmethod`` and return Plotly figures built from
    per-ROI result dicts containing 'x', 'y', and 'area' time series.
    """

    @staticmethod
    def plot_position(results):
        num = len(results)
        fig = sp.make_subplots(rows=num, cols=1)

        row_col = [(i+1, 1) for i in range(num)]
        for index, it in enumerate(results):
            r, c = row_col[index]
            fig.add_trace(go.Scatter(
                x=list(range(len(it['x']))),
                y=it['x'], 
                mode='lines', name=f'ROI{index+1}.x'), r, c)
            fig.add_trace(go.Scatter(
                x=list(range(len(it['y']))),
                y=it['y'],
                mode='lines', name=f'ROI{index+1}.y'), r, c)
        fig.update_layout(template="plotly_dark")
        # fig.update_layout(title_text="XY-coordinates of the colors over time")
        return fig
    
    @staticmethod
    def plot_speed(results):
        num = len(results)
        fig = sp.make_subplots(rows=num, cols=1)

        row_col = [(i+1, 1) for i in range(num)]
        for index, it in enumerate(results):
            r, c = row_col[index]

            speed = np.zeros(len(it['x']))
            dx = np.array(it['x'][1:] - it['x'][:-1])
            dy = np.array(it['y'][1:] - it['y'][:-1])
            speed[1:] = np.sqrt(dx*dx+dy*dy)

            fig.add_trace(go.Scatter(
                x=list(range(len(speed))),
                y=speed,
                mode='lines', name=f'ROI{index+1}.speed'), r, c)
        fig.update_layout(template="plotly_dark")
        # fig.update_layout(title_text="XY-coordinates of the colors over time")
        return fig
    

    @staticmethod
    def plot_area(results):
        num = len(results)
        fig = sp.make_subplots(rows=num, cols=1)

        row_col = [(i+1, 1) for i in range(num)]
        for index, it in enumerate(results):
            r, c = row_col[index]
            fig.add_trace(go.Scatter(
                x=list(range(len(it['area']))),
                y=it['area'], 
                mode='lines', name=f'ROI{index+1}.area'), r, c)
        fig.update_layout(template="plotly_dark")
        # fig.update_layout(title_text="area of the colors over time")
        return fig
    
    @staticmethod
    def create_pandas(results):
        """Create a DataFrame with per-ROI kinematics.
        
        Delegates to castle.utils.analysis_utils.create_kinematic_dataframe()
        to keep the canonical implementation in the utils layer.
        """
        from castle.utils.analysis_utils import create_kinematic_dataframe
        return create_kinematic_dataframe(results)