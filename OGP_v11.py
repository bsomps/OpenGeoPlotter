from PyQt5 import QtWidgets
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QListWidget, QListWidgetItem
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QFont, QColor
from PyQt5 import uic
import traceback
from matplotlib.widgets import SpanSelector
from collections import defaultdict
from matplotlib.patches import Rectangle
import json
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QColor, QPalette, QIntValidator
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
import copy
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QDialogButtonBox
from PyQt5.QtWidgets import QInputDialog
from mpl_toolkits.mplot3d import Axes3D
import xml.etree.ElementTree as ET
from matplotlib.text import Annotation
from PyQt5 import QtGui
from PyQt5.QtWidgets import QFrame
import math
import csv
import Test_1_rc
import OpenGeoUI2
import pandas as pd
from matplotlib.ticker import FuncFormatter
import pickle
print(pd.__version__)
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from ColumnSelectionDialog import Ui_ColumnSelectionDialog
import os
from functools import partial
from OpenGeoUI2 import Ui_MainWindow  
import sys
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
import factor_analyzer
from factor_analyzer import FactorAnalyzer
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.path import Path
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import traceback
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import mplstereonet
from mplstereonet.stereonet_math import pole, plane
from mplstereonet.stereonet_axes import StereonetAxes
import numpy as np
from matplotlib.patches import Polygon
import rasterio
from rasterio.features import geometry_mask, geometry_window
from shapely.geometry import LineString
from scipy.interpolate import griddata
from sklearn.cluster import DBSCAN
from pykrige.ok import OrdinaryKriging
from scipy.interpolate import Rbf
from scipy.spatial import ConvexHull
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QComboBox, QPushButton, QColorDialog, QLabel, QSpinBox, QListView, QScrollArea, QWidget, QDoubleSpinBox, QLineEdit, QGroupBox, QHBoxLayout, QGridLayout, QFileDialog, QTableView, QApplication, QSlider, QCheckBox, QTextEdit
from PyQt5.QtWidgets import QProgressDialog
from OGP_help import HELP_TEXT






class ColumnSelectorDialog(QDialog): # Lithology parameters window

    plot_requested = pyqtSignal(dict)

    def __init__(self, df):
        super().__init__()

        # Create a QVBoxLayout for the dialog
        self.dialog_layout = QVBoxLayout(self)

        # Create a QScrollArea
        self.scroll = QScrollArea(self)
        self.dialog_layout.addWidget(self.scroll)

        # widget of the QScrollArea
        self.scroll_content = QWidget(self.scroll)
        self.scroll.setWidget(self.scroll_content)
        self.scroll.setWidgetResizable(True)  # Enable the scroll area to resize the widget

        # QVBoxLayout for the content of the QScrollArea
        self.layout = QVBoxLayout(self.scroll_content)

        self.setWindowTitle("Column and Color Selector")

        # Select lithology column
        self.layout.addWidget(QLabel("Lithology Column:"))
        self.lithology_combo = QComboBox()
        self.lithology_combo.addItems(df.columns)
        self.layout.addWidget(self.lithology_combo)

        # Select from_depth column
        self.layout.addWidget(QLabel("From Depth Column:"))
        self.from_depth_combo = QComboBox()
        self.from_depth_combo.addItems(df.columns)
        self.layout.addWidget(self.from_depth_combo)

        # Select to_depth column
        self.layout.addWidget(QLabel("To Depth Column:"))
        self.to_depth_combo = QComboBox()
        self.to_depth_combo.addItems(df.columns)
        self.layout.addWidget(self.to_depth_combo)

        # Lithology column changed
        self.lithology_combo.currentTextChanged.connect(self.lithology_column_changed)
        self.from_depth_combo.currentTextChanged.connect(self.from_depth_column_changed)
        self.to_depth_combo.currentTextChanged.connect(self.to_depth_column_changed)

        # Hold color variables
        self.color_buttons = {}
        self.length_spin_boxes = {}
        self.df = df
        
        # Cancel or accept
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.dialog_layout.addWidget(self.buttons)
       
        
        self.setLayout(self.dialog_layout)  # Set dialog_layout as the layout of the dialog
    
    def lithology_column_changed(self, text):
        self.lithology_column = text
        self.update_lithology_controls()

    def from_depth_column_changed(self, text):
        self.from_depth_column = text

    def to_depth_column_changed(self, text):
        self.to_depth_column = text
        
    def update_lithology_controls(self):
        for widget in self.color_buttons.values():
            self.layout.removeWidget(widget)
            widget.setParent(None)

        for widget in self.length_spin_boxes.values():
            self.layout.removeWidget(widget)
            widget.setParent(None)

        self.color_buttons.clear()
        self.length_spin_boxes.clear()

        # Reference to self.lithology_column 
        unique_values = self.df[self.lithology_column].unique()
        for value in unique_values:
            color_button = QPushButton(f"Choose color for {value}")
            color_button.clicked.connect(lambda _, v=value: self.choose_color(v))
            self.layout.addWidget(color_button)
            self.color_buttons[value] = color_button

            length_spin_box = QDoubleSpinBox()
            length_spin_box.setRange(0, 0.5)  # Adjust the range
            length_spin_box.setSingleStep(0.1)  # Allow for decimal point precision
            self.layout.addWidget(QLabel(f"Choose length for {value}:"))
            self.layout.addWidget(length_spin_box)
            self.length_spin_boxes[value] = length_spin_box

    def choose_color(self, value):
        color = QColorDialog.getColor()
        if color.isValid():
            self.color_buttons[value].setText(f"{value} color: {color.name()}")

    def get_colors(self):
        colors = {}
        for value, button in self.color_buttons.items():
            if ":" in button.text():
                _, color = button.text().split(": ")
                colors[value] = color
            else:
                colors[value] = 'white'  
        return colors

    def get_lengths(self):
        return {value: spin_box.value() for value, spin_box in self.length_spin_boxes.items()}

    def get_parameters(self):
        parameters = {
            'lithology_column': self.lithology_combo.currentText(),
            'from_column': self.from_depth_combo.currentText(),
            'to_column': self.to_depth_combo.currentText(),
            'colors': self.get_colors(),
            'lengths': self.get_lengths()
        }
        
        return parameters or {}
        
        
class OrderDialog(QtWidgets.QDialog): # Change order for grpahic log window
    def __init__(self, hole_ids, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Change Order")
        layout = QtWidgets.QVBoxLayout(self)

        self.comboboxes = []
        for _ in hole_ids:
            combobox = QtWidgets.QComboBox(self)
            combobox.addItems(hole_ids)
            self.comboboxes.append(combobox)
            layout.addWidget(combobox)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_order(self):
        return [combobox.currentText() for combobox in self.comboboxes]


class PlotWindow(QtWidgets.QMainWindow): # Graphic log plot window
    def __init__(self, parent, data, hole_ids, parameters, initial_unit="ft"):
        super().__init__()
        
        self.setWindowModality(QtCore.Qt.NonModal)
       
        self.data = data # Store lithology column
        self.hole_ids = hole_ids  # Store the list of hole IDs
        self.parameters = parameters  # Store the selected parameters
        self.figure = Figure(figsize=(8, 12))
        self.canvas = FigureCanvas(self.figure)
        self.setCentralWidget(self.canvas)
        self.lith_depth_unit = initial_unit
        self.main_window_reference = parent
        
        
        self.save_button = QtWidgets.QPushButton(self)
        self.save_button.setText("Save plot")
        self.save_button.clicked.connect(self.save_plot)
        
        self.change_order_button = QtWidgets.QPushButton("Change Order", self)
        self.change_order_button.clicked.connect(self.change_order)
        

        # Set the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.change_order_button)
        layout.addWidget(self.save_button) 
        layout.addWidget(self.canvas)
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        
        # Set the size
        base_width = 500
        max_width = 2000  
        required_width = base_width * len(hole_ids)
        final_width = min(required_width, max_width)
        self.resize(final_width, 1200)
        self.create_graphic_log()

        
    def updateLithDepthUnit(self, value): # Choose m of ft
        
        self.lith_depth_unit = "ft" if value == 0 else "m" # Meter to ft slider
        
   
    def create_graphic_log(self): # Function to create graphic log
    
        # Clear Prevoius figure
        self.figure.clear()
        
        # Compute the maximum depth across all selected hole_ids
        max_depth = self.data[self.data['hole_id'].isin(self.hole_ids)][self.parameters['to_column']].max()
        
        num_holes = len(self.hole_ids)
        
        # Find hole_id and depth columns
        for idx, hole_id in enumerate(self.hole_ids):
            hole_data = self.data[self.data['hole_id'] == hole_id]
            ax = self.figure.add_subplot(1, num_holes, idx+1)
            
            if len(self.hole_ids) == 1:
                hole_min_depth = hole_data[self.parameters['from_column']].min()
                ax.set_ylim(hole_data[self.parameters['to_column']].max(), hole_min_depth)

            else:
                ax.set_ylim(max_depth, 0)
                
            
            if idx == 0:
                ax.set_ylabel(f'Depth ({self.lith_depth_unit})')  # only label y-axis for the leftmost plot

            else:
                ax.set_ylabel('')  # remove y-axis tick labels for other plots
                
            self.figure.subplots_adjust(left=0.15, wspace=0.45)
            
            # Sore variables
            previous_lithology = None
            previous_end_depth = None
            segment_start_depth = None
            y_positions = []  # List to store y-coordinates of plotted labels
            depth_range = hole_data[self.parameters['to_column']].max() - hole_data[self.parameters['from_column']].min()
            label_buffer_percentage = 0.015
            label_buffer = depth_range * label_buffer_percentage

            # Find lith columns and colors
            for _, row in hole_data.iterrows():
                lithology = row[self.parameters['lithology_column']]
                from_depth = row[self.parameters['from_column']]
                to_depth = row[self.parameters['to_column']]
                color = self.parameters['colors'].get(lithology, 'white')
                length = self.parameters['lengths'].get(lithology, 0.0)

                # Check if the lithology has changed
                if lithology != previous_lithology:
                    # If the previous lithology is not None, draw the span for the previous sequence
                    if previous_lithology is not None:
                        ax.axhspan(previous_end_depth, segment_start_depth, xmin=0, xmax=prev_length, facecolor=color_prev, edgecolor='k', linewidth=0.5, alpha=0.7)

                        y_center = (segment_start_depth + previous_end_depth) / 2
                        if all(abs(y - y_center) > label_buffer for y in y_positions):
                            ax.text(0.45, y_center, previous_lithology, fontsize=8)
                            y_positions.append(y_center)

                    # Reset segment_start_depth for the new lithology
                    segment_start_depth = from_depth

                color_prev = color  # Store the color of the current row to be used in the next iteration
                prev_length = length  # Store the length of the current row to be used in the next iteration
                previous_lithology = lithology
                previous_end_depth = to_depth

            # Plot the last segment
            if previous_lithology is not None:
                ax.axhspan(previous_end_depth, segment_start_depth, xmin=0, xmax=prev_length, facecolor=color_prev, edgecolor='k', linewidth=0.5, alpha=0.7)

                y_center = (segment_start_depth + previous_end_depth) / 2
                if all(abs(y - y_center) > label_buffer for y in y_positions):
                    ax.text(0.45, y_center, previous_lithology, fontsize=8)


            ax.set_xlim(0, 0.7)  # Full range of X-axis
            

            ax.set_xlabel('')
            ax.set_xticks([])

            if self.lith_depth_unit == "ft":
                ax.annotate(f'{max_depth} ft', xy=(0, max_depth), xytext=(10, -10), textcoords='offset points')
            else:
                ax.annotate(f'{max_depth} m', xy=(0, max_depth), xytext=(10, -10), textcoords='offset points')
            
            ax.set_title(f"Hole ID: {hole_id}")

        self.figure.subplots_adjust(left=0.15)
        self.canvas.draw()
        self.show()
        self.setWindowTitle("Graphic Log Generator")

    def change_order(self):
        dialog = OrderDialog(self.hole_ids, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            new_order = dialog.get_order()
            if len(set(new_order)) != len(self.hole_ids):
                # Ensure there are no duplicate selections
                QtWidgets.QMessageBox.warning(self, "Invalid Order", "Please select distinct holes for each order position.")
                return
            self.hole_ids = new_order
            self.create_graphic_log()
        
    def save_plot(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "All Files (*);;JPEG (*.jpeg);;PNG (*.png);;SVG (*.svg)", options=options)
        if file_name:
            self.figure.savefig(file_name, dpi=200)

    
    def closeEvent(self, event):
        if self in self.main_window_reference.plot_windows:
            self.main_window_reference.plot_windows.remove(self)
        event.accept()  # window close



class DownholePlotWindow(QtWidgets.QMainWindow):  # Plot window for downhole geochem
    def __init__(self, main_window, data, hole_id, column_data, column_name, depth_column, plot_bars=False):
        super().__init__()
        
        self.setWindowModality(QtCore.Qt.NonModal)
       
        # Store viariables
        self.data = data
        self.hole_id = hole_id
        self.column_data = column_data  
        self.column_name = column_name  
        self.main_window = main_window
        self.plot_bars = plot_bars
        self.figure = Figure(figsize=(8, 12))
        self.canvas = FigureCanvas(self.figure)
        self.setCentralWidget(self.canvas)
        
        self.save_button = QtWidgets.QPushButton(self)
        self.save_button.setText("Save plot")
        self.save_button.clicked.connect(self.save_plot)
        self.depth_column = depth_column  # Save depth column
        self.geochem_depth_unit = "ft"  # Default to feet

        # Set the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.save_button)  
        layout.addWidget(self.canvas)
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        
        self.resize(400, 1100)
        self.setGeometry(0, 0, 400, 1100)
        self.plot()
        
    def updategeochemDepthUnit(self, value):
        self.geochem_depth_unit = "ft" if value == 0 else "m" 
        self.update_labels()
        
    def closeEvent(self, event):
        # Disconnect the signal for updating y-axis label
        self.main_window.geochem_ft_m.valueChanged.disconnect(self.updategeochemDepthUnit)

        # Construct the key for this window
        window_key = f"{self.hole_id}_{self.column_name}"

        # Remove the reference of this window from the geochem_plot_windows dictionary
        if window_key in self.main_window.geochem_plot_windows:
            del self.main_window.geochem_plot_windows[window_key]

        event.accept()  # let the window close

        
    def plot(self):
        self.plot_data()
        self.update_labels()
        self.canvas.draw()
        self.show()

    def plot_data(self):

        # Create a mask where the column_data is not zero
        mask = self.column_data != 0

        # Use the mask to filter the x data
        y = self.data[self.depth_column]
        x = self.column_data[mask]
        y = y[mask]  # Ensures y-values correspond to the filtered x-values

        self.ax = self.figure.add_subplot(111)

        if self.plot_bars:
            width = 3  # Adjust as necessary
            self.ax.barh(y, x, align='center', height=width, color='gray')  # Using horizontal bars
        else:
            self.ax.plot(x, y)
        self.ax.invert_yaxis()  # To display depth with min at the top and max at the bottom
        
        # Set the y-axis limits based on the minimum and maximum depth values in the data
        self.ax.set_ylim(y.max(), y.min())  # Use the actual depth data

        # Set the x-axis limits based on the minimum and maximum values in the data
        upper_limit = x.max() + 0.10 * x.max()
        self.ax.set_xlim(0, upper_limit)


        self.setWindowTitle(f"Hole: {self.hole_id} - {self.column_name}")  # Updated
        
        # make the plot fit the pop up window 
        self.figure.subplots_adjust(left=0.18)

        

    def update_labels(self):
        self.ax.set_xlabel(self.column_name)
        self.ax.set_ylabel(f'Depth ({self.geochem_depth_unit})')
        self.ax.set_title(f"Hole ID: {self.hole_id}")
       
        
    def save_plot(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self,"Save Plot","","All Files (*);;JPEG (*.jpeg);;PNG (*.png)", options=options)
        if file_name:
            self.figure.savefig(file_name)

        
class CorrelationMatrixWindow(QtWidgets.QMainWindow): # window for correlation matrix
    def __init__(self, data, hole_id, tag='_ppm', parent=None):
        super(CorrelationMatrixWindow, self).__init__(parent)

        # Create the matplotlib Figure and FigCanvas objects.
        self.hole_id = hole_id
        self.fig, self.ax = plt.subplots(figsize=(9, 12))
        self.canvas = FigureCanvas(self.fig)
        
        
        # add save button
        self.save_button = QtWidgets.QPushButton(self)
        self.save_button.setText("Save plot")
        
        # Adjust the button size and position
        self.save_button.resize(100,30)  # Size of button
        self.save_button.move(20,20)
        
        self.save_button.clicked.connect(self.save_plot)
        
        # Set the layout
        layout = QtWidgets.QVBoxLayout()
        
        # Create and add the toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        
        
        layout.addWidget(self.save_button)
        layout.addWidget(self.canvas)
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        
        
        # Compute and draw the correlation matrix
        self.draw_correlation_matrix(data, tag)
        self.ax.set_title(f"Correlation Matrix - {self.hole_id}")
        
    def draw_correlation_matrix(self, data, tag):
        # Select columns with the specified tag in their names
        selected_columns = [col for col in data.columns if tag in col]
        data_selected = data[selected_columns]

        # Compute the correlation matrix
        corr_matrix = data_selected.corr()

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio, without annotations
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, 
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=self.ax)
                    
        self.canvas.draw()  
        self.show()  # Show the window
        
    def save_plot(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self,"Save Plot","","All Files (*);;JPEG (*.jpeg);;PNG (*.png)", options=options)
        if file_name:
            self.fig.savefig(file_name)

            

class ColumnSelectionDialog(QtWidgets.QDialog, Ui_ColumnSelectionDialog): # Window for downhole plot selection
    def __init__(self, parent=None):
        super(ColumnSelectionDialog, self).__init__(parent)
        self.setupUi(self)

        self.plot_button.clicked.connect(self.accept)

        # Initialize depth column combo box
        self.depth_column_combo = QtWidgets.QComboBox(self)

        # Initialize QLabel for depth column
        self.depth_column_label = QLabel("Select Depth Column")
        
        # Initialize QLabel for attributes
        self.attributes_label = QLabel("Select Attributes to Plot")
        
        # Initialize checkbox for bar plot
        self.plot_bars_checkbox = QtWidgets.QCheckBox("Plot using bars", self)

        # Layout to organize widgets
        layout = QVBoxLayout(self)

        # Add QLabel and QComboBox to layout
        layout.addWidget(self.depth_column_label)
        layout.addWidget(self.depth_column_combo)

        # Add QLabel and QListWidget to layout
        layout.addWidget(self.attributes_label)
        layout.addWidget(self.plot_bars_checkbox)
        layout.addWidget(self.column_listWidget)
        
        
        # Add the plot button
        layout.addWidget(self.plot_button)

        # Set layout
        self.setLayout(layout)

    def load_columns(self, columns):
        for column in columns:
            item = QListWidgetItem(column)
            item.setCheckState(Qt.Unchecked)
            self.column_listWidget.addItem(item)

    def load_depth_columns(self, depth_columns):
        self.depth_column_combo.addItems(depth_columns)

            


class CrossSection(QDialog): # Cross section window
    MAX_BAR_LENGTH = 50 # for auxiliary bar plot
    def __init__(self, data, hole_ids, azimuth, attribute_column=None, attributes_model=None, attributes_dict=None, DEM_data=None, remove_outliers=True, remove_outliers_auxiliary=True, checkbox_add_grid_lines=True, upper_quantile=75.0, lower_quantile=25.0, IQR=3.0, x_buffer=120.0, y_buffer=0.05, line_width=4, selected_hole_ids_for_labels=None):
        super().__init__()
        print(selected_hole_ids_for_labels)  

        # Storing the data, hole_ids, and azimuth as instance variables
        self.data = data
        self.hole_ids = hole_ids
        self.azimuth = azimuth
        self.attribute_column = attribute_column
        self.attributes_dict = attributes_dict or {}
        self.categorical_encodings = {}
        manage_attributes_dialog = ManageAttributesDialog(data)
        self.drag_region = None
        self.is_plan_view = False
        self.generate_contours_flag = False
        self.canvases = []
        self.DEM_data = DEM_data
        if DEM_data is not None:
            self.DEM_loaded = True
        else:
            self.DEM_loaded = False

        
        self.overlay_image_state = {
            'x': None,
            'y': None,
            'width': None,
            'height': None,
        }

        self.remove_outliers = remove_outliers
        self.upper_quantile = upper_quantile
        self.lower_quantile = lower_quantile
        self.IQR = IQR
        self.x_buffer = x_buffer
        self.y_buffer = y_buffer
        self.line_width = line_width 
        self.selected_hole_ids_for_labels = selected_hole_ids_for_labels
        print("CrossSection - selected_hole_ids_for_labels:", self.selected_hole_ids_for_labels)
        self.filtered_bar_data = self.data
        self.remove_outliers_auxiliary = remove_outliers_auxiliary
        self.bar_vmin = None
        self.bar_vmax = None
        self.checkbox_add_grid_lines = checkbox_add_grid_lines
        self.attributes_model = attributes_model
        self.setup_attribute_list_view()
        self.attributes_model.itemChanged.connect(self.on_attribute_selection_changed)
        self.pencil_mode = False
        self.sky_color = 'lightsteelblue'
        self.remove_topo_and_sky = False
        self.currently_drawing = False
        self.drawing_lines = []
        self.current_overlay_image_display = None
        self.overlay_image = None
        self.overlay_image_state = {'x': 0, 'y': 0, 'width': 0, 'height': 0}
        self.use_user_defined_grid = False
        self.filtered_grid_points = None
        self.y_axis_scale_factor = 1
        
        
        
        # Set up the main vertical layout for the QDialog
        main_layout = QVBoxLayout(self)

        # Create a Figure that will hold the plot
        self.figure = Figure(figsize=(10, 15))

        # Create a FigureCanvasQTAgg widget that will hold the Figure
        self.canvas = FigureCanvas(self.figure)

        # Connect press events and click events
        self.cid_key = self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.cid_move = self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self.on_mouse_release)

        self.dragging_overlay = False

        # Create and add the toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(self.toolbar)

        # Create a QHBoxLayout for canvas and buttons
        self.layout = QHBoxLayout()

        # Create a QVBoxLayout for the buttons
        button_layout = QVBoxLayout()
        
        # Add "Topo Line Settings" button
        self.topo_line_settings_button = QPushButton("Topo Line Settings", self)
        self.topo_line_settings_button.clicked.connect(self.on_topo_line_settings_clicked)
        button_layout.addWidget(self.topo_line_settings_button)
        
        # Add 'Y-axis Scale Factor' label and input
        y_axis_scale_label = QLabel("Vertical Exaggeration")
        button_layout.addWidget(y_axis_scale_label)

        # QLineEdit for scale factor input
        self.y_axis_scale_factor_input = QLineEdit(self)
        self.y_axis_scale_factor_input.setPlaceholderText("Enter vertical exaggeration (e.g., 2)")
        button_layout.addWidget(self.y_axis_scale_factor_input)

        

        # Add hover tool
        hover_tool_btn = QPushButton("Hover Tool")
        hover_tool_btn.clicked.connect(self.activate_hover_tool)
        button_layout.addWidget(hover_tool_btn)

        # Add image overlay 
        add_image_btn = QPushButton("Add Image Overlay")
        add_image_btn.clicked.connect(self.add_image_overlay)
        button_layout.addWidget(add_image_btn)

        # Add plan view 
        self.toggle_view_button = QPushButton("Change to Plan View", self)
        self.toggle_view_button.clicked.connect(self.toggle_view)
        button_layout.addWidget(self.toggle_view_button)

        # Add bar plot 
        self.secondary_bar_plot_button = QPushButton("Auxiliary Bar Plot", self)
        self.secondary_bar_plot_button.clicked.connect(self.secondary_bar_plot)
        button_layout.addWidget(self.secondary_bar_plot_button)
        
        # Add Generate Contours button
        self.generate_contours_button = QPushButton("Interpolate Contours (RBF)", self)
        self.generate_contours_button.clicked.connect(self.generate_contours)
        button_layout.addWidget(self.generate_contours_button)
        
        # Add Pencil tool button
        self.pencil_tool_button = QPushButton("Pencil", self)
        self.pencil_tool_button.setCheckable(True)  # Make the button toggleable
        self.pencil_tool_button.clicked.connect(self.toggle_pencil_tool)
        button_layout.addWidget(self.pencil_tool_button)
        
        # Add "Save to CSV" button
        self.save_to_csv_button = QPushButton("Export Plot to CSV", self)
        self.save_to_csv_button.clicked.connect(self.on_save_to_csv_clicked)
        button_layout.addWidget(self.save_to_csv_button)

       
        # Create the QLabel for the title
        attribute_list_label = QLabel("Change Attribute")
        attribute_list_label.setAlignment(Qt.AlignCenter)  # Center align the text
        
        # Set the font to bold
        font = attribute_list_label.font()
        font.setBold(True)
        attribute_list_label.setFont(font)

        # Add space above the label
        button_layout.addSpacing(20)  # Adjust the spacing value as needed

        # Add the QLabel to the layout
        button_layout.addWidget(attribute_list_label)

        # Add space below the label
        button_layout.addSpacing(10)  # Adjust the spacing value as needed

        # Add the attribute list
        button_layout.addWidget(self.attribute_list_view)
        
        
        # Add a label for the azimuth control
        azimuth_label = QLabel("Change Azimuth")
        azimuth_label.setAlignment(Qt.AlignCenter)  # Center align the text
        button_layout.addWidget(azimuth_label)
        
        # Set the font to bold
        font = azimuth_label.font()
        font.setBold(True)
        azimuth_label.setFont(font)

        # Add QSpinBox for azimuth
        self.azimuth_spin_box = QSpinBox(self)
        self.azimuth_spin_box.setRange(0, 360)  # Assuming azimuth range is 0-360 degrees
        # Set the value of azimuth spin box as an integer
        self.azimuth_spin_box.setValue(int(self.azimuth))
        self.azimuth_spin_box.valueChanged.connect(self.on_azimuth_changed)
        button_layout.addWidget(self.azimuth_spin_box)
        
        # Add "Redraw Plot" button
        self.redraw_plot_button = QPushButton("Redraw Plot", self)
        self.redraw_plot_button.clicked.connect(self.on_redraw_button_clicked)
        button_layout.addWidget(self.redraw_plot_button)
        
        # Add a stretch to push the buttons to the top
        button_layout.addStretch(1)
        
       
        # Add the QVBoxLayout to the QHBoxLayout
        self.layout.addLayout(button_layout, stretch=1)

        # Add the canvas to the QHBoxLayout
        self.layout.addWidget(self.canvas, stretch=5)  

        # Add the QHBoxLayout to the main QVBoxLayout
        main_layout.addLayout(self.layout)

        # Set the main QVBoxLayout as the layout for the QDialog
        self.setLayout(main_layout)

        # Resize the QDialog
        self.resize(1200, 825)
        
        # Include maximize and minimize buttons
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)

        self.plot()  # Create the plot
        self.setWindowTitle("Cross Section Visualizer")
        
        
    def set_y_axis_scale_factor(self):
        try:
            # Get scale factor from input and store it in the class variable
            self.y_axis_scale_factor = float(self.y_axis_scale_factor_input.text())

            # Check if scale factor is positive
            if self.y_axis_scale_factor <= 0:
                raise ValueError("Scale factor must be positive")

            self.plot()
            
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", str(e))
            self.y_axis_scale_factor = 1  # Reset to default if there's an error
        
    def on_save_to_csv_clicked(self):
        # Open a file dialog to choose where to save the CSV
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "",
                                                  "CSV Files (*.csv)", options=options)
        if fileName:
            try:
                # Save the extended data to CSV
                self.export_data.to_csv(fileName, index=False)
            except Exception as e:
                # Handle exceptions (e.g., IOError)
                print("Error saving file:", e)
        
    def toggle_pencil_tool(self):
        self.pencil_mode = self.pencil_tool_button.isChecked()
        if not self.pencil_mode:
            self.clear_drawings()
            
        
    def on_topo_line_settings_clicked(self):
        dialog = QDialog(self)
        layout = QVBoxLayout(dialog)

        # Color picker for the sky
        color_label = QLabel("Select Sky Color:")
        color_picker = QPushButton("Choose Color")
        color_picker.clicked.connect(lambda: self.choose_sky_color(color_picker))

        # Checkboxes for removing sky and both topo line and sky
        remove_sky_checkbox = QCheckBox("Remove Sky")
        remove_topo_and_sky_checkbox = QCheckBox("Remove Topo Line and Sky")

        # Offset input
        offset_label = QLabel("Topo Line Offset: Postive values to move up, negative values to move down")
        offset_input = QLineEdit()
        offset_input.setText("0")  # Set the default value to 0

        # OK and Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        # Add widgets to layout
        layout.addWidget(color_label)
        layout.addWidget(color_picker)
        layout.addWidget(remove_sky_checkbox)
        layout.addWidget(remove_topo_and_sky_checkbox)
        layout.addWidget(offset_label)
        layout.addWidget(offset_input)
        layout.addWidget(buttons)

        result = dialog.exec_()
        if result:
            # Process selected color
            selected_color = color_picker.property("color")
            if selected_color:
                self.sky_color = selected_color

            # Process remove sky
            if remove_sky_checkbox.isChecked():
                self.sky_color = 'white'

            # Process remove topo line and sky
            self.remove_topo_and_sky = remove_topo_and_sky_checkbox.isChecked()
           
            # Process offset
            try:
                offset_value = float(offset_input.text())
                self.offset = offset_value
                self.generate_contours_flag = False
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter a valid number.")

            self.plot()  # Replot with the new settings

    def choose_sky_color(self, button):
        color = QColorDialog.getColor()
        if color.isValid():
            button.setStyleSheet("background-color: %s;" % color.name())
            button.setProperty("color", color.name())
            
    
        
    def select_contour_column(self):
        # Get a list of numerical columns
        numerical_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if not numerical_columns:
            QMessageBox.warning(self, "Warning", "No numerical columns found in the data.")
            return

        # Create a custom dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Contour Options")
        layout = QVBoxLayout(dialog)

        # Column selection combo box
        column_label = QLabel("Choose a numerical column for contouring:")
        column_combo = QComboBox(dialog)
        column_combo.addItems(numerical_columns)
        layout.addWidget(column_label)
        layout.addWidget(column_combo)

        # Input for Q1 percentile
        Q1_input = QDoubleSpinBox(dialog)
        Q1_input.setRange(0, 100)
        Q1_input.setValue(25)  # Default value
        Q1_input.setSuffix("%")
        layout.addWidget(QLabel("Enter Q1 percentile:"))
        layout.addWidget(Q1_input)

        # Input for Q3 percentile
        Q3_input = QDoubleSpinBox(dialog)
        Q3_input.setRange(0, 100)
        Q3_input.setValue(75)  # Default value
        Q3_input.setSuffix("%")
        layout.addWidget(QLabel("Enter Q3 percentile:"))
        layout.addWidget(Q3_input)

        # Input for IQR scaling factor
        scale_input = QDoubleSpinBox(dialog)
        scale_input.setRange(0, 100)
        scale_input.setValue(1.5)  # Default value
        layout.addWidget(QLabel("Enter IQR scaling factor:"))
        layout.addWidget(scale_input)
        
        # Checkbox for disabling outlier capping
        disable_outliers_checkbox = QCheckBox("Disable outlier capping", dialog)
        layout.addWidget(disable_outliers_checkbox)
        

        # OK and Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        # Show the dialog and get the result
        if dialog.exec_():
            selected_column = column_combo.currentText()
            Q1 = Q1_input.value()
            Q3 = Q3_input.value()
            scale = scale_input.value()

            # Store the selected options as instance variables
            self.contour_column = selected_column
            self.outlier_Q1 = Q1
            self.outlier_Q3 = Q3
            # Store the state of the disable outliers checkbox
            self.disable_outliers = disable_outliers_checkbox.isChecked()
            self.outlier_scale = scale
            
            
           

    def generate_contours(self):
        # Call the method to select the contour column and set outlier parameters
        self.select_contour_column()

        # set the flag and replot
        if hasattr(self, 'contour_column'):  # Check if the column was successfully selected
            self.generate_contours_flag = True
            self.plot()  # Replot with contours
        
    def on_azimuth_changed(self, value):
        self.azimuth = float(value)

    def setup_attribute_list_view(self):
        self.attribute_list_view = QListView(self)
        self.attribute_list_view.setModel(self.attributes_model)
        self.attribute_list_view.selectionModel().selectionChanged.connect(self.on_attribute_selection_changed)
        
    def on_attribute_selection_changed(self, item):
        # Check if the item is checked
        if item.checkState() == Qt.Checked:
            # Uncheck all other items
            for row in range(self.attributes_model.rowCount()):
                other_item = self.attributes_model.item(row)
                if other_item != item:
                    other_item.setCheckState(Qt.Unchecked)

            # Store the selected attribute
            self.attribute_column = item.text()
            
    def on_redraw_button_clicked(self):
        
        # Create and configure the progress dialog
        self.progress_dialog = QProgressDialog("Redrawing plot...", "Cancel", 0, 0, self)
        self.progress_dialog.setModal(True)  # Make the dialog modal
        self.progress_dialog.setWindowTitle("Redrawing plot...")
        self.progress_dialog.setCancelButton(None)  # Optionally remove the cancel button
        self.progress_dialog.show()

        # Update the plot
        self.update_plot(self.attribute_column)

    def update_plot(self, attribute_column):
        self.attribute_column = attribute_column
        self.generate_contours_flag = False
        self.bar_column = None

        
        # Check if the y-axis scale factor input is empty
        if not self.y_axis_scale_factor_input.text().strip():
            # If the input is empty, assume scaling factor is 1
            self.y_axis_scale_factor = 1
        else:
            # If there is an input, use it as the scaling factor
            try:
                self.y_axis_scale_factor = float(self.y_axis_scale_factor_input.text())
                if self.y_axis_scale_factor <= 0:
                    raise ValueError("Scale factor must be positive")
            except ValueError:
                # Handle invalid input or reset to default
                self.y_axis_scale_factor = 1
                QMessageBox.warning(self, "Input Error", "Invalid Y-axis scale factor. Resetting to 1.")

        self.plot()  # Redraw the plot 

        self.redraw_image()  # Redraw the overlay image after updating the plot

        # Close the progress dialog
        self.progress_dialog.close()
  

        
    def add_image_overlay(self):
        options = QFileDialog.Options()
        image_path, _ = QFileDialog.getOpenFileName(self, "Open Image Overlay", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)", options=options)
        if image_path:
            self.overlay_image = plt.imread(image_path)
            
            # Get the initial dimensions of the image
            initial_height, initial_width = self.overlay_image.shape[:2]
            
            # Compute the initial x and y coordinates
            ax = self.figure.axes[0]  # The first axis is the cross-section plot
            x_center = (ax.get_xlim()[1] + ax.get_xlim()[0]) / 2
            y_center = (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2

            initial_x = x_center - (initial_width / 2)
            initial_y = y_center - (initial_height / 2)

            self.overlay_image_state = {
                'x': initial_x,
                'y': initial_y,
                'width': initial_width,
                'height': initial_height,
            }

            # Display the overlay image using the computed x and y values
            self.current_overlay_image_display = ax.imshow(self.overlay_image, extent=[initial_x, initial_x + initial_width, initial_y, initial_y + initial_height], zorder=0)
            self.canvas.draw()

    


    def on_mouse_press(self, event):
        if self.pencil_mode:
            if event.inaxes:  # Click is inside the plot
                # Start a new line
                line = Line2D([event.xdata], [event.ydata], color='black', zorder=6)
                self.figure.axes[0].add_line(line)
                self.drawing_lines.append(line)
                self.currently_drawing = True  # Start drawing
            else:  # Click is outside the plot
                self.clear_drawings()  # Clear drawings
                self.pencil_tool_button.setChecked(False)  # Turn off the pencil tool
                self.pencil_mode = False  # Update the pencil mode state
        else:
            # Ensure the mouse press is within the overlay image
            if (self.overlay_image_state['x'] <= event.xdata <= self.overlay_image_state['x'] + self.overlay_image_state['width'] and
                    self.overlay_image_state['y'] <= event.ydata <= self.overlay_image_state['y'] + self.overlay_image_state['height']):
                
                
                # Determine the region of the image that was clicked
                x_diff = event.xdata - self.overlay_image_state['x']
                y_diff = event.ydata - self.overlay_image_state['y']

                # margin within which a corner or edge is considered clicked
                margin = 50  # pixels

                # Explicitly handle each region
                if x_diff < margin and y_diff < margin:
                    print("Detected bottom_left corner")
                    self.drag_region = 'bottom_left'
                elif x_diff > self.overlay_image_state['width'] - margin and y_diff < margin:
                    print("Detected bottom_right corner")
                    self.drag_region = 'bottom_right'
                elif x_diff < margin and y_diff > self.overlay_image_state['height'] - margin:
                    print("Detected top_left corner")
                    self.drag_region = 'top_left'
                elif x_diff > self.overlay_image_state['width'] - margin and y_diff > self.overlay_image_state['height'] - margin:
                    print("Detected top_right corner")
                    self.drag_region = 'top_right'
                elif x_diff < margin:
                    print("Detected left edge")
                    self.drag_region = 'left'
                elif x_diff > self.overlay_image_state['width'] - margin:
                    print("Detected right edge")
                    self.drag_region = 'right'
                else:
                    print("Detected center of the image")
                    self.drag_region = 'center'
                
                # Store initial mouse press location and set dragging_overlay to True
                self.drag_start = (event.xdata, event.ydata)
                self.dragging_overlay = True
            
            self.canvas.setFocusPolicy(Qt.StrongFocus)
            self.canvas.setFocus()


    def on_mouse_move(self, event):
        if self.pencil_mode and self.currently_drawing and event.inaxes and self.drawing_lines:
            # Add point to the current line
            line = self.drawing_lines[-1]  # Get the last line
            xdata, ydata = line.get_data()
            xdata = np.append(xdata, event.xdata)
            ydata = np.append(ydata, event.ydata)
            line.set_data(xdata, ydata)
            self.canvas.draw_idle()
        elif self.dragging_overlay:
            if not self.dragging_overlay or event.xdata is None or event.ydata is None:
                return

            # Calculate how much the mouse has moved
            dx = event.xdata - self.drag_start[0]
            dy = event.ydata - self.drag_start[1]
                
            # Determine how to adjust the overlay based on the drag region
            if self.drag_region == 'center':
                self.overlay_image_state['x'] += dx
                self.overlay_image_state['y'] += dy

            # Redraw the overlay image with the updated position
            self.redraw_image()


            # Update the drag start position for the next move event
            self.drag_start = (event.xdata, event.ydata)
            print(f"Image moved/scaled to {self.overlay_image_state['x']}, {self.overlay_image_state['y']}.")
        
    def on_key_press(self, event):
    
        if any(val is None for val in self.overlay_image_state.values()):
            print('Some overlay image state values are missing.')
            return


        scale_factor = 1.05  # Scale the image by 5% each time

        if event.key == '=': # Scale uniform bigger
            self.overlay_image_state['width'] *= scale_factor
            self.overlay_image_state['height'] *= scale_factor

        elif event.key == '-': # Scale uniform smaller
            self.overlay_image_state['width'] /= scale_factor
            self.overlay_image_state['height'] /= scale_factor

        elif event.key == 'a':  # Scale left side in
            self.overlay_image_state['x'] += scale_factor
            self.overlay_image_state['width'] -= scale_factor

        elif event.key == 'w':  # Scale top side down
            self.overlay_image_state['y'] += scale_factor
            self.overlay_image_state['height'] -= scale_factor

        elif event.key == 's':  # Scale bottom side up
            self.overlay_image_state['y'] -= scale_factor
            self.overlay_image_state['height'] += scale_factor

        elif event.key == 'd':  # Scale right side in
            self.overlay_image_state['width'] -= scale_factor

        elif event.key == 'i':  # Scale top side up
            self.overlay_image_state['y'] -= scale_factor
            self.overlay_image_state['height'] -= scale_factor

        elif event.key == 'j':  # Scale left side out
            self.overlay_image_state['x'] -= scale_factor
            self.overlay_image_state['width'] += scale_factor

        elif event.key == 'l':  # Scale right side out
            self.overlay_image_state['width'] += scale_factor

        elif event.key == 'k':  # Scale bottom side down
            self.overlay_image_state['height'] -= scale_factor
            
        elif event.key == 'backspace':  # Detect backspace key press
            if self.current_overlay_image_display:  
                self.current_overlay_image_display.remove()  # Remove the displayed image
                self.current_overlay_image_display = None  # Set the display reference to None
                self.overlay_image = None  

        # Redraw the image with the new scale or after deletion
        self.redraw_image()


    def redraw_image(self):
        ax = self.figure.axes[0]
        if self.current_overlay_image_display:
            self.current_overlay_image_display.remove()

        # Only draw if overlay image is set
        if self.overlay_image is not None:
            self.current_overlay_image_display = ax.imshow(
                self.overlay_image, 
                extent=[
                    self.overlay_image_state['x'], 
                    self.overlay_image_state['x'] + self.overlay_image_state['width'], 
                    self.overlay_image_state['y'], 
                    self.overlay_image_state['y'] + self.overlay_image_state['height']
                ], 
                zorder=-1
            )
        self.canvas.draw()
        

    def on_mouse_release(self, event):
        if self.pencil_mode:
            # When in pencil mode, stop drawing on mouse release
            self.currently_drawing = False
        else:
            # Existing logic for releasing the overlay image
            self.dragging_overlay = False
            self.drag_region = None


        
    def clear_drawings(self):
        for line in self.drawing_lines:
            line.remove()
        self.drawing_lines.clear()
        self.canvas.draw_idle()


    def calculate_coordinates(self):
        # calculate x, y coordinates of points along the cross-section line
        # based on the azimuth and the original x, y coordinates
        self.data['x_cross'] = self.data['x'] * np.cos(np.deg2rad(self.azimuth)) - self.data['y'] * np.sin(np.deg2rad(self.azimuth))
        self.data['y_cross'] = self.data['x'] * np.sin(np.deg2rad(self.azimuth)) + self.data['y'] * np.cos(np.deg2rad(self.azimuth))
        


    def column_type(self, column, dataframe):
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            return 'continuous'
        else:
            return 'categorical'
            
    def set_DEM_data(self, DEM_data):
        self.DEM_data = DEM_data
        

    def run_DEM_calc(self):
        print("Executing run_DEM_calc")
        if not hasattr(self, 'DEM_data') or self.DEM_data is None:
            QMessageBox.warning(self, "No DEM", "Please load a DEM first")
            return

        if self.data is None:
            QMessageBox.warning(self, "No data", "Please load desurveyed data first")
            return
            
        
        if self.DEM_data is not None:
            self.DEM_loaded = True
        
        self.hole_ids = [self.model_cross.item(index).text() for index in range(self.model_cross.rowCount())
                                if self.model_cross.item(index).checkState() == Qt.Checked]

        start_point, end_point = self.create_profile_line(self.hole_ids, self.azimuth, self.max_z)
    
        # Transform to plotting coordinates
        transformed_start_point = self.transform_coordinates(*start_point)
        transformed_end_point = self.transform_coordinates(*end_point)
        
        # Extract the topographic profile using real-world coordinates
        profile = self.extract_topographic_profile(start_point, end_point)
        
        # Plot the profile using transformed coordinates
        self.plot_elevation_profile_on_existing_plot(ax, profile, transformed_start_point, transformed_end_point)
        
        
    def extract_topographic_profile(self, original_start_point, original_end_point):
        print("extract_topographic_profile method invoked.")
        line = LineString([original_start_point, original_end_point])
        ##print(f"Start Point: {original_start_point}")
        ##print(f"End Point: {original_end_point}")
        window = geometry_window(self.DEM_data, [line])
        cropped_data = self.DEM_data.read(window=window)

        # Create a mask of the DEM that represents the path of the line
        mask = geometry_mask([line], transform=self.DEM_data.window_transform(window), invert=True, out_shape=cropped_data.shape[-2:])

        profile = np.extract(mask, cropped_data)
        nodata_value = -999999.0
        profile = profile[profile != nodata_value]
        
        ##print("Elevation Profile Extracted")
        ##print("Profile Length:", len(profile))
        ##print("Profile values range from:", profile.min(), "to", profile.max())
        
        
        ##print("Elevation Profile:", profile)
        
        return profile
        
        
    def create_profile_line(self, hole_ids, azimuth, max_z, min_x=None, max_x=None):
    
        self.calculate_coordinates()
        
        # Filter the sheet by hole_ids
        filtered_df = self.data[self.data['hole_id'].isin(hole_ids)]
        
        # Find the row with the maximum z value
        max_z_row = filtered_df.loc[filtered_df['z'].idxmax()]
        
        # Extract x and y coordinates of the max_z_row
        x_origin, y_origin = max_z_row['x'], max_z_row['y']
        
        if self.azimuth >= 180:
            adjusted_azimuth = self.azimuth - 180
        else:
            adjusted_azimuth = self.azimuth
        
        ##print("Input azimuth:", azimuth)
        theta = np.radians(adjusted_azimuth)
        ##print("Theta in radians:", theta)  
        
        # Define the length of the line to extend from the origin
        line_length = 5000  # 2000 meters in EITHER direction
        
        # Calculate changes in x and y based on the new azimuth
        dx = line_length * np.cos(theta)
        dy = line_length * np.sin(theta)
        print("dx:", dx, "dy:", dy)
        
        # Calculate left and right endpoints of the profile line based on dx and dy
        x_left_end = x_origin - dx
        y_left_end = y_origin + dy  
        x_right_end = x_origin + dx
        y_right_end = y_origin - dy  
        
        # Print the calculated endpoints
        ##print("Calculated start point:", (x_left_end, y_left_end))
        ##print("Calculated end point:", (x_right_end, y_right_end))
            
        # Return the start and end coordinates of the profile line
        return ((x_left_end, y_left_end), (x_right_end, y_right_end))
        
        
    def transform_coordinates(self, x, y):
        x_cross = x * np.cos(np.deg2rad(self.azimuth)) - y * np.sin(np.deg2rad(self.azimuth))
        y_cross = x * np.sin(np.deg2rad(self.azimuth)) + y * np.cos(np.deg2rad(self.azimuth))
        return x_cross, y_cross

        
    def plot_elevation_profile_on_existing_plot(self, ax, adjusted_profile, start_point, end_point):
        # Calculate the x values (distance along the profile)
        x_values = np.linspace(start_point[0], end_point[0], len(adjusted_profile))
        
        # Print the x and y values
        ##print("X Values:", x_values)
        ##print("Adjusted Y Values (Elevation Profile):", adjusted_profile)
        
        # Adjust the alpha level based on whether to remove topo line and sky
        alpha_level = 0 if self.remove_topo_and_sky else 0.5
        
        # Plot the adjusted topographic profile on the given axes
        ax.plot(x_values, adjusted_profile, color='black', alpha=alpha_level, zorder=4)

        
    def secondary_bar_plot(self):
    
        # Get a list of numerical columns
        numerical_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if not numerical_columns:
            QMessageBox.warning(self, "Warning", "No numerical columns found in the data.")
            return

        # Create a dialog for the user to select a column
        column, ok = QInputDialog.getItem(self, "Select Column", "Choose a numerical column for the secondary bar plot:", numerical_columns, 0, False)
        
        if ok and column:
            self.bar_column = column  # Store the selected column as an instance variable
            self.plot()  # Re-draw the plot
            self.canvas.draw()

        
    def plot_bars(self, bar_column, ax):
    
        # Ensure bar_column is valid
        if bar_column not in self.data.columns:
            print(f"'{bar_column}' is not a valid column in the data.")
            return
        
        # Filter the data for only the selected holes
        selected_holes_data = self.data[self.data['hole_id'].isin(self.hole_ids)]
        
        # Get the min and max values of the bar column for normalization
        min_bar_value, max_bar_value = selected_holes_data[bar_column].min(), selected_holes_data[bar_column].max()
        print(f"Bar column: {bar_column}")
        print(f"Min bar value: {min_bar_value}, Max bar value: {max_bar_value}")
        
        # Determine the range of the bar_column values based on outlier removal
        filtered_data = selected_holes_data.copy()
        if self.remove_outliers and self.remove_outliers_auxiliary:  
            # Only filter outliers if both conditions are met
            filtered_data = filtered_data[pd.to_numeric(filtered_data[bar_column], errors='coerce').notna()]
            Q1_bar = filtered_data[bar_column].quantile(self.lower_quantile / 100)
            Q3_bar = filtered_data[bar_column].quantile(self.upper_quantile / 100)
            IQR_bar = Q3_bar - Q1_bar
            filtered_data = filtered_data[~((filtered_data[bar_column] < (Q1_bar - self.IQR * IQR_bar)) | (filtered_data[bar_column] > (Q3_bar + self.IQR * IQR_bar)))]
            self.bar_vmin = filtered_data[bar_column].min()
            self.bar_vmax = filtered_data[bar_column].max()
        else:
            # Use original, unfiltered data
            self.bar_vmin = selected_holes_data[bar_column].min()
            self.bar_vmax = selected_holes_data[bar_column].max()
        
        self.filtered_bar_data = filtered_data
        
        y_min, y_max = ax.get_ylim()

        # Compute the offset based on the y-axis range
        OFFSET_FRACTION = 0.01
        offset = OFFSET_FRACTION * (y_max - y_min) * self.y_axis_scale_factor

        # Compute the maximum bar length based on the y-axis range
        MAX_BAR_LENGTH_FRACTION = 0.05
        MAX_BAR_LENGTH = MAX_BAR_LENGTH_FRACTION * (y_max - y_min) * self.y_axis_scale_factor
        
        for hole_id in self.hole_ids:
            hole_data = filtered_data[filtered_data['hole_id'] == hole_id]
            if hole_data.empty:
                continue

            # Sort the hole_data based on depth for proper segment computation
            hole_data = hole_data.sort_values('z', ascending=True)
            
            for i in range(len(hole_data) - 1):
                # Get the start and end points of the current segment in the drill trace
                start_x, start_y = hole_data.iloc[i]['x_cross'], hole_data.iloc[i]['z']
                end_x, end_y = hole_data.iloc[i+1]['x_cross'], hole_data.iloc[i+1]['z']
                
                # Compute the direction vector of the segment
                delta_x, delta_y = end_x - start_x, end_y - start_y
                magnitude = np.sqrt(delta_x**2 + delta_y**2)
                normalized_dir_vector = (delta_x/magnitude, delta_y/magnitude)
                
                # Compute the perpendicular vector
                perp_vector = (-normalized_dir_vector[1], normalized_dir_vector[0])
                
                # If hole is dipping eastward, adjust (mirror) the perpendicular vector
                if delta_x < 0:
                    perp_vector = (-perp_vector[0], -perp_vector[1])
                    
                # If hole is vertical or near-vertical, force bars to stick to right side 
                VERTICAL_THRESHOLD = 0.5  # Small threshold value
                if abs(delta_x) < VERTICAL_THRESHOLD:
                    perp_vector = (1, 0)  # Force bars to the right side

                # Compute the midpoint of the segment and offset it using the perpendicular vector
                mid_x = (start_x + end_x) / 2 + perp_vector[0] * offset
                mid_y = (start_y + end_y) / 2 + perp_vector[1] * offset

                # Compute the end point for the bar, based on the value in the selected column
                bar_value = hole_data.iloc[i][bar_column]
                normalized_bar_value = (bar_value - self.bar_vmin) / (self.bar_vmax - self.bar_vmin)
                bar_length = MAX_BAR_LENGTH * normalized_bar_value
                
                end_x = mid_x + bar_length * perp_vector[0]
                end_y = mid_y + bar_length * perp_vector[1]
                
                # Plot the bar
                ax.plot([mid_x, end_x], [mid_y, end_y], color='gray', zorder=10, alpha=0.5)
                
                
                
    def add_bar_legend(self, ax, bar_column):
        # Check if bar_column is None; if so, do not draw the legend
        if bar_column is None:
            return
    
        x_position = -0.11  # places the legend slightly to the right of the plot area.
        
        # Start and end y-positions for the bar
        y_position_max = 0.15
        y_position_min = y_position_max - 0.1

        # Draw the single gray bar using ax.plot()
        line, = ax.plot([x_position, x_position], [y_position_min, y_position_max], color='gray', linewidth=6, transform=ax.transAxes, zorder=10)
        line.set_clip_on(False)

        # Add labels
        ax.text(x_position - 0.03, y_position_max, f'{self.bar_vmax:.2f}', va='bottom', ha='right', transform=ax.transAxes)
        ax.text(x_position - 0.03, y_position_min, f'{self.bar_vmin:.2f}', va='top', ha='right', transform=ax.transAxes)
        
        # Set the title for the legend
        ax.text(x_position, y_position_max + 0.05, bar_column, va='center', ha='right', transform=ax.transAxes, fontweight='bold')

    def calculate_tick_interval(self, min_value, max_value):
        # Determine the total distance
        distance = max_value - min_value

        # Decide on an optimal number of ticks (you can adjust this)
        desired_num_ticks = 10

        # Calculate the rough interval
        rough_interval = distance / desired_num_ticks

        # Adjust the interval to a 'nice' number for readability
        magnitude = 10 ** np.floor(np.log10(rough_interval))
        interval = round(rough_interval / magnitude) * magnitude

        # Handle special cases for very large or small intervals
        if interval == 0:
            interval = magnitude
        if distance / interval > 2 * desired_num_ticks:
            interval *= 2

        # Adjust for specific distance thresholds
        if distance > 2000:
            interval = max(interval, 500)
        elif distance > 600:
            interval = max(interval, 200)
        else:
            interval = max(interval, 100)

        return interval


    def plot(self): # Main drill hole plotting function
   
        # Save the pencil line data 
        pencil_line_data = []
        for line in self.drawing_lines:
            xdata, ydata = line.get_data()
            pencil_line_data.append((xdata, ydata))
        
        self.calculate_coordinates()
        self.figure.clear()  # Clear the entire figure
        ax = self.figure.add_subplot(111)
        
        # Re-add the pencil drawings if there
        self.drawing_lines = []  # Clear existing line references
        for xdata, ydata in pencil_line_data:
            line = Line2D(xdata, ydata, color='black', zorder=6)
            ax.add_line(line)
            self.drawing_lines.append(line)

        self.canvas.draw_idle()  # Redraw the canvas with the new plot and pencil lines
        
        min_x_plan, max_x_plan = float('inf'), float('-inf')
        min_x_cross, max_x_cross = float('inf'), float('-inf')
        min_z, max_z = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        
        self.min_x_cross, self.max_x_cross = float('inf'), float('-inf')
        self.min_z, self.max_z = float('inf'), float('-inf')

        # Filter the data based on selected hole_ids
        selected_data = self.data[self.data['hole_id'].isin(self.hole_ids)]

        # Update logic to determine actual axis limits
        for row in selected_data.itertuples():
            self.min_z = min(self.min_z, row.z)
            self.max_z = max(self.max_z, row.z)
        
        # Initialize
        filtered_data = None
        vmin, vmax = None, None
        
        # Set up normailzation
        if self.attribute_column and self.column_type(self.attribute_column, self.data) == 'continuous':
            if self.remove_outliers:
                filtered_data = self.data[pd.to_numeric(self.data[self.attribute_column], errors='coerce').notna()].copy()
                Q1 = filtered_data[self.attribute_column].quantile(self.lower_quantile / 100)
                Q3 = filtered_data[self.attribute_column].quantile(self.upper_quantile / 100)
                IQR = Q3 - Q1
                
                # Remove outliers from filtered_data for the other plots
                filtered_data = filtered_data[~((filtered_data[self.attribute_column] < (Q1 - self.IQR * IQR)) | (filtered_data[self.attribute_column] > (Q3 + self.IQR * IQR)))]
                
                vmin = filtered_data[self.attribute_column].min()
                vmax = filtered_data[self.attribute_column].max()
            else:
                # Use the original data without filtering for outliers
                vmin = self.data[self.attribute_column].min()
                vmax = self.data[self.attribute_column].max()
                
        if hasattr(self, 'bar_column') and hasattr(self, 'bar_vmin') and hasattr(self, 'bar_vmax'):
            pass  # bar_vmin and bar_vmax are already set as instance attributes
        else:
            self.bar_vmin, self.bar_vmax = None, None
     
                
        print(f"vmin: {vmin}, vmax: {vmax}")
        print(f"bar_vmin: {self.bar_vmin}, bar_vmax: {self.bar_vmax}")
        
        color_dict = {}
        
        
        # For RBF contour plotting
        if self.generate_contours_flag:
            # Check if the necessary parameters are set
            if not hasattr(self, 'contour_column') or not hasattr(self, 'outlier_Q1') or not hasattr(self, 'outlier_Q3') or not hasattr(self, 'outlier_scale'):
                QMessageBox.warning(self, "Warning", "Contour parameters not set.")
                return

            # Extract necessary data
            data = self.data[self.data['hole_id'].isin(self.hole_ids)]
            x_cross, z, values = data['x_cross'], data['z'], data[self.contour_column]

            # Convert to numpy arrays and remove NaNs
            x_cross, z, values = map(np.array, [x_cross, z, values])
            mask = np.isfinite(x_cross) & np.isfinite(z) & np.isfinite(values)
            x_cross, z, values = x_cross[mask], z[mask], values[mask]
            
            if not self.disable_outliers:
                Q1_percentile = np.percentile(values, self.outlier_Q1)
                Q3_percentile = np.percentile(values, self.outlier_Q3)
                IQR = Q3_percentile - Q1_percentile
                upper_threshold = Q3_percentile + self.outlier_scale * IQR
                lower_threshold = Q1_percentile - self.outlier_scale * IQR

                # Cap the outliers
                values = np.clip(values, lower_threshold, upper_threshold)

            # Extend the grid for the boundary
            boundary_width = 0.1  # adjust as needed
            extended_grid_x = np.linspace(min(x_cross) - boundary_width, max(x_cross) + boundary_width, num=50)
            extended_grid_z = np.linspace(min(z) - boundary_width, max(z) + boundary_width, num=50)
            extended_grid_x, extended_grid_z = np.meshgrid(extended_grid_x, extended_grid_z)

            # Perform RBF interpolation on the original data
            rbf = Rbf(x_cross, z, values, function='multiquadric', smooth=0.05)
            z_pred = rbf(extended_grid_x, extended_grid_z)

            # Create a mask for the original data range
            mask = ((extended_grid_x >= min(x_cross)) & (extended_grid_x <= max(x_cross)) & 
                    (extended_grid_z >= min(z)) & (extended_grid_z <= max(z)))

            # Apply the mask to set boundary values to zero
            z_pred[~mask] = 0  # Set values outside the original data range to zero

            # Plotting
            contourf = ax.contourf(extended_grid_x, extended_grid_z, z_pred, levels=10, cmap='Spectral_r', alpha=0.5, zorder=0)

            
            # levels to be made transparent
            num_transparent_levels = 3

            # Find collections corresponding to the first 3 contour levels and set their alpha to 0
            for collection in contourf.collections[:num_transparent_levels]:
                collection.set_alpha(0)  # Set alpha to 0 for these levels
                
        # New DataFrame to store the extended data
        self.extended_data = pd.DataFrame()

          
        # Drill hole plotting
        for hole_id in self.hole_ids:
            hole_data = self.data[self.data['hole_id'] == hole_id]
            if hole_data.empty:
                
                continue
                
            # Plot the full hole trace as a thin black line
            if self.is_plan_view:
                ax.plot(hole_data['x'], hole_data['y'], color='black', linewidth=0.5, alpha=0.5, zorder=4)
            else:
                ax.plot(hole_data['x_cross'], hole_data['z'], color='black', pickradius=50, linewidth=0.5, alpha=0.5, zorder=1)

            
            hole_data.sort_values('z', inplace=True, ascending=False)
            
            # Initialize a list to store colors for each point
            hole_colors = []
            
            # Contineous data plotting
            if self.attribute_column:
                if self.column_type(self.attribute_column, self.data) == 'continuous':
                    cmap = plt.get_cmap('Spectral_r')
                    norm = plt.Normalize(vmin=vmin, vmax=vmax)
                    for i in range(1, len(hole_data)):
                        current_value = hole_data[self.attribute_column].iloc[i]
                        
                        if pd.notna(current_value):  # Only plot when current data point has attribute data          
                            if self.is_plan_view:
                                x_values = hole_data['x'].iloc[i-1:i+1]
                                y_values = hole_data['y'].iloc[i-1:i+1]
                                
                                ax.plot(x_values, y_values, color=cmap(norm(current_value)), linewidth=self.line_width, zorder=5, picker=True, pickradius=50, label=str(i))
                                
                            else:
                                x_values = hole_data['x_cross'].iloc[i-1:i+1]
                                z_values = hole_data['z'].iloc[i-1:i+1]
                                
                                ax.plot(x_values, z_values, color=cmap(norm(current_value)), linewidth=self.line_width, zorder=5, picker=True, pickradius=50, label=str(i))
                            
                            
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                
                # Categorical data plotting
                else: 
                    if self.attribute_column in self.attributes_dict:
                        for i in range(1, len(hole_data)):
                            x_values = hole_data['x_cross' if not self.is_plan_view else 'x'].iloc[i-1:i+1]
                            y_values = hole_data['z' if not self.is_plan_view else 'y'].iloc[i-1:i+1]

                            if pd.notna(hole_data[self.attribute_column].iloc[i]):  # Only plot when attribute data exists
                                color_val = hole_data[self.attribute_column].iloc[i]

                                if color_val in self.attributes_dict[self.attribute_column]:
                                    settings = self.attributes_dict[self.attribute_column][color_val]
                                    
                                    # Check if line_width is 0, if so set to 0.5 and color to black
                                    line_width = settings["line_width"] if settings["line_width"] != 0 else 0.5
                                    color = settings["color"] if settings["line_width"] != 0 else "black"

                                    ax.plot(x_values, y_values, color=color, linewidth=line_width, zorder=5, picker=True, pickradius=50, label=str(i-1))
                                    
                                else:
                                    # Default when the color value is not found in the predefined attributes dictionary
                                    ax.plot(x_values, y_values, color="black", linewidth=0.5, alpha=0.5, picker=True, pickradius=50, zorder=4)
                            else:
                                # Default when the attribute data is missing
                                ax.plot(x_values, y_values, color="black", linewidth=0.5, alpha=0.5, picker=True, pickradius=50, zorder=4)

                    else:
                        # If the attribute column isn't in the predefined attributes dictionary, use default color mapping
                        unique_vals = self.data[self.attribute_column].unique()
                        colors = plt.cm.get_cmap('tab20', len(unique_vals))
                        color_dict = dict(zip(unique_vals, colors(range(len(unique_vals)))))

                        for i in range(1, len(hole_data)):
                            x_values = hole_data['x_cross' if not self.is_plan_view else 'x'].iloc[i-1:i+1]
                            y_values = hole_data['z' if not self.is_plan_view else 'y'].iloc[i-1:i+1]

                            if pd.notna(hole_data[self.attribute_column].iloc[i]):  # Only plot when attribute data exists
                                color_val = hole_data[self.attribute_column].iloc[i]

                                if color_val in color_dict:
                                    ax.plot(x_values, y_values, color=color_dict[color_val], linewidth=self.line_width, zorder=5, picker=True, pickradius=50, label=str(i-1))
                                else:
                                    # Default when the color value is not found in the generated color dictionary
                                    ax.plot(x_values, y_values, color="black", linewidth=0.5, alpha=0.5, zorder=1)
                            else:
                                # Default when the attribute data is missing
                                ax.plot(x_values, y_values, color="black", linewidth=0.5, alpha=0.5, picker=True, zorder=1)
            
            # No data plotting
            else:
                x_values = hole_data['x_cross' if not self.is_plan_view else 'x']
                y_values = hole_data['z' if not self.is_plan_view else 'y']
                ax.plot(x_values, y_values, color='black', linewidth=0.5, alpha=0.5, zorder=4)
                
                
            # Color collection for Continuous and Categorical data plotting
            for i in range(len(hole_data)):
                color = 'black'
                if self.attribute_column:
                    # Determine the color for continuous data
                    if self.column_type(self.attribute_column, self.data) == 'continuous':
                        current_value = hole_data[self.attribute_column].iloc[i]
                        if pd.notna(current_value):
                            cmap = plt.get_cmap('Spectral_r')
                            norm = plt.Normalize(vmin=vmin, vmax=vmax)
                            color = cmap(norm(current_value))
                        else:
                            color = 'black'  # Default color for missing data

                    # Determine the color for categorical data
                    elif self.attribute_column in self.attributes_dict:
                        color_val = hole_data[self.attribute_column].iloc[i]
                        if pd.notna(color_val) and color_val in self.attributes_dict[self.attribute_column]:
                            settings = self.attributes_dict[self.attribute_column][color_val]
                            # Set color to black if line_width is 0, else use the predefined color
                            color = 'black' if settings["line_width"] == 0 else settings["color"]
                        else:
                            color = 'black'  # Default color
                else:
                    color = 'black'  # Default color for no data

                # Append the color to the list
                hole_colors.append(mcolors.to_hex(color))

            # Update extended_hole_data with the colors
            self.extended_hole_data = hole_data.copy()
            self.extended_hole_data['color'] = hole_colors
            
            # Initialize a list to store line widths
            line_widths = []

            # Iterate over each row in hole_data
            for i, row in hole_data.iterrows():
                if self.attribute_column and pd.notna(row[self.attribute_column]):
                    # Check if the attribute_column is categorical
                    if self.column_type(self.attribute_column, self.data) == 'categorical':
                        # For categorical data, get line width from the predefined dictionary
                        color_val = row[self.attribute_column]
                        if self.attribute_column in self.attributes_dict and color_val in self.attributes_dict[self.attribute_column]:
                            settings = self.attributes_dict[self.attribute_column][color_val]
                            line_width = settings["line_width"] if settings["line_width"] != 0 else 0.5
                            line_widths.append(line_width)
                        else:
                            # If color_val is not in self.attributes_dict, use the default line width
                            line_widths.append(self.line_width)
                    else:
                        # If the attribute_column is not categorical, use the default line width
                        line_widths.append(self.line_width)

                else:
                    # If the attribute column does not exist or the data point has no attribute data
                    line_widths.append(0.5)

            # Assign the list of line widths to the extended_hole_data
            self.extended_hole_data['line_width'] = line_widths
            
            self.extended_data = pd.concat([self.extended_data, self.extended_hole_data])
            
            # Keep only necessary columns
            export_columns = ['hole_id', 'x', 'y', 'z', 'color', 'line_width']

            # Add the attribute column if it's valid
            if self.attribute_column and self.attribute_column in self.extended_data.columns:
                export_columns.append(self.attribute_column)

            # Keep only necessary columns
            self.export_data = self.extended_data[export_columns]

            # Find min and max's
            min_x_plan = min(min_x_plan, hole_data['x'].min())
            max_x_plan = max(max_x_plan, hole_data['x'].max())
            
            self.min_x_cross = min(self.min_x_cross, hole_data['x_cross'].min())
            self.max_x_cross = max(self.max_x_cross, hole_data['x_cross'].max())
            
            self.min_z = min(self.min_z, hole_data['z'].min())
            self.max_z = max(self.max_z, hole_data['z'].max())
            
            min_y = min(min_y, hole_data['y'].min())
            max_y = max(max_y, hole_data['y'].max())
           
            min_depth_index = hole_data['z'].idxmin()
            
            print(self.selected_hole_ids_for_labels, hole_id)
            
            if self.selected_hole_ids_for_labels is None or hole_id in self.selected_hole_ids_for_labels:
            
                if self.is_plan_view:
                    self.plot_label_within_limits(ax, hole_data['x'].loc[min_depth_index], hole_data['y'].loc[min_depth_index], str(hole_id))
                else:
                    self.plot_label_within_limits(ax, hole_data['x_cross'].loc[min_depth_index], hole_data['z'].loc[min_depth_index] - 1, str(hole_id))
            
        if not self.is_plan_view and hasattr(self, 'bar_column'):
            try:
                self.plot_bars(self.bar_column, ax)
            except Exception as e:
                print("Error while plotting bars:", e)

    
        
        if 'sm' in locals(): # Legend variables
            self.figure.colorbar(sm, ax=ax, orientation='vertical', label=self.attribute_column, shrink=0.5)
        
        if self.attribute_column and self.column_type(self.attribute_column, self.data) != 'continuous':
            if self.attribute_column in self.attributes_dict:
                legend_elements = []
                for val, attributes in self.attributes_dict[self.attribute_column].items():
                    if attributes['line_width'] != 0:
                        legend_element = Line2D([0], [0], color=attributes['color'], lw=4, label=val)
                        legend_elements.append(legend_element)
            else:
                legend_elements = [Line2D([0], [0], color=color_dict[val], lw=4, label=val) for val in color_dict]
            
            if legend_elements:  # Only add the legend if there are any elements to show
                ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.6))
                
        # Add the bar legend
        if hasattr(self, 'bar_column'):
            self.add_bar_legend(ax, self.bar_column)

        # buffer x and y/z 
        self.buffer_x = self.x_buffer
        self.buffer_yz = (self.max_z - self.min_z) * self.y_buffer

        if self.is_plan_view:
            ax.set_xlim([min_x_plan - self.buffer_x, max_x_plan + self.buffer_x])
            ax.set_ylim([min_y - self.buffer_yz, max_y + self.buffer_yz])
            ax.set_xlabel('Easting', labelpad=15)
            ax.set_ylabel('Northing', labelpad=15)
            ax.set_aspect('equal')
            
        else:
            # For cross-section view
            min_x_with_buffer = self.min_x_cross - self.buffer_x
            max_x_with_buffer = self.max_x_cross + self.buffer_x

            ax.set_xlim([min_x_with_buffer, max_x_with_buffer])
            ax.set_ylim([self.min_z - self.buffer_yz, self.max_z + self.buffer_yz])
            ax.set_xlabel('Distance along cross-section line (m)', labelpad=20)
            ax.set_ylabel('Elevation / Depth (m)', labelpad=20)
            ax.tick_params(axis='x', labelsize=8)

            # Apply the y-axis scale factor
            if self.y_axis_scale_factor != 1:
                ax.set_aspect(self.y_axis_scale_factor)
            else:
                ax.set_aspect('equal')

            # Calculate and set dynamic ticks using distance
            tick_interval = self.calculate_tick_interval(min_x_with_buffer, max_x_with_buffer)
            offset = -min_x_with_buffer  # Adjust offset for buffer
            tick_values = np.arange(min_x_with_buffer, max_x_with_buffer, tick_interval)
            tick_labels = [f"{int(tick + offset)}" for tick in tick_values]  # Adjust for offset
            ax.set_xticks(tick_values)
            ax.set_xticklabels(tick_labels)
            
            # Apply the y-axis scale factor
            if self.y_axis_scale_factor != 1:
                ax.set_aspect(self.y_axis_scale_factor) 
            else:
                ax.set_aspect('equal')

            # colors for sky
            colors = ["white", self.sky_color]
            cmap_bg = LinearSegmentedColormap.from_list("", colors)

            # If the DEM is loaded
            if self.DEM_loaded:
                print("DEM_loaded is True. Proceeding with topographic profile plotting.")

                # Create the profile line using the new version of create_profile_line
                start_point, end_point = self.create_profile_line(self.hole_ids, self.azimuth, self.max_z)

                # Transform these to plotting coordinates
                transformed_start_point = self.transform_coordinates(*start_point)
                transformed_end_point = self.transform_coordinates(*end_point)

                # Extract the topographic profile
                profile = self.extract_topographic_profile(start_point, end_point)

                # Adjust the profile values to raise the topographic profile
                adjusted_profile = [value + getattr(self, 'offset', 0) for value in profile]
                
                # Adjust the alpha level based on whether to remove topo line and sky
                alpha_level = 0 if self.remove_topo_and_sky else 1

                # Plot the adjusted topographic profile using transformed coordinates
                self.plot_elevation_profile_on_existing_plot(ax, adjusted_profile, transformed_start_point, transformed_end_point)
                
                # Calculate the x values for the topographic profile in transformed coordinates
                x_start_transformed = start_point[0] * np.cos(np.deg2rad(self.azimuth)) - start_point[1] * np.sin(np.deg2rad(self.azimuth))
                x_end_transformed = end_point[0] * np.cos(np.deg2rad(self.azimuth)) - end_point[1] * np.sin(np.deg2rad(self.azimuth))

                # Calculate the x values for the topographic profile using transformed coordinates
                x_values = np.linspace(x_start_transformed, x_end_transformed, len(adjusted_profile))

                # Define the upper boundary for the gradient
                upper_boundary = np.full_like(adjusted_profile, self.max_z + self.buffer_yz)
                

                # Use fill_between to fill the area above the topographic profile
                ax.fill_between(x_values, adjusted_profile, upper_boundary, color=self.sky_color, alpha=alpha_level, zorder=3)



            else:
                # If the DEM is not loaded, plot the gradient
                alpha_level = 0 if self.remove_topo_and_sky else 0.5  # Adjust alpha for gradient
                
                gradient = np.linspace(0, 1, 256)
                gradient = np.vstack((gradient, gradient))
                gradient = np.transpose(gradient)
                ax.imshow(gradient, aspect='auto', cmap=cmap_bg, extent=[self.min_x_cross - self.buffer_x, self.max_x_cross + self.buffer_x, self.max_z, self.max_z + self.buffer_yz], origin='lower', alpha=alpha_level)
                ax.axhline(self.max_z, color='k', linestyle='--', alpha=alpha_level)
            
            # Apply the y-axis scale factor
            if self.y_axis_scale_factor != 1:
                ax.set_aspect(self.y_axis_scale_factor)
            else:
                ax.set_aspect('equal')

        # Set up the titles
        if self.is_plan_view:
            ax.set_title('Plan View')
        else:
            if self.azimuth == 90:
                ax.set_title('Cross-Section View (Facing East)')
            elif self.azimuth == 180:
                ax.set_title('Cross-Section View (Facing South)')
            elif self.azimuth == 270:
                ax.set_title('Cross-Section View (Facing West)')
            elif self.azimuth == 0 or self.azimuth == 360:
                ax.set_title('Cross-Section View (Facing North)')
            else:
                ax.set_title('Cross-Section View (Facing Azimuth: {} degrees)'.format(self.azimuth))
             
        # If grid lines added     
        if self.DEM_loaded:
            if self.checkbox_add_grid_lines:
                ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3, color='gray', zorder=-1)
        else:
            if self.checkbox_add_grid_lines:
                # Draw vertical grid lines
                for tick in ax.get_xticks():
                    ax.plot([tick, tick], [ax.get_ylim()[0], self.max_z], linestyle='-', linewidth=0.5, alpha=0.3, color='gray', zorder=-1)
                
                # Draw horizontal grid lines
                for tick in ax.get_yticks():
                    if tick < self.max_z:
                        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [tick, tick], linestyle='-', linewidth=0.5, alpha=0.3, color='gray', zorder=-1)
                        
        # Check if the scale factor is not zero (and not one, since one means no scaling)
        if self.y_axis_scale_factor != 0 and self.y_axis_scale_factor != 1:
            annotation_text = f"Vertical Exaggeration: {self.y_axis_scale_factor}x"
            # Place the annotation on the plot without a bounding box
            # Adjust the x and y coordinates to place it in the lower left corner
            # You may need to adjust these values based on your specific plot layout
            ax.annotate(annotation_text, xy=(0.02, 0.02), xycoords='axes fraction',
                        fontsize=7, style='italic')
                
        self.canvas.draw()
        self.canvases.append(self.canvas)
        
    
        
    def plot_label_within_limits(self, ax, x, y, text, fontsize=8): # Plots the labels inside the plot area
        x_min, x_max = ax.get_xlim()  # Get x-limits
        y_min, y_max = ax.get_ylim()  # Get y-limits

        # Calculate the buffer as 5% of the y-axis range (which represents depth or the vertical dimension in the plot)
        buffer_y = 0.025 * (y_max - y_min)
        
        # Adjust the y-coordinate for the label to be slightly below
        label_y = y - buffer_y
        if label_y < y_min:
            label_y = y_min

        # horizontal alignment based on the hole's orientation
        if x > (x_max + x_min) / 2:  # If the hole is on the right half of the plot
            ha = "right"
        else:
            ha = "left"

        # Plot the label
        ax.text(x, label_y, text, fontsize=fontsize, zorder=6, ha=ha)



        
    def toggle_view(self):
        self.is_plan_view = not self.is_plan_view  # Toggle the view attribute
        if self.is_plan_view:
            self.toggle_view_button.setText("Change to Cross Section View")
        else:
            self.toggle_view_button.setText("Change to Plan View")
        
        self.generate_contours_flag = False
        self.plot()  # Replot the data

        
    def activate_hover_tool(self):
        # Display a dropdown to select a column
        columns = self.data.columns.tolist()
        column_name, ok = QInputDialog.getItem(self, "Select a Column", "Choose a column:", columns, 0, False)
        
        if ok and column_name:
            # Activate hover functionality on the cross-section plot
            self.add_hover_functionality(column_name)
                
    def add_hover_functionality(self, column_name):
         # Filter the data for the plotted hole_ids
        filtered_data = self.data[self.data['hole_id'].isin(self.hole_ids)].reset_index(drop=True)
        for canvas in self.canvases:
            ax = canvas.figure.axes[0]  # the first axis is the cross-section plot
            canvas.mpl_connect('motion_notify_event', partial(self.updated_on_hover, data=filtered_data, ax=ax, canvas=canvas, is_plan_view=self.is_plan_view, column_name=column_name))

        
    def updated_on_hover(self, event, data, ax, canvas, is_plan_view, column_name):
        # Check if the event is within the axes
        if event.inaxes == ax:
            
            # Compute squared distances based on the view
            if is_plan_view:
                distances_squared = (data["x"] - event.xdata) ** 2 + (data["y"] - event.ydata) ** 2
            else:
                distances_squared = (data["x_cross"] - event.xdata) ** 2 + (data["z"] - event.ydata) ** 2
            
            # Get the index of the closest data point
            closest_index = distances_squared.idxmin()
            
            # Print the hover coordinates and closest data point coordinates
            print(f"Mouse hover coordinates (canvas): x={event.xdata}, y={event.ydata}")
            if is_plan_view:
                print(f"Closest data point coordinates: x={data.iloc[closest_index]['x']}, y={data.iloc[closest_index]['y']}")
            else:
                print(f"Closest data point coordinates: x_cross={data.iloc[closest_index]['x_cross']}, z={data.iloc[closest_index]['z']}")
            
            # Compute the actual distance to the closest point
            closest_distance = np.sqrt(distances_squared[closest_index])
            
            # Define a threshold distance
            threshold_distance = 5

            if closest_distance < threshold_distance:
                # If there's an existing annotation, remove it
                if hasattr(ax, "_last_annotation"):
                    ax._last_annotation.remove()
                
                # Get the column name and value for the closest data point
                value = data.at[closest_index, column_name]

                # Add the annotation based on the view
                if is_plan_view:
                    annotation = ax.annotate(
                        f"{column_name}: {value}", 
                        (data.iloc[closest_index]["x"], data.iloc[closest_index]["y"]),
                        backgroundcolor='#FFFFE0',
                        color="black",
                        fontsize='small',
                        fontstyle='italic',
                        zorder=6
                    )
                else:
                    annotation = ax.annotate(
                        f"{column_name}: {value}", 
                        (data.iloc[closest_index]["x_cross"], data.iloc[closest_index]["z"]),
                        backgroundcolor='#FFFFE0',
                        color="black",
                        fontsize='small',
                        fontstyle='italic',
                        zorder=6
                    )
                
                # Store the annotation for later removal
                ax._last_annotation = annotation
                
                # Redraw the figure to show the annotation
                canvas.draw()
            else:
                # Remove any existing annotation if the cursor is not close to a data point
                if hasattr(ax, "_last_annotation"):
                    ax._last_annotation.remove()
                    delattr(ax, "_last_annotation")
                    canvas.draw()

   
class ManageAttributesDialog(QDialog): # attribute manager button for cross section
    def __init__(self, data, initial_attributes=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Attributes")

        self.data = data
        self.attributes = initial_attributes or {} 
        self.attributes_dict = {}

        self.column_combo_box = QComboBox()
        self.column_combo_box.addItem("-- Select a Column --")
        self.column_combo_box.addItems([col for col in self.data.columns if col not in ['x', 'y', 'z', 'hole_id']])
        
        self.save_button = QPushButton("Save")
        self.save_button.setEnabled(False)

        main_content = QWidget()   
        self.main_layout = QVBoxLayout(main_content)  # Set layout for main content

        self.main_layout.addWidget(self.column_combo_box)
        self.main_layout.addWidget(self.save_button)

        scroll = QScrollArea(self)  # Create a QScrollArea
        scroll.setWidget(main_content)  # Add the main content to the scroll area
        scroll.setWidgetResizable(True)  # Make the scroll area content resizable
        
        layout = QVBoxLayout(self)  # Main dialog layout
        layout.addWidget(scroll)  # Add the scroll area to the main dialog
        self.setLayout(layout)

        self.column_combo_box.currentTextChanged.connect(self.update_values_widget)
        self.save_button.clicked.connect(self.save)

        self.values_widget = None

    def update_values_widget(self, column): # Update based on user input
        if self.values_widget is not None:
            self.main_layout.removeWidget(self.values_widget)
            sip.delete(self.values_widget)

        if column != "-- Select a Column --":
            unique_values = self.data[column].unique()
            self.values_widget = QGroupBox("Unique Values")

            layout = QVBoxLayout()
            self.values_widget.setLayout(layout)
            self.value_widgets = {}

            for value in unique_values:
                sub_widget = QWidget()
                sub_layout = QHBoxLayout()
                sub_widget.setLayout(sub_layout)

                color_button = QPushButton("Choose color")
                line_width_spin_box = QDoubleSpinBox()
                line_width_spin_box.setValue(4.0)

                sub_layout.addWidget(QLabel(str(value)))
                sub_layout.addWidget(color_button)
                sub_layout.addWidget(line_width_spin_box)

                self.value_widgets[value] = {"color_button": color_button, "line_width_spin_box": line_width_spin_box}
                layout.addWidget(sub_widget)

                color_button.clicked.connect(lambda _, button=color_button: self.choose_color(button))

            self.main_layout.insertWidget(1, self.values_widget)
            self.save_button.setEnabled(True)  # Enable save button once unique values are loaded.
            
            # Resize the dialog to a specific width and height:
            self.resize(400, 600)
        else:
            self.save_button.setEnabled(False)

    def choose_color(self, button):
        color = QColorDialog.getColor()
        if color.isValid():
            button.setStyleSheet(f"background-color: {color.name()}")

    def save(self):
        column_name = self.column_combo_box.currentText()
        self.attributes[column_name] = {}

        for value, widgets in self.value_widgets.items():
            color = widgets["color_button"].palette().button().color().name()
            if color == '#000000':
                color = 'black'
            line_width = widgets["line_width_spin_box"].value()
            
            self.attributes[column_name][value] = {"color": color, "line_width": line_width}

        self.accept()

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Parameters saved!")
        msg.setWindowTitle("Success")
        msg.exec_()

    def get_attributes_dict(self):
        return self.attributes
        
class ScatterPlotWindow(QMainWindow): # Scatter plot window
    def __init__(self, data):
        super(ScatterPlotWindow, self).__init__()

        self.figure = Figure(figsize=(10, 10)) 
        self.canvas = FigureCanvas(self.figure)

        # Create a central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # Set the layout for the central widget
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Create and add the toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        # Create and add the save button
        self.save_button = QtWidgets.QPushButton("Save plot")
        self.save_button.clicked.connect(self.save_plot)
        layout.addWidget(self.save_button)

        # Add canvas to the layout
        layout.addWidget(self.canvas)

        self.data = data
        self.setWindowTitle("XY Scatter Plot")
        

    def plot_scatter(self, x, y, column, color_ramp=None):
        ax = self.figure.add_subplot(111)

        # Clear the current plot 
        ax.clear()

        # Filter outliers using the IQR method ####### add to settings!!
        Q1 = self.data[x].quantile(0.25)
        Q3 = self.data[x].quantile(0.75)
        IQR = Q3 - Q1
        filtered_data = self.data[(self.data[x] >= Q1 - 1.5 * IQR) & (self.data[x] <= Q3 + 1.5 * IQR)]
        
        Q1 = self.data[y].quantile(0.25)
        Q3 = self.data[y].quantile(0.75)
        IQR = Q3 - Q1
        filtered_data = filtered_data[(filtered_data[y] >= Q1 - 1.5 * IQR) & (filtered_data[y] <= Q3 + 1.5 * IQR)]

        # Plot the scatter plot on the subplot
        if column: 
            if pd.api.types.is_numeric_dtype(self.data[column]):
                # If column data type is numeric
                if color_ramp in ['cool', 'Spectral_r', 'bwr', 'jet', 'viridis_r']:
                    sc = ax.scatter(self.data[x], self.data[y], c=self.data[column], cmap=plt.get_cmap(color_ramp))
                    cbar = plt.colorbar(sc)
                    cbar.set_label(column)
                else:
                    QMessageBox.warning(self, "Warning", "Please choose a numeric color ramp for a numeric column")
                    return
            else:
                # If column data type is categorical
                if color_ramp in ['tab10', 'tab20']:
                    categories = self.data[column].unique()
                    for category in categories:
                        data_category = self.data[self.data[column] == category]
                        ax.scatter(data_category[x], data_category[y], label=category, cmap=plt.get_cmap(color_ramp))
                    ax.legend()
                else:
                    QMessageBox.warning(self, "Warning", "Please choose a categorical color ramp for a non-numeric column")
                    return
        else:
            ax.scatter(self.data[x], self.data[y], color="black")
        

        # Set the title and labels
        ax.set_title(f"{x} vs {y}")
        ax.set_xlabel(x)
        ax.set_ylabel(y)

        # Force the canvas to update and draw the new plot
        self.canvas.draw()
        
        
    def save_plot(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self,"Save Plot","","All Files (*);;JPEG (*.jpeg);;PNG (*.png)", options=options)
        if file_name:
            self.figure.savefig(file_name)

        

class ScatterPlotDialog(QDialog): # Window for scatter plot inputs
    def __init__(self, data, selected_hole_ids):
        super().__init__()

        self.data = data
        self.selected_hole_ids = selected_hole_ids

        self.layout = QVBoxLayout(self)
        self.setWindowTitle("XY Scatter Plot")
        

        self.xaxis_combo = QComboBox()
        self.xaxis_combo.addItems(self.data.columns)
        self.layout.addWidget(QLabel("Select x-axis column"))
        self.layout.addWidget(self.xaxis_combo)

        self.yaxis_combo = QComboBox()
        self.yaxis_combo.addItems(self.data.columns)
        self.layout.addWidget(QLabel("Select y-axis column"))
        self.layout.addWidget(self.yaxis_combo)

        self.column_combo = QComboBox()
        self.column_combo.addItem("None")  # None option first
        self.column_combo.addItems(self.data.columns)
        self.column_combo.currentIndexChanged.connect(self.update_color_ramp_combo)
        self.layout.addWidget(QLabel("Select column for stylizing points"))
        self.layout.addWidget(self.column_combo)

        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(['10 equal bins', 'quantile'])
        self.layout.addWidget(QLabel("Select interpolation method"))
        self.layout.addWidget(self.interpolation_combo)

        self.color_ramp_combo = QComboBox()
        self.color_ramp_combo.addItems(['cool', 'Spectral_r', 'bwr', 'jet', 'viridis_r'])
        self.layout.addWidget(QLabel("Select color ramp"))
        self.layout.addWidget(self.color_ramp_combo)

        plot_button = QPushButton("Plot")
        plot_button.clicked.connect(self.plot_scatter)
        self.layout.addWidget(plot_button)

        self.setLayout(self.layout)

        # Reference to ScatterPlotWindow
        self.scatter_plot_window = None
        
    def update_color_ramp_combo(self):
        column = self.column_combo.currentText()

        # Clear the combo box
        self.color_ramp_combo.clear()

        # Add items based on the column data type
        if column != "None":
            if pd.api.types.is_numeric_dtype(self.data[column]):
                self.color_ramp_combo.addItems(['cool', 'Spectral_r', 'bwr', 'jet', 'viridis_r'])
            else:
                self.color_ramp_combo.addItems(['tab10', 'tab20'])

    def plot_scatter(self):
        filtered_data = self.data[self.data['hole_id'].isin(self.selected_hole_ids)]
        x = self.xaxis_combo.currentText()
        y = self.yaxis_combo.currentText()
        column = self.column_combo.currentText()

        # Filter out rows where x, y, or column are zero
        if column != "None":
            filtered_data = filtered_data[(filtered_data[x] != 0) & (filtered_data[y] != 0) & (filtered_data[column] != 0)]
        else:
            filtered_data = filtered_data[(filtered_data[x] != 0) & (filtered_data[y] != 0)]
            
        if column == "None":  # Handle the None option
            column = None

        interpolation = self.interpolation_combo.currentText()
        color_ramp = self.color_ramp_combo.currentText() if self.color_ramp_combo.currentText() != "None" else None

        self.scatter_plot_window = ScatterPlotWindow(filtered_data)
        self.scatter_plot_window.plot_scatter(x, y, column, color_ramp)
        self.scatter_plot_window.show()

        # Close the dialog after the scatter plot window is shown
        self.close()


class StereonetPlotWindow: # Stereonet window
    def __init__(self, data_structure, strike_column, dip_column, plot_type, color_coding_column=None):
        self.data_structure = data_structure
        self.strike_column = strike_column
        self.dip_column = dip_column
        self.plot_type = plot_type
        self.color_coding_column = color_coding_column if color_coding_column != "None" else None
        self.create_stereonet_plot()

    def create_stereonet_plot(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='stereonet')

        if self.color_coding_column is not None:  # If color coding column is not 'None'
            # Use the tab10 color map
            cmap = plt.get_cmap("tab10")
            color_coding_data = self.data_structure[self.color_coding_column]
            categories = pd.Categorical(color_coding_data).categories
            color_code_dict = {category: cmap(i % cmap.N) for i, category in enumerate(categories)}
            colors = [color_code_dict[val] for val in color_coding_data]
        else:
            colors = ['black'] * len(self.data_structure)

        for strike, dip, color in zip(self.data_structure[self.strike_column], self.data_structure[self.dip_column], colors):
            if self.plot_type in ["Poles", "Both"]:
                ax.pole(strike, dip, 'o', color=color)
            if self.plot_type in ["Planes", "Both"]:
                ax.plane(strike, dip, '-', color=color)

        # If there is color coding, add a legend
        if self.color_coding_column is not None:
            legend_handles = []
            for category in categories:
                legend_handles.append(mpatches.Patch(color=color_code_dict[category], label=category))
            ax.legend(handles=legend_handles, loc='upper left')

        ax.grid(True, alpha=0.5)

        plt.show()



class StereonetDialog(QDialog): # Stereonet user inputs
    def __init__(self, parent=None):
        super(StereonetDialog, self).__init__(parent)

        self.strike_label = QLabel("Strike Column:", self)
        self.strike_combobox = QComboBox(self)

        self.dip_label = QLabel("Dip Column:", self)
        self.dip_combobox = QComboBox(self)
        
        self.type_label = QLabel("Plot Type:", self)
        self.type_combobox = QComboBox(self)
        self.type_combobox.addItems(['Poles', 'Planes', 'Both'])

        self.color_coding_label = QLabel("Color Coding (optional):", self)
        self.color_coding_combobox = QComboBox(self)
        self.color_coding_combobox.addItem("None")

        grid_layout = QGridLayout(self)
        grid_layout.addWidget(self.strike_label, 0, 0)
        grid_layout.addWidget(self.strike_combobox, 0, 1)
        grid_layout.addWidget(self.dip_label, 1, 0)
        grid_layout.addWidget(self.dip_combobox, 1, 1)
        grid_layout.addWidget(self.type_label, 2, 0)
        grid_layout.addWidget(self.type_combobox, 2, 1)
        grid_layout.addWidget(self.color_coding_label, 3, 0)
        grid_layout.addWidget(self.color_coding_combobox, 3, 1)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        grid_layout.addWidget(button_box, 4, 0, 1, 2)

        self.setLayout(grid_layout)
        self.setWindowTitle("Stereonet Plot Configuration")
        
        
class RosePlot: # Rose plot window
    def __init__(self, df, hole_ids):
        self.df = df
        self.hole_ids = hole_ids

    def plot(self, strike_column, dip_column, color_column=None):
        # Filter DataFrame based on selected hole IDs
        df = self.df[self.df['hole_id'].isin(self.hole_ids)]

        # Convert strike angles to radians
        strike_angles = np.radians(df[strike_column])

        # Determine colors for each row in DataFrame
        if color_column:
            unique_values = df[color_column].unique()
            colormap = plt.cm.get_cmap('viridis', len(unique_values))
            color_dict = dict(zip(unique_values, [colormap(i) for i in range(len(unique_values))]))
            colors = df[color_column].map(color_dict)
        else:
            colors = 'blue'

        # Create rose diagram
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, projection='polar')

        # Bin strike angles and get the counts for each bin
        strike_bins = np.linspace(0, 2*np.pi, 36)
        counts, bin_edges = np.histogram(strike_angles, bins=strike_bins)
        
        # Compute bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        ax.bar(bin_centers, counts, color=colors, width=2*np.pi/36)

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        # If there is color coding, add a legend
        if color_column:
            legend_handles = []
            for category, color in color_dict.items():
                legend_handles.append(mpatches.Patch(color=color, label=category))
            ax.legend(handles=legend_handles, loc='upper left')

        plt.show()


    def dialog(self):
        # Create and configure dialog
        dialog = RoseDiagramDialog()

        # Populate combo boxes with column names from DataFrame
        dialog.strike_combobox.addItems(self.df.columns)
        dialog.dip_combobox.addItems(self.df.columns)
        dialog.color_coding_combobox.addItems(['None'] + list(self.df.columns))

        # Show dialog, user to make a selection
        result = dialog.exec_()
        if result:
            # Get selected columns
            strike_column = dialog.strike_combobox.currentText()
            dip_column = dialog.dip_combobox.currentText()
            color_column = dialog.color_coding_combobox.currentText() if dialog.color_coding_combobox.currentText() != 'None' else None

            # Call plot function
            self.plot(strike_column, dip_column, color_column)
            
class RoseDiagramDialog(QDialog): # Rose diagram window
    def __init__(self, parent=None):
        super(RoseDiagramDialog, self).__init__(parent)

        self.strike_label = QLabel("Strike Column:", self)
        self.strike_combobox = QComboBox(self)

        self.dip_label = QLabel("Dip Column:", self)
        self.dip_combobox = QComboBox(self)

        self.color_coding_label = QLabel("Color Coding (optional):", self)
        self.color_coding_combobox = QComboBox(self)
        self.color_coding_combobox.addItem("None")

        grid_layout = QGridLayout(self)
        grid_layout.addWidget(self.strike_label, 0, 0)
        grid_layout.addWidget(self.strike_combobox, 0, 1)
        grid_layout.addWidget(self.dip_label, 1, 0)
        grid_layout.addWidget(self.dip_combobox, 1, 1)
        grid_layout.addWidget(self.color_coding_label, 2, 0)
        grid_layout.addWidget(self.color_coding_combobox, 2, 1)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        grid_layout.addWidget(button_box, 3, 0, 1, 2)

        self.setLayout(grid_layout)
        self.setWindowTitle("Rose Diagram Configuration")



class FactorAnalysisDialog(QDialog): # Factor analysis window
    create_factor_analysis = pyqtSignal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Factor Analysis")

        self.layout = QVBoxLayout(self)

        # User inputs
        self.factors_label = QLabel("Number of factors:")
        self.factors_spin_box = QSpinBox()
        self.factors_spin_box.setMinimum(1)
        self.factors_spin_box.setMaximum(10)

        self.column_ending_label = QLabel("Column ending tag:")
        self.column_ending_line_edit = QLineEdit()

        self.factor_scores_check_box = QCheckBox("Get Factor Scores")
        self.factor_scores_check_box.setChecked(False)

        # Adding threshold input field
        self.threshold_label = QLabel("Factor Contribution Threshold:")
        self.threshold_line_edit = QLineEdit()
        self.threshold_line_edit.setText("0.35")  # Default threshold value as placeholder

        self.run_button = QPushButton("Run Factor Analysis")
        self.run_button.clicked.connect(self.run_factor_analysis)

        self.layout.addWidget(self.factors_label)
        self.layout.addWidget(self.factors_spin_box)
        self.layout.addWidget(self.column_ending_label)
        self.layout.addWidget(self.column_ending_line_edit)
        self.layout.addWidget(self.factor_scores_check_box)
        
        # Adding the threshold elements to the layout
        self.layout.addWidget(self.threshold_label)
        self.layout.addWidget(self.threshold_line_edit)
        
        self.layout.addWidget(self.run_button)



    def run_factor_analysis(self):
        # Collect user values
        number_of_factors = self.factors_spin_box.value()
        column_ending_tag = self.column_ending_line_edit.text()
        get_factor_scores = self.factor_scores_check_box.isChecked() 

        selected_hole_ids = []
        for index in range(self.parent.model_geochem.rowCount()):
            item = self.parent.model_geochem.item(index)
            if item.checkState() == Qt.Checked:  # If the hole_id is checked
                hole_id = item.text()
                selected_hole_ids.append(hole_id)
                
        selected_columns = [col for col in self.parent.data_geochem.columns if col.endswith(column_ending_tag)]
        
        if not selected_hole_ids:
            QMessageBox.warning(self, "No Holes Selected", "Please select at least one hole to run the factor analysis.")
            return

        if not selected_columns:
            QMessageBox.warning(self, "No Matching Columns", "No columns match the provided ending tag. Please provide a valid ending tag.")
            return

        selected_data = self.parent.data_geochem[self.parent.data_geochem['hole_id'].isin(selected_hole_ids)][selected_columns]

        if selected_data.empty:
            QMessageBox.warning(self, "No Matching Data", "No data matches the selected holes and column ending tag. Please select different holes or provide a different ending tag.")
            return
            
        
        # Scale the data with mean imputation ##### change from mean imputation?
        scaler = StandardScaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(selected_data.fillna(0)), columns=selected_data.columns)  ### This one for column means:  scaled_data = pd.DataFrame(scaler.fit_transform(selected_data.fillna(selected_data.mean())), columns=selected_data.columns)
        
        
        if scaled_data.isna().sum().sum() > 0:
            QMessageBox.warning(self, "NaNs in Scaled Data", "The scaled data contains NaN values. Please handle these before running factor analysis.")
            return  # Stop execution

        if np.isinf(scaled_data).sum().sum() > 0:
            QMessageBox.warning(self, "Infs in Scaled Data", "The scaled data contains inf values. Please handle these before running factor analysis.")
            return  # Stop execution

        # Check if the correlation matrix contains NaNs or infs
        corr_mtx = scaled_data.corr()
        if corr_mtx.isna().sum().sum() > 0:
            QMessageBox.warning(self, "NaNs in Correlation Matrix", "The correlation matrix contains NaN values. Check your data for columns with constant values.")
            return  # Stop execution

        if np.isinf(corr_mtx).sum().sum() > 0:
            QMessageBox.warning(self, "Infs in Correlation Matrix", "The correlation matrix contains inf values. Please handle these before running factor analysis.")
            return

        # Run the factor analysis on the scaled data
        fa = FactorAnalyzer(n_factors=number_of_factors, rotation='varimax')
        fa.fit(scaled_data)

        # Get the factor loadings and variance
        loadings = fa.loadings_
        variance = fa.get_factor_variance()

        # Save the loadings to a DataFrame
        factor_results = pd.DataFrame(loadings, columns=[f'Factor {i+1}' for i in range(number_of_factors)], index=selected_columns)

        # Determine the variables for each factor
        threshold = float(self.threshold_line_edit.text())
        variable_rows = []
        for i in range(number_of_factors):
            factor = f'Factor {i+1}'
            variables = factor_results.index[factor_results[factor].abs() > threshold].tolist()
            variable_rows.append(pd.Series([', '.join(variables)], index=[factor], name=f'{factor} Variables'))

        # Create a DataFrame for the variance
        variance_df = pd.DataFrame([variance[0]], columns=[f'Factor {i+1}' for i in range(number_of_factors)], index=['Variance'])
        
        hole_ids_df = pd.DataFrame([', '.join(selected_hole_ids)], columns=['selected_hole_ids'], index=[''])

        # Concatenate the factor results and variance DataFrames
        factor_results = pd.concat([factor_results] + [variance_df] + variable_rows + [hole_ids_df])
        
        if get_factor_scores:
            factor_scores = fa.transform(scaled_data)
            factor_scores_df = pd.DataFrame(factor_scores, columns=[f'Factor {i+1}' for i in range(number_of_factors)])

            # Add the factor scores as new columns in the original DataFrame
            for i in range(1, number_of_factors+1):
                column_name = f'Factor {i} Score'
                self.parent.data_geochem[column_name] = factor_scores_df[f'Factor {i}']
                nan_rows = self.parent.data_geochem[selected_columns].isnull().all(axis=1)
                self.parent.data_geochem.loc[nan_rows, column_name] = np.nan

            # Prompt the user to save the factor scores to a CSV
            filename, _ = QFileDialog.getSaveFileName(self, "Save Factor Scores", "", "CSV Files (*.csv)")
            if filename:
                self.parent.data_geochem[self.parent.data_geochem['hole_id'].isin(selected_hole_ids)].to_csv(filename, index=False)

        # Prompt the user to save the DataFrame to a CSV
        filename, _ = QFileDialog.getSaveFileName(self, "Save Factor Loading Results", "", "CSV Files (*.csv)")
        if filename:
            factor_results.to_csv(filename, index=True)

        # Emit the signal at the end
        self.create_factor_analysis.emit(number_of_factors, column_ending_tag)
        
        self.close()


class CustomPlotWindow(QWidget): # Custom plot window
    def __init__(self, json_file_path, data):
        super().__init__()

        # Open and load the JSON file
        with open(json_file_path, 'r') as f:
            self.plot_info = json.load(f)

        self.data = data
        self.extract_plot_limits_from_json()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.canvas)

        self.setLayout(self.layout)
        
    def extract_plot_limits_from_json(self):
        all_x_values = []
        all_y_values = []
        for polygon in self.plot_info.get('polygons', []):
            for point in polygon['points']:
                all_x_values.append(point[0])
                all_y_values.append(point[1])
        x_min, x_max = min(all_x_values), max(all_x_values)
        y_min, y_max = min(all_y_values), max(all_y_values)
        return (x_min, x_max), (y_min, y_max)


    def plot_custom(self, x, y, log_x, log_y, color_column=None, color_ramp=None, data_points=True):
        # Clear the current contents of the figure
        self.figure.clear()

        # Create an axis
        ax = self.figure.add_subplot(111)
        x_limits, y_limits = self.extract_plot_limits_from_json()
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)

        # Plot the data
        if x and y and data_points:
            if color_column: 
                if pd.api.types.is_numeric_dtype(self.data[color_column]):
                    sc = ax.scatter(self.data[x], self.data[y], c=self.data[color_column], cmap=plt.get_cmap(color_ramp))
                    cbar = plt.colorbar(sc)
                    cbar.set_label(color_column)
                else:
                    categories = self.data[color_column].unique()
                    for category in categories:
                        data_category = self.data[self.data[color_column] == category]
                        ax.scatter(data_category[x], data_category[y], label=category, cmap=plt.get_cmap(color_ramp))
                    ax.legend()
            else:
                ax.scatter(self.data[x], self.data[y])

        # Use the plot information from the JSON file
        ax.set_title(self.plot_info.get('title', ''))
        ax.set_xlabel(self.plot_info.get('x_label', ''))
        ax.set_ylabel(self.plot_info.get('y_label', ''))
       

        # Set the x scale
        if log_x:
            ax.set_xscale('log')

        # Set the y scale
        if log_y:
            ax.set_yscale('log')

        # Add the labels
        labels = self.plot_info.get('labels', {})
        for label, (x_pos, y_pos) in labels.items():
            ax.text(x_pos, y_pos, label, ha='center', va='center', fontsize=8)
        
        # Plot the line segments
        for line_segment in self.plot_info.get('lines', []):
            print(line_segment) 
            ax.plot(*zip(*line_segment), color='black')
            
        # Extract polygons
        polygons = self.plot_info.get('polygons', [])

        # Plot each polygon
        for polygon in polygons:
            points = polygon['points']
            style = polygon['style']
           
            # Extract style information
            fillColor = style.get('fillColor', '#FFFFFF')
            fillOpacity = style.get('fillOpacity', 1)
            borderColor = style.get('borderColor', '#000000')
            borderWidth = style.get('borderWidth', 1)

            # Plot the filled polygon
            ax.fill(*zip(*points), color=fillColor, alpha=fillOpacity)

            # Plot the border of the polygon
            ax.plot(*zip(*points, points[0]), color=borderColor, linewidth=borderWidth)
            
            # Add label/name to the polygon
            if "name" in polygon and "label" in polygon:
                label_info = polygon['label']
                ax.annotate(polygon['name'], 
                            tuple(label_info['position']),
                            color=label_info.get('color', 'black'), 
                            ha=label_info.get('ha', 'center'), 
                            va=label_info.get('va', 'center'),
                            fontsize=label_info.get('fontsize', 9))

        if x and y:
            self.data['polygon value'] = "None"
            for index, row in self.data.iterrows():
                point = (row[x], row[y])
                for polygon in polygons:
                    poly_path = Path(polygon['points'])
                    if poly_path.contains_point(point):
                        self.data.at[index, 'polygon value'] = polygon['name']
                        break
                        
        # Apply the formatter
        formatter = FuncFormatter(format_func)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        
        # Redraw the canvas
        self.canvas.draw()

        self.close()
        
def format_func(value, tick_number):
    # Format the numbers as 0.01, 0.1, 1, etc.
    return f"{value:.2f}"
        
class CustomPlotDialog(QMainWindow): # Custom plot user inputs 
    def __init__(self, data, json_file_path, parent=None):
        super().__init__(parent)

        self.data = data
        self.json_file_path = json_file_path

        self.plot_window = CustomPlotWindow(json_file_path, data)

        self.x_select = QComboBox()
        self.x_select.addItems(list(self.data.columns))
        self.y_select = QComboBox()
        self.y_select.addItems(list(self.data.columns))

        self.log_x_checkbox = QCheckBox('Logarithmic X-axis')
        self.log_y_checkbox = QCheckBox('Logarithmic Y-axis')

        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.plot)
        self.save_csv_button = QPushButton("Save CSV")
        self.save_csv_button.clicked.connect(self.save_to_csv)
        

        # create a central widget to hold our layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        
        layout.addWidget(QLabel("Select X-axis:"))
        layout.addWidget(self.x_select)
        layout.addWidget(self.log_x_checkbox)
        layout.addWidget(QLabel("Select Y-axis:"))
        layout.addWidget(self.y_select)
        layout.addWidget(self.log_y_checkbox)
        layout.addWidget(self.plot_button)
        layout.addWidget(self.plot_window.canvas)
        layout.addWidget(self.save_csv_button)
        
        # Add color column selection
        self.color_select = QComboBox()
        self.color_select.addItem("None")
        self.color_select.addItems(list(self.data.columns))
        self.color_select.currentIndexChanged.connect(self.update_color_ramp_combo)
        layout.addWidget(QLabel("Select color column:"))
        layout.addWidget(self.color_select)

        # Add color ramp selection
        self.color_ramp_select = QComboBox()
        self.update_color_ramp_combo()
        layout.addWidget(QLabel("Select color ramp:"))
        layout.addWidget(self.color_ramp_select)

        # Add checkbox for polygon plotting
        self.polygon_checkbox = QCheckBox("Attribute to Plot Polygons") # attributes each point to the polygon it falls in
        layout.addWidget(self.polygon_checkbox)

        # set the central widget
        self.setCentralWidget(central_widget)
        
        # Draw a blank plot for preview
        self.plot_window.figure.clear()
        ax = self.plot_window.figure.add_subplot(111)
        x_limits, y_limits = self.plot_window.extract_plot_limits_from_json()
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        self.plot_window.canvas.draw()
                
        self.plot_window.plot_custom("", "", False, False, data_points=False)

        
        self.plot_window.plot_custom("", "", False, False, data_points=False)
        
        self.resize(800, 800)
        
    def update_color_ramp_combo(self):
        column = self.color_select.currentText()

        # Clear the combo box
        self.color_ramp_select.clear()

        # Add the appropriate items based on the column data type
        if column != "None":
            if pd.api.types.is_numeric_dtype(self.data[column]):
                self.color_ramp_select.addItems(['cool', 'Spectral_r', 'bwr', 'jet', 'viridis_r'])
            else:
                self.color_ramp_select.addItems(['tab10', 'tab20'])

    def plot(self): # Plotting Function
        x = self.x_select.currentText()
        y = self.y_select.currentText()
        color_column = self.color_select.currentText() if self.color_select.currentText() != "None" else None
        color_ramp = self.color_ramp_select.currentText() if self.color_ramp_select.currentText() != "None" else None
        log_x = self.log_x_checkbox.isChecked()
        log_y = self.log_y_checkbox.isChecked()
        self.plot_window.plot_custom(x, y, log_x, log_y, color_column, color_ramp)

    def save_to_csv(self):
        if self.polygon_checkbox.isChecked():
            csv_file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV files (*.csv)")
            if csv_file_path:
                self.plot_window.data.to_csv(csv_file_path, index=False)
                QMessageBox.information(self, "Success", f"File saved to {csv_file_path}")
                
        
class DesurveyCalc(QDialog): # Desurvey settings window
    def __init__(self, parent=None):
        super(DesurveyCalc, self).__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # --- Drill hole data ---
        self.btn_drill_data = QPushButton('Upload drill hole data', self)
        self.btn_drill_data.clicked.connect(self.load_drill_data_csv)
        layout.addWidget(self.btn_drill_data)

        self.drill_columns_label = QLabel("Choose columns for drill hole data:")
        layout.addWidget(self.drill_columns_label)

        # Dropdowns for column selection for drill hole data
        self.cb_hole_id = QComboBox(self)
        self.cb_depth = QComboBox(self)
        layout.addWidget(QLabel("Hole ID:"))
        layout.addWidget(self.cb_hole_id)
        layout.addWidget(QLabel("Depth:"))
        layout.addWidget(self.cb_depth)
        
        # --- Survey data ---
        self.btn_survey_data = QPushButton('Upload survey data', self)
        self.btn_survey_data.clicked.connect(self.load_survey_data_csv)
        layout.addWidget(self.btn_survey_data)

        self.survey_columns_label = QLabel("Choose columns for survey data:")
        layout.addWidget(self.survey_columns_label)

        self.cb_survey_hole_id = QComboBox(self)
        self.cb_depth_sur = QComboBox(self)
        self.cb_azimuth = QComboBox(self)
        self.cb_dip = QComboBox(self)
        
        
        layout.addWidget(QLabel("Hole ID:"))
        layout.addWidget(self.cb_survey_hole_id)
        layout.addWidget(QLabel("Depth:"))
        layout.addWidget(self.cb_depth_sur)
        layout.addWidget(QLabel("Azimuth:"))
        layout.addWidget(self.cb_azimuth)
        layout.addWidget(QLabel("Dip:"))
        layout.addWidget(self.cb_dip)
        
        
        # --- Collar data ---
        self.btn_collar_data = QPushButton('Upload collar data', self)
        self.btn_collar_data.clicked.connect(self.load_collar_data_csv)
        layout.addWidget(self.btn_collar_data)

        self.collar_columns_label = QLabel("Choose columns for collar data:")
        layout.addWidget(self.collar_columns_label)

        self.cb_collar_hole_id = QComboBox(self)
        self.cb_easting = QComboBox(self)
        self.cb_northing = QComboBox(self)
        self.cb_elevation = QComboBox(self)
        
        layout.addWidget(QLabel("Hole ID:"))
        layout.addWidget(self.cb_collar_hole_id)
        layout.addWidget(QLabel("Easting (X):"))
        layout.addWidget(self.cb_easting)
        layout.addWidget(QLabel("Northing (Y):"))
        layout.addWidget(self.cb_northing)
        layout.addWidget(QLabel("Elevation (Z):"))
        layout.addWidget(self.cb_elevation)
        
        # Add the "Get Desurveyed CSV" button
        self.btn_get_csv = QPushButton('Get Desurveyed CSV', self)
        self.btn_get_csv.clicked.connect(self.generate_csv)
        layout.addWidget(self.btn_get_csv)

        self.setLayout(layout)
    
    def generate_csv(self):
        print("generate_csv triggered!")
        
        result_data = self.calculate_desurveyed_data()

        if not result_data.empty:
            options = QFileDialog.Options()
            filePath, _ = QFileDialog.getSaveFileName(self, "Save File", "", "CSV Files (*.csv);;All Files (*)", options=options)
            
            if filePath:
                # Ensuring the file extension
                if not filePath.endswith('.csv'):
                    filePath += '.csv'
                
                result_data.to_csv(filePath, index=False)
                QMessageBox.information(self, "Success", "Desurveyed data saved successfully!")
            else:
                QMessageBox.warning(self, "Cancelled", "Save operation was cancelled!")
        else:
            QMessageBox.warning(self, "Error", "Failed to generate desurveyed data!")
            

    def load_drill_data_csv(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        if fname:
            self.drill_data = pd.read_csv(fname)
            self.update_combo_boxes(self.drill_data, [self.cb_hole_id, self.cb_depth])
            
    def load_survey_data_csv(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        if fname:
            self.survey_data = pd.read_csv(fname)
            self.update_combo_boxes(self.survey_data, 
                                    [self.cb_depth_sur, self.cb_azimuth, self.cb_dip, self.cb_survey_hole_id])

    def load_collar_data_csv(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        if fname:
            self.collar_data = pd.read_csv(fname)
            self.update_combo_boxes(self.collar_data, [self.cb_collar_hole_id, self.cb_easting, self.cb_northing, self.cb_elevation])
            
    def update_combo_boxes(self, data, combo_boxes):
        columns = data.columns
        for cb in combo_boxes:
            cb.clear()
            cb.addItems(columns)
            
    @staticmethod
    def vlookup_approx(value, lookup_array, result_array):
        """
        Emulate the behavior of VLOOKUP with TRUE flag for approximate match.
        Return the closest value that is less than or equal to the lookup value.
        If no match is found, return the last value in the result_array.
        """
        matches = [i for i, x in enumerate(lookup_array) if x <= value]
        if matches:
            index = matches[-1]
            return result_array[index]
        else:
            return result_array[-1]  # Return the last value if no match is found

            
    def calculate_desurveyed_data(self):
        print("calculate_desurveyed_data function started")
        print(self.drill_data.columns)
        print(self.survey_data.columns)
        
        # Extracting data from the UI elements
        hole_id_col_drill = self.cb_hole_id.currentText()
        depth_col_drill = self.cb_depth.currentText()
        
        hole_id_col_survey = self.cb_survey_hole_id.currentText()
        depth_col_survey = self.cb_depth_sur.currentText()
        azimuth_col_survey = self.cb_azimuth.currentText()
        dip_col_survey = self.cb_dip.currentText()

        hole_id_col_collar = self.cb_collar_hole_id.currentText()
        easting_col_collar = self.cb_easting.currentText()
        northing_col_collar = self.cb_northing.currentText()
        elevation_col_collar = self.cb_elevation.currentText()
        
        self.drill_data['x'] = np.nan
        self.drill_data['y'] = np.nan
        self.drill_data['z'] = np.nan
        
        unique_holes = self.drill_data[hole_id_col_drill].unique()
        print(f"Found {len(unique_holes)} unique hole IDs: {unique_holes}")
        
        try:
            for hole_id in unique_holes:
                # Filter collar, survey, and drill data by hole ID
                collar_data_filtered = self.collar_data[self.collar_data[hole_id_col_collar] == hole_id]
                survey_data_filtered = self.survey_data[self.survey_data[hole_id_col_survey] == hole_id]
                drill_data_filtered = self.drill_data[self.drill_data[hole_id_col_drill] == hole_id]

                # Identify the minimum depth value for this hole
                min_depth = drill_data_filtered[depth_col_drill].min()

                # Initial coordinates for each hole
                prev_x = collar_data_filtered.iloc[0][easting_col_collar]
                prev_y = collar_data_filtered.iloc[0][northing_col_collar]
                prev_z = collar_data_filtered.iloc[0][elevation_col_collar]

                # Populate the row corresponding to the minimum depth value with collar coordinates
                idx_in_main_df = self.drill_data.index[(self.drill_data[hole_id_col_drill] == hole_id) & (self.drill_data[depth_col_drill] == min_depth)].tolist()[0]
                self.drill_data.at[idx_in_main_df, 'x'] = prev_x
                self.drill_data.at[idx_in_main_df, 'y'] = prev_y
                self.drill_data.at[idx_in_main_df, 'z'] = prev_z
            
                # Initialize coordinates
                prev_depth_F2 = min_depth
                # Loop through drill data for the current hole
                for index in range(len(drill_data_filtered)):
                    next_row = drill_data_filtered.iloc[index]  
                    depth_F3 = next_row[depth_col_drill]

                    # Get values for depth using the vlookup approximation
                    azimuth_F3 = self.vlookup_approx(depth_F3, survey_data_filtered[depth_col_survey].values, survey_data_filtered[azimuth_col_survey].values)
                    dip_F3 = self.vlookup_approx(depth_F3, survey_data_filtered[depth_col_survey].values, survey_data_filtered[dip_col_survey].values)

                    # Apply the calculation (geometric transformation) for x, y, z using the depth difference
                    delta_x = (depth_F3 - prev_depth_F2) * np.sin(np.radians(azimuth_F3)) * np.cos(np.radians(dip_F3))
                    delta_y = (depth_F3 - prev_depth_F2) * np.cos(np.radians(azimuth_F3)) * np.cos(np.radians(dip_F3))
                    delta_z = (depth_F3 - prev_depth_F2) * np.sin(np.radians(dip_F3))

                    # Cumulatively add the deltas to the previous coordinates
                    new_x = prev_x + delta_x
                    new_y = prev_y + delta_y
                    new_z = prev_z + delta_z

                    # Store these new values for the next iteration
                    prev_x, prev_y, prev_z = new_x, new_y, new_z
                    prev_depth_F2 = depth_F3  # Update depth for next iteration

                    # Update the original DataFrame with new x, y, and z values
                    idx_in_main_df = self.drill_data.index[(self.drill_data[hole_id_col_drill] == hole_id) & (self.drill_data[depth_col_drill] == depth_F3)].tolist()[0]
                    self.drill_data.at[idx_in_main_df, 'x'] = new_x
                    self.drill_data.at[idx_in_main_df, 'y'] = new_y
                    self.drill_data.at[idx_in_main_df, 'z'] = new_z

        except Exception as e:
            # Convert the exception message to a string before concatenating
            QMessageBox.warning(self, "Error", f"Failed to generate desurveyed data! Error: {str(e)}")
            return self.drill_data

        return self.drill_data
             
class PlotSettingsDialog(QDialog): # Settings window for cross section plot
    def __init__(
        self, hole_ids, remove_outliers=True, remove_outliers_auxiliary=False,
        add_grid_lines=False, upper_quantile=75, lower_quantile=25, IQR=3,
        x_buffer=120, y_buffer=0.05, line_width=4, selected_hole_ids_for_labels=None
    ):
        super(PlotSettingsDialog, self).__init__()
        self.hole_ids = hole_ids
        
        self.setWindowTitle('Plot Settings')
        
        # Create layout managers
        main_layout = QVBoxLayout()

        # Create UI Elements for "Remove Outliers"
        self.outliers_layout = QVBoxLayout()
        self.checkbox_remove_outliers = QCheckBox("Remove Outliers?")
        
        # Create the new checkbox for "Apply to Auxiliary Bar Plot"
        self.checkbox_aux_plot = QCheckBox("Apply to Auxiliary Bar Plot")
        
        # Create a horizontal layout to contain both checkboxes side by side
        self.checkboxes_layout = QHBoxLayout()
        self.checkboxes_layout.addWidget(self.checkbox_remove_outliers)
        self.checkboxes_layout.addWidget(self.checkbox_aux_plot)
        
        # Add the horizontal layout containing checkboxes to the outliers layout
        self.outliers_layout.addLayout(self.checkboxes_layout)
        
        self.line_edit_upper_quantile = QLineEdit()
        self.line_edit_lower_quantile = QLineEdit()
        self.line_edit_IQR = QLineEdit()
        self.outliers_layout.addWidget(self.checkbox_remove_outliers)
        self.outliers_layout.addWidget(QLabel(" Assign Upper Quantile:"))
        self.outliers_layout.addWidget(self.line_edit_upper_quantile)
        self.outliers_layout.addWidget(QLabel("Assign Lower Quantile:"))
        self.outliers_layout.addWidget(self.line_edit_lower_quantile)
        self.outliers_layout.addWidget(QLabel("Assign IQR Scaling Factor:"))
        self.outliers_layout.addWidget(self.line_edit_IQR)

        # Create UI Elements for "Define x and y Axis Limits"
        self.axis_limits_layout = QVBoxLayout()
        self.line_edit_x_buffer = QLineEdit()
        self.line_edit_y_buffer = QLineEdit()
        self.axis_limits_layout.addWidget(QLabel("X axis Buffer:"))
        self.axis_limits_layout.addWidget(self.line_edit_x_buffer)
        self.axis_limits_layout.addWidget(QLabel("Y axis Buffer:"))
        self.axis_limits_layout.addWidget(self.line_edit_y_buffer)
        
        self.line_width_layout = QVBoxLayout()
        self.line_edit_line_width = QLineEdit()
        self.line_width_layout.addWidget(QLabel("Line Width (1-5):"))
        self.line_width_layout.addWidget(self.line_edit_line_width)
        
        self.hole_labels_layout = QVBoxLayout()
        self.hole_labels_layout.addWidget(QLabel("hole_id labels"))
        
        # Create a QListView to display hole_ids with checkboxes
        self.hole_id_list_view = QListView()
        self.hole_id_list_model = QStandardItemModel(self.hole_id_list_view)
        
        #"Select All/Deselect All" checkbox
        self.select_all_checkbox = QCheckBox("Select All/Deselect All")
        self.hole_labels_layout.addWidget(self.select_all_checkbox)
        
        # add a "add grid lines" checkbox
        self.grid_layout = QVBoxLayout()
        self.checkbox_add_grid_lines = QCheckBox("Add Grid Lines")
        self.grid_layout.addWidget(self.checkbox_add_grid_lines)
        
        # Add hole_ids to the list view with a checkbox next to each
        for hole_id in self.hole_ids:
            item = QStandardItem(hole_id)
            item.setCheckable(True)
            item.setCheckState(Qt.Checked)  # Check all items by default
            self.hole_id_list_model.appendRow(item)
        
        self.hole_id_list_view.setModel(self.hole_id_list_model)
        self.hole_labels_layout.addWidget(self.hole_id_list_view)

        # Add layouts to main layout
        main_layout.addLayout(self.outliers_layout)
        main_layout.addLayout(self.axis_limits_layout)
        main_layout.addLayout(self.line_width_layout)
        main_layout.addLayout(self.hole_labels_layout)
        main_layout.addLayout(self.grid_layout) 

        # Set the main layout
        self.setLayout(main_layout)

        # Initialize default values and connect signals
        self.connect_signals()
        self.select_all_checkbox.stateChanged.connect(self.toggle_select_all_hole_ids)

        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)
        
        # Set the initial state of the dialog's controls using passed-in settings
        self.init_defaults(
            remove_outliers=remove_outliers,
            remove_outliers_auxiliary=remove_outliers_auxiliary,
            add_grid_lines=add_grid_lines,
            upper_quantile=upper_quantile,
            lower_quantile=lower_quantile,
            IQR=IQR,
            x_buffer=x_buffer,
            y_buffer=y_buffer,
            line_width=line_width,
            selected_hole_ids_for_labels=selected_hole_ids_for_labels
        )

        # Connect signals for buttons
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
    def init_defaults(self,
                      remove_outliers=True,
                      remove_outliers_auxiliary=False,
                      add_grid_lines=False,
                      upper_quantile=75,
                      lower_quantile=25,
                      IQR=3,
                      x_buffer=120,
                      y_buffer=0.05,
                      line_width=4,
                      selected_hole_ids_for_labels=None):
        # Set defaults or use passed-in settings
        self.checkbox_remove_outliers.setChecked(remove_outliers)
        self.checkbox_aux_plot.setChecked(remove_outliers_auxiliary)
        self.checkbox_add_grid_lines.setChecked(add_grid_lines)
        self.line_edit_upper_quantile.setText(str(upper_quantile))
        self.line_edit_lower_quantile.setText(str(lower_quantile))
        self.line_edit_IQR.setText(str(IQR))
        self.line_edit_x_buffer.setText(str(x_buffer))
        self.line_edit_y_buffer.setText(str(y_buffer))
        self.line_edit_line_width.setText(str(line_width))

        # Initialize hole_id list view checkboxes
        if selected_hole_ids_for_labels is not None:
            for row in range(self.hole_id_list_model.rowCount()):
                item = self.hole_id_list_model.item(row)
                if item.text() in selected_hole_ids_for_labels:
                    item.setCheckState(Qt.Checked)
                else:
                    item.setCheckState(Qt.Unchecked)
        
    
    def toggle_select_all_hole_ids(self, state):
        # Select or deselect all hole_id checkboxes based on the state of the select_all_checkbox
        for row in range(self.hole_id_list_model.rowCount()):
            item = self.hole_id_list_model.item(row)
            item.setCheckState(state)
            
    def get_selected_hole_ids(self):
        # Get a list of selected hole_ids
        selected_hole_ids = []
        for row in range(self.hole_id_list_model.rowCount()):
            item = self.hole_id_list_model.item(row)
            if item.checkState() == Qt.Checked:
                selected_hole_ids.append(item.text())
        return selected_hole_ids

        
    def get_selected_line_width(self):
        line_width = float(self.line_edit_line_width.text())
        return max(1.0, min(line_width, 5.0))  # Limit between 1 and 5


    def connect_signals(self):
        self.checkbox_remove_outliers.toggled.connect(self.toggle_remove_outliers)

    def toggle_remove_outliers(self, checked):
        self.line_edit_upper_quantile.setEnabled(checked)
        self.line_edit_lower_quantile.setEnabled(checked)
        self.line_edit_IQR.setEnabled(checked)


class CombinedPlotWindow(QtWidgets.QMainWindow): # Downline line plot and graphic plot merge
    def __init__(self, main_window, plot_windows, geochem_plot_windows):
        super().__init__()
        self.setWindowModality(QtCore.Qt.NonModal)
        
        self.figure = Figure(figsize=(15, 12))
        self.canvas = FigureCanvas(self.figure)
        self.setCentralWidget(self.canvas)
        self.main_window = main_window
        
        self.main_window = main_window
        self.plot_windows = plot_windows
        self.geochem_plot_windows = geochem_plot_windows
        
        self.save_button = QtWidgets.QPushButton(self)
        self.save_button.setText("Save plot")
        self.save_button.clicked.connect(self.save_plot)
        
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.save_button)

        self.merge_plots()
        self.horizontal_lines = [ax.axhline(y=0, color='red') for ax in self.axs]  # Initial horizontal line at y=0 for each axis
        self.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.draw()
        self.show()
        
        
    def merge_plots(self):
        # Assumes all geochem plots have the same depth (Y-axis) range
        num_plots = 1 + len(self.geochem_plot_windows)
        
        # Create subplots side by side
        self.axs = self.figure.subplots(1, num_plots)
        print("Total plot_windows:", len(self.plot_windows))
        print("Total geochem_plot_windows:", len(self.geochem_plot_windows))
        print("Total axes (axs):", len(self.axs))
        
        # If only one plot, make it to a list
        if not isinstance(self.axs, np.ndarray):
            self.axs = [self.axs]

           
        # Extract y-axis limits from the graphic log
        graphic_log_window = self.plot_windows[0]  # Since there's only one graphic log plot
        ax = graphic_log_window.figure.axes[0]
        global_min_depth, global_max_depth = ax.get_ylim()
        y_ticks_graphic_log = ax.get_yticks()
        y_ticklabels_graphic_log = [label.get_text() for label in ax.get_yticklabels()]

        # Merge lithology plots
        for i, window in enumerate(self.plot_windows):
            ax = self.axs[i]
            
            ax.set_ylim(global_max_depth, global_min_depth)
            ax.margins(y=0)
            
            ax.set_yticks(y_ticks_graphic_log)
            ax.set_yticklabels(y_ticklabels_graphic_log)

            # Copy lines
            for line in window.figure.axes[0].lines:
                ax.plot(line.get_xdata(), line.get_ydata(), color=line.get_color())
                
            
            # Copy patches 
            for patch in window.figure.axes[0].patches:
                if isinstance(patch, patches.Rectangle):
                    new_patch = patches.Rectangle((patch.get_x(), patch.get_y()), patch.get_width(), patch.get_height(), 
                                                  facecolor=patch.get_facecolor(), edgecolor=patch.get_edgecolor(), 
                                                  linewidth=patch.get_linewidth(), alpha=patch.get_alpha())
                elif isinstance(patch, patches.Polygon):
                    xy = patch.get_xy()
                    new_patch = patches.Polygon(xy, closed=True, 
                                                facecolor=patch.get_facecolor(), edgecolor=patch.get_edgecolor(),
                                                linewidth=patch.get_linewidth(), alpha=patch.get_alpha())
                else:
                    # If the patch type is not recognized, skip it.
                    continue
                ax.add_patch(new_patch)


            
            # Copy texts over
            for text in window.figure.axes[0].texts:
                ax.text(text.get_position()[0], text.get_position()[1], text.get_text(), fontsize=text.get_fontsize(), color=text.get_color())

            # Copy annotations (e.g., depth annotations)
            for annotation in window.figure.axes[0].texts:
                new_annotation = ax.annotate(annotation.get_text(), 
                                             annotation.get_position(),
                                             color=annotation.get_color(),
                                             fontsize=annotation.get_fontsize(),
                                             ha=annotation.get_ha(),
                                             va=annotation.get_va())

            
            # Match axis settings
            ax.set_xlim(window.figure.axes[0].get_xlim())
            ax.set_ylim(window.figure.axes[0].get_ylim())
            ax.set_xticks(window.figure.axes[0].get_xticks())
            ax.set_yticks(window.figure.axes[0].get_yticks())
            ax.set_title(window.figure.axes[0].get_title())
            ax.set_ylabel(window.figure.axes[0].get_ylabel())

        
        for i, window in enumerate(self.geochem_plot_windows.values()):
            ax = self.axs[len(self.plot_windows) + i]
            
            ax.set_ylim(global_max_depth, global_min_depth)
            ax.margins(y=0)
            
            # Remove the y-axis labels but keep the tick marks
            ax.set_yticklabels([''] * len(ax.get_yticks()))
            ax.set_title(window.column_name)
            
            # Copy the content of the geochem plot window's axis to the new axis
            for line in window.figure.axes[0].lines:
                ax.plot(line.get_xdata(), line.get_ydata(), color=line.get_color())
            
            # Copy bars if present
            for bar in window.figure.axes[0].patches:
                if isinstance(bar, patches.Rectangle):
                    new_bar = patches.Rectangle((bar.get_x(), bar.get_y()), bar.get_width(), bar.get_height(), 
                                                facecolor=bar.get_facecolor(), edgecolor=bar.get_edgecolor(), 
                                                linewidth=bar.get_linewidth(), alpha=bar.get_alpha())
                    ax.add_patch(new_bar)

            # Ensure the x-axis limits match the original geochem plot
            ax.set_xlim(window.figure.axes[0].get_xlim())

        for ax in self.axs:
            ax.set_ylim(global_max_depth, global_min_depth)
            ax.invert_yaxis()


    def closeEvent(self, event):
        self.main_window.combined_plot_window = None
        event.accept()  # let the window close
        
    def save_plot(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self,"Save Plot", "", "All Files (*);;JPEG (*.jpeg);;PNG (*.png);;SVG (*.svg)", options=options)
        if file_name:
            self.figure.savefig(file_name, dpi=200)
            
    def on_click(self, event):
        # Update the line position for each axis
        for line in self.horizontal_lines:
            line.set_ydata([event.ydata, event.ydata])
        self.canvas.draw()

            
            
class UnorientedPlot: # plot window for small circles/unoriented core structures 
    def __init__(self, data_model, model_structure):
        self.data_model = data_model
        self.model_structure = model_structure
        self.tca_column = None
        self.selected_hole_ids = []
        self.setup_ui()

    def setup_ui(self):
    
        # QDialog to act as the main window for this class
        self.dialog = QDialog()
        
        # Setup a vertical layout
        main_layout = QVBoxLayout(self.dialog)
        
        # Add a label and dropdown for Drill Hole Azimuth column selection
        azimuth_label = QLabel("Please choose Drill Hole Azimuth column:", self.dialog)
        self.azimuth_dropdown = QComboBox(self.dialog)
        self.azimuth_dropdown.addItems(list(self.data_model.columns))
        main_layout.addWidget(azimuth_label)
        main_layout.addWidget(self.azimuth_dropdown)

        # Add a label and dropdown for Drill Hole Dip column selection
        dip_label = QLabel("Please choose Drill Hole Dip column:", self.dialog)
        self.dip_dropdown = QComboBox(self.dialog)
        self.dip_dropdown.addItems(list(self.data_model.columns))
        main_layout.addWidget(dip_label)
        main_layout.addWidget(self.dip_dropdown)

        # Add a label and dropdown for TCA column selection
        tca_label = QLabel("Please choose TCA column:", self.dialog)
        self.tca_dropdown = QComboBox(self.dialog)
        self.tca_dropdown.addItems(list(self.data_model.columns))
        main_layout.addWidget(tca_label)
        main_layout.addWidget(self.tca_dropdown)

        # Add a label and dropdown for structure column selection
        structure_label = QLabel("Please select structure column:", self.dialog)
        self.structure_dropdown = QComboBox(self.dialog)
        self.structure_dropdown.addItems(list(self.data_model.columns))
        main_layout.addWidget(structure_label)
        main_layout.addWidget(self.structure_dropdown)

        # Add a label and list widget for structure values
        structure_values_label = QLabel("Please select structure values:", self.dialog)
        self.structure_values_list_widget = QListWidget(self.dialog)
        self.structure_dropdown.currentTextChanged.connect(self.update_structure_values_list)
        
        # Populate structure_values_list_widget with all unique structure values in the dataframe
        all_structure_values = self.data_model.dropna().astype(str).values.ravel()
        unique_structure_values = np.unique(all_structure_values)

        for structure_value in unique_structure_values:
            item = QListWidgetItem(structure_value)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.structure_values_list_widget.addItem(item)

        main_layout.addWidget(structure_values_label)
        main_layout.addWidget(self.structure_values_list_widget)

     
        # Add a Plot button
        self.plot_button = QPushButton("Plot", self.dialog)
        self.plot_button.clicked.connect(self.create_unoriented_plot)
        main_layout.addWidget(self.plot_button)
        

        # Checkbox for structure_values_list_widget
        self.structure_values_select_all_checkbox = QCheckBox("Select All/Deselect All", self.dialog)
        self.structure_values_select_all_checkbox.stateChanged.connect(self.toggle_structure_values_selection)
        main_layout.addWidget(self.structure_values_select_all_checkbox)
        main_layout.addWidget(self.structure_values_list_widget)


        self.dialog.setLayout(main_layout)
        self.dialog.exec_()
        

    def toggle_hole_id_selection(self, state):
        for index in range(self.hole_id_list_widget.count()):
            item = self.hole_id_list_widget.item(index)
            if state == Qt.Checked:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)

    def toggle_structure_values_selection(self, state):
        for index in range(self.structure_values_list_widget.count()):
            item = self.structure_values_list_widget.item(index)
            if state == Qt.Checked:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)

    def update_structure_values_list(self):
        # Clear previous items
        self.structure_values_list_widget.clear()

        # Get selected structure column
        selected_col = self.structure_dropdown.currentText()

        # Populate structure_values_list_widget with unique values from the selected structure column
        unique_structure_values = self.data_model[selected_col].dropna().unique()
        for structure_value in unique_structure_values:
            item = QListWidgetItem(str(structure_value))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.structure_values_list_widget.addItem(item)

  
    def create_unoriented_plot(self):
        # Fetch the selected TCA column
        self.tca_column = self.tca_dropdown.currentText()
        self.azimuth_column = self.azimuth_dropdown.currentText()
        self.dip_column = self.dip_dropdown.currentText()

        # Fetch the selected structure values
        selected_structures = []
        for index in range(self.structure_values_list_widget.count()):
            item = self.structure_values_list_widget.item(index)
            if item.checkState() == Qt.Checked:
                selected_structures.append(item.text())

        # Define the structure column
        structure_col = self.structure_dropdown.currentText()

        # Fetch the selected hole_ids and filter the data_model based on selected hole_ids and structure types
        self.selected_hole_ids = [self.model_structure.item(index).text() for index in range(self.model_structure.rowCount()) if self.model_structure.item(index).checkState() == Qt.Checked]
        filtered_data = self.data_model[self.data_model['hole_id'].isin(self.selected_hole_ids) & self.data_model[structure_col].isin(selected_structures)]
        
      
        # Create the stereonet plot based on the filtered data
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='stereonet')
        
        structure_col = self.structure_dropdown.currentText()
        cmap = plt.get_cmap("tab10")
        unique_structures = filtered_data[structure_col].unique()
        structure_color_dict = {structure: cmap(i % cmap.N) for i, structure in enumerate(unique_structures)}

        legend_handles = []  # List to store legend handles

        for structure_type, color in structure_color_dict.items():
            sub_data = filtered_data[filtered_data[structure_col] == structure_type]

            for i in range(len(sub_data) - 1):
                azimuth = filtered_data[self.azimuth_column].iloc[i]
                plunge = filtered_data[self.dip_column].iloc[i]
                plunge = abs(plunge)
                
                tca = sub_data[self.tca_column].iloc[i]
                tca = 90-tca 
                
                strike = (azimuth + 90) % 360
                dip = 90 - plunge

                # Plot the pole
                ax.pole(strike, dip, marker='o', color=color, alpha=0.5)

                # Plot the TCA circle using the cone method with an unfilled circle
                circle = ax.cone(plunge, azimuth, tca, color=color, facecolor='none', alpha=0.5, bidirectional=False, label=structure_type)

            # Create a Line2D object for the legend (only for the first iteration to avoid duplicate legend entries)
            legend_handles.append(Line2D([0], [0], color=color, marker='o', linestyle='', label=structure_type))

        # Add the legend outside the plot
        ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.05, 0.5))

        ax.grid(True, alpha=0.5)
        plt.show()
        self.dialog.accept()
        
class InputDialog(QDialog): # Dialog for cross section topo line settings
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Raise or Lower Topo Line (meters)")
        layout = QVBoxLayout(self)

        # Label
        self.label = QLabel("Enter value (positive to raise, negative to lower):")
        layout.addWidget(self.label)

        # Textbox
        self.textbox = QLineEdit(self)
        self.textbox.setText("0")  # Default value
        layout.addWidget(self.textbox)

        # OK and Cancel buttons
        self.buttons = QPushButton("OK", self)
        self.buttons.clicked.connect(self.accept)
        layout.addWidget(self.buttons)

    def get_value(self):
        return self.textbox.text()











class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow): # START MAIN WINDOW
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # Set up the GUI -----  OpenGeoUI.py
        
        
        # Set the window icon
        self.setWindowIcon(QtGui.QIcon(r"C:\Users\bsomp\Downloads\opvicon.ico"))
        
        
        # Initialize models for the list views
        self.model_lithology = QStandardItemModel(self.hole_id_listView_lithology)
        self.model_geochem = QStandardItemModel(self.hole_id_listView_geochem)
        self.model_structure = QStandardItemModel(self.hole_id_listView_structure)
        self.model_hole_id_cross = QStandardItemModel(self.hole_id_listView_cross)  # Model for hole_id_listView_cross
        self.model_attri = QStandardItemModel(self.listview_attri)  # Model for listview_attri
        self.model_cross = QStandardItemModel(self.hole_id_listView_cross)
        

        # Initialize dataframes for different CSVs
        self.data_lithology = None
        self.data_geochem = None
        self.data_structure = None
        self.data_desurveyed = None  
        
        
        # Initialize user inputs 
        self.attributes_dict = {}
        self.scatter_plot_windows = []
        self.views = []
        self.DEM_data = None
        self.selected_parameters = None
        self.combined_plot_window = None
        
        
        self.factor_analysis_dialog = FactorAnalysisDialog(self)
        self.help_button.clicked.connect(self.on_help_button_clicked)
        self.current_save_file = None
        
        
        self.checkbox_add_grid_lines = False
        self.remove_outliers_auxiliary = False
        self.remove_outliers = True
        self.upper_quantile = 75.0
        self.lower_quantile = 25.0
        self.IQR = 3.0
        self.slider_state = "ft"
       
        
        # Default Values
        self.x_buffer = 120.0
        self.y_buffer = 0.05
        
        self.line_width = 4
        
        self.selected_hole_ids_for_labels = [self.model_cross.item(row).text() for row in range(self.model_cross.rowCount())]
        

         
        # Create a Figure that will hold the plot
        self.figure = Figure(figsize=(8, 12))

        # Create a FigureCanvasQTAgg widget that will hold the Figure
        self.canvas = FigureCanvas(self.figure)

        # Connect main window buttons to their respective methods
        self.import_lithology_csv_button.clicked.connect(self.import_csv_method_lithology)
        self.import_geochem_csv_button.clicked.connect(self.import_csv_method_geochem)
        self.import_structure_csv_button.clicked.connect(self.import_csv_method_structure)
        self.create_graphic_logs_button.clicked.connect(self.create_graphic_log)
        self.downhole_plots_button.clicked.connect(self.create_downhole_plots)
        self.correlation_matrices.clicked.connect(self.create_correlation_matrices)
        self.manage_attributes_button.clicked.connect(self.manage_attributes)
        self.upload_des_data.clicked.connect(self.load_desurveyed_data)
        self.create_cross_section_button.clicked.connect(self.create_cross_section)
        self.stereonet_button.clicked.connect(self.create_stereonet_dialog)
        self.alpha_beta_converter_button.clicked.connect(self.convert_alpha_beta)
        self.view_3d_button.clicked.connect(self.create_3d_plot)
        self.rose_diagram_button.clicked.connect(self.create_rose_plot)
        self.XYscatter_plots_button.clicked.connect(self.create_scatter_plots)
        self.stylize_logs_button.clicked.connect(self.stylize_logs)  
        self.clear_parameters_button.clicked.connect(self.clear_parameters)    
        self.show_lith_button.clicked.connect(self.display_lithology_csv)
        self.show_geochem_button.clicked.connect(self.display_geochem_csv)
        self.show_structure_button.clicked.connect(self.display_structure_csv)
        self.show_desurveyed_button.clicked.connect(self.display_desurveyed_csv)
        self.lith_ft_m.valueChanged.connect(self.updateLithDepthUnit)
        self.geochem_ft_m.valueChanged.connect(self.updateGeochemDepthUnit)
        self.factor_analysis_button.clicked.connect(self.factor_analysis_button_clicked)
        self.help_button.clicked.connect(self.on_help_button_clicked)
        self.actionOpen.triggered.connect(self.open_state)
        self.actionSave.triggered.connect(self.save_state)
        self.actionSave_As.triggered.connect(self.save_state_as)
        self.custom_plot_button.clicked.connect(self.select_custom_plot)
        self.desurvey_button.clicked.connect(self.open_desurvey_calc)
        self.DEM_button.clicked.connect(self.load_DEM)
        self.plot_settings_button.clicked.connect(self.on_plot_settings_clicked)
        self.merge_plots.clicked.connect(self.merge_open_plots)
        self.unoriented_button.clicked.connect(self.on_unoriented_clicked)
        self.actionExport.triggered.connect(self.on_export_clicked)


        # Initialize plot windows
        self.lith_plot_windows = {}
        self.geochem_plot_windows = {}
        self.plot_windows = []
        
        
        # Create a list to hold the CrossSection instances
        self.cross_section_dialogs = []
        
        self.current_file_name = None
        self.actionSave.setEnabled(False)
        
        
    def on_export_clicked(self):
        # Show a dialog to let the user choose which dataset to export
        choice, ok = QInputDialog.getItem(self, "Select Dataset", "Choose a dataset to export:",
                                          ["Geochem CSV", "Lithology CSV", "Structure CSV", "Desurveyed CSV"], 0, False)
        if ok and choice:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "Select File to Export", "",
                                                       "CSV Files (*.csv)", options=options)
            if file_path:
                if choice == "Geochem CSV":
                    self.export_to_csv(self.data_geochem, file_path)
                elif choice == "Lithology CSV":
                    self.export_to_csv(self.data_lithology, file_path)
                elif choice == "Structure CSV":
                    self.export_to_csv(self.data_structure, file_path)
                elif choice == "Desurveyed CSV":
                    self.export_to_csv(self.data_desurveyed, file_path)
   

    def export_to_csv(self, dataset, file_path):
        try:
            dataset.to_csv(file_path, index=False)
            QMessageBox.information(self, "Export Successful", f"File successfully exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export file. Error: {e}")
            
    

    @pyqtSlot()
    def on_unoriented_clicked(self):
        # Check if data_structure is empty
        if self.data_structure is None or self.data_structure.empty:
            # Show a warning message
            warning_msg = QMessageBox()
            warning_msg.setIcon(QMessageBox.Warning)
            warning_msg.setText("Please upload structure CSV first.")
            warning_msg.setWindowTitle("Warning")
            warning_msg.exec_()
            return  # Stop execution of the function
    
        plot_window = UnorientedPlot(self.data_structure, self.model_structure)

    def merge_open_plots(self):
        """
        Combine the open lithology and geochemistry plots into a single window.
        """

        # Check if there are any open windows to merge
        if not self.plot_windows and not self.geochem_plot_windows:
            QMessageBox.warning(self, "No Open Plots", "Please open some plots to merge.")
            return

        # Check if more than one lithology plot is open
        if len(self.plot_windows) > 1:
            QMessageBox.warning(self, "Multiple Lithology Plots", "Please only have one lithology plot open.")
            return
            
        # Check if more than one lithology plot is open
        if len(self.plot_windows) < 1:
            QMessageBox.warning(self, "No Lithology Plots", "Please have one lithology plot open.")
            return

        # Check if the hole_id from the lithology plot doesn't match the hole_id from the geochemistry plots
        lith_hole_id = self.plot_windows[0].hole_ids[0]
        for geochem_plot in self.geochem_plot_windows.values():
            if geochem_plot.hole_id != lith_hole_id:
                QMessageBox.warning(self, "Mismatched Hole IDs", "Ensure the hole_id matches between lithology and geochemistry plots.")
                return

        # Check if multiple hole_ids have been selected for the geochemistry plots
        if len(self.geochem_plot_windows) > 1:
            unique_hole_ids = set([plot.hole_id for plot in self.geochem_plot_windows.values()])
            if len(unique_hole_ids) > 1:
                QMessageBox.warning(self, "Multiple Hole IDs", "Please select only one hole_id for geochemistry plots.")
                return

        # Check if only one type of plot is open
        if not self.plot_windows or not self.geochem_plot_windows:
            QMessageBox.warning(self, "Insufficient Plots", "Please open at least one lithology and one geochemistry plot.")
            return

        # Check if the previously created merged plot window is still open
        if self.combined_plot_window:
            QMessageBox.warning(self, "Previous Merged Plot", "Please close the previously created merged plot window first.")
            return

        # Check if more than 7 geochemistry plots are open
        if len(self.geochem_plot_windows) > 7:
            QMessageBox.warning(self, "Too Many Geochemistry Plots", "Please open no more than 7 geochemistry plots.")
            return
        
        # Close any existing combined plot window
        if self.combined_plot_window:
            self.combined_plot_window.close()
        
        # Create the new combined plot window
        self.combined_plot_window = CombinedPlotWindow(self, self.plot_windows, self.geochem_plot_windows)
        print(f"Merging {len(self.plot_windows)} graphic log plots.")



        
    def updateLithDepthUnit(self, value): # Change from ft to meters
        if value == 0:
            self.lith_depth_unit = "feet"
            self.slider_state = "ft"
        else:
            self.lith_depth_unit = "meters"
            self.slider_state = "m"
        
        
    def on_plot_settings_clicked(self):
    
        # Get all hole_ids from the model
        all_hole_ids = [self.model_cross.item(row).text() for row in range(self.model_cross.rowCount())]

        # Create the PlotSettingsDialog with all hole_ids and current settings
        plot_settings_dialog = PlotSettingsDialog(all_hole_ids)
        plot_settings_dialog.init_defaults(
            remove_outliers=self.remove_outliers,
            remove_outliers_auxiliary=self.remove_outliers_auxiliary,
            add_grid_lines=self.checkbox_add_grid_lines,   # Note: Make sure to call isChecked()
            upper_quantile=self.upper_quantile,
            lower_quantile=self.lower_quantile,
            IQR=self.IQR,
            x_buffer=self.x_buffer,
            y_buffer=self.y_buffer,
            line_width=self.line_width,
            selected_hole_ids_for_labels=self.selected_hole_ids_for_labels
)
            
        if plot_settings_dialog.exec_():
            try:
                # Retrieve and store the settings
                self.remove_outliers = plot_settings_dialog.checkbox_remove_outliers.isChecked()
                # Retrieve and store the settings for "Apply to Auxiliary Bar Plot"
                self.remove_outliers_auxiliary = plot_settings_dialog.checkbox_aux_plot.isChecked()
                self.checkbox_add_grid_lines = plot_settings_dialog.checkbox_add_grid_lines.isChecked()
                
                upper_quantile_text = plot_settings_dialog.line_edit_upper_quantile.text()
                self.upper_quantile = float(upper_quantile_text) if upper_quantile_text else 75.0  # default value
                
                lower_quantile_text = plot_settings_dialog.line_edit_lower_quantile.text()
                self.lower_quantile = float(lower_quantile_text) if lower_quantile_text else 25.0  # default value
                
                IQR_text = plot_settings_dialog.line_edit_IQR.text()
                self.IQR = float(IQR_text) if IQR_text else 3  # default value
                
                x_buffer_text = plot_settings_dialog.line_edit_x_buffer.text()
                self.x_buffer = float(x_buffer_text) if x_buffer_text else 120.0  # default value
                
                y_buffer_text = plot_settings_dialog.line_edit_y_buffer.text()
                self.y_buffer = float(y_buffer_text) if y_buffer_text else 0.05  # default value
                
                line_width_text = plot_settings_dialog.line_edit_line_width.text()
                self.line_width = float(line_width_text) if line_width_text else 4.0  # default value
                self.line_width = max(1.0, min(self.line_width, 5.0))  # Limit between 1 and 5
              
                self.selected_hole_ids_for_labels = plot_settings_dialog.get_selected_hole_ids()
                


            except ValueError:
                QMessageBox.critical(self, "Error", "Invalid input in one or more fields")


        
    def load_DEM(self):
        dem_file, _ = QFileDialog.getOpenFileName(
            self, "Open DEM File", "", "DEM Files (*.tif);;All Files (*)"
        )
        if not dem_file:  # User cancelled the file dialog
            return

        print("DEM file path:", dem_file)
        try:
            dem_data_temp = rasterio.open(dem_file)
            if dem_data_temp is None:
                print("Failed to load DEM data with rasterio.")
                return
            else:
                self.DEM_data = dem_data_temp
        except Exception as e:
            print("Error while loading DEM with rasterio:", str(e))
            return

        ###print("DEM Data Loaded")
        ###print("Shape:", self.DEM_data.shape)
        ###print("DEM data type:", self.DEM_data.dtypes)
        ###print("DEM values range from:", self.DEM_data.read().min(), "to", self.DEM_data.read().max())
        
        self.cross_section_dialogs = []

        # Inform the user of the successful upload
        QMessageBox.information(self, "Success", "DEM Upload Successful")

        for cs_dialog in self.cross_section_dialogs:
            cs_dialog.DEM_data = self.DEM_data
            cs_dialog.DEM_loaded = True
            cs_dialog.run_DEM_calc()
           
        
    def open_desurvey_calc(self):
        try:
            self.desurvey_dialog = DesurveyCalc(self)
            result = self.desurvey_dialog.exec_()
            
            if result == QDialog.Accepted:
       
                pass
            
        except Exception as e:
            print("Error:", e)
        
    def select_custom_plot(self):

        if self.data_geochem is None:
            QMessageBox.warning(self, "No data", "Please load geochemistry data first")
            return

        # Count checked items in self.model_geochem, excluding the 'Select All' option
        checked_items = sum(
            self.model_geochem.item(index).checkState() == Qt.Checked
            for index in range(1, self.model_geochem.rowCount())  # Start from index 1
        )

        if checked_items == 0:
            QMessageBox.warning(self, "No selection", "Please select at least one hole.")
            return


        # Get the selected hole IDs
        selected_hole_ids = []
        for index in range(1, self.model_geochem.rowCount()):
            item = self.model_geochem.item(index)
            if item.checkState() == Qt.Checked:
                selected_hole_ids.append(item.text())

        # Create a new data frame with only the selected holes
        filtered_data = self.data_geochem[self.data_geochem['hole_id'].isin(selected_hole_ids)]
        filtered_data['polygon value'] = "None"

        # Show a file dialog for the user to select the JSON file
        json_file_path, _ = QFileDialog.getOpenFileName(self, "Open JSON file", "", "JSON files (*.json)")

        if json_file_path:  # if user didn't pick a file, do nothing
            # Create and show the CustomPlotDialog with the filtered data
            self.custom_plot_dialog = CustomPlotDialog(filtered_data, json_file_path)
            self.custom_plot_dialog.show()


        
    def factor_analysis_button_clicked(self): 
        if self.data_geochem is None or self.data_geochem.empty:
            # Show a warning message
            warning_msg = QMessageBox()
            warning_msg.setIcon(QMessageBox.Warning)
            warning_msg.setText("Please upload geochem CSV first.")
            warning_msg.setWindowTitle("Warning")
            warning_msg.exec_()
            return  # Stop execution of the function
            
        # Count checked items in self.model_geochem, excluding the 'Select All' option
        checked_items = sum(
            self.model_geochem.item(index).checkState() == Qt.Checked
            for index in range(1, self.model_geochem.rowCount())  # Start from index 1
        )

        if checked_items == 0:
            QMessageBox.warning(self, "No selection", "Please select at least one hole.")
            return

            
        self.factor_analysis_dialog = FactorAnalysisDialog(self)
        self.factor_analysis_dialog.show()
        

        
    def create_rose_plot(self):
        # Check if data_structure is empty
        if self.data_structure is None or self.data_structure.empty:
            # Show a warning message
            warning_msg = QMessageBox()
            warning_msg.setIcon(QMessageBox.Warning)
            warning_msg.setText("Please upload structure CSV first.")
            warning_msg.setWindowTitle("Warning")
            warning_msg.exec_()
            return  # Stop execution of the function
            
        # Count checked items in self.model_geochem and exclude the 'Select All' option
        checked_items = sum(
            self.model_structure.item(index).checkState() == Qt.Checked
            for index in range(1, self.model_structure.rowCount())  # Start from index 1
        )

        if checked_items == 0:
            QMessageBox.warning(self, "No selection", "Please select at least one hole.")
            return

    
        hole_ids = []
        for index in range(1, self.model_structure.rowCount()):
            item = self.model_structure.item(index)
            if item.checkState() == Qt.Checked:  # If the hole_id is checked
                hole_id = item.text()
                hole_ids.append(hole_id)

        # Create RosePlot object
        rose_plot = RosePlot(self.data_structure, hole_ids)

        # Call dialog function
        rose_plot.dialog()

    def convert_alpha_beta(self):
        # Check if data_structure is empty
        if self.data_structure is None or self.data_structure.empty:
            # Show a warning message
            warning_msg = QMessageBox()
            warning_msg.setIcon(QMessageBox.Warning)
            warning_msg.setText("Please upload structure CSV first.")
            warning_msg.setWindowTitle("Warning")
            warning_msg.exec_()
            return  # Stop execution of the function
            
        checked_items = sum(self.model_structure.item(index).checkState() == Qt.Checked for index in range(self.model_structure.rowCount()))
        if checked_items == 0:
            QMessageBox.warning(self, "No selection", "Please select at least one hole.")
            return
    
        if 'alpha' in self.data_structure.columns and 'beta' in self.data_structure.columns:
            if 'drill_hole_azimuth' not in self.data_structure.columns or 'drill_hole_dip' not in self.data_structure.columns:
                QMessageBox.warning(self, 'Warning', 'No azimuth or inclination columns found.')
                return

            self.data_structure["alpha_rad"] = np.deg2rad(self.data_structure["alpha"])
            self.data_structure["beta_rad"] = np.deg2rad(self.data_structure["beta"])
            self.data_structure["I_rad"] = np.deg2rad(self.data_structure["drill_hole_dip"])
            self.data_structure["B_rad"] = np.deg2rad(self.data_structure["drill_hole_azimuth"])

            self.data_structure["n_x_BH"] = np.cos(self.data_structure["beta_rad"]) * np.cos(self.data_structure["alpha_rad"])
            self.data_structure["n_y_BH"] = np.sin(self.data_structure["beta_rad"]) * np.cos(self.data_structure["alpha_rad"])
            self.data_structure["n_z_BH"] = np.sin(self.data_structure["alpha_rad"])

            self.data_structure[["n_x_G", "n_y_G", "n_z_G"]] = self.data_structure.apply(self.transform_to_global, axis=1)

            self.data_structure[["trend", "plunge"]] = self.data_structure.apply(self.compute_trend_and_plunge, axis=1)

            self.data_structure[["strike_converted", "dip_converted"]] = self.data_structure.apply(self.compute_strike_and_dip, axis=1)

            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_name, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "", "CSV Files (*.csv)", options=options)

            if file_name:
                self.data_structure.to_csv(file_name, index=False)
        else:
            QMessageBox.warning(self, 'Warning', 'No alpha or beta columns found.')

    def rotation_matrices(self, row):
        I = row["I_rad"]
        B = row["B_rad"]


        Y_rot = np.array([
            [np.cos(np.pi/2 - I), 0, np.sin(np.pi/2 - I)],
            [0, 1, 0],
            [-np.sin(np.pi/2 - I), 0, np.cos(np.pi/2 - I)]
        ])
        
        Z_rot = np.array([
            [np.cos(np.pi/2 - B), -np.sin(np.pi/2 - B), 0],
            [np.sin(np.pi/2 - B), np.cos(np.pi/2 - B), 0],
            [0, 0, 1]
        ])
        

        return Y_rot, Z_rot

    def transform_to_global(self, row):
        n_BH = np.array([row["n_x_BH"], row["n_y_BH"], row["n_z_BH"]])
        Y_rot, Z_rot = self.rotation_matrices(row)
        n_G = Z_rot @ Y_rot @ n_BH
        return pd.Series(n_G)

    @staticmethod
    def adjust_trend(drill_hole_azimuth):
        temp = 360 - drill_hole_azimuth
        hemispheres = np.array([0, 180, 360])
        differences = np.abs(hemispheres - temp)
        min_difference_index = np.argmin(differences)
        closest_hemisphere = hemispheres[min_difference_index]
        final_subtraction_value = temp - (closest_hemisphere - temp)
        return final_subtraction_value
        
    @staticmethod
    def hemisphere_adjustment(drill_hole_azimuth, alpha, beta):
        if (alpha < 30 and (beta < 75 or beta > 285)):
            # Hemisphere adjustment is reversed
            if 90 <= drill_hole_azimuth <= 270:
                # return adjustment for upper hemisphere
                return 90
            else:
                # return adjustment for lower hemisphere
                return -90
        else:
            if 90 <= drill_hole_azimuth <= 270:
                # lower hemisphere
                return -90
            else:
                # upper hemisphere
                return 90


    def compute_trend_and_plunge(self, row):
        trend = np.rad2deg(np.arctan2(row["n_y_G"], row["n_x_G"])) - self.adjust_trend(row["drill_hole_azimuth"]) + self.hemisphere_adjustment(row["drill_hole_azimuth"], row["alpha"], row["beta"])
        while trend < 0: 
            trend += 360  # ensure trend is between 0 and 360 degrees
        plunge = np.rad2deg(np.arcsin(-row["n_z_G"]))
        return pd.Series([trend, plunge])

    def compute_strike_and_dip(self, row):
        dip = 90 - abs(row["plunge"])
        strike = np.where(row["trend"] >= 90, row["trend"] - 90, row["trend"] + 270)
        return pd.Series([strike, dip])

            
    def create_stereonet_dialog(self):
    
        # Check if data_structure is empty
        if self.data_structure is None or self.data_structure.empty:
            # Show a warning message
            warning_msg = QMessageBox()
            warning_msg.setIcon(QMessageBox.Warning)
            warning_msg.setText("Please upload structure CSV first.")
            warning_msg.setWindowTitle("Warning")
            warning_msg.exec_()
            return  # Stop execution of the function
            
        # Count checked items in self.model_geochem, excluding the 'Select All' option
        checked_items = sum(
            self.model_structure.item(index).checkState() == Qt.Checked
            for index in range(1, self.model_structure.rowCount())  # Start from index 1
        )

        if checked_items == 0:
            QMessageBox.warning(self, "No selection", "Please select at least one hole.")
            return


        dialog = StereonetDialog(self)
        
        if self.data_structure is not None:
            column_names = self.data_structure.columns.tolist()
            dialog.strike_combobox.addItems(column_names)
            dialog.dip_combobox.addItems(column_names)
            dialog.color_coding_combobox.addItems(column_names)

        if dialog.exec_() == QDialog.Accepted:
            strike_column = dialog.strike_combobox.currentText()
            dip_column = dialog.dip_combobox.currentText()
            plot_type = dialog.type_combobox.currentText()
            color_coding_column = dialog.color_coding_combobox.currentText()

            # Check if the selected color coding column is numerical
            if color_coding_column != "None" and self.data_structure[color_coding_column].dtype in ['int64', 'float64']:
                warning_msg = QMessageBox()
                warning_msg.setIcon(QMessageBox.Warning)
                warning_msg.setText("Color coding column cannot be numerical.")
                warning_msg.setWindowTitle("Warning")
                warning_msg.exec_()
                return  # Stop execution of the function

            # Filter the data for selected hole ids
            selected_data = pd.DataFrame()  # Initialize an empty DataFrame
            for index in range(1, self.model_structure.rowCount()):
                item = self.model_structure.item(index)
                if item.checkState() == Qt.Checked:  # If the hole_id is checked
                    hole_id = item.text()

                    # Filter the DataFrame to get the data of the selected hole_id
                    hole_data = self.data_structure[self.data_structure['hole_id'] == hole_id]

                    # Append the data of the selected hole_id to the accumulated DataFrame
                    selected_data = pd.concat([selected_data, hole_data])

            self.stereonet_plot_window = StereonetPlotWindow(selected_data, strike_column, dip_column, plot_type, color_coding_column)

        
    def create_scatter_plots(self):

        if self.data_geochem is None:
            QMessageBox.warning(self, "No data", "Please load geochemistry data first")
            return
            
        # Count checked items in self.model_geochem
        checked_items = sum(
            self.model_geochem.item(index).checkState() == Qt.Checked
            for index in range(1, self.model_geochem.rowCount())  # Start from index 1
        )

        if checked_items == 0:
            QMessageBox.warning(self, "No selection", "Please select at least one hole.")
            return


        # Get the selected hole IDs
        selected_hole_ids = []
        for index in range(1, self.model_geochem.rowCount()):
            item = self.model_geochem.item(index)
            if item.checkState() == Qt.Checked:
                selected_hole_ids.append(item.text())

        # Create and show the ScatterPlotDialog
        self.scatter_plot_dialog = ScatterPlotDialog(self.data_geochem, selected_hole_ids)
        self.scatter_plot_dialog.exec_()
        self.scatter_plot_dialog.close()

        
    def manage_attributes(self): # for cross section
        if self.data_desurveyed is None:
            QMessageBox.warning(self, "No data", "Please load desurveyed data first")
            return

        self.manage_attributes_dialog = ManageAttributesDialog(self.data_desurveyed, self.attributes_dict, self)   
        self.manage_attributes_dialog.exec_()

        self.attributes_dict = self.manage_attributes_dialog.get_attributes_dict()
      
    def load_desurveyed_data(self): # for cross section
        filename, _ = QFileDialog.getOpenFileName(self, "Select desurveyed data file", "", "CSV files (*.csv)")
        if filename:
            print("Selected file:", os.path.basename(filename))
            try:
                self.data_desurveyed = pd.read_csv(filename, header=0)
                
                # Create a new QStandardItemModel and set it as the model for hole_id_listView_cross
                self.model_cross = QStandardItemModel(self.hole_id_listView_cross)
                self.hole_id_listView_cross.setModel(self.model_cross)
                self.add_check_all_option(self.model_cross)

                # Get the unique hole IDs from the data
                unique_hole_ids = self.data_desurveyed['hole_id'].unique()
                
                # Print out the unique hole IDs to check they are correct
                print("Unique hole IDs: ", unique_hole_ids)

                # Populate the hole_id_listView_cross with the unique hole IDs
                for hole_id in unique_hole_ids:
                    item = QStandardItem(str(hole_id))  # Convert hole_id to string
                    item.setCheckable(True)
                    self.model_cross.appendRow(item)
                    
                # Initialize self.selected_hole_ids_for_labels with all hole IDs
                self.selected_hole_ids_for_labels = list(map(str, unique_hole_ids))

                # Print out self.selected_hole_ids_for_labels to check it is populated correctly
                print("MainWindow - selected_hole_ids_for_labels after loading data: ", self.selected_hole_ids_for_labels)
                    
                
                # Get the column names from the data, exclude x, y, z, and hole_id columns
                column_names = [col for col in self.data_desurveyed.columns if col not in ['x', 'y', 'z', 'hole_id']]

                # Clear listview_attri
                self.model_attri.clear()

                # Populate listview_attri with column names
                for col_name in column_names:
                    item = QStandardItem(col_name)
                    item.setCheckable(True)
                    item.setForeground(QtGui.QBrush(QtGui.QColor("black")))  # Set the text color to black
                    self.model_attri.appendRow(item)
                
                self.listview_attri.setModel(self.model_attri)  # Set the model for listview_attri

            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    
    def get_selected_ppm_column(self): # for cross section
        for index in range(self.model_attri.rowCount()):
            item = self.model_attri.item(index)
            if item.checkState() == Qt.Checked:
                return item.text()
        return None


    def create_cross_section(self):
        print("MainWindow - In create_cross_section: ", self.selected_hole_ids_for_labels)
        if self.data_desurveyed is None:
            QMessageBox.warning(self, "No data", "Please load desurveyed data first")
            return


        selected_hole_ids = []
        
        # Iterate over the items of the hole_id_listView_cross
        for index in range(1, self.model_cross.rowCount()):
            item = self.model_cross.item(index)
            if item.checkState() == Qt.Checked:  # If the hole_id is checked
                selected_hole_ids.append(item.text())

        if not selected_hole_ids:
            QMessageBox.warning(self, "No hole IDs selected", "Please select at least one hole ID")
            return

        azimuth, ok = QInputDialog.getDouble(self, "Input", "Enter the direction to view (0-360):", min=0, max=360)
        

        if not ok:
            
            return
        
        # Get all selected attribute columns
        selected_attributes = [self.model_attri.item(index).text() for index in range(self.model_attri.rowCount())
                              if self.model_attri.item(index).checkState() == Qt.Checked]

        # If no attributes are selected, default to a list containing None
        if not selected_attributes:
            selected_attributes = [None]

        try:
            # Loop over all selected attributes (or just once with None) and create a CrossSection for each
            for attribute in selected_attributes:
                print("MainWindow - Before creating CrossSection: ", self.selected_hole_ids_for_labels)
                cross_section = CrossSection(
                    data=self.data_desurveyed, 
                    hole_ids=selected_hole_ids, 
                    azimuth=azimuth, 
                    attribute_column=attribute,  
                    attributes_model=self.model_attri, 
                    attributes_dict=self.attributes_dict, 
                    DEM_data=self.DEM_data if hasattr(self, 'DEM_data') else None, 
                    remove_outliers=self.remove_outliers, 
                    upper_quantile=self.upper_quantile, 
                    lower_quantile=self.lower_quantile, 
                    IQR=self.IQR, 
                    x_buffer=self.x_buffer, 
                    y_buffer=self.y_buffer, 
                    line_width=self.line_width, 
                    selected_hole_ids_for_labels=self.selected_hole_ids_for_labels, 
                    remove_outliers_auxiliary=self.remove_outliers_auxiliary, 
                    checkbox_add_grid_lines=self.checkbox_add_grid_lines
                )  
                cross_section.show()  # Show the CrossSection window immediately
                self.cross_section_dialogs.append(cross_section)  # Append the dialog to the list
        except Exception as e:
            print(e)
            QMessageBox.critical(self, "Error", str(e))


           
    def import_csv_method(self, model): # all except desurvey
        data = None
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "CSV Files (*.csv)", options=options)
        if file_name:
            print(f"Opening file: {file_name}")
            data = pd.read_csv(file_name, header=0)

            # Check if 'hole_id' column exists in the data
            if 'hole_id' not in data.columns:
                QMessageBox.warning(self, 'Warning', "Please make sure CSV contains 'hole_id' column.")
                return None

            # Clear the hole_id_listWidget and then populate it with the hole-ids
            model.clear()
            hole_ids = data['hole_id'].unique()
            for hole_id in hole_ids:
                item = QStandardItem(str(hole_id))  # Create a QStandardItem
                item.setCheckable(True)  # Make the item checkable
                item.setForeground(QtGui.QBrush(QtGui.QColor("black")))  # Set the text color to black
                model.appendRow(item)  # Add each hole_id to the listWidget


        return data
        
        
        
    def column_type(self, column, dataframe): # Determine column type
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            return 'continuous'
        else:
            return 'categorical'

        
    def create_3d_plot(self):
        if self.data_desurveyed is None:
            QMessageBox.warning(self, "No data", "Please load desurveyed data first")
            return
        
        selected_hole_ids = [
            self.model_cross.item(index).text() 
            for index in range(1, self.model_cross.rowCount())  # Start from index 1
            if self.model_cross.item(index).checkState() == Qt.Checked
        ]

                                
        selected_attribute = None
        for index in range(1, self.model_attri.rowCount()):
            item = self.model_attri.item(index)
            if item.checkState() == Qt.Checked:
                selected_attribute = item.text()

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        if selected_attribute and selected_attribute not in self.attributes_dict:
            if self.column_type(selected_attribute, self.data_desurveyed) == 'continuous':
            
                # calculate the IQR and remove outliers if the selected attribute is continuous.
                data_copy = self.data_desurveyed.copy()
                Q1 = data_copy[selected_attribute].quantile(0.25)
                Q3 = data_copy[selected_attribute].quantile(0.75)
                IQR = Q3 - Q1
                data_copy = data_copy[~((data_copy[selected_attribute] < (Q1 - 3 * IQR)) | (data_copy[selected_attribute] > (Q3 + 3 * IQR)))]

                attribute_min = data_copy[selected_attribute].min()
                attribute_max = data_copy[selected_attribute].max()
                cmap = plt.get_cmap('Spectral_r')
                norm = plt.Normalize(vmin=attribute_min, vmax=attribute_max)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
            else:
                sm = None
        else:
            sm = None

        column_type = None
        if selected_attribute:
            column_type = self.column_type(selected_attribute, self.data_desurveyed)
        
        # Categorical data plotting
        color_dict = None
        for hole_id in selected_hole_ids:
            hole_data = self.data_desurveyed[self.data_desurveyed['hole_id'] == hole_id]
            hole_data.sort_values('z', inplace=True, ascending=False)

            if selected_attribute and selected_attribute in hole_data.columns:
                if selected_attribute in self.attributes_dict:
                    for i in range(1, len(hole_data)):
                        x_values = hole_data['x'].iloc[i-1:i+1]
                        y_values = hole_data['y'].iloc[i-1:i+1]
                        z_values = hole_data['z'].iloc[i-1:i+1]
                        attribute_value = hole_data[selected_attribute].iloc[i]

                        if attribute_value in self.attributes_dict[selected_attribute]:
                            color = self.attributes_dict[selected_attribute][attribute_value]["color"]
                            ax.plot(x_values, y_values, z_values, color=color, linewidth=3)
                else:
                    if column_type == 'categorical':
                        unique_vals = self.data_desurveyed[selected_attribute].unique()
                        colors = plt.cm.get_cmap('tab10', len(unique_vals))
                        color_dict = dict(zip(unique_vals, colors(range(len(unique_vals)))))

                        for i in range(1, len(hole_data)):
                            x_values = hole_data['x'].iloc[i-1:i+1]
                            y_values = hole_data['y'].iloc[i-1:i+1]
                            z_values = hole_data['z'].iloc[i-1:i+1]
                            attribute_value = hole_data[selected_attribute].iloc[i]
                            color = color_dict[attribute_value]
                            ax.plot(x_values, y_values, z_values, color=color, linewidth=3)
                    elif sm is not None:
                        for i in range(1, len(hole_data)):
                            x_values = hole_data['x'].iloc[i-1:i+1]
                            y_values = hole_data['y'].iloc[i-1:i+1]
                            z_values = hole_data['z'].iloc[i-1:i+1]
                            attribute_value = hole_data[selected_attribute].iloc[i]
                            color = sm.to_rgba(attribute_value)
                            ax.plot(x_values, y_values, z_values, color=color, linewidth=3)
            else:
                ax.plot(hole_data['x'], hole_data['y'], hole_data['z'], color='black', linewidth=0.8)
                

            bottom_of_hole = hole_data.loc[hole_data['z'].idxmin()]
            ax.text(bottom_of_hole['x'], bottom_of_hole['y'], bottom_of_hole['z'], hole_id, fontsize=8)
                
        if sm is not None:
            fig.colorbar(sm, ax=ax, orientation='vertical', label=selected_attribute, shrink=0.5)
                
        if selected_attribute and column_type != 'continuous':
            if selected_attribute in self.attributes_dict:
                legend_elements = [Line2D([0], [0], color=self.attributes_dict[selected_attribute][val]['color'], lw=4, label=val) for val in self.attributes_dict[selected_attribute]]
            elif color_dict is not None:
                legend_elements = [Line2D([0], [0], color=color_dict[val], lw=4, label=val) for val in color_dict.keys()]
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.7, 0.5))
            
        plt.get_current_fig_manager().window.showMaximized()
        plt.show()


    def add_check_all_option(self, model): # Check all option for the qlistviews
        check_all_item = QStandardItem("Select All")

        # Set font to italics
        font = QFont()
        font.setItalic(True)
        check_all_item.setFont(font)

        # Set the text color with reduced opacity
        color = QColor(0, 0, 0, 200)  # Black color with alpha set to 5
        check_all_item.setForeground(color)

        check_all_item.setCheckable(True)
        model.insertRow(0, check_all_item)

        model.itemChanged.connect(lambda item, model=model: self.on_item_changed(item, model))

    def on_item_changed(self, item, model):
        if item.row() == 0 and item.text() == "Select All":
            check_state = item.checkState()
            for row in range(1, model.rowCount()):
                child_item = model.item(row)
                if child_item.isCheckable():
                    child_item.setCheckState(check_state)

    def import_csv_method_lithology(self): # Lithology Import
        self.data_lithology = self.import_csv_method(self.model_lithology)
        if self.data_lithology is not None:
            self.hole_id_listView_lithology.setModel(self.model_lithology)
            self.add_check_all_option(self.model_lithology)

    def import_csv_method_geochem(self): # geochem Import
        self.data_geochem = self.import_csv_method(self.model_geochem)
        if self.data_geochem is not None:
            self.hole_id_listView_geochem.setModel(self.model_geochem)
            self.add_check_all_option(self.model_geochem)

    def import_csv_method_structure(self): # structure Import
        self.data_structure = self.import_csv_method(self.model_structure)
        if self.data_structure is not None:
            self.hole_id_listView_structure.setModel(self.model_structure)
            self.add_check_all_option(self.model_structure)

    def create_downhole_plots(self):
        
        if self.data_geochem is None:
            QMessageBox.warning(self, "No data", "Please load geochemistry data first")
            return
            
        checked_items = sum(
            self.model_geochem.item(index).checkState() == Qt.Checked
            for index in range(1, self.model_geochem.rowCount())  # Start from index 1
        )

        if checked_items == 0:
            QMessageBox.warning(self, "No selection", "Please select at least one hole.")
            return

        data = self.data_geochem

        dialog = ColumnSelectionDialog(self)
        dialog.load_columns(self.data_geochem.columns)
        dialog.load_depth_columns(self.data_geochem.columns)
        
        for index in range(dialog.column_listWidget.count()):
            print(dialog.column_listWidget.item(index).text())
        
        if dialog.exec_():
            plot_bars = dialog.plot_bars_checkbox.isChecked()
            selected_columns = []
            for index in range(dialog.column_listWidget.count()):
                item = dialog.column_listWidget.item(index)
                if item.checkState() == Qt.Checked:
                    selected_columns.append(item.text())
            
            # Iterate over the items of the QListView_geochem
            for index in range(1, self.model_geochem.rowCount()):
                item = self.model_geochem.item(index)
                if item.checkState() == Qt.Checked:  # If the hole_id is checked
                    hole_id = item.text()
                    print(f"Hole ID: {hole_id}")
                    # Filter the DataFrame to get the data of the selected hole_id
                    hole_data = self.data_geochem[self.data_geochem['hole_id'] == hole_id]
                    
                    selected_depth_column = dialog.depth_column_combo.currentText()
                    # Check if the selected depth column is numerical
                    if not np.issubdtype(self.data_geochem[selected_depth_column].dtype, np.number):
                        QMessageBox.warning(self, "Invalid selection", "Please select a numerical column for depth.")
                        return

                    # For each selected column, create a DownholePlotWindow and add it to the list of PlotWindows
                    for column in selected_columns:
                        print(f"Creating plot for column: {column}")
                        # Create a copy of the data for this column and fill NaNs with 0
                        column_data = hole_data[column].fillna(0)

                        # Create a unique key for the window. It could be based on hole_id and column.
                        window_key = f"{hole_id}_{column}"

                        # Create a new window and add it to the dictionary using the unique key.
                        new_plot_window = DownholePlotWindow(self, hole_data, hole_id, column_data, column, selected_depth_column, plot_bars=plot_bars)
                        self.geochem_plot_windows[window_key] = new_plot_window

                        # Set the initial depth unit based on the slider
                        new_plot_window.updategeochemDepthUnit(self.geochem_ft_m.value())

                        # Connect the QSlider valueChanged signal to the new window's updategeochemDepthUnit methods
                        self.geochem_ft_m.valueChanged.connect(new_plot_window.updategeochemDepthUnit)
                        

    def create_graphic_log(self):
        if self.data_lithology is None:
            QMessageBox.warning(self, "No data", "Please load lithology data first")
            return

        if self.selected_parameters is None:
            QMessageBox.warning(self, "No parameters", "Please stylize logs first")
            return
            
        # Count checked items in self.model_lithology
        checked_items = sum(
            self.model_lithology.item(index).checkState() == Qt.Checked
            for index in range(1, self.model_lithology.rowCount())  # Start from index 1
        )

        if checked_items == 0:
            QMessageBox.warning(self, "No selection", "Please select at least one hole.")
            return

        # List comprehension for selected_hole_ids, starting from index 1
        selected_hole_ids = [
            self.model_lithology.item(index).text()
            for index in range(1, self.model_lithology.rowCount())  # Start from index 1
            if self.model_lithology.item(index).checkState() == Qt.Checked
        ]

        selected_hole_data = self.data_lithology[self.data_lithology['hole_id'].isin(selected_hole_ids)]


        # Determine the unit from the slider's current position
        current_unit = "ft" if self.lith_ft_m.value() == 0 else "m"
        
        plot_window = PlotWindow(self, selected_hole_data, selected_hole_ids, self.selected_parameters, current_unit)
        self.lith_ft_m.valueChanged.connect(plot_window.updateLithDepthUnit)
        plot_window.show()
        self.plot_windows.append(plot_window)  # Append the window to the list to keep a reference
        print(f"Total graphic log plot windows: {len(self.plot_windows)}")
        print("Selected hole IDs for lithology:", selected_hole_ids)




            
    def stylize_logs(self):
        if self.data_lithology is None:
            QMessageBox.warning(self, "No data", "Please load lithology data first")
            return

        data = self.data_lithology

        dialog = ColumnSelectorDialog(data)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            self.selected_parameters = dialog.get_parameters()  # Store the user's selections in an instance variable
            print("Stored parameters:", self.selected_parameters)  
        else:
            QMessageBox.warning(self, "No parameters selected", "Please select parameters")

        QMessageBox.information(self, "Success", "Log parameters saved!")

        
    def clear_parameters(self):
        self.selected_parameters = None
        QMessageBox.information(self, "Parameters cleared", "Stylization parameters have been cleared.")
    
    def create_correlation_matrices(self):
        if self.data_geochem is None:
            QMessageBox.warning(self, "No data", "Please load geochemistry data first")
            return
            
        checked_items = sum(
            self.model_geochem.item(index).checkState() == Qt.Checked
            for index in range(1, self.model_geochem.rowCount())  # Start from index 1
        )

        if checked_items == 0:
            QMessageBox.warning(self, "No selection", "Please select at least one hole.")
            return

        # Get the ending tag from the user
        ending_tag, ok = QInputDialog.getText(self, "Choose Ending Tag", "Enter the ending tag for the columns (e.g., _ppm):")
        if not ok or not ending_tag:
            QMessageBox.warning(self, "No ending tag", "You did not enter an ending tag.")
            return

        # Count the number of columns that contain the user-specified ending tag in their name
        selected_columns = [col for col in self.data_geochem.columns if ending_tag in col]

        if len(selected_columns) < 2:
            QMessageBox.warning(self, "Insufficient data", f"Please add {ending_tag} values first")  
            return

        # Combine data from all selected holes
        selected_data = pd.DataFrame()
        selected_hole_ids = []

        for index in range(1, self.model_geochem.rowCount()):
            item = self.model_geochem.item(index)
            if item.checkState() == Qt.Checked:
                hole_id = item.text()
                selected_hole_ids.append(hole_id)
                hole_data = self.data_geochem[self.data_geochem['hole_id'] == hole_id][selected_columns]
                selected_data = pd.concat([selected_data, hole_data])

        # Check if there is any data to plot
        if selected_data.empty:
            QMessageBox.warning(self, "No data", "No data available for the selected holes and tag.")
            return

        # Determine the title based on the number of selected holes
        if len(selected_hole_ids) == 1:
            title = selected_hole_ids[0]  # Single selected hole
        else:
            title = "Combined Holes"  # Multiple holes

        # Create a correlation matrix with the combined data
        self.geochem_plot_windows[title] = CorrelationMatrixWindow(selected_data, title, ending_tag)

            
    def display_lithology_csv(self):  # Open the table
        if self.data_lithology is None:
            QMessageBox.warning(self, "Warning", "Please upload lithology data first.")
        else:
            self.display_data(self.data_lithology)

    def display_geochem_csv(self):  # Open the table
        if self.data_geochem is None:
            QMessageBox.warning(self, "Warning", "Please upload geochemistry data first.")
        else:
            self.display_data(self.data_geochem)

    def display_structure_csv(self):  # Open the table
        if self.data_structure is None:
            QMessageBox.warning(self, "Warning", "Please upload structure data first.")
        else:
            self.display_data(self.data_structure)

    def display_desurveyed_csv(self):  # Open the table
        if self.data_desurveyed is None:
            QMessageBox.warning(self, "Warning", "Please upload desurveyed data first.")
        else:
            self.display_data(self.data_desurveyed)

    def display_data(self, data): # table display properties
        model = pandasModel(data)
        view = QTableView()
        view.setModel(model)

        # Set the font of the horizontal header to be bold
        font = view.horizontalHeader().font()
        font.setBold(True)
        view.horizontalHeader().setFont(font)

        # Set the font of the vertical header to be bold
        font = view.verticalHeader().font()
        font.setBold(True)
        view.verticalHeader().setFont(font)

        view.resize(1100, 600)
        view.show()
        self.views.append(view)
        
    def updateLithDepthUnit(self, value): 
        if value == 0:
            self.lith_depth_unit = "feet"
            self.slider_state = "ft" 
        else:
            self.lith_depth_unit = "meters"
            self.slider_state = "m"  

        # Update all PlotWindows
        for lith_plot_window in self.lith_plot_windows.values():
            lith_plot_window.updateLithDepthUnit(value)


    def updateGeochemDepthUnit(self, value):
        if value == 0:
            self.geochem_depth_unit = "feet"
        else:
            self.geochem_depth_unit = "meters"

        # Update all DownholePlotWindows 
        for geochem_plot_window in self.geochem_plot_windows.values():
            if isinstance(geochem_plot_window, DownholePlotWindow):  
                geochem_plot_window.updategeochemDepthUnit(value)
    
    def on_help_button_clicked(self):
        self.help_dialog = HelpDialog(HELP_TEXT)
        self.help_dialog.show_help()
        self.help_dialog.show()


        
    def save_state(self):
        # Logic to save the state of your application
        hole_ids_lithology = self.get_model_data(self.model_lithology)
        hole_ids_geochem = self.get_model_data(self.model_geochem)
        hole_ids_structure = self.get_model_data(self.model_structure)
        hole_ids_cross = self.get_model_data(self.model_cross)
        attributes = self.get_model_data(self.model_attri)
        with open(self.current_save_file, 'wb') as f:
            pickle.dump((self.data_lithology, self.data_geochem, self.data_structure, self.data_desurveyed, 
                         self.selected_parameters, self.attributes_dict, 
                         hole_ids_lithology, hole_ids_geochem, hole_ids_structure, hole_ids_cross, attributes, self.slider_state), f)
                         
        if self.current_save_file is None:
            # If there is no current save file, show the "Save As" dialog
            self.save_state_as()
        else:
            # Otherwise, save to the current save file
            self.save_state_to_file(self.current_save_file)
            
        # Show message box after saving
        QMessageBox.information(self, "Save Successful", f"File saved to {self.current_save_file}")

    def save_state_as(self):
        # Get data from the models
        hole_ids_lithology = self.get_model_data(self.model_lithology)
        hole_ids_geochem = self.get_model_data(self.model_geochem)
        hole_ids_structure = self.get_model_data(self.model_structure)
        hole_ids_cross = self.get_model_data(self.model_cross)
        attributes = self.get_model_data(self.model_attri)

        # Open a file dialog to get the save file name from the user
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                  "Pickle Files (*.pkl)", options=options)
        
        # If the user selected a file
        if fileName:
            self.current_save_file = fileName  # Update the current save file
            self.update_window_title() # Update the window title
            self.save_state_to_file(fileName)  # Save the state of the application
            
            # Show message box after saving
            QMessageBox.information(self, "Save Successful", f"File saved as {fileName}")

    def save_state_to_file(self, file_name):
        # Get data from the models
        hole_ids_lithology = self.get_model_data(self.model_lithology)
        hole_ids_geochem = self.get_model_data(self.model_geochem)
        hole_ids_structure = self.get_model_data(self.model_structure)
        hole_ids_cross = self.get_model_data(self.model_cross)
        attributes = self.get_model_data(self.model_attri)
        
        # Save the state of your application
        with open(file_name, 'wb') as f:
            pickle.dump((self.data_lithology, self.data_geochem, self.data_structure, self.data_desurveyed,
                         self.selected_parameters, self.attributes_dict, 
                         hole_ids_lithology, hole_ids_geochem, hole_ids_structure, hole_ids_cross, attributes, self.slider_state), f)
        

    def open_state(self): # Open a saved project
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "Pickle Files (*.pkl)", options=options)
        if fileName:
            try:
                with open(fileName, 'rb') as f:
                    loaded_data = pickle.load(f)

                if len(loaded_data) == 11:
                    self.data_lithology, self.data_geochem, self.data_structure, self.data_desurveyed, \
                    self.selected_parameters, self.attributes_dict, \
                    hole_ids_lithology, hole_ids_geochem, hole_ids_structure, hole_ids_cross, attributes = loaded_data
                    self.DEM_loaded = False
                    # You can set a default slider_state here if needed
                    self.slider_state = "ft"  # or whatever default value you'd like
                else:
                    self.data_lithology, self.data_geochem, self.data_structure, self.data_desurveyed, \
                    self.selected_parameters, self.attributes_dict, \
                    hole_ids_lithology, hole_ids_geochem, hole_ids_structure, hole_ids_cross, attributes, self.slider_state = loaded_data


                # Load model data and add 'check all' option if data is present
                self.load_model_data_and_check_all(self.model_lithology, hole_ids_lithology)
                self.load_model_data_and_check_all(self.model_geochem, hole_ids_geochem)
                self.load_model_data_and_check_all(self.model_structure, hole_ids_structure)
                self.load_model_data_and_check_all(self.model_cross, hole_ids_cross)
                self.load_model_data(self.model_attri, attributes)

                
                # Update the current save file to the file that was just opened
                self.current_save_file = fileName
                self.update_window_title()  # Update the window title

                # Enable the "Save" action now that we have a file to save to
                self.actionSave.setEnabled(True)
                
                # Update selected_hole_ids_for_labels with all unique hole IDs from the loaded data
                if self.data_desurveyed is not None:
                    unique_hole_ids = self.data_desurveyed['hole_id'].unique()
                    self.selected_hole_ids_for_labels = list(map(str, unique_hole_ids))

            except Exception as e:
                QMessageBox.critical(self, "Error", "Failed to load file. Error: " + str(e))
                
        self.hole_id_listView_lithology.setModel(self.model_lithology)
        self.hole_id_listView_geochem.setModel(self.model_geochem)
        self.hole_id_listView_structure.setModel(self.model_structure)
        self.hole_id_listView_cross.setModel(self.model_cross)
        self.listview_attri.setModel(self.model_attri)

        if self.slider_state == "ft":
            self.lith_ft_m.setValue(0)
        else:
            self.lith_ft_m.setValue(1)
            
    def load_model_data_and_check_all(self, model, data):
        self.load_model_data(model, data)
        if data:  # Check if the data list is not empty
            self.add_check_all_option(model)

    def update_window_title(self):
        base_title = "Open Geo Plotter"
        if hasattr(self, 'current_save_file') and self.current_save_file:
            file_name = os.path.basename(self.current_save_file)
            full_title = f"{base_title} - [{file_name}]"
        else:
            full_title = base_title
        self.setWindowTitle(full_title)

    def get_model_data(self, model):
        data = []
        for i in range(model.rowCount()):
            item_text = model.item(i).text()
            if item_text != "Select All":  # Exclude "Select All" from saved state
                data.append(item_text)
        return data


    def load_model_data(self, model, data):
        model.clear()
        for item in data:
            new_item = QStandardItem(item)
            new_item.setCheckable(True)  # Make the item checkable
            model.appendRow(new_item)

    

class HelpDialog(QDialog):
    def __init__(self, help_text):
        super().__init__()

        self.setWindowTitle("Help")

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlainText(help_text)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.text_edit)
        self.setLayout(self.layout)
        
        self.setFixedSize(800, 600)

    def show_help(self):
        self.text_edit.setPlainText(HELP_TEXT)



class pandasModel(QStandardItemModel):
    def __init__(self, data, parent=None):
        QStandardItemModel.__init__(self, parent)
        self._data = data
        for row in self._data.values:
            items = [QStandardItem(str(i)) for i in row]
            self.appendRow(items)
        self.setHorizontalHeaderLabels(self._data.columns)

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size
        
        
def exception_hook(exctype, value, traceback):
    error_message = f"{exctype.__name__}: {value}"
    QMessageBox.critical(None, "An unexpected error occurred", error_message)
    sys.__excepthook__(exctype, value, traceback)
    
                       

    
    
if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)  # enables high dpi scaling
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)  # uses high dpi pixmaps
    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(r"C:\Users\bsomp\Downloads\opvicon.ico"))
    
    # Set the global stylesheet
    stylesheet = """
       QListView {
           color: black;
       }
    """
    app.setStyleSheet(stylesheet)

    window = MainWindow()
    window.showMaximized()
    window.setWindowTitle("Open Geo Plotter")
    
    sys.excepthook = exception_hook

    sys.exit(app.exec_())