# -*- coding: utf-8 -*-
"""
Created on Fri May 19 19:06:57 2023

@author: xiety
"""

# -*- coding: utf-8 -*-
#!!! pip install PyQt5


from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton,QComboBox, QLabel, QTextEdit, QMainWindow, QCheckBox, QLineEdit, QScrollArea, QSizePolicy, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QFont
#from PyQt5.QtCore import Qt
#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import pickle
import csv
import sys
import os
import subprocess
import time
import re


dir_file=(os.popen("pwd").read())
Current_dir=max(dir_file.split('\n'))
root_dir = os.path.expandvars('$HOME')
#code_path="%s/bin/QE-batch/"%root_dir
code_path=sys.path[0]

exchange_dir="exchange_dir"

str_names_out=[]
depth=1
stuff = os.path.abspath(os.path.expanduser(os.path.expandvars(exchange_dir)))
for root,dirnames,filenames in os.walk(stuff):
    if root[len(stuff):].count(os.sep) < depth:
        for dirname in dirnames:
            if "-exhange_file-extractor" in dirname:
                str_names_out.append(dirname.split("-exhange_file-extractor")[0])

def load_stylesheet():
    with open("styles.qss", "r") as f:
        return f.read()

class CheckBoxApp(QWidget):
    def __init__(self,str_names):
        super().__init__()
        # Load and apply QSS
        #stylesheet = load_stylesheet()
        #self.setStyleSheet(stylesheet)
        # 设置全局字体大小
        font = QFont()
        font.setPointSize(14)  # 设置字体大小为 14 点
        app.setFont(font)
        self.str_names=str_names        
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout()

        # 创建全选复选框
        button_cal=QPushButton("Calculate Chosen Structures")
        layout.addWidget(button_cal)
        self.all_checkbox = QCheckBox('全选', self)
        self.all_checkbox.stateChanged.connect(self.check_all)
        layout.addWidget(self.all_checkbox)
        button_cal.clicked.connect(lambda : self.button_func())
        # 创建其他复选框
        self.checkboxes = []
        for i in self.str_names:
            checkbox = QCheckBox(i, self)
            layout.addWidget(checkbox)
            # 不再连接update_all_checkbox_state
            self.checkboxes.append(checkbox)
        
        self.setLayout(layout)
        self.setWindowTitle('Chose the file you need to calculate')
        self.show()

    def check_all(self, state):
        # 如果全选复选框被选中，则选中所有其他复选框，否则取消选中
        is_checked = True if state == 2 else False
        for checkbox in self.checkboxes:
            checkbox.setChecked(is_checked)
    def button_func(self):
        self.str_names_selected = []
        for checkbox in self.checkboxes:
            if checkbox.isChecked():
                self.str_names_selected.append(checkbox.text())
        self.anashow = AnalyzeShow(self.str_names_selected)
        self.anashow.show()

analyze_windows=[]
class AnalyzeShow(QMainWindow):
    def __init__(self, str_names):
        super().__init__() 
        self.str_names=str_names
        print(self.str_names)
        #self.setGeometry(100, 100, 300, 400)
        #self.resize(300, 400)
        self.title = f"Analyze Data"
        self.setWindowTitle(self.title)
        self.initUI()
        analyze_windows.append(self)
    def initUI(self):
        central_widget = QWidget(self)  # 创建一个中央部件
        self.setCentralWidget(central_widget)  # 将中央部件设置为窗口的中央部件
        vbox = QVBoxLayout(central_widget)

        splitter_prop = QScrollArea()
        splitter_diy = QScrollArea()
        splitter_diy.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # 设置大小策略为固定长拉伸宽
        # 创建一个小布局
        self.inner_widget_prop = QWidget()
        self.inner_layout_prop = QVBoxLayout(self.inner_widget_prop)
        inner_widget_diy = QWidget()
        inner_layout_diy = QVBoxLayout(inner_widget_diy)

        button_input = QPushButton("查看数据")
        ####Here you can input your function which could create data and also put the output data in the search_data.py
        self.button_surr = QPushButton('计算周围原子信息')
        self.button_surr.clicked.connect(lambda : self.refresh_checkbox(exe_code=f"{code_path}/get_surrounding_by_elements_multi.py"))
        self.inner_layout_prop.addWidget(self.button_surr)
       
        self.button_surr_statistic = QPushButton('统计周围原子信息')
        self.button_surr_statistic.clicked.connect(lambda : self.refresh_checkbox(exe_code=f"{code_path}/separate_sur_atom_info.py",judge_file=["data.save/SURROUNDING_ATOMS_of_",".txt"],warn_mess="还未统计周围原子信息！"))
        self.button_surr_statistic.clicked.connect(lambda : self.refresh_checkbox(exe_code=f"{code_path}/separate_sur_atom_info_lst.py"))
        self.inner_layout_prop.addWidget(self.button_surr_statistic)

        self.button_octahedral = QPushButton('计算八面体信息')
        self.button_octahedral.clicked.connect(lambda : self.refresh_checkbox(exe_code=f"{code_path}/atomic_info_extractor.py"))
        self.inner_layout_prop.addWidget(self.button_octahedral)


        ##################################################################################################################
        self.all_checkbox = QCheckBox('全选', self)
        self.all_checkbox.stateChanged.connect(self.check_all)
        self.inner_layout_prop.addWidget(self.all_checkbox)

        self.generate_checkbox()

        inner_layout_diy.addWidget(button_input)

        splitter_prop.setWidget(self.inner_widget_prop)
        splitter_prop.setWidgetResizable(True)  # 使内部小部件可以自动调整大小
        splitter_diy.setWidget(inner_widget_diy)
        splitter_diy.setWidgetResizable(True)  # 使内部小部件可以自动调整大小

        vbox.addWidget(splitter_prop)
        vbox.addWidget(splitter_diy)
     
        self.setGeometry(100,100,500, self.height())
        button_input.clicked.connect(lambda : self.button_click_analyze(button_input))

    def check_all(self, state):
        # 如果全选复选框被选中，则选中所有其他复选框，否则取消选中
        is_checked = True if state == 2 else False
        for checkbox in self.checkboxes:
            checkbox.setChecked(is_checked)

    def generate_checkbox(self):
        av_props=[]
        tot_props=[]
        for i in self.str_names:
            av_props_i=[]
            tot_props_i=[]
            av_prop_lines=os.popen("python %s/search_data.py %s \'{1}\' magnetic"%(code_path,i)).readlines()
            for lines in av_prop_lines:
                if "#AV_PROP#" in lines:
                    av_props_i=lines.strip("#AV_PROP#").strip().strip("\n").split("\t")
                elif "#TOT_PROP#" in lines:
                    tot_props_i=lines.strip("#TOT_PROP#").strip().strip("\n").split("\t")
            for j in av_props_i:
                if j not in av_props:
                    av_props.append(j)
            for j in tot_props_i:
                if j not in tot_props:
                    tot_props.append(j)
            
        self.checkboxes=[]
        for props in tot_props:
            checkbox=QCheckBox(props)
            if props not in av_props:
                checkbox.setEnabled(False)
            #checkbox.stateChanged.connect(self.update_all_checkbox_state)
            self.checkboxes.append(checkbox)
            self.inner_layout_prop.addWidget(checkbox)

    def refresh_checkbox(self,exe_code=None,judge_file=None,warn_mess=None):
        for i in self.str_names:
            if (judge_file!=None):
                #print(f"{i}".join(judge_file))
                if not os.path.isfile(f"{i}".join(judge_file)):
                    QMessageBox.critical(self, "错误", f"{i}"+warn_mess)
                else:
                    os.system(f"python {exe_code} {i}")
                    print(f"python {exe_code} {i}")
            else:
                os.system(f"python {exe_code} {i}")
                print(f"python {exe_code} {i}")

        # 清除现有的复选框
        for checkbox in self.checkboxes:
            self.inner_layout_prop.removeWidget(checkbox)
            checkbox.deleteLater()  # 清除并删除复选框
        self.checkboxes.clear()

        # 重新获取数据并创建复选框
        self.generate_checkbox()

    def button_click_analyze(self,button):
        selected_elements = []
        for checkbox in self.checkboxes:
            if checkbox.isChecked():
                selected_elements.append(checkbox.text()+"*")
        if selected_elements:
            properties="+".join(selected_elements)
            self.anaVis = AnaVisual(self.str_names,properties)
            self.anaVis.show()
        else:
            print("No props selected!")


ana_visual_windows=[]
class AnaVisual(QMainWindow):
    def __init__(self, str_names, properties):
        super().__init__() 
        self.setWindowTitle(f"Pearson & Spearman Calculator")
        self.setGeometry(100, 100, 800, 600)
        self.str_names=str_names
        self.table_data=[]
        for i,v in enumerate(str_names):
            filename=f"AnaVisual_Delta_E_{v}.data"
            os.system(f"python %s/exchange_E_extractor.py %s {properties} > %s"%(code_path,v,filename)) 
            print(f"python %s/exchange_E_extractor.py %s {properties} > %s"%(code_path,v,filename)) 
            with open(filename, 'r') as file:
                reader = csv.reader(file, delimiter='\t')
                lst_buffer=[]
                for j,row in enumerate(reader):
                    if row[0].startswith('#'):
                        continue
                    if (i!=0)&(j==0):
                        continue
                    else:
                        lst_buffer.append(row)
                data_raw = list(lst_buffer)
                table_data_i=[[value for value in row if value] for row in data_raw]
                for j in table_data_i:
                    self.table_data.append(j)

        self.result_labels = []  # To keep track of the result labels
        self.initUI()
        ana_visual_windows.append(self)

    def initUI(self):
        #self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Add table
        # Table inside a QScrollArea
        self.scroll_area = QScrollArea(self)
        self.table = QTableWidget(len(self.table_data), len(self.table_data[0]))
        self.scroll_area.setWidget(self.table)
        self.scroll_area.setWidgetResizable(True)
        main_layout.addWidget(self.scroll_area)

        for row in range(len(self.table_data)):
            for col in range(len(self.table_data[row])):
                self.table.setItem(row if row > 0 else 0, col, QTableWidgetItem(str(self.table_data[row][col])))

        # Button to save table data
        save_btn = QPushButton("保存表格内数据")
        save_btn.clicked.connect(self.save_table_data)
        main_layout.addWidget(save_btn)

        # Layout 1
        checkbox_layout = QHBoxLayout()
        main_layout.addLayout(checkbox_layout)
        # Checkboxes inside a QScrollArea
        checkbox_scroll_area = QScrollArea(self)
        checkbox_container = QWidget()
        checkbox_layout = QHBoxLayout()
        checkbox_container.setLayout(checkbox_layout)
        checkbox_scroll_area.setWidget(checkbox_container)
        checkbox_scroll_area.setWidgetResizable(True)
        main_layout.addWidget(checkbox_scroll_area)

        self.all_others_checkbox = QCheckBox("与其他所有")
        self.all_others_checkbox.stateChanged.connect(self.checkbox_changed)
        checkbox_layout.addWidget(self.all_others_checkbox)
        self.checkboxes = []
        for header in self.table_data[0]:
            checkbox = QCheckBox(header)
            checkbox.stateChanged.connect(self.checkbox_changed)
            checkbox_layout.addWidget(checkbox)
            self.checkboxes.append(checkbox)

        # Layout for buttons
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)

        self.pearson_btn = QPushButton("计算pearson系数")
        self.pearson_btn.clicked.connect(self.calculate_pearson)
        button_layout.addWidget(self.pearson_btn)

        self.spearman_btn = QPushButton("计算spearman系数")
        self.spearman_btn.clicked.connect(self.calculate_spearman)
        button_layout.addWidget(self.spearman_btn)
        
        
        # Layout 2
        self.results_scroll_area = QScrollArea(self)
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout()
        self.results_container.setLayout(self.results_layout)
        self.results_scroll_area.setWidget(self.results_container)
        self.results_scroll_area.setWidgetResizable(True)
        main_layout.addWidget(self.results_scroll_area)
        
        self.default_label_note = QLabel("注：Pearson & Spearman系数越接近±1时呈正/负相关，接近0时呈现不相关")
        self.results_layout.addWidget(self.default_label_note)
        self.default_label = QLabel("性质1\t性质2\t相关系数类型\t相关系数值")
        self.results_layout.addWidget(self.default_label)
        # Clear button
        self.clear_btn = QPushButton("清除结果")
        self.clear_btn.clicked.connect(self.clear_results)
        main_layout.addWidget(self.clear_btn)
        # Sort button
        self.sort_btn = QPushButton("排序")
        self.sort_btn.clicked.connect(self.sort_results)
        main_layout.addWidget(self.sort_btn)

    def sort_results(self):
        # Sort the labels based on correlation value
        self.result_labels.sort(key=lambda label: float(label.text().split('\t')[-1]), reverse=True)

        # Clear the layout
        for i in reversed(range(self.results_layout.count())):
            widget = self.results_layout.itemAt(i).widget()
            if widget is not None:
                self.results_layout.removeWidget(widget)
                widget.setParent(None)

        # Re-add the default label
        self.results_layout.addWidget(self.default_label_note)
        self.results_layout.addWidget(self.default_label)

        # Re-add the sorted labels
        for label in self.result_labels:
            self.results_layout.addWidget(label)


    def checkbox_changed(self):
        checked_boxes = [checkbox for checkbox in self.checkboxes if checkbox.isChecked()]
        if self.all_others_checkbox.isChecked():
            if len(checked_boxes) == 1:
                for checkbox in self.checkboxes:
                    if checkbox not in checked_boxes:
                        checkbox.setEnabled(False)
            elif len(checked_boxes) < 1:
                for checkbox in self.checkboxes:
                    checkbox.setEnabled(True)

        elif not self.all_others_checkbox.isChecked():
            if len(checked_boxes) == 2:
                for checkbox in self.checkboxes:
                    if checkbox not in checked_boxes:
                        checkbox.setEnabled(False)
            elif len(checked_boxes) < 2:
                for checkbox in self.checkboxes:
                    checkbox.setEnabled(True)

    def calculate_pearson(self):
        self.calculate_correlation('pearson')

    def calculate_spearman(self):
        self.calculate_correlation('spearman')

    def calculate_correlation(self, method):
        selected_cols = [i for i, checkbox in enumerate(self.checkboxes) if checkbox.isChecked()]

        if self.all_others_checkbox.isChecked() and len(selected_cols) == 1:
            for i, _ in enumerate(self.checkboxes):
                if i != selected_cols[0]:
                    try:
                        col1_data = [float(row[selected_cols[0]]) for row in self.table_data[1:]]
                        col2_data = [float(row[i]) for row in self.table_data[1:]]
                    except ValueError:
                        #QMessageBox.critical(self, "错误", f"{self.table_data[0][selected_cols[0]]} 或 {self.table_data[0][selected_cols[1]]} 内容不是纯数字！")
                        continue
                    #col1_data = [float(row[selected_cols[0]]) for row in self.table_data[1:]]
                    #col2_data = [float(row[i]) for row in self.table_data[1:]]

                    if method == 'pearson':
                        if np.std(col1_data) == 0 or np.std(col2_data) == 0:
                            result=0
                        else:
                            result = np.corrcoef(col1_data, col2_data)[0, 1]
                        method_name = "pearson"
                    elif method == 'spearman':
                        if np.std(col1_data) == 0 or np.std(col2_data) == 0:
                            result=0
                        else:
                            rank_col1 = [sorted(col1_data).index(i)+1 for i in col1_data]
                            rank_col2 = [sorted(col2_data).index(i)+1 for i in col2_data]
                            result = np.corrcoef(rank_col1, rank_col2)[0, 1]
                        method_name = "spearman"
                
                    label_text = f"{self.table_data[0][selected_cols[0]]}\t{self.table_data[0][i]}\t{method_name}\t{result:.3f}"
                    result_label = QLabel(label_text)
                    self.results_layout.addWidget(result_label)
                    self.result_labels.append(result_label) 

        #if len(selected_cols) != 2:
        #    return
           
        elif not self.all_others_checkbox.isChecked() and len(selected_cols) == 2:
            #selected_cols = [i for i, checkbox in enumerate(self.checkboxes) if checkbox.isChecked()]
            try:
                col1_data = [float(row[selected_cols[0]]) for row in self.table_data[1:]]
                col2_data = [float(row[selected_cols[1]]) for row in self.table_data[1:]]
            except ValueError:
                QMessageBox.critical(self, "错误", f"{self.table_data[0][selected_cols[0]]} 或 {self.table_data[0][selected_cols[1]]} 内容不是纯数字！")
                return
            #col1_data = [row[selected_cols[0]] for row in self.table_data[1:]]
            #col2_data = [row[selected_cols[1]] for row in self.table_data[1:]]
            #if np.std(col1_data) == 0 or np.std(col2_data) == 0:
                #QMessageBox.critical(self, "错误", "一个或两个选中的变量的标准差为0，不能计算相关系数。")
                #return    
            if method == 'pearson':
                if np.std(col1_data) == 0 or np.std(col2_data) == 0:
                    result=0
                else:
                    result = np.corrcoef(col1_data, col2_data)[0, 1]
                method_name = "pearson"
            elif method == 'spearman':
                if np.std(col1_data) == 0 or np.std(col2_data) == 0:
                    result=0
                else:
                    rank_col1 = [sorted(col1_data).index(i)+1 for i in col1_data]
                    rank_col2 = [sorted(col2_data).index(i)+1 for i in col2_data]
                    result = np.corrcoef(rank_col1, rank_col2)[0, 1]
                method_name = "spearman"
        
            label_text = f"{self.table_data[0][selected_cols[0]]}\t{self.table_data[0][selected_cols[1]]}\t{method_name}\t{result:.3f}"
            result_label = QLabel(label_text)
            self.results_layout.addWidget(result_label)
            self.result_labels.append(result_label) 
        
    def save_table_data(self):
        # 使用 QFileDialog 获取保存文件的路径
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save_Table_Data', f"Searched_data_.csv", 'CSV Files (*.csv);;Excel Files (*.xls);;All Files (*)', options=options)

        if file_path:
            try:
                # 打开文件以写入数据
                with open(file_path, 'w' , encoding='utf-8-sig') as file:
                    table = self.centralWidget().findChild(QTableWidget)
                    if table:
                        # 遍历表格中的数据并写入文件
                        for row in range(table.rowCount()):
                            row_data = []
                            for col in range(table.columnCount()):
                                item = table.item(row, col)
                                if item:
                                    row_data.append(item.text())
                                else:
                                    row_data.append('')
                            file.write(','.join(row_data) + '\n')
                print(f'Table data saved to {file_path}')
            except Exception as e:
                print(f'Error saving table data: {str(e)}')

    def clear_results(self):
        self.result_labels = []  #Clear: To keep track of the result labels
        for i in reversed(range(self.results_layout.count())):
            widget = self.results_layout.itemAt(i).widget()
            if widget is not None and widget != self.default_label and widget != self.clear_btn and widget != self.sort_btn and widget != self.default_label_note:
                widget.deleteLater()




# Create an application and a main window instance
app = QApplication(sys.argv)
window = CheckBoxApp(str_names_out)
window.show()
sys.exit(app.exec_())

