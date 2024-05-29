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
if os.path.exists("%s/flat.save"%code_path)==0:
    os.system("mkdir %s/flat.save"%code_path)
if os.path.isfile("%s/flat.save/FLAT_INFO"%code_path)==0:
    #print("AAA")
    os.mknod("%s/flat.save/FLAT_INFO"%code_path)
    f_initializing=open("%s/flat.save/FLAT_INFO"%code_path,"w")
    f_initializing.writelines("flat_number=1           # 1 for\"zf_normal\",2 for \"spst_pub\",3 for \"dm_pub_cpu\"\n")
    f_initializing.writelines("node_num=1\n")
    f_initializing.writelines("ppn_num=0               # 0 for default\n")
    f_initializing.writelines("wall_time=\"116:00:00\"\n")
    f_initializing.close()

if os.path.exists("data.save")==0:
    os.system("mkdir data.save")

def read_flat_state():
    flat_info=open("%s/flat.save/FLAT_INFO"%code_path).readlines()
    for lines in flat_info:
        if lines.find("flat_number=")!=-1:
            flat_number=int(lines.split("flat_number=")[-1].split("#")[0])
        if lines.find("node_num=")!=-1:
            node_num=int(lines.split("node_num=")[-1].split("#")[0])
        if lines.find("ppn_num")!=-1:
            ppn_num=int(lines.split("ppn_num=")[-1].split("#")[0])
        if lines.find("wall_time=")!=-1:
            wall_time=lines.split("wall_time=")[-1].split("#")[0].strip("\n").strip("\"")
    
    if flat_number==1:
        type_flat="zf_normal"
    if flat_number==2:
        type_flat="spst_pub"
    if flat_number==3:
        type_flat="dm_pub_cpu"
    return flat_number,node_num,ppn_num,wall_time,type_flat

#initialize some variables
flat_number,node_num,ppn_num,wall_time,type_flat=read_flat_state()
show_y_l_bands=4
show_y_r_bands=4

#show the available orbital for input element,used in pdos simple mode and detailed mode
def atomic_orbit(element_in,str_name):
    common_str_in_output_file="BTO.pdos_atm#"
    common_str_in_output_file2=element_in
    #print(common_str_in_output_file2)
    output_file="accumulated_pdos_file"+common_str_in_output_file2
    dir_file=(os.popen("pwd").read())
    dir_current=max(dir_file.split('\n'))+'/'+str_name
    
    file_list=[]
    for root,dirs,files in os.walk(dir_current):
        for file in files:
            if common_str_in_output_file in file:
                if common_str_in_output_file2 in file.split('#')[1]:
                    file_list.append(file)
    
    file_dic=[]
    for i,files in enumerate(file_list):
        file_tail_tick=files.split("#")[-1].strip("\n").strip(")").split("(")
        file_tail="-"+file_tail_tick[0]+"-"+file_tail_tick[1]
        if file_tail not in file_dic:
            file_dic.append(file_tail)

    def sort_orbit_num(orbit):
        return ''.join([i for i in orbit if i.isdigit()])
    file_dic.sort(key=sort_orbit_num)
    for i,v in enumerate(file_dic):
        file_dic[i]=f"{element_in}{v}"
    #print(file_dic)
    return file_dic

#show the atomic number range and available orbital under this psp,used in pdos manual mode
def show_pdos_detail(str_name):
    common_str_in_output_file="BTO.pdos_atm#"
    dir_file=(os.popen("pwd").read())
    dir_current=max(dir_file.split('\n'))+'/'+str_name
    
    file_list_tot=[]
    for root,dirs,files in os.walk(dir_current):
        for file in files:
            if common_str_in_output_file in file:
                file_list_tot.append(file)
    
    file_dic=[]
    for i,files in enumerate(file_list_tot): #read all the pdos files in the dir and store them in [atomic_number,atomic_element,orbital_number,orbital_type]
        atomic_nb_i=int(files.split("#")[1].split("(")[0])
        atomic_ele_i=files.split("#")[1].split("(")[1].split(")")[0]
        orbit_nb_i=int(files.split("#")[-1].split("(")[0])
        orbit_tp_i=files.split("#")[-1].split("(")[1].split(")")[0]
        file_dic.append([atomic_nb_i,atomic_ele_i,orbit_nb_i,orbit_tp_i])
    file_dic.sort(key=lambda x:(x[0],x[2]))
    orbit_proj=[['s',['s'],[4]],['p',['p','pz','px','py'],[2,4,6,8]],['d',['d','dz2','dzx','dzy','dx2_y2','dxy'],[2,4,6,8,10,12]]]
    ele_type=list(set([x[1] for x in file_dic]))
    ele_range_buffer=[]
    for i,v in enumerate(file_dic):
        if v[1] not in [x[0] for x in ele_range_buffer]:
            ele_range_buffer.append([v[1],[v[0]]])
        elif v[1] in [x[0] for x in ele_range_buffer]:
            if v[0] not in ele_range_buffer[[x[0] for x in ele_range_buffer].index(v[1])][1]:
                ele_range_buffer[[x[0] for x in ele_range_buffer].index(v[1])][1].append(v[0])
    ele_range=[]
    for i in ele_range_buffer:
        ele_range.append([i[0],min(i[1]),max(i[1])])
    ele_orbit=[]# this list stores the info like [['Ni1', ['-1-s', '-2-p', '-3-s', '-4-d']], ['Li', ['-1-s', '-2-s']], ['O', ['-1-s', '-2-p']], ['Co', ['-1-s', '-2-p', '-3-s', '-4-d']], ['Ni2', ['-1-s', '-2-p', '-3-s', '-4-d']]]
    for i,v in enumerate(file_dic):
        v_orb='-'+str(v[2])+'-'+v[3]
        if v[1] not in [x[0] for x in ele_orbit]:
            ele_orbit.append([v[1],[v_orb]])
        elif v[1] in [x[0] for x in ele_orbit]:
            if v_orb not in ele_orbit[[x[0] for x in ele_orbit].index(v[1])][1]:
                ele_orbit[[x[0] for x in ele_orbit].index(v[1])][1].append(v_orb)
    show_results=[]
    for i in range(len(ele_range)):
        ele_i_in_orb=ele_orbit[[x[0] for x in ele_orbit].index(ele_range[i][0])][1]
        ele_str_orb=",".join(ele_i_in_orb)
        show_results.append(f"{ele_range[i][0]}:({ele_range[i][1]},{ele_range[i][2]})[{ele_str_orb}]")

    return "\n".join(show_results)
    

def load_stylesheet():
    with open("styles.qss", "r") as f:
        return f.read()

# Define the main window class
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # Load and apply QSS
        #stylesheet = load_stylesheet()
        #self.setStyleSheet(stylesheet)
        # 设置全局字体大小
        font = QFont()
        font.setPointSize(14)  # 设置字体大小为 14 点
        app.setFont(font)

        self.resize(600, 400)
        self.setWindowTitle('QE-batch_V-9.2')
        self.move(100, 100)
        label = QLabel("开发版",self)
        label.move(0,0)
        button = QPushButton('上传文件', self)
        button.move(50, 100)
        #label1 = QLabel("\"一键计算\"会计算当前目录下的\n所有vasp文件并建立文件夹",self)
        #label1.setGeometry(30,260,180,100)
        button_mode = QPushButton('一键计算', self)
        button_mode.move(50, 250)

        # Add a new button named "new_be"
        button_new_be = QPushButton("查看计算进度", self)
        button_new_be.move(50, 300)


        # Connect the other buttons' clicked signals to your existing methods
        button.clicked.connect(lambda:os.system("rz"))
        label_mode_chose = QLabel("选择模式",self)
        label_mode_chose.setGeometry(50,170,100,30)
        self.combo_box = QComboBox(self)
        self.combo_box.setGeometry(50, 200, 100, 30)
        # Add items to the QComboBox widget
        self.combo_box.addItem("scf")
        self.combo_box.addItem("relax")
        self.combo_box.addItem("bands")
        self.combo_box.addItem("pdos")
        self.combo_box.addItem("BaderCharge")
        # Set the default item to "mode"
        self.combo_box.setCurrentText("relax")
        self.mode = self.combo_box.currentText()
        self.combo_box.activated.connect(self.on_activated)

        #flat_chosen combo
        label_flat_chose = QLabel("选择计算平台",self)
        label_flat_chose.setGeometry(300,70,100,30)
        self.combo_flat = QComboBox(self)
        self.combo_flat.setGeometry(300, 100, 100, 30)
        self.combo_flat.addItem("zf_normal")
        self.combo_flat.addItem("spst_pub")
        self.combo_flat.addItem("dm_pub_cpu")
        self.combo_flat.setCurrentText(type_flat)
        self.flat = self.combo_flat.currentText()
        self.combo_flat.activated.connect(self.on_activated_flat)
        ppn_lineedit=QLineEdit(str(ppn_num), self)
        node_lineedit=QLineEdit(str(node_num), self)
        wallT_lineedit=QLineEdit(wall_time, self)

        label_node = QLabel("node number",self)
        label_node.setGeometry(300,125,100,30)
        node_lineedit.setGeometry(300,150,100,30)

        label_ppn = QLabel("ppn number (0 for default)",self)
        label_ppn.setGeometry(300,175,180,30)
        ppn_lineedit.setGeometry(300,200,100,30)

        label_wallT = QLabel("Wall Time [format 116:00:00]",self)
        label_wallT.setGeometry(300,225,220,30)
        wallT_lineedit.setGeometry(300,250,180,30)

        button_flat_save = QPushButton('保存', self)
        button_flat_save.move(300, 300)

        button_flat_save.clicked.connect(lambda:self.flat_save(ppn_lineedit,node_lineedit,wallT_lineedit))



        # Connect the activated signal to a function
        #button_mode.clicked.connect(lambda:os.system("cp /hpc/data/home/spst/zhengfan/open/replace/in_%s ."%self.mode))
        button_mode.clicked.connect(lambda:os.system("python %s/run-all.py %s"%(code_path,self.mode)))
        # Connect the button's clicked signal to a method that will open the new window
        button_new_be.clicked.connect(lambda:self.read_relax(self.mode))

    def flat_save(self,ppn_lineedit,node_lineedit,wallT_lineedit):
            
        type_flat_new = self.combo_flat.currentText()
        ppn_num_new=ppn_lineedit.text()
        node_num_new=node_lineedit.text()
        wallT_new=wallT_lineedit.text()
        #flat_number,node_num,ppn_num,wall_time,type_flat=read_flat_state()
        print(type_flat_new,node_num_new,ppn_num_new,wallT_new)
        if type_flat_new=="zf_normal":
            flat_number_new=1
        if type_flat_new=="spst_pub":
            flat_number_new=2
        if type_flat_new=="dm_pub_cpu":
            flat_number_new=3

        f_flat_change=open("%s/flat.save/FLAT_INFO_1"%code_path,"w")
        f_flat_change.writelines("flat_number=%d           # 1 for\"zf_normal\",2 for \"spst_pub\",3 for \"dm_pub_cpu\"\n"%flat_number_new)
        f_flat_change.writelines("node_num=%s\n"%node_num_new)
        f_flat_change.writelines("ppn_num=%s               # 0 for default\n"%ppn_num_new)
        f_flat_change.writelines("wall_time=\"%s\"\n"%wallT_new)
        f_flat_change.close()
        os.system("mv %s/flat.save/FLAT_INFO_1 %s/flat.save/FLAT_INFO"%(code_path,code_path))

    # Define a function that will be executed when an item is selected

    def on_activated(self):
        # Get the selected item
        self.mode = self.combo_box.currentText()
        # Print the selected item
        print ("inner",self.mode)
        #return self.mode
    def on_activated_flat(self):
        # Get the selected item
        self.flat = self.combo_box.currentText()
        # Print the selected item
        print ("inner",self.mode)
    def read_relax(self,mode):
        read_back=os.popen("python %s/read_relax_E_for_UI.py %s"%(code_path,mode)).readlines()
        for i in read_back: 
            if i.find("pv_result_out generated!")!=-1:
                self.show_data(mode)
    # Define a method that will open the new window and close the current one (optional)
    def show_data(self,mode):
        self.data_show = DataShow(mode)
        self.data_show.show()
        # Uncomment the next line if you want to close the current window
        # self.close()

class DataShow(QWidget):
    def __init__(self,mode):
        super().__init__()
        self.title = "%s results"%mode
        self.mode=mode
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 900, 500)
        layout = QVBoxLayout()
        if (self.mode=="pdos")|(self.mode=="relax")|(self.mode=="scf"):
            colomnout=QHBoxLayout()
            colomnout.addStretch()
            label_x_rg = QLabel("X range:")
            label_y_rg = QLabel("Y range:")
            label_space =QLabel("        ")
            label_con_x = QLabel(" ~ ")
            label_con_y = QLabel(" ~ ")
            lineEdit_x_l = QLineEdit()
            lineEdit_x_r = QLineEdit()
            lineEdit_y_l = QLineEdit()
            lineEdit_y_r = QLineEdit()
            colomnout.addWidget(label_x_rg)
            colomnout.addWidget(lineEdit_x_l)
            colomnout.addWidget(label_con_x)
            colomnout.addWidget(lineEdit_x_r)
            colomnout.addWidget(label_space)
            colomnout.addWidget(label_y_rg)
            colomnout.addWidget(lineEdit_y_l)
            colomnout.addWidget(label_con_y)
            colomnout.addWidget(lineEdit_y_r)
            if self.mode=='pdos':
                linepara = QLineEdit("2;1;0;1")
                paralabel = QLabel("Line width;Fill VB;Show Ef;Fill pattern")
                colomnout.addWidget(paralabel)
                colomnout.addWidget(linepara)
                # Add new text_line editer
                hidden_col=QHBoxLayout()
                hidden_col.addStretch()
                hidden_edit=QLineEdit("(1-2)-1-s")
                hidden_label=QLabel("Input text:")
                hidden_col.addWidget(hidden_label)
                hidden_col.addWidget(hidden_edit)
                hidden_col.addStretch()
                hidden_edit.hide()
                hidden_label.hide() 
                # Add items to the QComboBox widget
                self.pdos_combo_box = QComboBox(self)
                self.pdos_combo_box.setGeometry(50, 200, 100, 30)
                self.pdos_combo_box.addItem("simple mode")
                self.pdos_combo_box.addItem("detailed orbital mode")
                self.pdos_combo_box.addItem("manual mode")
                # Set the default item to "simple mode"
                self.pdos_combo_box.setCurrentText("simple mode")
                self.pdos_mode = self.pdos_combo_box.currentText()
                #self.pdos_combo_box.activated.connect(self.pdos_on_active)
                layout.addWidget(self.pdos_combo_box)
             
            layout.addLayout(colomnout)
            if self.mode=="pdos":
                layout.addLayout(hidden_col)

        elif self.mode=="bands":
            colomnout=QHBoxLayout()
            colomnout.addStretch()
            label_y_rg = QLabel("Y range:")
            label_minus_y = QLabel(" - ")
            label_con_y = QLabel(" ~ ")
            lineEdit_y_l = QLineEdit("4")
            lineEdit_y_r = QLineEdit("4")
            colomnout.addWidget(label_y_rg)
            colomnout.addWidget(label_minus_y)
            colomnout.addWidget(lineEdit_y_l)
            colomnout.addWidget(label_con_y)
            colomnout.addWidget(lineEdit_y_r)
            layout.addLayout(colomnout)

        elif self.mode=="BaderCharge":
            pass
        table = QTableWidget()
        layout.addWidget(table)
        self.setLayout(layout)


        with open('pv_result_out', 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            data = list(reader)

        table.setRowCount(len(data))
        max_column=max([len(i) for i in data])
        #print(max_column)
        table.setColumnCount(max_column+1)
        button_clicked = [False] * len(data) # initialize list of boolean flags

        #print(button_clicked)

        #for i in range(table.rowCount()):
            #for j in range(table.columnCount()-1):
        for i in range(len(data)):
            for j in range(len(data[i])):
                table.setItem(i, j, QTableWidgetItem(str(data[i][j])))
            #print(data[i][-2])
        if self.mode=='pdos':
            self.table_1 = QTableWidget(table.rowCount(),table.columnCount())
            self.table_2 = QTableWidget(table.rowCount(),table.columnCount())
            self.table_3 = QTableWidget(table.rowCount(),table.columnCount())
            for ii in range(table.rowCount()):
                for jj in range(table.columnCount()):
                    self.table_1.setItem(ii, jj, QTableWidgetItem(table.item(ii,jj)))
                    self.table_2.setItem(ii, jj, QTableWidgetItem(table.item(ii,jj)))
                    self.table_3.setItem(ii, jj, QTableWidgetItem(table.item(ii,jj)))
            self.table_1.setGeometry(table.geometry())
            self.table_2.setGeometry(table.geometry())
            self.table_3.setGeometry(table.geometry())
            layout.addWidget(self.table_1,layout.indexOf(table))
            layout.addWidget(self.table_2,layout.indexOf(table))
            layout.addWidget(self.table_3,layout.indexOf(table))
            self.table_1.hide()
            self.table_2.hide()
            self.table_3.hide()

        for i in range(table.rowCount()):
            self.checkBox_1=locals()
            self.checkBox_2=locals()
            self.hbox_pdos_1=locals()
            self.hbox_pdos_2=locals()
            self.hbox_pdos_layer_2=locals()
            self.widget_pdos_1=locals()
            self.widget_pdos_2=locals()
            self.elements_1=locals()
            self.elements_2=locals()
            #continue calculation
            if (data[i][-2].find("out of timing")!=-1)|(data[i][-2].find("ERROR")!=-1)|(data[i][-2].find("not converged")!=-1):
                button = QPushButton("续算")
                button.clicked.connect(lambda state,x=i:self.button_click_error(button,table,button_clicked,x))
                button_check_error = QPushButton("查看错误")
                button_check_error.clicked.connect(lambda state,x=i:self.button_click_check_error(button_check_error,table,self.mode,x))
                button_force = QPushButton("应力")
                button_Energy = QPushButton("能量")
                button_force.clicked.connect(lambda state,x=i:self.button_click_curve(button_force,table,"force",x,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r))
                button_Energy.clicked.connect(lambda state,x=i:self.button_click_curve(button_Energy,table,"Energy",x,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r))

                layout_Button_relax = QHBoxLayout()
                layout_Button_relax.addWidget(button_force)
                layout_Button_relax.addWidget(button_Energy)
                widget_Button_relax = QWidget()
                widget_Button_relax.setLayout(layout_Button_relax)

                layout_Button_err = QHBoxLayout()
                layout_Button_err.addWidget(button)
                layout_Button_err.addWidget(button_check_error)
                widget_Button_err = QWidget()
                widget_Button_err.setLayout(layout_Button_err)
                table.setCellWidget(i, table.columnCount()-1, widget_Button_relax)
                table.setCellWidget(i, table.columnCount()-2, widget_Button_err)
                table.resizeRowsToContents()
                table.resizeColumnsToContents()
            #Initial Calculation
            if (data[i][-2].find("No Calculation")!=-1):
                button = QPushButton("计算")
                button.clicked.connect(lambda state,x=i:self.button_click_init(button,table,button_clicked,x))
                button_pdos1 = QPushButton("计算")
                button_pdos1.clicked.connect(lambda state,x=i:self.button_click_init(button_pdos1,table,button_clicked,x))
                button_pdos2 = QPushButton("计算")
                button_pdos2.clicked.connect(lambda state,x=i:self.button_click_init(button_pdos2,table,button_clicked,x))
                table.setCellWidget(i, table.columnCount()-1, button)
                table.resizeRowsToContents()
                table.resizeColumnsToContents()
                if self.mode=='pdos':
                    self.table_1.setCellWidget(i, self.table_1.columnCount()-1, button_pdos1)
                    self.table_2.setCellWidget(i, self.table_2.columnCount()-1, button_pdos2)
                    self.table_1.resizeRowsToContents()
                    self.table_2.resizeRowsToContents()
                    self.table_1.resizeColumnsToContents()
                    self.table_2.resizeColumnsToContents()
            #relax process check
            if (self.mode=="relax")&((data[i][-2].find("DONE")!=-1)|(data[i][-2].find("running")!=-1)):
                button_force = QPushButton("应力")
                button_Energy = QPushButton("能量")
                button_force.clicked.connect(lambda state,x=i:self.button_click_curve(button_force,table,"force",x,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r))
                button_Energy.clicked.connect(lambda state,x=i:self.button_click_curve(button_Energy,table,"Energy",x,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r))
                layout_Button_relax = QHBoxLayout()
                layout_Button_relax.addWidget(button_force)
                layout_Button_relax.addWidget(button_Energy)
                widget_Button_relax = QWidget()
                widget_Button_relax.setLayout(layout_Button_relax)
                table.setCellWidget(i, table.columnCount()-1,widget_Button_relax)
                table.resizeRowsToContents()
                table.resizeColumnsToContents()
            #scf process check
            if (self.mode=="scf")&((data[i][-2].find("DONE")!=-1)|(data[i][-2].find("running")!=-1)):
                button_E_er = QPushButton("E误差")
                button_E_er.clicked.connect(lambda state,x=i:self.button_click_curve(button_E_er,table,"scf-estima",x,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r))
                table.setCellWidget(i, table.columnCount()-1, button_E_er)
                table.resizeRowsToContents()
                table.resizeColumnsToContents()
            #bands draw
            if (self.mode=="bands")&(data[i][-2].find("DONE")!=-1):
                button = QPushButton("计算能带")
                #button = QPushButton("画能带")
                #button.clicked.connect(lambda state,x=i:self.button_click_bands(button,table,button_clicked,x,lineEdit_y_l,lineEdit_y_r,show_y_l_bands,show_y_r_bands))
                button.clicked.connect(lambda state,x=i:self.button_click_bands(button,table,button_clicked,x,lineEdit_y_l,lineEdit_y_r))
                table.setCellWidget(i, table.columnCount()-1, button)
                table.resizeRowsToContents()
                table.resizeColumnsToContents()
            #badercharge show
            if (self.mode=="BaderCharge")&(data[i][-2].find("DONE")!=-1):
                infile=table.item(i, 0).text().strip()
                #judge spin mode
                file_input_tag=[]
                for root,dirs,files in os.walk(Current_dir+"/"+infile):
                    for file in files:
                        if "in_scf" in file or "in_relax" in file:
                            file_input_tag.append(file)
                if len(file_input_tag)==0:
                    raise ValueError("No input found!The input shall be named with \'in_scf\' or '\in_relax\'!")
                spin_mode=0
                input_tag=open(infile+"/"+file_input_tag[0]).readlines()
                for lines in input_tag:
                    if "nspin" in lines and "2" in lines and "!" not in lines:
                        spin_mode=1

                if (os.path.isfile(f"data.save/bader_charge_of_{infile}.data")==1)&(((os.path.isfile(f"data.save/Mag_of_{infile}.txt")==1)&(spin_mode==1))|(spin_mode==0)):
                    button_bader = QPushButton("计算电荷")
                    button_analyze = QPushButton("分析数据")
                    button_bader.clicked.connect(lambda state,x=i:self.button_click_badercharge(button_bader,table,button_clicked,x))
                    button_analyze.clicked.connect(lambda state,x=i:self.button_click_analyze(button_analyze,table,button_clicked,x))
                    layout_Button_bader = QHBoxLayout()
                    layout_Button_bader.addWidget(button_bader)
                    layout_Button_bader.addWidget(button_analyze)
                    widget_Button_bader = QWidget()
                    widget_Button_bader.setLayout(layout_Button_bader)
                    table.setCellWidget(i, table.columnCount()-1,widget_Button_bader)
                else:
                    button_bader = QPushButton("计算电荷")
                    button_bader.clicked.connect(lambda state,x=i:self.button_click_badercharge(button_bader,table,button_clicked,x))
                    table.setCellWidget(i, table.columnCount()-1, button_bader)
                table.resizeRowsToContents()
                table.resizeColumnsToContents()
            #pdos draw
            if (self.mode=="pdos")&(data[i][-2].find("DONE")!=-1):
                #reading elemental species from in_pdos
                start_reading=0
                ele_spe_li=[]
                self.elements_1["el_a"+str(i)]=["total"]
                self.elements_2["el_b"+str(i)]=[["total"]]
                str_name=table.item(i, 0).text().strip()
                if os.path.isfile("%s/in_scf_%s"%(str_name,str_name))==1:
                    f_read_ele=open("%s/in_scf_%s"%(str_name,str_name)).readlines()
                elif os.path.isfile("%s/in_relax_%s"%(str_name,str_name))==1:
                    f_read_ele=open("%s/in_relax_%s"%(str_name,str_name)).readlines()
                
                for read_ele_lines in f_read_ele:
                    if read_ele_lines.find("ATOMIC_SPECIES")!=-1:
                        start_reading=1
                        continue
                    elif read_ele_lines.find("K_POINTS")!=-1:
                        start_reading=0
                    if start_reading==1:
                        ele_spe_new=[x for x in read_ele_lines.split() if len(x)>0][0]
                        if ele_spe_new not in ele_spe_li:
                            ele_spe_li.append(ele_spe_new)
                for ele_spe in ele_spe_li:
                    ele_2_buffer=[]
                    self.elements_1["el_a"+str(i)].append(ele_spe)
                    ele_2_buffer.append(ele_spe)
                    ele_2_buffer.extend(atomic_orbit(ele_spe,str_name))
                    self.elements_2["el_b"+str(i)].append(ele_2_buffer)
                    #self.elements_2["el_b"+str(i)].append(ele_spe)
                    #self.elements_2["el_b"+str(i)].extend(atomic_orbit(ele_spe,str_name))

                self.hbox_pdos_1["hbx_a"+str(i)] = QHBoxLayout()
                self.hbox_pdos_1["hbx_a"+str(i)].addStretch()
                self.button_1 = QPushButton("计算PDOS")
                table.setCellWidget(i, table.columnCount()-1, self.button_1)
                for ele in range(len(self.elements_1["el_a"+str(i)])):
                    self.checkBox_1["chbx_a"+str(i)]=QCheckBox(self.elements_1["el_a"+str(i)][ele])
                    self.hbox_pdos_1["hbx_a"+str(i)].addWidget(self.checkBox_1["chbx_a"+str(i)])
                self.widget_pdos_1["wg_a"+str(i)] = QWidget()
                self.widget_pdos_1["wg_a"+str(i)].setLayout(self.hbox_pdos_1["hbx_a"+str(i)])
                
                created_elements = set() 
                self.button_2 = QPushButton("计算PDOS")
                #for ele_symbol in [x.split("-")[0] for x in self.elements_2["el_b"+str(i)]]:
                #    if ele_symbol not in created_elements:
                #        checkbox = QCheckBox(ele_symbol)
                #        self.hbox_pdos_2["hbx_b"+str(i)].addWidget(checkbox)
                #        created_elements.add(ele_symbol)
                #self.widget_pdos_2["wg_b"+str(i)] = QWidget()
                #self.widget_pdos_2["wg_b"+str(i)].setLayout(self.hbox_pdos_2["hbx_b"+str(i)])

                #self.widget_pdos_2["wg_b"+str(i)] = QWidget()
                #self.hbox_pdos_layer_2["hbx_lo_b"+str(i)] = QVBoxLayout()
                #for ele in range(len(self.elements_2["el_b"+str(i)])):
                #    self.hbox_pdos_2["hbx_b"+str(i)+"-"+str(ele)] = QHBoxLayout()
                #    self.hbox_pdos_2["hbx_b"+str(i)+"-"+str(ele)].addStretch()
                #    for ele_2 in range(len(self.elements_2["el_b"+str(i)][ele])):
                #        self.checkBox_2["chbx_b"+str(i)+"-"+str(ele)+"-"+str(ele_2)]=QCheckBox(self.elements_2["el_b"+str(i)][ele][ele_2])
                #        self.hbox_pdos_2["hbx_b"+str(i)+"-"+str(ele)].addWidget(self.checkBox_2["chbx_b"+str(i)+"-"+str(ele)+"-"+str(ele_2)])
                #    self.hbox_pdos_layer_2["hbx_lo_b"+str(i)].addLayout(self.hbox_pdos_2["hbx_b"+str(i)+"-"+str(ele)])
                #self.widget_pdos_2["wg_b"+str(i)].setLayout(self.hbox_pdos_layer_2["hbx_lo_b"+str(i)])
                
                self.widget_pdos_2["wg_b"+str(i)] = QWidget()
                self.hbox_pdos_layer_2["hbx_lo_b"+str(i)] = QVBoxLayout()
                for ele in range(len(self.elements_2["el_b"+str(i)])):
                    self.hbox_pdos_2["hbx_b"+str(i)+"-"+str(ele)] = QHBoxLayout()
                    self.hbox_pdos_2["hbx_b"+str(i)+"-"+str(ele)].addStretch()
                    for ele_2 in range(len(self.elements_2["el_b"+str(i)][ele])):
                        self.checkBox_2["chbx_b"+str(i)+"-"+str(ele)+"-"+str(ele_2)]=QCheckBox(self.elements_2["el_b"+str(i)][ele][ele_2])
                        self.hbox_pdos_2["hbx_b"+str(i)+"-"+str(ele)].addWidget(self.checkBox_2["chbx_b"+str(i)+"-"+str(ele)+"-"+str(ele_2)])
                        #self.hbox_pdos_2["hbx_b"+str(i)+"-"+str(ele)].setAlignment(Qt.AlignLeft)
                        self.hbox_pdos_2["hbx_b"+str(i)+"-"+str(ele)].addStretch()
                    self.hbox_pdos_layer_2["hbx_lo_b"+str(i)].addLayout(self.hbox_pdos_2["hbx_b"+str(i)+"-"+str(ele)])
                    #self.hbox_pdos_layer_2["hbx_lo_b"+str(i)].setAlignment(Qt.AlignLeft)
                self.widget_pdos_2["wg_b"+str(i)].setLayout(self.hbox_pdos_layer_2["hbx_lo_b"+str(i)])

                self.table_2.setCellWidget(i, table.columnCount()-2, self.widget_pdos_2["wg_b"+str(i)])
                self.button_2.clicked.connect(lambda state,x=i:self.button_click_pdos_detail(self.button_2,self.table_2,button_clicked,x,self.elements_2["el_b"+str(x)],self.hbox_pdos_2,self.checkBox_2,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara))
                #self.button_2.clicked.connect(lambda state,x=i:self.button_click_pdos(self.button_2,self.table_2,button_clicked,x,self.elements_2["el_b"+str(x)],self.hbox_pdos_2["hbx_b"+str(x)],self.checkBox_2["chbx_b"+str(x)],lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara))
                self.table_2.setCellWidget(i, table.columnCount()-1, self.button_2)
                self.table_2.resizeRowsToContents()
                self.table_2.resizeColumnsToContents()

                self.table_1.setCellWidget(i, table.columnCount()-2, self.widget_pdos_1["wg_a"+str(i)])
                self.button_1.clicked.connect(lambda state,x=i:self.button_click_pdos(self.button_1,self.table_1,button_clicked,x,self.elements_1["el_a"+str(x)],self.hbox_pdos_1["hbx_a"+str(x)],self.checkBox_1["chbx_a"+str(x)],lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara))
                self.table_1.setCellWidget(i, table.columnCount()-1, self.button_1)
                #uncommand the followings when use PyQy5.15.9 or higher version, but for 5.15.2 its the best
                self.table_1.resizeColumnsToContents()
                self.table_1.resizeRowsToContents()
                 
                self.button_3 = QPushButton("计算PDOS")
                self.table_3.setItem(i, table.columnCount()-2, QTableWidgetItem(show_pdos_detail(table.item(i, 0).text().strip())))
                self.button_3.clicked.connect(lambda state,x=i:self.button_click_pdos_diy(self.button_3,self.table_3,button_clicked,x,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara,hidden_edit,hidden_label))
                self.table_3.setCellWidget(i, table.columnCount()-1, self.button_3)
                self.table_3.resizeRowsToContents()
                self.table_3.resizeColumnsToContents()

                table.hide()
                self.table_1.show()
                self.pdos_combo_box.activated.connect(lambda state,x=i:self.pdos_on_active(table,button_clicked,x,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara,hidden_edit,hidden_label))
                
    def pdos_on_active(self,table,button_clicked,i,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara,hidden_edit,hidden_label):
        # Get the selected item
        self.pdos_mode = self.pdos_combo_box.currentText()
        # Print the selected item
        print ("Pdos mode:",self.pdos_mode)
        if self.pdos_mode=="simple mode":
            self.table_1.show()
            self.table_2.hide()
            self.table_3.hide()
            hidden_edit.hide()
            hidden_label.hide()
            #self.button_2.hide()
            #self.button_1.show()
            #self.widget_pdos_2["wg_b"+str(i)].hide()
            #self.widget_pdos_1["wg_a"+str(i)].show()
            #table.setCellWidget(i, table.columnCount()-2, self.widget_pdos_1["wg_a"+str(i)])
            #self.button_1.clicked.connect(lambda state,x=i:self.button_click_pdos(self.button_1,table,button_clicked,x,self.elements_1["el_a"+str(x)],self.hbox_pdos_1["hbx_a"+str(x)],self.checkBox_1["chbx_a"+str(x)],lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara))
            #table.setCellWidget(i, table.columnCount()-1, self.button_1)

        elif self.pdos_mode=="detailed orbital mode":
            self.table_1.hide()
            self.table_2.show()
            self.table_3.hide()
            hidden_edit.hide()
            hidden_label.hide()
            #self.button_1.hide()
            #self.button_2.show()
            #self.widget_pdos_1["wg_a"+str(i)].hide()
            #self.widget_pdos_2["wg_b"+str(i)].show()
            #table.setCellWidget(i, table.columnCount()-2, self.widget_pdos_2["wg_b"+str(i)])
            #self.button_2.clicked.connect(lambda state,x=i:self.button_click_pdos(self.button_2,table,button_clicked,x,self.elements_2["el_b"+str(x)],self.hbox_pdos_2["hbx_b"+str(x)],self.checkBox_2["chbx_b"+str(x)],lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara))
            #table.setCellWidget(i, table.columnCount()-1, self.button_2)

        elif self.pdos_mode=="manual mode":
            self.table_1.hide()
            self.table_2.hide()
            self.table_3.show()
            hidden_edit.show()
            hidden_label.show()


    def button_click_check_error(self,button_check_error,table,mode,x):
        str_name=table.item(x, 0).text().strip()
        #button= self.sender()#To change the button's name when it is clicked, you can use the sender() method to get the button that was clicked and then use the setText() method to change its name
        #button.setText("已续算")
        print("%s/out_%s_%s"%(str_name,mode,str_name))
        self.showtext=TextShow("%s/out_%s_%s"%(str_name,mode,str_name))
        self.showtext.show()
    def button_click_error(self,button,table,button_clicked,x):
        #nonlocal button_clicked # use nonlocal to access the list of boolean flags
        if button_clicked[x]==False:
            os.system("python %s/continue_cal.py %s %s"%(code_path,table.item(x, 0).text(),self.mode)) 
            print(table.item(x, 0).text())
            button= self.sender()#To change the button's name when it is clicked, you can use the sender() method to get the button that was clicked and then use the setText() method to change its name
            #button.setText("已续算")
            button.hide()
            button_clicked[x] = True
            #table.setItem(x, table.columnCount()-1, QTableWidgetItem("已续算")) # set text of corresponding cell

    def button_click_init(self,button,table,button_clicked,x):
        #nonlocal button_clicked # use nonlocal to access the list of boolean flags
        if button_clicked[x]==False:
            os.system("python %s/replace.py %s.vasp %s 0 0"%(code_path,table.item(x, 0).text(),self.mode)) 
            print(table.item(x, 0).text())
            button= self.sender()#To change the button's name when it is clicked, you can use the sender() method to get the button that was clicked and then use the setText() method to change its name
            button.hide()
            button_clicked[x] = True
            table.setItem(x, table.columnCount()-1, QTableWidgetItem("已计算")) # set text of corresponding cell

    #def button_click_bands(self,button,table,button_clicked,x,lineEdit_y_l,lineEdit_y_r,show_y_l_bands,show_y_r_bands):
    def button_click_bands(self,button,table,button_clicked,x,lineEdit_y_l,lineEdit_y_r):
        global show_y_l_bands
        global show_y_r_bands
        str_name=table.item(x, 0).text().strip()
        if (str(show_y_l_bands)!=lineEdit_y_l.text().strip())|(str(show_y_r_bands)!=lineEdit_y_r.text().strip()):
            for i in range(table.rowCount()):
                button_clicked[i]=False
            show_y_l_bands,show_y_r_bands=lineEdit_y_l.text().strip(),lineEdit_y_r.text().strip()
        if button_clicked[x]==False:
            vb_num=bands_read("%s/out_scf_"%str_name+str_name)
            print("python %s/titledbandstrv5.1.py out_bands_%s %d %s %s"%(code_path,str_name,vb_num,show_y_l_bands,show_y_r_bands))
            os.chdir(str_name)
            if (show_y_l_bands==0)|(show_y_r_bands==0):
                show_y_l_bands=4
                show_y_r_bands=4
                print("No data,plz input a range!")
            data_bands=os.popen("python %s/titledbandstrv5.1.py out_bands_%s %d %s %s"%(code_path,str_name,vb_num,show_y_l_bands,show_y_r_bands)).readlines()
            os.chdir(Current_dir)
            for j_i in data_bands:
                if j_i.find("Fermi energy =")!=-1:
                    Fermi_E=float(j_i.split("\n")[0].split("Fermi energy = ")[-1].strip())
                elif j_i.find("Band gap =")!=-1:
                    Band_gap=float(j_i.split("\n")[0].split("Band gap =")[-1].strip())
            button=self.sender()
            button.setText("查看能带")
            button_clicked[x] = True
            table.setItem(x, table.columnCount()-2, QTableWidgetItem("Band_Gap=%.2f Fermi_E=%.2f"%(Band_gap,Fermi_E)))
     
        if button_clicked[x]==True:
            self.bands_show = BandsShow(str_name)
            self.bands_show.show()
                                   
    def button_click_badercharge(self,button,table,button_clicked,x):
        str_name=table.item(x, 0).text().strip()
        if button_clicked[x]==False:
            original_charge=self.bader_process(str_name) 
            button=self.sender()
            button.setText("查看电荷")
            button_clicked[x] = True
            table.setItem(x, table.columnCount()-2, QTableWidgetItem(original_charge))
     
        if button_clicked[x]==True:
            self.bader_show = BaderShow(str_name)
            self.bader_show.show()

    def button_click_analyze(self,button,table,button_clicked,x):
        str_name=table.item(x, 0).text().strip()
        self.analyze_show = AnalyzeShow(str_name)
        self.analyze_show.show()

    def bader_process(self,str_name):
        file_input_tag=[]
        for root,dirs,files in os.walk(Current_dir+"/"+str_name):
            for file in files:
                if "in_scf" in file or "in_relax" in file:
                    file_input_tag.append(file)
        if len(file_input_tag)==0:
            raise ValueError("No input found!The input shall be named with \'in_scf\' or '\in_relax\'!")
        spin_mode=0
        input_tag=open(str_name+"/"+file_input_tag[0]).readlines()
        for lines in input_tag:
            if "nspin" in lines and "2" in lines and "!" not in lines:
                spin_mode=1

        infile=str_name
        if (os.path.isfile(f"data.save/bader_charge_of_{infile}.data")==1)&(os.path.isfile(f"data.save/Lowdin_of_{infile}.txt")==1):
            if ((spin_mode==1)&(os.path.isfile(f"data.save/Mag_of_{infile}.txt")==1))|(spin_mode==0):
                print(f"Quick read charge of {infile}")
                bader_file=f"{str_name}/Bader_{infile}"
                in_bader=open(bader_file).readlines()
                valence_charge=[]
                for i,v in enumerate(in_bader):
                    #print(v)
                    pattern=r'^\s+\d+\s+[A-Za-z0-9]+\s+\d+(?:\.\d*)?\s*\n?$'
                    if re.match(pattern,v):
                        valence_line=[x for x in v.strip("\n").split() if len(x)>0]
                        #print(valence_line)
                        valence_charge.append([valence_line[1],float(valence_line[2])])
                outshit=''
                for i in valence_charge:
                    outshit+=f"{i[0]}-{[i[1]]}\n"
                return outshit
        os.chdir(str_name)
        jj=os.popen(f"bader Bader_{str_name}.cube").readlines()
        os.chdir(Current_dir)


        magfile=f"data.save/Mag_of_{str_name}.txt"
        lowdin_file=f"data.save/Lowdin_of_{str_name}.txt"

        if os.path.isfile(f"{str_name}/in_scf_{infile}"):
            print("scf done in {str_name}")
            atomic_file=f"{str_name}/in_scf_{infile}"
            if not os.path.isfile(magfile) and (spin_mode==1):
                read_mag=os.popen(f"python {code_path}/read_mag.py {str_name} scf").readlines()
                os.system(f"mv Mag_of_{str_name}.txt data.save")
            if not os.path.isfile(lowdin_file):
                if os.path.isfile(f"{str_name}/out_pdos_{str_name}"):
                    read_mag=os.popen(f"python {code_path}/read_lowdin.py {str_name} scf").readlines()
                    os.system(f"mv Lowdin_of_{str_name}.txt data.save")
                else:
                    raise ValueError("PDOS has not been calculated in {str_name}!")
        elif os.path.isfile(f"{str_name}/in_relax_{infile}"):
            print("relax done in {str_name}")
            atomic_file=f"{str_name}/in_relax_{infile}"
            if not os.path.isfile(magfile) and (spin_mode==1):
                read_mag=os.popen(f"python {code_path}/read_mag.py {str_name} relax").readlines()
                os.system(f"mv Mag_of_{str_name}.txt data.save")
            if not os.path.isfile(lowdin_file):
                if os.path.isfile(f"{str_name}/out_pdos_{str_name}"):
                    read_mag=os.popen(f"python {code_path}/read_lowdin.py {str_name} relax").readlines()
                    os.system(f"mv Lowdin_of_{str_name}.txt data.save")
                else:
                    raise ValueError("PDOS has not been calculated in {str_name}!")
        else:
            raise ValueError("No scf or relaxation found!")

        bader_file=f"{str_name}/Bader_{infile}"
        acf=f"{str_name}/ACF.dat"
        if spin_mode==1:
            in_mag=open(magfile).readlines()     #Magnetic file get by read_mag.py
        in_lowdin=open(lowdin_file).readlines()
        in_spe=open(atomic_file).readlines()    
        in_charge=open(acf).readlines()      #bader charge file get by bader 
        in_bader=open(bader_file).readlines() #bader charge file get by pp.x calculation
        
        atomic_number=0
        element_number=0
        for i,v in enumerate(in_spe):
            if 'nat' in v and '!' not in v:
                atomic_number=int(v.split("=")[-1].strip("\n").strip().strip(",").strip())
            if 'ntyp' in v and '!' not in v:
                element_number=int(v.split("=")[-1].strip("\n").strip().strip(",").strip())
        if (atomic_number==0)|(element_number==0):
            raise ValueError(f"No \'nat\' or \'ntyp\' found in {atomic_file}")
        elements=[]             #read elements from in_scf/in_relax it is elements related to EACH atomic number
        start_read_in_spe=0
        read_count_in_spe=0
        for i,v in enumerate(in_spe):
            if "ATOMIC_POSITIONS" in v:
                start_read_in_spe=1
                continue
            if (start_read_in_spe==1)&(read_count_in_spe<atomic_number):
                elements.append([x for x in v.split() if len(x)>0][0])

        element_type=[]             #read element_type from in_scf/in_relax it is element with no duplicates
        start_read_ele=0
        read_count_ele=0
        for i,v in enumerate(in_spe):
            if "ATOMIC_SPECIES" in v:
                start_read_ele=1
                continue
            if (start_read_ele==1)&(read_count_ele<element_number):
                element_type.append([x for x in v.split() if len(x)>0][0])

        element_type_number=[]
        for i in elements:
            if i in [x[0] for x in element_type_number]:
                element_type_number[[x[0] for x in element_type_number].index(i)][1]+=1
            else:
                element_type_number.append([i,1])        

        start_read_in_bader=0
        bader_charge_lst=[]   #bader charge of each atomic number
        for i,v in enumerate(in_charge):
            if "#" in v:
                continue
            if (start_read_in_bader==0)&(v.find("-------------")!=-1):
                start_read_in_bader=1
                continue
            if (start_read_in_bader==1)&(v.find("-------------")!=-1):
                start_read_in_bader=0
                continue
            if  start_read_in_bader==1:
                bader_charge_lst.append(float([x for x in v.strip("\n").split() if len(x)>0][4]))
        
        valence_charge=[]  #total charge read from Bader_{str_name} which gets from pesuedo potentials
        for i,v in enumerate(in_bader):
            #print(v)
            pattern=r'^\s+\d+\s+[A-Za-z0-9]+\s+\d+(?:\.\d*)?\s*\n?$'
            if re.match(pattern,v):
                valence_line=[x for x in v.strip("\n").split() if len(x)>0]
                #print(valence_line)
                valence_charge.append([valence_line[1],float(valence_line[2])])
        for i,v in enumerate(valence_charge):  #replace the element incase the spin up and spin down is not required by Bader_{infile} file
            v[0]=element_type[i]        

        total_charge=0
        for i in valence_charge:
            print(i[0],i[1],element_type_number[[x[0] for x in element_type_number].index(i[0])][0],element_type_number[[x[0] for x in element_type_number].index(i[0])][1])
            total_charge+=i[1]*element_type_number[[x[0] for x in element_type_number].index(i[0])][1]
        print("Calculated total charge:",total_charge)

        lowdin_charge=[]     #lowdin charge read from out_pdos
        for i,v in enumerate(in_lowdin):
            if "#" in v:
                continue
            else:
                lowdin_line=[x for x in v.strip("\n").split("\t") if len(x)>0]
                lowdin_charge.append([float(lowdin_line[1])])

        lowdin_sum=0
        for i in lowdin_charge:
            lowdin_sum+=i[0]

        for i in lowdin_charge:
            i[0]=float("%.6f"%(i[0]*total_charge/lowdin_sum))

        if spin_mode==1:
            magnetics=[]     #magnetics read from out_scf/out_relax
            for i,v in enumerate(in_mag):
                mag_line=[x for x in v.strip("\n").split(" ") if len(x)>0]
                magnetics.append([float(mag_line[-1])])

        ionic_charge=[]      #calculated by valence_charge-bader/lowdin charge
        #print(valence_charge,lowdin_charge)
        for i,v in enumerate(elements):
            if v in [x[0] for x in valence_charge]:
                ionic_charge.append(["%.6f"%(valence_charge[[x[0] for x in valence_charge].index(v)][1]-bader_charge_lst[i]),
                                     "%.6f"%(valence_charge[[x[0] for x in valence_charge].index(v)][1]-lowdin_charge[i][0]) ] )
        
        
        out_filenm=f"bader_charge_of_{infile}.data"
        f_out=open(out_filenm,"w")
        for i in range(len(ionic_charge)):
            if spin_mode==1:
                f_out.writelines(f"{elements[i]}-{i+1}    {lowdin_charge[i][0]}    {bader_charge_lst[i]}    {ionic_charge[i][1]}    {ionic_charge[i][0]}    {magnetics[i][0]}\n")           
            elif spin_mode==0:
                f_out.writelines(f"{elements[i]}-{i+1}    {lowdin_charge[i][0]}    {bader_charge_lst[i]}    {ionic_charge[i][1]}    {ionic_charge[i][0]}\n")           
            #print(f"{elements[i]}    {bader_charge_lst[i]}    {ionic_charge[i]}")           
        f_out.close()
        os.system(f"mv {out_filenm} data.save")
        outshit=''
        for i in valence_charge:
            outshit+=f"{i[0]}-{[i[1]]}\n"
        return outshit
      
    def get_selected_element_detail(self,elements,hbox_pdos,i,checkBox):
        selectedChoices = []
        for ele in range(len(self.elements_2["el_b"+str(i)])):
            for ele_2 in range(len(self.elements_2["el_b"+str(i)][ele])):
                checkBox = self.checkBox_2["chbx_b"+str(i)+"-"+str(ele)+"-"+str(ele_2)]
                if (checkBox is not None and checkBox.isChecked()):
                    selectedChoices.append(self.elements_2["el_b"+str(i)][ele][ele_2])
        #element_i=[]
        #hbox_pdos_i=[]
        #for ele in range(len(elements)):
        #    element_i.extend(elements[ele])
        #    hbox_pdos_i.extend(hbox_pdos[ele])
        #print("total",elements)
        #print(checkBox)
        #for ele in range(len(elements_i)):
        #    checkBox = hbox_pdos_i.itemAt(ele+1).widget()
        #    if (checkBox is not None and checkBox.isChecked()):
        #        selectedChoices.append(elements_i[ele])
        #print("sele",selectedChoices)
        return selectedChoices

    def button_click_pdos_detail(self,button,table,button_clicked,x,elements,hbox_pdos,checkBox,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara):
        str_name=table.item(x, 0).text().strip()
        #print("in button clicked",elements)
        if button_clicked[x]==False:
            button=self.sender()
            button.setText("查看PDOS")
            button_clicked[x] = True
            self.pdos_show(str_name,self.get_selected_element_detail(elements,hbox_pdos,x,checkBox),lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara)
        elif button_clicked[x]==True:
            self.pdos_show(str_name,self.get_selected_element_detail(elements,hbox_pdos,x,checkBox),lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara) 

    def get_selected_element(self,elements,hbox_pdos,i,checkBox):
        selectedChoices = []
        #print("total",elements)
        #print(checkBox)
        for ele in range(len(elements)):
            checkBox = hbox_pdos.itemAt(ele+1).widget()
            if (checkBox is not None and checkBox.isChecked()):
                selectedChoices.append(elements[ele])
        #print("sele",selectedChoices)
        return selectedChoices

    def button_click_pdos(self,button,table,button_clicked,x,elements,hbox_pdos,checkBox,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara):
        str_name=table.item(x, 0).text().strip()
        #print("in button clicked",elements)
        if button_clicked[x]==False:
            button=self.sender()
            button.setText("查看PDOS")
            button_clicked[x] = True
            self.pdos_show(str_name,self.get_selected_element(elements,hbox_pdos,x,checkBox),lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara)
        elif button_clicked[x]==True:
            self.pdos_show(str_name,self.get_selected_element(elements,hbox_pdos,x,checkBox),lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara) 

    def button_click_pdos_diy(self,button,table,button_clicked,x,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara,hidden_edit,hidden_label):
        str_name=table.item(x, 0).text().strip()
        #print("in button clicked",elements)
        if button_clicked[x]==False:
            button=self.sender()
            button.setText("查看PDOS")
            button_clicked[x] = True
            self.pdos_show_diy(str_name,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara,hidden_edit,hidden_label)
        elif button_clicked[x]==True:
            self.pdos_show_diy(str_name,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara,hidden_edit,hidden_label) 

    def pdos_show(self,str_name,elements_sel,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara):

            file_input_tag=[]
            for root,dirs,files in os.walk(Current_dir+"/"+str_name):
                for file in files:
                    if "in_scf" in file or "in_relax" in file:
                        file_input_tag.append(file)
            if len(file_input_tag)==0:
                raise ValueError("No input found!The input shall be named with \'in_scf\' or '\in_relax\'!")
            spin_mode=0
            input_tag=open(str_name+"/"+file_input_tag[0]).readlines()
            for lines in input_tag:
                if "nspin" in lines and "2" in lines and "!" not in lines:
                    spin_mode=1   

            f_gnu=open("gnu_buffer_pdos","w")
            f_gnu.writelines("set xlabel \"E-Ef\"\n")
            f_gnu.writelines("set ylabel \"pDOS\"\n")
            if len(linepara.text())>0:
                line_width=linepara.text().split(";")[0]
                fill_fermi=linepara.text().split(";")[1]
                show_ef=linepara.text().split(";")[2]
                pattern_fs=linepara.text().split(";")[3]
            '''
            else:
                line_width='1'
                fill_fermi='0'
                show_ef='0'
                pattern_fs='1'
            '''
            #print(line_width,fill_fermi,show_ef,pattern_fs)
            if fill_fermi=='1':
                #print("FILL FERMI")
                f_gnu.writelines("set style fill pattern %s noborder\n"%pattern_fs)
            if (len(lineEdit_x_l.text())>0)&(input_type_judge(lineEdit_x_l.text())!="str"):
                pdos_x_l=lineEdit_x_l.text()
            else:
                pdos_x_l="*"
            if (len(lineEdit_x_r.text())>0)&(input_type_judge(lineEdit_x_r.text())!="str"):
                pdos_x_r=lineEdit_x_r.text()
            else:
                pdos_x_r="*"
            if (len(lineEdit_y_l.text())>0)&(input_type_judge(lineEdit_y_l.text())!="str"):
                pdos_y_l=lineEdit_y_l.text()
            else:
                pdos_y_l="*"
            if (len(lineEdit_y_r.text())>0)&(input_type_judge(lineEdit_y_r.text())!="str"):
                pdos_y_r=lineEdit_y_r.text()
            else:
                pdos_y_r="*"
            f_gnu.writelines("set xrange [%s:%s]\n"%(str(pdos_x_l),str(pdos_x_r)))
            f_gnu.writelines("set yrange [%s:%s]\n"%(str(pdos_y_l),str(pdos_y_r)))
            f_gnu.writelines("set title \"PDOS of %s\"\n"%str_name)
            if "total" in elements_sel:
                if os.path.isfile("%s/out_scf_%s"%(str_name,str_name))==1:
                    Fermi_E=float(os.popen("grep Fermi %s/out_scf_%s"%(str_name,str_name)).readlines()[-1].split("is")[-1].split("ev")[0].strip())
                elif os.path.isfile("%s/out_relax_%s"%(str_name,str_name))==1:
                    Fermi_E=float(os.popen("grep Fermi %s/out_relax_%s"%(str_name,str_name)).readlines()[-1].split("is")[-1].split("ev")[0].strip())
                print("Fermi Energy:",Fermi_E)
                out_pdos="\"%s/BTO.pdos_tot\" u ($1-%f):2 w l lw %s lc \"black\" t \"Total-up\","%(str_name,Fermi_E,line_width)
                if spin_mode==1:
                    out_pdos+="\"%s/BTO.pdos_tot\" u ($1-%f):(-1*$3) w l lw %s lc \"black\" t \"Total-down\","%(str_name,Fermi_E,line_width)
                if fill_fermi=='1':
                    if spin_mode==1:
                        out_pdos+="\"%s/BTO.pdos_tot\" u ($1-%f):(($1-%f)<0 ? $2 : 1/0):(($1-%f)<0 ? -1*$3 : 1/0) with filledcurves lc \'black\' notitle,"%(str_name,Fermi_E,Fermi_E,Fermi_E)
                    elif spin_mode==0:
                        out_pdos+="\"%s/BTO.pdos_tot\" u ($1-%f):(($1-%f)<0 ? $2 : 1/0) with filledcurves x1 lc \'black\' notitle,"%(str_name,Fermi_E,Fermi_E)
                elements_sel_ele=elements_sel.copy()
                elements_sel_ele.remove("total")
            elif "total" not in elements_sel:
                out_pdos=""
                elements_sel_ele=elements_sel.copy()
            #calculate the uncalculated files
            os.chdir(str_name)
            print("change dir into:",str_name)
            for ele in range(len(elements_sel_ele)):
                if (os.path.isfile("accumulated_pdos_file%s"%elements_sel_ele[ele])==0)&(elements_sel_ele[ele]!="total"):
                    print("calculating :%s"%elements_sel_ele[ele])
                    data_pdos=os.popen("python %s/sum_qe_pdos.py %s"%(code_path,elements_sel_ele[ele])).readlines()
            os.chdir(Current_dir)

            
            if show_ef=='1':
                #print("show eF!")
                out_pdos+="\"%s/BTO.pdos_tot\" u ($1-$1):($3) w l lt -1 lc 'black' lw %s t \"Fermi-level\","%(str_name,line_width)
                if spin_mode==1:
                    out_pdos+="\"%s/BTO.pdos_tot\" u ($1-$1):(-1*$3) w l lt -1 lc 'black' lw %s notitle,"%(str_name,line_width)
            for ele in range(len(elements_sel_ele)):
                #out_pdos+="\"%s/accumulated_pdos_file%s\" u 1:2 w l lc %d lw %s t \"%s-up\","%(str_name,elements_sel_ele[ele],ele+1,line_width,elements_sel_ele[ele])
                #out_pdos+="\"%s/accumulated_pdos_file%s\" u 1:(-1*$3) w l lc %d lw %s t \"%s-down\","%(str_name,elements_sel_ele[ele],ele+1,line_width,elements_sel_ele[ele])
                out_pdos+="\"%s/accumulated_pdos_file%s\" u 1:2 w l lc %d lw %s t \"%s\","%(str_name,elements_sel_ele[ele],ele+1,line_width,elements_sel_ele[ele])
                if spin_mode==1:
                    out_pdos+="\"%s/accumulated_pdos_file%s\" u 1:(-1*$3) w l lc %d lw %s notitle,"%(str_name,elements_sel_ele[ele],ele+1,line_width)
                if fill_fermi=='1':
                    if spin_mode==1:
                        out_pdos+="\"%s/accumulated_pdos_file%s\" u 1:($1<0 ? $2 : 1/0):($1<0 ? -1*$3 : 1/0) with filledcurves lc %d notitle,"%(str_name,elements_sel_ele[ele],ele+1)
                    elif spin_mode==0:
                        out_pdos+="\"%s/accumulated_pdos_file%s\" u 1:($1<0 ? $2 : 1/0) with filledcurves x1 lc %d notitle,"%(str_name,elements_sel_ele[ele],ele+1)
                    #out_pdos+="\"%s/accumulated_pdos_file%s\" u 1:($1<0 ? -1*$5 : 1/0) with filledcurves x1 lc %d notitle,"%(str_name,elements_sel_ele[ele],ele+1)
            f_gnu.writelines("p"+out_pdos+"\n")
            f_gnu.close()
            #time.sleep(1)
            subprocess.call(['gnuplot', '-e', 'l \"gnu_buffer_pdos\"','--persist'])        

    def pdos_show_diy(self,str_name,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara,hidden_edit,hidden_label):
            f_gnu=open("gnu_buffer_pdos","w")
            f_gnu.writelines("set xlabel \"E-Ef\"\n")
            f_gnu.writelines("set ylabel \"pDOS\"\n")
            if len(linepara.text())>0:
                line_width=linepara.text().split(";")[0]
                fill_fermi=linepara.text().split(";")[1]
                show_ef=linepara.text().split(";")[2]
                pattern_fs=linepara.text().split(";")[3]
            #print(line_width,fill_fermi,show_ef,pattern_fs)
            if fill_fermi=='1':
                #print("FILL FERMI")
                f_gnu.writelines("set style fill pattern %s noborder\n"%pattern_fs)
            if (len(lineEdit_x_l.text())>0)&(input_type_judge(lineEdit_x_l.text())!="str"):
                pdos_x_l=lineEdit_x_l.text()
            else:
                pdos_x_l="*"
            if (len(lineEdit_x_r.text())>0)&(input_type_judge(lineEdit_x_r.text())!="str"):
                pdos_x_r=lineEdit_x_r.text()
            else:
                pdos_x_r="*"
            if (len(lineEdit_y_l.text())>0)&(input_type_judge(lineEdit_y_l.text())!="str"):
                pdos_y_l=lineEdit_y_l.text()
            else:
                pdos_y_l="*"
            if (len(lineEdit_y_r.text())>0)&(input_type_judge(lineEdit_y_r.text())!="str"):
                pdos_y_r=lineEdit_y_r.text()
            else:
                pdos_y_r="*"
            f_gnu.writelines("set xrange [%s:%s]\n"%(str(pdos_x_l),str(pdos_x_r)))
            f_gnu.writelines("set yrange [%s:%s]\n"%(str(pdos_y_l),str(pdos_y_r)))
            f_gnu.writelines("set title \"PDOS of %s\"\n"%str_name)
                        
            spell_in=hidden_edit.text().strip()
            #calculate the uncalculated files
            os.chdir(str_name)
            print("change dir into:",str_name)
            data_pdos=os.popen("python %s/sum_qe_pdos_diy.py \'%s\'"%(code_path,spell_in)).readlines()
            #print(data_pdos)
            #if "ValueError" in data_pdos:
                #QMessageBox.critical(self, "错误", data_pdos)
            os.chdir(Current_dir)
            return_lst=[]
            for returns in data_pdos:
                if "###" in returns:
                    returns_seg=[x for x in returns.strip("\n").split(" ") if len(x) > 0]
                    return_lst.append([returns_seg[1],returns_seg[2],returns_seg[3],returns_seg[4]])

            file_input_tag=[]
            for root,dirs,files in os.walk(Current_dir+"/"+str_name):
                for file in files:
                    if "in_scf" in file or "in_relax" in file:
                        file_input_tag.append(file)
            if len(file_input_tag)==0:
                raise ValueError("No input found!The input shall be named with \'in_scf\' or '\in_relax\'!")
            spin_mode=0
            input_tag=open(str_name+"/"+file_input_tag[0]).readlines()
            for lines in input_tag:
                if "nspin" in lines and "2" in lines and "!" not in lines:
                    spin_mode=1   

            out_pdos=''
            if show_ef=='1':
                #print("show eF!")
                out_pdos+="\"%s/BTO.pdos_tot\" u ($1-$1):($3) w l lt -1 lc 'black' lw %s t \"Fermi-level\","%(str_name,line_width)
                if spin_mode==1:
                    out_pdos+="\"%s/BTO.pdos_tot\" u ($1-$1):(-1*$3) w l lt -1 lc 'black' lw %s notitle,"%(str_name,line_width)
            for segs in return_lst:
                #out_pdos+="\"%s/%s\" u 1:%d w l lc %d lw %s t \"%s-up\","%(str_name,segs[1],int(segs[2]),return_lst.index(segs),line_width,segs[3])
                #out_pdos+="\"%s/%s\" u 1:(-1*$%d) w l lc %d lw %s t \"%s-down\","%(str_name,segs[1],int(segs[2])+1,return_lst.index(segs),line_width,segs[3])
                out_pdos+="\"%s/%s\" u 1:%d w l lc %d lw %s t \"%s\","%(str_name,segs[1],int(segs[2]),return_lst.index(segs)+1,line_width,segs[3])
                if spin_mode==1:
                    out_pdos+="\"%s/%s\" u 1:(-1*$%d) w l lc %d lw %s notitle,"%(str_name,segs[1],int(segs[2])+1,return_lst.index(segs)+1,line_width)
                if fill_fermi=='1':
                    if spin_mode==1:
                        out_pdos+="\"%s/%s\" u 1:($1<0 ? $%d : 1/0):($1<0 ? -1*$%d : 1/0) with filledcurves lc %d notitle,"%(str_name,segs[1],int(segs[2]),int(segs[2])+1,return_lst.index(segs)+1)
                    elif spin_mode==0:
                        out_pdos+="\"%s/%s\" u 1:($1<0 ? $%d : 1/0) with filledcurves x1 lc %d notitle,"%(str_name,segs[1],int(segs[2]),return_lst.index(segs)+1)
            f_gnu.writelines("p"+out_pdos+"\n")
            f_gnu.close()
            #time.sleep(1)
            subprocess.call(['gnuplot', '-e', 'l \"gnu_buffer_pdos\"','--persist'])        

    def button_click_curve(self,button,table,key_words,x,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r):
        
        str_name=table.item(x, 0).text().strip()
        if key_words=="force":
            grep_words="Gradient error"
        elif key_words=="Energy":
            grep_words="!"
        elif key_words=="scf-estima":
            grep_words="estimated"
        print("grep \"%s\" %s/out_%s_%s"%(grep_words,str_name,self.mode,str_name))
        data_grep=os.popen("grep \"%s\" %s/out_%s_%s"%(grep_words,str_name,self.mode,str_name)).readlines()
        data_printout=[]
        for lines in data_grep:
            if self.mode=="relax":
                data_printout.append([jj for jj in lines.split("=")[-1].split() if len(jj)>0])
            if self.mode=="scf":
                data_printout.append([jj for jj in lines.split("<")[-1].split() if len(jj)>0])
        if len(data_grep)>0:
            if os.path.exists("data.save")==0:
                os.system("mkdir data.save")
            f_data=open("data.save/%s_%s_%s"%(str_name,self.mode,key_words),"w")
            for i in range(len(data_printout)):
                f_data.writelines("%s\n"%data_printout[i][0])
            f_data.close()
            f_gnu=open("gnu_buffer_curves","w")
            if (len(lineEdit_x_l.text())>0)&(input_type_judge(lineEdit_x_l.text())!="str"):
                pdos_x_l=lineEdit_x_l.text()
            else:
                pdos_x_l="*"
            if (len(lineEdit_x_r.text())>0)&(input_type_judge(lineEdit_x_r.text())!="str"):
                pdos_x_r=lineEdit_x_r.text()
            else:
                pdos_x_r="*"
            if (len(lineEdit_y_l.text())>0)&(input_type_judge(lineEdit_y_l.text())!="str"):
                pdos_y_l=lineEdit_y_l.text()
            else:
                pdos_y_l="*"
            if (len(lineEdit_y_r.text())>0)&(input_type_judge(lineEdit_y_r.text())!="str"):
                pdos_y_r=lineEdit_y_r.text()
            else:
                pdos_y_r="*"
            f_gnu.writelines("set xrange [%s:%s]\n"%(str(pdos_x_l),str(pdos_x_r)))
            f_gnu.writelines("set yrange [%s:%s]\n"%(str(pdos_y_l),str(pdos_y_r)))
            f_gnu.writelines("set title \"%s of %s\"\n"%(key_words,str_name))
            f_gnu.writelines("set xlabel \"Time step\"\n")
            f_gnu.writelines("set ylabel \"%s\(%s\)\"\n"%(key_words,data_printout[0][1]))
            f_gnu.writelines("p \"data.save/%s_%s_%s\" u 1 w l t \"%s\"\n"%(str_name,self.mode,key_words,key_words))
            f_gnu.close()
            #time.sleep(1)
            #proc = subprocess.Popen(['gnuplot','-p'], 
            #                 shell=True,
            #                 stdin=subprocess.PIPE,
            #                 )
            #proc.stdin.write('load \'gnu_buffer\''.encode())
            #plot_gnu = subprocess.Popen(['gnuplot'], stdin=subprocess.PIPE)      
            #plot_gnu.communicate("l \"gnu_buffer\"".encode())
            #subprocess.call(['gnuplot', '-e', 'set terminal x11; l \"gnu_buffer\"','--persist'])        
            subprocess.call(['gnuplot', '-e', 'l \"gnu_buffer_curves\"','--persist'])        

class TextShow(QMainWindow):
    def __init__(self, filename):
        super().__init__()
         
        self.initUI(filename)

    def initUI(self, filename):
        textEdit = QTextEdit()
        with open(filename) as f:
            text = f.read()
            textEdit.setText(text)

        self.setCentralWidget(textEdit)
        self.show()

bader_windows=[]
class BaderShow(QMainWindow):
    def __init__(self, str_name):
        super().__init__() 
        self.resize(800, 1000)
        self.title = f"BaderCharge of {str_name}"
        self.setWindowTitle(self.title)
        self.filename=f"data.save/bader_charge_of_{str_name}.data"
        #judge spin mode
        file_input_tag=[]
        for root,dirs,files in os.walk(Current_dir+"/"+str_name):
            for file in files:
                if "in_scf" in file or "in_relax" in file:
                    file_input_tag.append(file)
        if len(file_input_tag)==0:
            raise ValueError("No input found!The input shall be named with \'in_scf\' or '\in_relax\'!")
        spin_mode=0
        input_tag=open(str_name+"/"+file_input_tag[0]).readlines()
        for lines in input_tag:
            if "nspin" in lines and "2" in lines and "!" not in lines:
                spin_mode=1

        self.spin_mode=spin_mode
        self.initUI()
        bader_windows.append(self)

    def initUI(self):

        table_bader = QTableWidget()
        self.setCentralWidget(table_bader)
        #layout.addWidget(table)
        #self.setLayout(layout)
        with open(self.filename, 'r') as file:
            reader = csv.reader(file, delimiter=' ')
            data_raw = list(reader)
        data_charge = [[value for value in row if value] for row in data_raw]
        if self.spin_mode==1:
            data=[["ATOM","Lowdin Charge(Ionic)","Bader Charge(Ionic)","Lowdin Charge Change","Bader Charge Change","Magnetization"]]
            for i,v in enumerate(data_charge):
                data.append([v[0],v[1],v[2],v[3],v[4],v[5]])
        elif self.spin_mode==0:
            data=[["ATOM","Lowdin Charge(Ionic)","Bader Charge(Ionic)","Lowdin Charge Change","Bader Charge Change"]]
            for i,v in enumerate(data_charge):
                data.append([v[0],v[1],v[2],v[3],v[4]])

        table_bader.setRowCount(len(data))
        max_column=max([len(i) for i in data])
        table_bader.setColumnCount(max_column+1)
        for i in range(len(data)):
            for j in range(len(data[i])):
                table_bader.setItem(i, j, QTableWidgetItem(str(data[i][j])))
        table_bader.resizeColumnsToContents()
        self.show()


analyze_windows=[]
class AnalyzeShow(QMainWindow):
    def __init__(self, str_name):
        super().__init__() 
        self.str_name=str_name
        #self.setGeometry(100, 100, 300, 400)
        #self.resize(300, 400)
        self.title = f"Analyze Data of {str_name}"
        self.setWindowTitle(self.title)
        self.initUI()
        analyze_windows.append(self)
    def initUI(self):
        central_widget = QWidget(self)  # 创建一个中央部件
        self.setCentralWidget(central_widget)  # 将中央部件设置为窗口的中央部件
        vbox = QVBoxLayout(central_widget)

        splitter_prop = QScrollArea()
        splitter_cluster = QScrollArea()
        splitter_diy = QScrollArea()
        splitter_pdos_diy = QScrollArea()
        splitter_diy.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # 设置大小策略为固定长拉伸宽
        splitter_pdos_diy.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # 创建一个小布局
        self.inner_widget_prop = QWidget()
        self.inner_layout_prop = QVBoxLayout(self.inner_widget_prop)
        inner_widget_cluster = QWidget()
        inner_layout_cluster = QVBoxLayout(inner_widget_cluster)
        inner_widget_diy = QWidget()
        inner_layout_diy = QVBoxLayout(inner_widget_diy)
        inner_widget_pdos_diy = QWidget()
        inner_layout_pdos_diy = QVBoxLayout(inner_widget_pdos_diy)

        button_input = QPushButton("查看数据")
        label_input = QLabel("Input Text")
        lineEdit_input = QLineEdit("{1,3,6}")
        diy_button_pdos = QPushButton("查看PDOS(轨道在上面)")
        ####Here you can input your function which could create data and also put the output data in the search_data.py
        self.button_surr = QPushButton('计算周围原子信息')
        self.button_surr.clicked.connect(lambda : self.refresh_checkbox(f"{code_path}/get_surrounding_by_elements_multi.py {self.str_name}"))
        self.inner_layout_prop.addWidget(self.button_surr)
       
        self.button_surr_statistic = QPushButton('统计周围原子信息')
        self.button_surr_statistic.clicked.connect(lambda : self.refresh_checkbox(f"{code_path}/separate_sur_atom_info.py {self.str_name}"))
        self.button_surr_statistic.clicked.connect(lambda : self.refresh_checkbox(f"{code_path}/separate_sur_atom_info_lst.py {self.str_name}"))
        self.inner_layout_prop.addWidget(self.button_surr_statistic)
        if not os.path.isfile(f"data.save/SURROUNDING_ATOMS_of_{self.str_name}.txt"):
            self.button_surr_statistic.hide()
        else:
            self.button_surr_statistic.show()

        self.button_octahedral = QPushButton('计算八面体信息')
        self.button_octahedral.clicked.connect(lambda : self.refresh_checkbox(f"{code_path}/atomic_info_extractor.py {self.str_name}"))
        self.inner_layout_prop.addWidget(self.button_octahedral)


        ##################################################################################################################
        self.all_checkbox = QCheckBox('全选', self)
        self.all_checkbox.stateChanged.connect(self.check_all)
        self.inner_layout_prop.addWidget(self.all_checkbox)

        self.generate_checkbox()
        #pdos box
        colomnout_pdos=QHBoxLayout()
        colomnout_pdos.addStretch()
        label_x_rg_pdos = QLabel("X range:")
        label_y_rg_pdos = QLabel("Y range:")
        label_space_pdos =QLabel("        ")
        label_con_x_pdos = QLabel(" ~ ")
        label_con_y_pdos = QLabel(" ~ ")
        lineEdit_x_l_pdos = QLineEdit()
        lineEdit_x_r_pdos = QLineEdit()
        lineEdit_y_l_pdos = QLineEdit()
        lineEdit_y_r_pdos = QLineEdit()
        colomnout_pdos.addWidget(label_x_rg_pdos)
        colomnout_pdos.addWidget(lineEdit_x_l_pdos)
        colomnout_pdos.addWidget(label_con_x_pdos)
        colomnout_pdos.addWidget(lineEdit_x_r_pdos)
        colomnout_pdos.addWidget(label_space_pdos)
        colomnout_pdos.addWidget(label_y_rg_pdos)
        colomnout_pdos.addWidget(lineEdit_y_l_pdos)
        colomnout_pdos.addWidget(label_con_y_pdos)
        colomnout_pdos.addWidget(lineEdit_y_r_pdos)
        linepara_pdos = QLineEdit("2;1;0;1")
        paralabel_pdos = QLabel("Line width;Fill VB;Show Ef;Fill pattern")
        colomnout_pdos.addWidget(paralabel_pdos)
        colomnout_pdos.addWidget(linepara_pdos)
        hidden_col_pdos=QHBoxLayout()
        hidden_col_pdos.addStretch()
        hidden_edit_pdos=QLineEdit("-tot")
        hidden_label_pdos=QLabel("轨道:（-1-s,tot）")
        hidden_button_pdos = QPushButton("查看PDOS")
        hidden_col_pdos.addWidget(hidden_label_pdos)
        hidden_col_pdos.addWidget(hidden_edit_pdos)
        hidden_col_pdos.addWidget(hidden_button_pdos)
        hidden_col_pdos.addStretch()
        inner_layout_pdos_diy.addLayout(colomnout_pdos)
        inner_layout_pdos_diy.addLayout(hidden_col_pdos)

        inner_layout_diy.addWidget(label_input)
        inner_layout_diy.addWidget(lineEdit_input)
        inner_layout_diy.addWidget(diy_button_pdos)
        inner_layout_diy.addWidget(button_input)

        if os.path.isfile(f"data.save/{self.str_name}_cluster_data.pkl"):
            loaded_data={}
            with open(f"data.save/{self.str_name}_cluster_data.pkl", "rb") as file:
                loaded_data = pickle.load(file)
            spe_df=loaded_data['spe_df']
            self.spe_df=spe_df
        #print(self.spe_df)

        self.checkboxes_cluster=[]
        self.all_checkbox_cluster = QCheckBox('全选', self)
        self.all_checkbox_cluster.stateChanged.connect(self.check_all_cluster)
        inner_layout_cluster.addWidget(self.all_checkbox_cluster)
        if os.path.isfile(f"data.save/{self.str_name}_cluster_data.pkl"):
            for key,v in spe_df.items():
                checkbox_cluster=QCheckBox(f"{key}:{v}")
                #checkbox_cluster.stateChanged.connect(self.update_all_checkbox_state_cluster)
                self.checkboxes_cluster.append(checkbox_cluster)
                inner_layout_cluster.addWidget(checkbox_cluster)

        button_input_cluster = QPushButton("查看此类原子")
        #inner_layout_cluster.addWidget(button_input_cluster)

        splitter_prop.setWidget(self.inner_widget_prop)
        splitter_prop.setWidgetResizable(True)  # 使内部小部件可以自动调整大小
        splitter_cluster.setWidget(inner_widget_cluster)
        splitter_cluster.setWidgetResizable(True)  # 使内部小部件可以自动调整大小
        splitter_diy.setWidget(inner_widget_diy)
        splitter_diy.setWidgetResizable(True)  # 使内部小部件可以自动调整大小
        splitter_pdos_diy.setWidget(inner_widget_pdos_diy)
        splitter_pdos_diy.setWidgetResizable(True)  # 使内部小部件可以自动调整大小

        vbox.addWidget(splitter_prop)
        vbox.addWidget(splitter_cluster)
        vbox.addWidget(splitter_pdos_diy)
        vbox.addWidget(button_input_cluster)
        vbox.addWidget(splitter_diy)
  
     
        self.setGeometry(100,100,500, self.height())
        button_input.clicked.connect(lambda : self.button_click_analyze(button_input,lineEdit_input))
        button_input_cluster.clicked.connect(lambda : self.button_click_analyze_cluster(button_input_cluster))
        hidden_button_pdos.clicked.connect(lambda : self.button_click_pdos_diy(hidden_button_pdos,lineEdit_x_l_pdos,lineEdit_x_r_pdos,lineEdit_y_l_pdos,lineEdit_y_r_pdos,linepara_pdos,hidden_edit_pdos,hidden_label_pdos))
        diy_button_pdos.clicked.connect(lambda : self.button_click_pdos_diy_diy(hidden_button_pdos,lineEdit_x_l_pdos,lineEdit_x_r_pdos,lineEdit_y_l_pdos,lineEdit_y_r_pdos,linepara_pdos,hidden_edit_pdos,hidden_label_pdos,lineEdit_input))

    def check_all_cluster(self, state):
        # 如果全选复选框被选中，则选中所有其他复选框，否则取消选中
        is_checked = True if state == 2 else False
        for checkbox in self.checkboxes_cluster:
            checkbox.setChecked(is_checked)
    def update_all_checkbox_state_cluster(self):
        # 当任何一个复选框的状态改变时，更新全选复选框的状态
        all_checked = all(checkbox.isChecked() for checkbox in self.checkboxes_cluster)
        self.all_checkbox_cluster.setChecked(all_checked)

    def check_all(self, state):
        # 如果全选复选框被选中，则选中所有其他复选框，否则取消选中
        is_checked = True if state == 2 else False
        for checkbox in self.checkboxes:
            checkbox.setChecked(is_checked)
    def update_all_checkbox_state(self):
        # 当任何一个复选框的状态改变时，更新全选复选框的状态
        all_checked = all(checkbox.isChecked() for checkbox in self.checkboxes)
        self.all_checkbox.setChecked(all_checked)

    def button_click_pdos_diy(self,button_pdos,lineEdit_x_l_pdos,lineEdit_x_r_pdos,lineEdit_y_l_pdos,lineEdit_y_r_pdos,linepara_pdos,hidden_edit_pdos,hidden_label_pdos):
        str_name_i=self.str_name
        selected_clusters = []
        for checkbox_clu in self.checkboxes_cluster:
            if checkbox_clu.isChecked():
                selected_clusters.append(checkbox_clu.text())
        send_words_i=[]
        for i,v in enumerate(selected_clusters):
            key_i=v.split(":")[0]
            send_words_i.extend(self.spe_df[key_i])
        if not send_words_i:
            print("No atom clusters selected!")
            return
        atom_info="{%s}"%(",".join([str(x) for x in send_words_i]))
        self.pdos_show_diy_pdos(str_name_i,lineEdit_x_l_pdos,lineEdit_x_r_pdos,lineEdit_y_l_pdos,lineEdit_y_r_pdos,linepara_pdos,hidden_edit_pdos,hidden_label_pdos,atom_info) 

    def button_click_pdos_diy_diy(self,button_pdos,lineEdit_x_l_pdos,lineEdit_x_r_pdos,lineEdit_y_l_pdos,lineEdit_y_r_pdos,linepara_pdos,hidden_edit_pdos,hidden_label_pdos,line_Edit):
        str_name_i=self.str_name 
        atom_info=line_Edit.text().strip()
        self.pdos_show_diy_pdos(str_name_i,lineEdit_x_l_pdos,lineEdit_x_r_pdos,lineEdit_y_l_pdos,lineEdit_y_r_pdos,linepara_pdos,hidden_edit_pdos,hidden_label_pdos,atom_info) 

    def pdos_show_diy_pdos(self,str_name,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara,hidden_edit,hidden_label,atom_info):
            f_gnu=open("gnu_buffer_pdos","w")
            f_gnu.writelines("set xlabel \"E-Ef\"\n")
            f_gnu.writelines("set ylabel \"pDOS\"\n")
            if len(linepara.text())>0:
                line_width=linepara.text().split(";")[0]
                fill_fermi=linepara.text().split(";")[1]
                show_ef=linepara.text().split(";")[2]
                pattern_fs=linepara.text().split(";")[3]
            #print(line_width,fill_fermi,show_ef,pattern_fs)
            if fill_fermi=='1':
                #print("FILL FERMI")
                f_gnu.writelines("set style fill pattern %s noborder\n"%pattern_fs)
            if (len(lineEdit_x_l.text())>0)&(input_type_judge(lineEdit_x_l.text())!="str"):
                pdos_x_l=lineEdit_x_l.text()
            else:
                pdos_x_l="*"
            if (len(lineEdit_x_r.text())>0)&(input_type_judge(lineEdit_x_r.text())!="str"):
                pdos_x_r=lineEdit_x_r.text()
            else:
                pdos_x_r="*"
            if (len(lineEdit_y_l.text())>0)&(input_type_judge(lineEdit_y_l.text())!="str"):
                pdos_y_l=lineEdit_y_l.text()
            else:
                pdos_y_l="*"
            if (len(lineEdit_y_r.text())>0)&(input_type_judge(lineEdit_y_r.text())!="str"):
                pdos_y_r=lineEdit_y_r.text()
            else:
                pdos_y_r="*"
            f_gnu.writelines("set xrange [%s:%s]\n"%(str(pdos_x_l),str(pdos_x_r)))
            f_gnu.writelines("set yrange [%s:%s]\n"%(str(pdos_y_l),str(pdos_y_r)))
            f_gnu.writelines("set title \"PDOS of %s\"\n"%str_name)
                        
            spell_in=atom_info+hidden_edit.text().strip()
            #calculate the uncalculated files
            os.chdir(str_name)
            print("change dir into:",str_name)
            data_pdos=os.popen("python %s/sum_qe_pdos_diy.py \'%s\'"%(code_path,spell_in)).readlines()
            os.chdir(Current_dir)
            return_lst=[]
            for returns in data_pdos:
                if "###" in returns:
                    returns_seg=[x for x in returns.strip("\n").split(" ") if len(x) > 0]
                    return_lst.append([returns_seg[1],returns_seg[2],returns_seg[3],returns_seg[4]])

            file_input_tag=[]
            for root,dirs,files in os.walk(Current_dir+"/"+str_name):
                for file in files:
                    if "in_scf" in file or "in_relax" in file:
                        file_input_tag.append(file)
            if len(file_input_tag)==0:
                raise ValueError("No input found!The input shall be named with \'in_scf\' or '\in_relax\'!")
            spin_mode=0
            input_tag=open(str_name+"/"+file_input_tag[0]).readlines()
            for lines in input_tag:
                if "nspin" in lines and "2" in lines and "!" not in lines:
                    spin_mode=1   

            out_pdos=''
            if show_ef=='1':
                #print("show eF!")
                out_pdos+="\"%s/BTO.pdos_tot\" u ($1-$1):($3) w l lt -1 lc 'black' lw %s t \"Fermi-level\","%(str_name,line_width)
                if spin_mode==1:
                    out_pdos+="\"%s/BTO.pdos_tot\" u ($1-$1):(-1*$3) w l lt -1 lc 'black' lw %s notitle,"%(str_name,line_width)
            for segs in return_lst:
                #out_pdos+="\"%s/%s\" u 1:%d w l lc %d lw %s t \"%s-up\","%(str_name,segs[1],int(segs[2]),return_lst.index(segs),line_width,segs[3])
                #out_pdos+="\"%s/%s\" u 1:(-1*$%d) w l lc %d lw %s t \"%s-down\","%(str_name,segs[1],int(segs[2])+1,return_lst.index(segs),line_width,segs[3])
                out_pdos+="\"%s/%s\" u 1:%d w l lc %d lw %s t \"%s\","%(str_name,segs[1],int(segs[2]),return_lst.index(segs)+1,line_width,segs[3])
                if spin_mode==1:
                    out_pdos+="\"%s/%s\" u 1:(-1*$%d) w l lc %d lw %s notitle,"%(str_name,segs[1],int(segs[2])+1,return_lst.index(segs)+1,line_width)
                if fill_fermi=='1':
                    if spin_mode==1:
                        out_pdos+="\"%s/%s\" u 1:($1<0 ? $%d : 1/0):($1<0 ? -1*$%d : 1/0) with filledcurves lc %d notitle,"%(str_name,segs[1],int(segs[2]),int(segs[2])+1,return_lst.index(segs)+1)
                    elif spin_mode==0:
                        out_pdos+="\"%s/%s\" u 1:($1<0 ? $%d : 1/0) with filledcurves x1 lc %d notitle,"%(str_name,segs[1],int(segs[2]),return_lst.index(segs)+1)
            f_gnu.writelines("p"+out_pdos+"\n")
            f_gnu.close()
            #time.sleep(1)
            subprocess.call(['gnuplot', '-e', 'l \"gnu_buffer_pdos\"','--persist'])        

    def generate_checkbox(self):
        av_prop_lines=os.popen("python %s/search_data.py %s \'{1}\' magnetic"%(code_path,self.str_name)).readlines()
        av_props=[]
        tot_props=[]
        for lines in av_prop_lines:
            if "#AV_PROP#" in lines:
                #print("***",lines)
                av_props=lines.strip("#AV_PROP#").strip().strip("\n").split("\t")
            elif "#TOT_PROP#" in lines:
                tot_props=lines.strip("#TOT_PROP#").strip().strip("\n").split("\t")
        #print(av_props)
        #print(tot_props)
        self.checkboxes=[]
        for props in tot_props:
            checkbox=QCheckBox(props)
            if props not in av_props:
                checkbox.setEnabled(False)
            #checkbox.stateChanged.connect(self.update_all_checkbox_state)
            self.checkboxes.append(checkbox)
            self.inner_layout_prop.addWidget(checkbox)

    def refresh_checkbox(self,pre_code):
        # previous script execute:
        os.system(f"python {pre_code}")
        # some button calculation logics
        if not os.path.isfile(f"data.save/SURROUNDING_ATOMS_of_{self.str_name}.txt"):
            self.button_surr_statistic.hide()
        else:
            self.button_surr_statistic.show()

        # 清除现有的复选框
        for checkbox in self.checkboxes:
            self.inner_layout_prop.removeWidget(checkbox)
            checkbox.deleteLater()  # 清除并删除复选框
        self.checkboxes.clear()

        # 重新获取数据并创建复选框
        self.generate_checkbox()


    def button_click_analyze(self,button,lineEdit):
        filename=f"AnaVisual.data"
        send_words=lineEdit.text().strip()
        selected_elements = []
        for checkbox in self.checkboxes:
            if checkbox.isChecked():
                selected_elements.append(checkbox.text())
        if selected_elements:
            properties="+".join(selected_elements)
            print("python %s/search_data.py %s \'%s\' %s > %s"%(code_path,self.str_name,send_words,properties,filename)) 
            self.anaVis = AnaVisual(self.str_name,send_words,properties)
            self.anaVis.show()
        else:
            print("No props selected!")

    def button_click_analyze_cluster(self,button):
        filename=f"AnaVisual.data"
        selected_clusters = []
        for checkbox_clu in self.checkboxes_cluster:
            if checkbox_clu.isChecked():
                selected_clusters.append(checkbox_clu.text())
        send_words_i=[]
        for i,v in enumerate(selected_clusters):
            key_i=v.split(":")[0]
            send_words_i.extend(self.spe_df[key_i])
        if not send_words_i:
            print("No atom clusters selected!")
            return
        send_words="{%s}"%(",".join([str(x) for x in send_words_i]))
        selected_elements = []
        for checkbox in self.checkboxes:
            if checkbox.isChecked():
                selected_elements.append(checkbox.text())
        if selected_elements:
            properties="+".join(selected_elements)
            print("python %s/search_data.py %s \'%s\' %s > %s"%(code_path,self.str_name,send_words,properties,filename)) 
            self.anaVis = AnaVisual(self.str_name,send_words,properties)
            self.anaVis.show()
        else:
            print("No props selected!")



ana_visual_windows=[]
class AnaVisual(QMainWindow):
    def __init__(self, str_name, send_words, properties):
        super().__init__() 
        #self.resize(800, 1000)
        #self.title = f"ana_visual of {str_name}"
        #self.setWindowTitle(self.title)
        self.setWindowTitle(f"Pearson & Spearman Calculator of {str_name}")
        self.setGeometry(100, 100, 800, 600)
        self.send=send_words
        self.filename=f"AnaVisual.data"
        self.str_name=str_name
        os.system(f"python %s/search_data.py %s \'%s\' {properties} > %s"%(code_path,str_name,send_words,self.filename)) 
        #vbox.setCentralWidget(table_analyze)
        with open(self.filename, 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            lst_buffer=[]
            for row in reader:
                if row[0].startswith('#'):
                    continue
                else:
                    lst_buffer.append(row)
            data_raw = list(lst_buffer)
        self.table_data = [[value for value in row if value] for row in data_raw]
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
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save_Table_Data', f"Searched_Data_of_{self.str_name}.csv", 'CSV Files (*.csv);;Excel Files (*.xls);;All Files (*)', options=options)

        if file_path:
            try:
                # 打开文件以写入数据
                with open(file_path, 'w') as file:
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



class BandsShow(QWidget):
    def __init__(self,str_name):
        super().__init__()
        self.str_name=str_name
        self.pic="%s/out_bands_%s.png"%(str_name,str_name)
        self.initUI()
    def initUI(self):
        hbox = QHBoxLayout(self)
        lbl = QLabel(self)
        pixmap = QPixmap(self.pic)  # 按指定路径找到图片
        lbl.setPixmap(pixmap)  # 在label上显示图片
        lbl.setScaledContents(True)  # 让图片自适应label大小
        hbox.addWidget(lbl)
        self.setLayout(hbox)
        self.move(300, 200)
        self.setWindowTitle('bands of %s'%self.str_name)
        self.show()

       
def bands_read(file_bands):
    bands_in_scf=[]
    readed=0
    f=open(file_bands,"r").readlines()
    for i in range(len(f)):
        if "occupation numbers" in f[i]:
            if readed==0:
                #print(i)
                for j in range(i+1,len(f)):
                    bands_line=f[j].split()
                    if len(bands_line)==0:
                        readed=1
                        break
                    elif len(bands_line)>0:
                        for k in bands_line:
                            if float(k)!=0:
                                bands_in_scf.append(k)
    bands_in_valence=int(len(bands_in_scf))
    return bands_in_valence

def input_type_judge(input_cont):
    input_type=""
    try:
        float(input_cont)
    except:
        input_type="str"
    else:
        try:
            int(input_cont)
        except:
            input_type="float"
        else:
            input_type="int"
    return input_type


# Create an application and a main window instance
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())

