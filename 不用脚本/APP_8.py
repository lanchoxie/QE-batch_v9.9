# -*- coding: utf-8 -*-
"""
Created on Fri May 19 19:06:57 2023

@author: xiety
"""

# -*- coding: utf-8 -*-
#!!! pip install PyQt5


from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton,QComboBox, QLabel, QTextEdit, QMainWindow, QCheckBox, QLineEdit
from PyQt5.QtGui import QPixmap
#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import csv
import sys
import os
import subprocess
import time



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



# Define the main window class
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(600, 400)
        self.setWindowTitle('QE-batch_V-2.0')
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
        self.setGeometry(100, 100, 500, 300)
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
                linepara = QLineEdit("1;0;1;1")
                paralabel = QLabel("Line width;Fill VB;Show Ef;Fill pattern")
                colomnout.addWidget(paralabel)
                colomnout.addWidget(linepara)
                
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

        if self.mode=="bands":
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
            for ii in range(table.rowCount()):
                for jj in range(table.columnCount()):
                    self.table_1.setItem(ii, jj, QTableWidgetItem(table.item(ii,jj)))
                    self.table_2.setItem(ii, jj, QTableWidgetItem(table.item(ii,jj)))
            self.table_1.setGeometry(table.geometry())
            self.table_2.setGeometry(table.geometry())
            layout.addWidget(self.table_1,layout.indexOf(table))
            layout.addWidget(self.table_2,layout.indexOf(table))
            self.table_1.hide()
            self.table_2.hide()

        for i in range(table.rowCount()):
            self.checkBox_1=locals()
            self.checkBox_2=locals()
            self.hbox_pdos_1=locals()
            self.hbox_pdos_2=locals()
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
            #Initial Calculation
            if (data[i][-2].find("No Calculation")!=-1):
                button = QPushButton("计算")
                button.clicked.connect(lambda state,x=i:self.button_click_init(button,table,button_clicked,x))
                button_pdos1 = QPushButton("计算")
                button_pdos1.clicked.connect(lambda state,x=i:self.button_click_init(button_pdos1,table,button_clicked,x))
                button_pdos2 = QPushButton("计算")
                button_pdos2.clicked.connect(lambda state,x=i:self.button_click_init(button_pdos2,table,button_clicked,x))
                table.setCellWidget(i, table.columnCount()-1, button)
                self.table_1.setCellWidget(i, self.table_1.columnCount()-1, button_pdos1)
                self.table_2.setCellWidget(i, self.table_2.columnCount()-1, button_pdos2)
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
            #scf process check
            if (self.mode=="scf")&((data[i][-2].find("DONE")!=-1)|(data[i][-2].find("running")!=-1)):
                button_E_er = QPushButton("E误差")
                button_E_er.clicked.connect(lambda state,x=i:self.button_click_curve(button_E_er,table,"scf-estima",x,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r))
                table.setCellWidget(i, table.columnCount()-1, button_E_er)
            #bands draw
            if (self.mode=="bands")&(data[i][-2].find("DONE")!=-1):
                button = QPushButton("计算能带")
                #button = QPushButton("画能带")
                #button.clicked.connect(lambda state,x=i:self.button_click_bands(button,table,button_clicked,x,lineEdit_y_l,lineEdit_y_r,show_y_l_bands,show_y_r_bands))
                button.clicked.connect(lambda state,x=i:self.button_click_bands(button,table,button_clicked,x,lineEdit_y_l,lineEdit_y_r))
                table.setCellWidget(i, table.columnCount()-1, button)
            #pdos draw
            if (self.mode=="pdos")&(data[i][-2].find("DONE")!=-1):
                #reading elemental species from in_pdos
                start_reading=0
                ele_spe_li=[]
                self.elements_1["el_a"+str(i)]=["total"]
                self.elements_2["el_b"+str(i)]=["total"]
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
                    self.elements_1["el_a"+str(i)].append(ele_spe)
                    self.elements_2["el_b"+str(i)].extend(atomic_orbit(ele_spe,str_name))

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
                self.hbox_pdos_2["hbx_b"+str(i)] = QHBoxLayout()
                self.hbox_pdos_2["hbx_b"+str(i)].addStretch()
                self.button_2 = QPushButton("计算PDOS")
                #for ele_symbol in [x.split("-")[0] for x in self.elements_2["el_b"+str(i)]]:
                #    if ele_symbol not in created_elements:
                #        checkbox = QCheckBox(ele_symbol)
                #        self.hbox_pdos_2["hbx_b"+str(i)].addWidget(checkbox)
                #        created_elements.add(ele_symbol)
                #self.widget_pdos_2["wg_b"+str(i)] = QWidget()
                #self.widget_pdos_2["wg_b"+str(i)].setLayout(self.hbox_pdos_2["hbx_b"+str(i)])

                for ele in range(len(self.elements_2["el_b"+str(i)])):
                    self.checkBox_2["chbx_b"+str(i)]=QCheckBox(self.elements_2["el_b"+str(i)][ele])
                    self.hbox_pdos_2["hbx_b"+str(i)].addWidget(self.checkBox_2["chbx_b"+str(i)])
                self.widget_pdos_2["wg_b"+str(i)] = QWidget()
                self.widget_pdos_2["wg_b"+str(i)].setLayout(self.hbox_pdos_2["hbx_b"+str(i)])
                
                self.table_2.setCellWidget(i, table.columnCount()-2, self.widget_pdos_2["wg_b"+str(i)])
                self.button_2.clicked.connect(lambda state,x=i:self.button_click_pdos(self.button_2,table,button_clicked,x,self.elements_2["el_b"+str(x)],self.hbox_pdos_2["hbx_b"+str(x)],self.checkBox_2["chbx_b"+str(x)],lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara))
                self.table_2.setCellWidget(i, table.columnCount()-1, self.button_2)
                self.table_2.resizeRowsToContents()

                self.table_1.setCellWidget(i, table.columnCount()-2, self.widget_pdos_1["wg_a"+str(i)])
                self.button_1.clicked.connect(lambda state,x=i:self.button_click_pdos(self.button_1,table,button_clicked,x,self.elements_1["el_a"+str(x)],self.hbox_pdos_1["hbx_a"+str(x)],self.checkBox_1["chbx_a"+str(x)],lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara))
                self.table_1.setCellWidget(i, table.columnCount()-1, self.button_1)
                self.table_1.resizeRowsToContents()

                table.hide()
                self.table_1.show()
                self.pdos_combo_box.activated.connect(lambda state,x=i:self.pdos_on_active(table,button_clicked,x,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara))
                
    def pdos_on_active(self,table,button_clicked,i,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara):
        # Get the selected item
        self.pdos_mode = self.pdos_combo_box.currentText()
        # Print the selected item
        print ("Pdos mode:",self.pdos_mode)
        if self.pdos_mode=="simple mode":
            self.table_1.show()
            self.table_2.hide()
            #self.button_2.hide()
            #self.button_1.show()
            #self.widget_pdos_2["wg_b"+str(i)].hide()
            #self.widget_pdos_1["wg_a"+str(i)].show()
            #table.setCellWidget(i, table.columnCount()-2, self.widget_pdos_1["wg_a"+str(i)])
            #self.button_1.clicked.connect(lambda state,x=i:self.button_click_pdos(self.button_1,table,button_clicked,x,self.elements_1["el_a"+str(x)],self.hbox_pdos_1["hbx_a"+str(x)],self.checkBox_1["chbx_a"+str(x)],lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara))
            #table.setCellWidget(i, table.columnCount()-1, self.button_1)

        if self.pdos_mode=="detailed orbital mode":
            self.table_1.hide()
            self.table_2.show()
            #self.button_1.hide()
            #self.button_2.show()
            #self.widget_pdos_1["wg_a"+str(i)].hide()
            #self.widget_pdos_2["wg_b"+str(i)].show()
            #table.setCellWidget(i, table.columnCount()-2, self.widget_pdos_2["wg_b"+str(i)])
            #self.button_2.clicked.connect(lambda state,x=i:self.button_click_pdos(self.button_2,table,button_clicked,x,self.elements_2["el_b"+str(x)],self.hbox_pdos_2["hbx_b"+str(x)],self.checkBox_2["chbx_b"+str(x)],lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara))
            #table.setCellWidget(i, table.columnCount()-1, self.button_2)

#def copy button
    def keyPressEvent(self, event):
        if event.matches('Ctrl+Insert'):
            selected = self.table.selectedRanges()
            s = '\t' + '\t'.join([str(self.table.horizontalHeaderItem(i).text()) for i in range(selected[0].leftColumn(), selected[0].rightColumn() + 1)]) + '\n'
            for r in range(selected[0].topRow(), selected[0].bottomRow() + 1):
                s += str(self.table.verticalHeaderItem(r).text()) + '\t'
                for c in range(selected[0].leftColumn(), selected[0].rightColumn() + 1):
                    try:
                        s += str(self.table.item(r,c).text()) + '\t'
                    except AttributeError:
                        s += '\t'
                s = s[:-1] + '\n'    
            QApplication.clipboard().setText(s)
        if event.matches('Ctrl'):
            donothing=1


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

    def pdos_show(self,str_name,elements_sel,lineEdit_x_l,lineEdit_x_r,lineEdit_y_l,lineEdit_y_r,linepara):
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
                out_pdos="\"%s/BTO.pdos_tot\" u ($1-%f):2 w l lw %s lc \"black\" t \"Total-up\","%(str_name,Fermi_E,line_width)
                out_pdos+="\"%s/BTO.pdos_tot\" u ($1-%f):(-1*$3) w l lw %s lc \"black\" t \"Total-down\","%(str_name,Fermi_E,line_width)
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
                out_pdos+="\"%s/BTO.pdos_tot\" u ($1-$1):(-1*$3) w l lt -1 lc 'black' lw %s notitle,"%(str_name,line_width)
            for ele in range(len(elements_sel_ele)):
                out_pdos+="\"%s/accumulated_pdos_file%s\" u 1:2 w l lc %d lw %s t \"%s-up\","%(str_name,elements_sel_ele[ele],ele+1,line_width,elements_sel_ele[ele])
                out_pdos+="\"%s/accumulated_pdos_file%s\" u 1:(-1*$3) w l lc %d lw %s t \"%s-down\","%(str_name,elements_sel_ele[ele],ele+1,line_width,elements_sel_ele[ele])
                if fill_fermi=='1':
                    out_pdos+="\"%s/accumulated_pdos_file%s\" u 1:($1<0 ? $2 : 1/0):($1<0 ? -1*$3 : 1/0) with filledcurves lc %d notitle,"%(str_name,elements_sel_ele[ele],ele+1)
                    #out_pdos+="\"%s/accumulated_pdos_file%s\" u 1:($1<0 ? -1*$5 : 1/0) with filledcurves x1 lc %d notitle,"%(str_name,elements_sel_ele[ele],ele+1)
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

