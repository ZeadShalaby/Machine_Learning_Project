# -*- coding: utf-8 -*-
"""
Created on Fri May 19 20:33:04 2023

@author: zead shalaby
"""

import tkinter as tk
from tkinter import filedialog ,messagebox,ttk
from tkinter import *
import pandas as pd
import numpy as np
from tkinter import messagebox
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tkinter import filedialog
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tkinter import filedialog
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.svm import SVC 





###########################################################


# the window 

window = tk.Tk()
    
window.title("Machine Learning")
window.geometry("1300x600")
window.config(background="black")


########################################################################################################################################
########################################((  DATA_SHOW , DESCRIPTION , SIMPLEIMPUTER , DATA_NULL ,Polt_Data  ))####################################
########################################################################################################################################
   
# function graph to plot data


    
    
def openFile():
    filepath = filedialog.askopenfilename(initialdir="/Downloads",title="Select A File",filetype=(("csv files","*.csv"),("All Files","*.*")))
    try:
        exel_data = r"{}".format(filepath)
        global df
        global count
        count = 1
        df=pd.read_csv(exel_data)
        #print(df.describe())
        labelfile.config(text=filepath) 
   
    except ValueError:
        tk.messagebox.showerror("information","the file you have chosen is invalid")
        return None
    except FileNotFoundError:
        tk.messagebox.showerror("information","No such file as {filepath}")
        return None
    
 
   


    
#cleardata

    clear_data( )
    tv1["column"] = list(df.columns)
    tv1["show"]  = "headings"
    for column in tv1["column"]:
        tv1.heading(column, text=column)
        
    df_rows = df.to_numpy().tolist()
    for row in df_rows:
        tv1.insert("","end", values=row)
        
   
        
#####
   
    tv2["column"] = list(df.describe().columns)
    tv2["show"]  = "headings"
    for column in tv2["column"]:
        tv2.heading(column, text=column)
        
    df_row =df.describe().to_numpy().tolist()
    for row in df_row : 
        
        tv2.insert("","end",values=row)
  
    
#####

    
               
#funtion

def clear_data():
  tv1.delete(*tv1.get_children())
  tv2.delete(*tv2.get_children())
  tv3.delete(*tv3.get_children())



def isnull():
    global simple
    simple =  df.isnull().sum()
    print(simple)

    tv3["column"] = list(simple)
    tv3["show"]  = "headings"
    for column in tv3["column"]:
        tv3.heading(column, text=column)
    
    
    df_row =simple.to_numpy().tolist()
    for row in df_row : 
        tv3.insert("","end",values=row) 
        
   
def simple():
    
             
         my_imput = SimpleImputer(missing_values = np.nan , strategy = 'mean')
         my_imput.fit(df.iloc[: ,0:1])
         df.iloc[:,0:1] = my_imput.transform(df.iloc[: ,0:1])
         
         my_imput1 = SimpleImputer(missing_values = np.nan , strategy = 'mean')
         my_imput1.fit(df.iloc[: ,1:2])
         df.iloc[:,1:2] = my_imput1.transform(df.iloc[: ,1:2])
         
         
         my_imput2 = SimpleImputer(missing_values = np.nan , strategy = 'constant')
         my_imput2.fit(df.iloc[: ,4:5])
         df.iloc[:,4:5] = my_imput2.transform(df.iloc[: ,4:5])
         
         my_imput3 = SimpleImputer(missing_values = np.nan , strategy = 'constant')
         my_imput3.fit(df.iloc[: ,5:6])
         df.iloc[:,5:6] = my_imput3.transform(df.iloc[: ,5:6])
        
        
    
         global simplee
         simplee = df.isnull().sum()    
         print(simplee)
      
         tv4["column"] = list(simplee)
         tv4["show"]  = "headings"
         for column in tv4["column"]:
             tv4.heading(column, text=column)
         
             
         df_row =simplee.to_numpy().tolist()
         for row in df_row : 
             tv4.insert("","end",values=row)  
             
             
def  Classification():
    
      
      messagebox.showerror("Warning","Cant Show Please Preprocessing Data First .!!")
     
        
def Evaulation():
    
    messagebox.showerror("Warning","Cant Show Please Preprocessing Data First .!!")
            

def plot():
    x = df.iloc[:,1].values
    y = df.iloc[:,2].values



    figure = plt.figure(figsize=(10, 9), dpi=70)
    ax = figure.add_subplot(111)

    ax.plot(x, y, c='b', label='Data')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Data Plot')

    ax.legend()

    canvas = FigureCanvasTkAgg(figure, master=plot_frame)
    
    canvas.draw()
    canvas.get_tk_widget().pack()
    

             

########################################################################################################################################
########################################################((  PREPROCESSING DATA   ))#####################################################
########################################################################################################################################

    
def preprocessing(): 
   if df is NONE:
     n= 2
   else :
    n =0
 

  
    window.destroy()
    window_preprocessing = tk.Tk()
    
    window_preprocessing.title("Preprocessing")
    window_preprocessing.geometry("1300x600")
    window_preprocessing.config(background="black")
    
   
    # function Preprocessing
    
    def label_encode():
        global df_encoded
        if df is not None:
            label_encoder = LabelEncoder()
            df_encoded = df.apply(label_encoder.fit_transform)
            print(df_encoded)
            tv5["column"] = list(df_encoded)
            tv5["show"]  = "headings"
            for column in tv5["column"]:

                tv5.heading(column, text=column)
            
                
            df_row =df_encoded.to_numpy().tolist()
            for row in df_row : 
                tv5.insert("","end",values=row) 
        else:
           print("nothing")

         
      
      
     
       
       
           
    def one_hot_encode():
      global df_encoded, df_encoded_one_hot
      if df_encoded is not None:
          one_hot_encoder = OneHotEncoder(sparse=False)
          df_encoded_one_hot = pd.DataFrame(one_hot_encoder.fit_transform(df_encoded))
         
          tv6["column"] = list(df_encoded_one_hot)
          tv6["show"]  = "headings"
          for column in tv6["column"]:

              tv6.heading(column, text=column)
          
              
          df_row =df_encoded_one_hot.to_numpy().tolist()
          for row in df_row : 
              tv6.insert("","end",values=row) 
      else:
          print("nothing")       
           
          
             
           
       

            
            
    def  StanderScaler():
      if df_encoded is NONE:
        print("none do that")
      else:
          
        global object_cols
        object_cols = ['text']
        
       

        global df_new
        df_new = df_encoded
        scaler = StandardScaler()                   
        df_scaled = pd.DataFrame(scaler.fit_transform(df_new))
      
        tv7["column"] = list(df_scaled)
        tv7["show"]  = "headings"
        for column in tv7["column"]:

            tv7.heading(column, text=column)
        
            
        df_row =df_scaled.to_numpy().tolist()
        for row in df_row : 
            tv7.insert("","end",values=row) 
     
        
        
        
        
    def  MinMaxScaler(): 
     if df_encoded is NONE:
        print("none do that")
     else:
            
        from sklearn.preprocessing import MinMaxScaler

        scaler_s = MinMaxScaler()
        
        global data_new
#       data_new = scaler_s.fit_transform(newscaler0)                       # data after StanderScaler
        data_new =   df_encoded  # origin data after encoding
        print(data_new)

        tv8["column"] = list(data_new)
        tv8["show"]  = "headings"
        for column in tv8["column"]:
            
            tv8.heading(column, text=column)
            
                        
        df_mini = pd.DataFrame(scaler_s.fit_transform(data_new))
      
        df_row =df_mini.to_numpy().tolist()
        for row in df_row : 
            tv8.insert("","end",values=row)   
            
     
    def  Classification():
         messagebox.showerror("Warning","Cant Show Please Evalution Data First .!!")

         
########################################################################################################################################
########################################################((   Evaulation DATA   ))#####################################################
########################################################################################################################################


         
    def Evaulation():
        if data_new is None:
         print("non do that")    
        else:
             
         window_preprocessing.destroy()
         window_evaulation = tk.Tk()
             
         window_evaulation.title("Evaulation")
         window_evaulation.geometry("1300x600")
         window_evaulation.config(background="black")
         
         
         def  run_regression():
            error = str(var.get())
            test_size =train_test_slider.get()/100
            test_alpha=Alpha_slider.get()
            object_cols = ['text']
            data = data_new
            print(data)
            X=data.iloc[:,1]
            y=data.iloc[:,2]
            print(X,y)
            
# Split data into train and test sets

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size , random_state = 8)
            X_train = np.array(X_train).reshape(-1, 1)
            X_test = np.array(X_test).reshape(-1, 1)
                
#LinearRegression and fit 
            global lr
            lr = LinearRegression()
            lr.fit(X_train, y_train)  
            y_pred = lr.predict(X_test)
                
#predict on train and test data
            y_train_pred = lr.predict(X_train.reshape(-1, 1)).flatten()
            y_test_pred = lr.predict(X_test.reshape(-1, 1)).flatten()
         
            try:
              if error =="1":
                  print("MSE","Test Size : ",test_size,"Test Alpha : ",test_alpha)
                  result_train.config(text=mean_absolute_error(y_train,y_train_pred))
                  result_test.config(text=mean_absolute_error(y_test,y_test_pred))

              elif error == "2":
                  print("MAS","Test Size : ",test_size,"Test Alpha : ",test_alpha)
                  result_train.config(text=mean_squared_error(y_train,y_train_pred))
                  result_test.config(text=mean_squared_error(y_test,y_test_pred))

              elif error == "3":
                  print("RMSE","Test Size : ",test_size,"Test Alpha : ",test_alpha)  
                  result_train.config(text=np.sqrt(mean_squared_error(y_train, y_train_pred)))
                  result_test.config(text=np.sqrt(mean_squared_error(y_test,y_test_pred)))

              else:
                  tk.messagebox.showerror("information","Choise the Gender Error First")
              
            except FileNotFoundError:
              tk.messagebox.showerror("information","No such file as {error}")
             
       
            
       
           
         def  train_error():
            error = str(var.get())
            test_size =train_test_slider.get()/100
            test_alpha=Alpha_slider.get()
            object_cols = ['text']
            data = data_new
            print(data)
            X=data.iloc[:,1]
            y=data.iloc[:,2]
            print(X,y)
            
# Split data into train and test sets

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size , random_state = 8)
            X_train = np.array(X_train).reshape(-1, 1)
            X_test = np.array(X_test).reshape(-1, 1)
                
#LinearRegression and fit 
            global lr
            lr = LinearRegression()
            lr.fit(X_train, y_train)  
            y_pred = lr.predict(X_test)
                
#predict on train and test data
            y_train_pred = lr.predict(X_train.reshape(-1, 1)).flatten()
            y_test_pred = lr.predict(X_test.reshape(-1, 1)).flatten()
         
            try:
              if error =="1":
                  print("MSE","Test Size : ",test_size,"Test Alpha : ",test_alpha)
                  result_train.config(text=mean_absolute_error(y_train,y_train_pred))

              elif error == "2":
                  print("MAS","Test Size : ",test_size,"Test Alpha : ",test_alpha)
                  result_train.config(text=mean_squared_error(y_train,y_train_pred))
              elif error == "3":
                  print("RMSE","Test Size : ",test_size,"Test Alpha : ",test_alpha)  
                  result_train.config(text=np.sqrt(mean_squared_error(y_train, y_train_pred)))
              else:
                  tk.messagebox.showerror("information","Choise the Gender Error First")
              
            except FileNotFoundError:
              tk.messagebox.showerror("information","No such file as {error}")
             
 
# show test             
         
         def  test_error():
           
                
               error = str(var.get())
               test_size =train_test_slider.get()/100
               test_alpha=Alpha_slider.get()
               object_cols = ['text']
               data = data_new
               print(data)
               X=data.iloc[:,1]
               y=data.iloc[:,2]
               print(X,y)
               
   # Split data into train and test sets
               global X_train,y_train,y_train_pred,y_test_pred 
               X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size , random_state = 8)
               X_train = np.array(X_train).reshape(-1, 1)
               X_test = np.array(X_test).reshape(-1, 1)
                   
   #LinearRegression and fit 
               global lr
               lr = LinearRegression()
               lr.fit(X_train, y_train)  
               y_pred = lr.predict(X_test)
                   
   #predict on train and test data

               y_train_pred = lr.predict(X_train.reshape(-1, 1)).flatten()
               y_test_pred = lr.predict(X_test.reshape(-1, 1)).flatten()
            
               try:
                 if error =="1":
                     print("MSE","Test Size : ",test_size,"Test Alpha : ",test_alpha)
                     result_test.config(text=mean_absolute_error(y_test,y_test_pred))

                 elif error == "2":
                     print("MAS","Test Size : ",test_size,"Test Alpha : ",test_alpha)
                     result_test.config(text=mean_squared_error(y_test,y_test_pred))
                 elif error == "3":
                     print("RMSE","Test Size : ",test_size,"Test Alpha : ",test_alpha)  
                     result_test.config(text=np.sqrt(mean_squared_error(y_test,y_test_pred)))
                 else:
                     tk.messagebox.showerror("information","Choise the Gender Error First")
                 
               except FileNotFoundError:
                 tk.messagebox.showerror("information","No such file as {error}")
                

            #######################################
           
       
         def  intercept_coef():
           
             intercept_label.config(text=lr.intercept_)
             coef_label.config(text=lr.coef_)

           
        
         def show_train( value):
             pass
         
         
         def show_plot():
             x = X_train
             y = y_train



             figure = plt.figure(figsize=(10, 9), dpi=70)
             ax = figure.add_subplot(111)

             ax = plt.axes()
             ax.set(xlabel = 'petal length' , ylabel = 'petal width' , title = 'matplotlib_Data ( X_train,  y_pred_train )')
             plt.scatter(X_train, y_train,color="black")
             plt.plot(X_train, y_train_pred, color ='red')


             ax.legend()

             canvas = FigureCanvasTkAgg(figure, master=plot_frame)
             
             canvas.draw()
             canvas.get_tk_widget().pack()
             
          

             
             
            
##############################################################################################################################################
########################################################((   Classification DATA   ))##############################################################
##############################################################################################################################################


   
         def  Classification_affter_evaulation():
               if data_new is None:
                print("non do that")    
               else:        
                   window_evaulation.destroy()
                   window_classification = tk.Tk()
                        
                   window_classification.title("Classification")
                   window_classification.geometry("1030x600")
                   window_classification.config(background="black")                  
                    
                   
                    
                   
                    
                   
                    
                   global n
                   n=0
                   
                   def calculate_confusion_matrix():
                       # df -> after preprocessing -> data_new
                       test_size = slider_svm_knn.get()/100;
                       x = data_new.iloc[:, 0:-1].values
                       y = data_new.iloc[:, -1].values
                       x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
                       clf = KNeighborsClassifier(n_neighbors=3)
                       clf.fit(x_train, y_train) 
                       y_pred = clf.predict(x_test)
                       cm = confusion_matrix(y_test, y_pred)
                       result_Confusion_Matrix.config(text="Confusion Matrix:\n" + str(cm))
                       global n                      
                       n=2

                     
                   def calculate_precision():
                       if n == 0 :
                           messagebox.showerror("information","calculate_confusion_matrix First")
                       else:
                           test_size = slider_svm_knn.get()/100;
                           x = data_new.iloc[:, 0:-1].values
                           y = data_new.iloc[:, -1].values
                           x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
                           clf = KNeighborsClassifier(n_neighbors=3)
                           clf.fit(x_train, y_train) 
                           y_pred = clf.predict(x_test)
                           precision = precision_score(y_test, y_pred)
                           result_Calculate_Precision.config(text= str(precision))
                           global m
                           m=2

                   def calculate_accuracy():
                       if m+n == 4:
                           test_size = slider_svm_knn.get()/100;
                           x = data_new.iloc[:, 0:-1].values
                           y = data_new.iloc[:, -1].values
                           x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
                           clf = KNeighborsClassifier(n_neighbors=3)
                           clf.fit(x_train, y_train) 
                           y_pred = clf.predict(x_test)
                           accuracy = accuracy_score(y_test, y_pred)
                           result_Calculate_Accuracy.config(text= str(accuracy))
                       else:
                           messagebox.showerror("information","calculate_precision First")

                   
                                           
                   def calculate_confusion_matrix_svm():
                            test_size = slider_svm_knn.get()/100;
                            x = data_new.iloc[:,:-1].values
                            y = data_new.iloc[:, -1].values
                            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
                            clf = SVC(kernel="linear")
                            clf.fit(x_train, y_train) 
                            y_pred = clf.predict(x_test)
                            cm = confusion_matrix(y_test, y_pred)
                            result_Confusion_Matrix_svm.config(text="Confusion Matrix:\n" + str(cm))
                    
                   def calculate_precision_svm():
                            test_size = slider_svm_knn.get()/100;
                            x = data_new.iloc[:, :-1].values
                            y = data_new.iloc[:, -1].values
                            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
                            clf = SVC(kernel="linear")
                            clf.fit(x_train, y_train) 
                            y_pred = clf.predict(x_test)
                            precision = precision_score(y_test, y_pred)
                            result_Calculate_Precision_svm.config(text= str(precision))
                    
                   def calculate_accuracy_svm():
                            test_size = slider_svm_knn.get()/100;
                            x = data_new.iloc[:, 0:-1].values
                            y = data_new.iloc[:, -1].values
                            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
                            clf = SVC(kernel="linear")
                            clf.fit(x_train, y_train) 
                            y_pred = clf.predict(x_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            result_Calculate_Accuracy_svm.config(text= str(accuracy))
                   
                   
                    
                   def New_Data():
                       
                       window_classification.destroy()
                   
                    
                   
                    
                   
                   
                # Dashboard 
           
                   side_frame_classification = tk.Frame(window_classification,bg="#808080")
                   side_frame_classification.pack(side="left" ,fill="y")
                   label = tk.Label(side_frame_classification,bg="#808080",fg="#808080",font=25)
                   label.pack(pady=50,padx=65) 
                

                # Button Confusion Matrix                             
                   button = Button(side_frame_classification,text="Confusion Matrix",fg = 'black' , bg = '#808080' ,command=calculate_confusion_matrix)
                   button.pack()
                   button.place(width=138,height=30,x=0,y=40)

                # Button Calculate Precision 
                   button = Button(side_frame_classification,text="Calculate Precision",fg = 'black' , bg = '#808080' ,command=calculate_precision)
                   button.pack()
                   button.place(width=138,height=30,x=0,y=130)

                # Button Calculate Accuracy
                   button = Button(side_frame_classification,text="Calculate Accuracy",fg = 'black' , bg = '#808080' ,command=calculate_accuracy)
                   button.pack()
                   button.place(width=138,height=30,x=0,y=210)
                   
   # Svm Button
                   
                # Button Confusion Matrix_svm                             
                   button = Button(side_frame_classification,text="Confusion_M_svm",fg = 'black' , bg = '#808080' ,command=calculate_confusion_matrix_svm)
                   button.pack()
                   button.place(width=138,height=30,x=0,y=290) 

                # Button Calculate Precision_svm
                   button = Button(side_frame_classification,text="Precision_svm",fg = 'black' , bg = '#808080' ,command=calculate_precision_svm)
                   button.pack()
                   button.place(width=138,height=30,x=0,y=370)

                # Button Calculate Accuracy_svm
                   button = Button(side_frame_classification,text="Accuracy_svm",fg = 'black' , bg = '#808080' ,command=calculate_accuracy_svm)
                   button.pack()
                   button.place(width=138,height=30,x=0,y=450)

                #New_Data
                   button = Button(side_frame_classification,text="Finsh",fg = 'black' , bg = '#808080' ,command=New_Data)
                   button.pack()
                   button.place(width=138,height=30,x=0,y=530) 
                
                 
                
                # label Confusion_Matrix
                 
                   Confusion_Matrix = Label(window_classification,text="Confusion_Matrix",bd=1,relief="sunken",justify="left",fg = 'red' , bg = 'black')
                   Confusion_Matrix.place(width=110,height=24,x=185,y=90)
                
                   result_Confusion_Matrix= Label(window_classification,bd=1,relief="sunken",justify="left")
                   result_Confusion_Matrix.place(width=100,height=55,x=310,y=75)
                
                # label Calculate_Precision

                   Calculate_Precision = Label(window_classification,text="Calculate_Precision",bd=1,relief="sunken",justify="left",fg = 'red' , bg = 'black')
                   Calculate_Precision.place(width=110,height=24,x=185,y=175)
           
                
                   result_Calculate_Precision = Label(window_classification,bd=1,relief="sunken",justify="left")
                   result_Calculate_Precision.place(width=250,height=24,x=310,y=175)
                
                # label Calculate_Accuracy

                   Calculate_Accuracy = Label(window_classification,text="Calculate_Accuracy",bd=1,relief="sunken",justify="left",fg = 'red' , bg = 'black')
                   Calculate_Accuracy.place(width=110,height=24,x=185,y=275)
           
                
                   result_Calculate_Accuracy = Label(window_classification,bd=1,relief="sunken",justify="left")
                   result_Calculate_Accuracy.place(width=250,height=24,x=310,y=275)
                
                    
                
#######################  Calcuate SVM     ####################     
      
                
                # label Confusion_Matrix_svm
                 
                   Confusion_Matrix_svm = Label(window_classification,text="Confusion_Matrix",bd=1,relief="sunken",justify="left",fg = 'red' , bg = 'black')
                   Confusion_Matrix_svm.place(width=110,height=24,x=870,y=90)
                
                   result_Confusion_Matrix_svm = Label(window_classification,bd=1,relief="sunken",justify="left")
                   result_Confusion_Matrix_svm.place(width=100,height=55,x=760,y=75)
                
                # label Calculate_Precision_svm

                   Calculate_Precision_svm = Label(window_classification,text="Calculate_Precision",bd=1,relief="sunken",justify="left",fg = 'red' , bg = 'black')
                   Calculate_Precision_svm.place(width=110,height=24,x=870,y=175)
           
                
                   result_Calculate_Precision_svm = Label(window_classification,bd=1,relief="sunken",justify="left")
                   result_Calculate_Precision_svm.place(width=250,height=24,x=610,y=175)
                
                # label Calculate_Accuracy_svm

                   Calculate_Accuracy_svm = Label(window_classification,text="Calculate_Accuracy",bd=1,relief="sunken",justify="left",fg = 'red' , bg = 'black')
                   Calculate_Accuracy_svm.place(width=110,height=24,x=870,y=275)
           
                
                   result_Calculate_Accuracy_svm = Label(window_classification,bd=1,relief="sunken",justify="left")
                   result_Calculate_Accuracy_svm.place(width=250,height=24,x=610,y=275)
                
                
                
                # Slider num 
                
                   slider_svm_knn = Scale(window_classification, from_=0, to=100,length=200,orient=HORIZONTAL)
                   slider_svm_knn.set(3)
                   slider_svm_knn.pack()
                   slider_svm_knn.place(height=40,width=350,x=405,y=490)
                   labe3 = Label(window_classification,text="Test Size : ",fg = 'red',relief="sunken" , bg = 'black')
                   labe3.pack()
                   labe3.place(width=80,height=24,x=220,y=490)
                   
                   window_classification.mainloop()
            
           
         
         
         
         
         
##############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################

         
         
        
        # Dashboard 

         side_frame_evaulation = tk.Frame(window_evaulation,bg="#808080")
         side_frame_evaulation.pack(side="left" ,fill="y")
         label = tk.Label(side_frame_evaulation,bg="#808080",fg="#808080",font=25)
         label.pack(pady=50,padx=65) 
        
         # Frame Plot Data
         plot_frame = LabelFrame(window_evaulation,text="Plot Data",fg = 'black' , bg = '#696969')  
         plot_frame.pack(fill="both",expand="yes",padx=40,pady=10 )
         plot_frame.place(height=300,width=550,x=700,y=100)


        # Button run_regression                             
         button = Button(side_frame_evaulation,text="Run_Regression",fg = 'black' , bg = '#808080' ,command=run_regression)
         button.pack()
         button.place(width=138,height=30,x=0,y=40)

        # train_error 
         button = Button(side_frame_evaulation,text="Train_Error",fg = 'black' , bg = '#808080' ,command=train_error)
         button.pack()
         button.place(width=138,height=30,x=0,y=130)

        # test_error
         button = Button(side_frame_evaulation,text="Test_Error",fg = 'black' , bg = '#808080' ,command=test_error)
         button.pack()
         button.place(width=138,height=30,x=0,y=210)

        #intercept&coef
         button = Button(side_frame_evaulation,text="intercept_coef",fg = 'black' , bg = '#808080' ,command=intercept_coef)
         button.pack()
         button.place(width=138,height=30,x=0,y=290)

        #Classification
         button = Button(side_frame_evaulation,text="Classification",fg = 'black' , bg = '#808080' ,command=Classification_affter_evaulation)
         button.pack()
         button.place(width=138,height=30,x=0,y=360)
         
         #Classification
         button = Button(side_frame_evaulation,text="Plot_Data",fg = 'black' , bg = '#808080' ,command=show_plot)
         button.pack()
         button.place(width=138,height=30,x=0,y=440)

     
         # Slider Test
         train_test_slider = Scale(window_evaulation,  from_=0, to=100,length=200, orient=HORIZONTAL)
         train_test_slider.set(75)
         train_test_slider.pack()
         train_test_slider.place(x=280,y=55,width=350)
         
       
         # Slider Test
         slider_value_label = Label(window_evaulation, text="Test Size",relief="sunken",fg = 'red' , bg = 'black')
         slider_value_label.pack()
         slider_value_label.place(x=180,y=65)
       
         # Slider Alpha
         Alpha_slider = Scale(window_evaulation, from_=0, to=100,length=200, orient=HORIZONTAL)
         Alpha_slider.set(8)
         Alpha_slider.pack()
         Alpha_slider.place(x=280,y=180,width=350)
         
        
    # # #,relief="sunken"   make border  label  # # #
         
         
         # Slider Alpha
         slider_Alpha_label = Label(window_evaulation, text="Alpha Size",relief="sunken",fg = 'red' , bg = 'black')
        
         slider_Alpha_label.pack()
         slider_Alpha_label.place(x=180,y=188)

# TVPLOT DATA 

         tvplot = ttk.Treeview(plot_frame)
         tvplot.place(relheight=1,relwidth=1)
        
         treescrolly = tk.Scrollbar(plot_frame,orient="vertical",command=tvplot.yview)     
         treescrollx = tk.Scrollbar(plot_frame,orient="horizontal",command=tvplot.xview)     
         tvplot.configure(xscrollcommand=treescrollx.set,yscrollcommand=treescrolly.set)
         treescrollx.pack(side="bottom",fill="x")
         treescrolly.pack(side="right",fill="y")
         # Radio_Button


         var = IntVar()
         R1 = Radiobutton(window_evaulation, text="MSE", variable=var, value=1)
         R1.pack( anchor = W )
         R1.place(width=90,x=345,y=300)
         R2 = Radiobutton(window_evaulation, text="MAE", variable=var, value=2)
         R2.pack( anchor = W )
         R2.place(width=90,x=415,y=300)

         R3 = Radiobutton(window_evaulation, text="RMSE", variable=var, value=3)
         R3.pack( anchor = W)
         R3.place(width=90,x=485,y=300)
         label = Label(window_evaulation,text="Gender Erorr : ",relief="sunken",fg = 'red' , bg = 'black')
         label.pack()
         label.place(width=70,height=24,x=180,y=300)
        
        # label Error train
         
         text_result_train = Label(window_evaulation,text="Error_Train",bd=1,relief="sunken",justify="left",fg = 'red' , bg = 'black')
         text_result_train.place(width=70,height=24,x=180,y=400)
        
         result_train = Label(window_evaulation,bd=1,relief="sunken",justify="left")
         result_train.place(width=357,height=24,x=280,y=400)
        
        # label Error test

         text_result_test = Label(window_evaulation,text="Error_Test",bd=1,relief="sunken",justify="left",fg = 'red' , bg = 'black')
         text_result_test.place(width=70,height=24,x=180,y=475)
   
        
         result_test = Label(window_evaulation,bd=1,relief="sunken",justify="left")
         result_test.place(width=357,height=24,x=280,y=480)
        
        # label intercept
        
         intercept = Label(window_evaulation,text="intercept",bd=1,relief="sunken",justify="left",fg = 'red' , bg = 'black')
         intercept.place(width=70,height=24,x=180,y=550)
   
        # label coefficient
     
   
         intercept_label = Label(window_evaulation,bd=1,relief="sunken",justify="left")
         intercept_label.place(width=100,height=24,x=280,y=550)
         
         coef = Label(window_evaulation,text="coefficient",bd=1,relief="sunken",justify="left",fg = 'red' , bg = 'black')
         coef.place(width=70,height=24,x=455,y=550)
   
        
         coef_label = Label(window_evaulation,bd=1,relief="sunken",justify="left")
         coef_label.place(width=100,height=24,x=537,y=550)
         
         window_evaulation.mainloop() 
        
         
        
        
        
        
##############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################

       
            
    def  Features():
        
       # pca - keep 90% of variance
      x= data_new.iloc[:, :-1].values   # featrures
        
      y= data_new.iloc[:, -1].values 
      
     
      svc = SVC(kernel="linear", C=1)
      rfe = RFE(estimator=svc, n_features_to_select=4, step=1)
      fit=rfe.fit(x, y)

      model_features = fit.transform(x)

      print( model_features)
      print(fit.support_)
      
      tv9["column"] = list(model_features)   # fit.support_
      tv9["show"]  = "headings"
      for column in tv9["column"]:
          
          tv9.heading(column, text=column)
          
                      
      df_feature = model_features     #fit.support_
    
      df_row =df_feature
      for row in df_row : 
          tv9.insert("","end",values=row)   
          

      
      
    #  pca = PCA(0.50)
     # principal_components = pca.fit_transform(data_new)
     # principal_df = pd.DataFrame(data = principal_components)
     # print(principal_df.shape)

      #print( principal_df)
     
    
    # Dashboard 

    side_frame_preprocessing = tk.Frame(window_preprocessing,bg="#808080")
    side_frame_preprocessing.pack(side="left" ,fill="y")
    label = tk.Label(side_frame_preprocessing,bg="#808080",fg="#808080",font=25)
    label.pack(pady=50,padx=65) 
     

    # Button LabelEncoder                             
    button = Button(side_frame_preprocessing,text="LabelEncoder",fg = 'black' , bg = '#808080' ,command=label_encode)
    button.pack()
    button.place(width=138,height=30,x=0,y=40)

    # Burron OneHotEncoder 
    button = Button(side_frame_preprocessing,text="OneHotEncoder",fg = 'black' , bg = '#808080' ,command=one_hot_encode)
    button.pack()
    button.place(width=138,height=30,x=0,y=130)

    # Button StanderScaler
    button = Button(side_frame_preprocessing,text="StandardScaler",fg = 'black' , bg = '#808080' ,command=StanderScaler)
    button.pack()
    button.place(width=138,height=30,x=0,y=210)

    #MinMaxScaler
    button = Button(side_frame_preprocessing,text="MinMaxScaler",fg = 'black' , bg = '#808080' ,command=MinMaxScaler)
    button.pack()
    button.place(width=138,height=30,x=0,y=290)

    #Features
    button = Button(side_frame_preprocessing,text="Features",fg = 'black' , bg = '#808080' ,command=Features)
    button.pack()
    button.place(width=138,height=30,x=0,y=360)
     
    # Button Evaulation
    button = Button(side_frame_preprocessing,text="Evaulation",fg = 'black' , bg = '#808080' ,command=Evaulation)
    button.pack() 
    button.place(width=138,height=30,x=0,y=430)


    # Button Classification
    button = Button(side_frame_preprocessing,text="Classification",fg = 'black' , bg = '#808080' ,command=Classification)
    button.pack()
    button.place(width=138,height=30,x=0,y=510)

  
    # labelencoder_frame
    
    labelencoder_frame = LabelFrame(window_preprocessing,text="LabelEncoder",fg = 'black' , bg = '#696969')  
    labelencoder_frame.pack(fill="both",expand="yes",padx=40,pady=10 )
    labelencoder_frame.place(height=250,width=500,x=170,y=30)
    
    # onehotencoder_frame
    
    onehotencoder_frame = LabelFrame(window_preprocessing,text="OneHotEncoder",fg = 'black' , bg = '#696969')  
    onehotencoder_frame.pack(fill="both",expand="yes",padx=40,pady=10 )
    onehotencoder_frame.place(height=250,width=500,x=170,y=320)   
    
    # StanderScaler_frame
    
    StanderScaler_frame = LabelFrame(window_preprocessing,text="StanderScaler",fg = 'black' , bg = '#696969')  
    StanderScaler_frame.pack(fill="both",expand="yes",padx=40,pady=10 )
    StanderScaler_frame.place(height=180,width=500,x=770,y=30)
    
    # MinMaxScaler
    
    MinMaxScaler_frame = LabelFrame(window_preprocessing,text="MinMaxScaler",fg = 'black' , bg = '#696969')  
    MinMaxScaler_frame.pack(fill="both",expand="yes",padx=40,pady=10 )
    MinMaxScaler_frame.place(height=180,width=500,x=770,y=230)
    
    # Return Feature
    Features_frame = LabelFrame(window_preprocessing,text="Features",fg = 'black' , bg = '#696969')  
    Features_frame.pack(fill="both",expand="yes",padx=40,pady=10)
    Features_frame.place(height=150,width=500,x=770,y=430)
    
  # treeview 5 
    tv5 = ttk.Treeview(labelencoder_frame)
    tv5.place(relheight=1,relwidth=1)
    
  # scroll  
    treescrolly = tk.Scrollbar(labelencoder_frame,orient="vertical",command=tv5.yview)     
    treescrollx = tk.Scrollbar(labelencoder_frame,orient="horizontal",command=tv5.xview)     
    tv5.configure(xscrollcommand=treescrollx.set,yscrollcommand=treescrolly.set)
    treescrollx.pack(side="bottom",fill="x")
    treescrolly.pack(side="right",fill="y")
    
  # treeview 6  
    tv6 = ttk.Treeview(onehotencoder_frame)
    tv6.place(relheight=1,relwidth=1)
    
  # scroll  
    treescrolly = tk.Scrollbar(onehotencoder_frame,orient="vertical",command=tv6.yview)     
    treescrollx = tk.Scrollbar(onehotencoder_frame,orient="horizontal",command=tv6.xview)     
    tv6.configure(xscrollcommand=treescrollx.set,yscrollcommand=treescrolly.set)
    treescrollx.pack(side="bottom",fill="x")
    treescrolly.pack(side="right",fill="y")
    
    
    
    ## Treeview Widget 2

    tv7 = ttk.Treeview(StanderScaler_frame)
    tv7.place(relheight=1,relwidth=1)

    treescrolly = tk.Scrollbar(StanderScaler_frame,orient="vertical",command=tv7.yview)     
    treescrollx = tk.Scrollbar(StanderScaler_frame,orient="horizontal",command=tv7.xview)     
    tv7.configure(xscrollcommand=treescrollx.set,yscrollcommand=treescrolly.set)
    treescrollx.pack(side="bottom",fill="x")
    treescrolly.pack(side="right",fill="y")
    
    # treeview 8
    tv8 = ttk.Treeview(MinMaxScaler_frame)
    tv8.place(relheight=1,relwidth=1)
      
   # scroll  
    treescrolly = tk.Scrollbar(MinMaxScaler_frame,orient="vertical",command=tv8.yview)     
    treescrollx = tk.Scrollbar(MinMaxScaler_frame,orient="horizontal",command=tv8.xview)     
    tv8.configure(xscrollcommand=treescrollx.set,yscrollcommand=treescrolly.set)
    treescrollx.pack(side="bottom",fill="x")
    treescrolly.pack(side="right",fill="y")
      
    # treeview 9
    tv9 = ttk.Treeview(Features_frame)
    tv9.place(relheight=1,relwidth=1)
  
# scroll  
    treescrolly = tk.Scrollbar(Features_frame,orient="vertical",command=tv9.yview)     
    treescrollx = tk.Scrollbar(Features_frame,orient="horizontal",command=tv9.xview)     
    tv9.configure(xscrollcommand=treescrollx.set,yscrollcommand=treescrolly.set)
    treescrollx.pack(side="bottom",fill="x")
    treescrolly.pack(side="right",fill="y")
    
    
    
    
    

    window_preprocessing.mainloop()



########################################################################################################################################
########################################################################################################################################
########################################################################################################################################


 ###############
#  gui project  #
 ###############
 
# Dashboard 

side_frame = tk.Frame(window,bg="#808080")
side_frame.pack(side="left" ,fill="y")
label = tk.Label(side_frame,bg="#808080",fg="#808080",font=25)
label.pack(pady=50,padx=65) 
 
# Button Upload #

button = Button(side_frame,text="upload file",fg = 'red' , bg = 'black' ,command=openFile)
button.pack()
button.place(width=138,height=30,x=0,y=0)
v = StringVar()
labelfile = Label(window,text="No File Selected",fg='red',bg='black')
labelfile.pack()
labelfile.place(x=140,y=5)
b=labelfile["text"]

# Button Null Data
button = Button(side_frame,text="Null_Data",fg = 'black' , bg = '#808080' ,command=isnull)
button.pack()
button.place(width=138,height=30,x=0,y=80)

# Burron SimpleImputer 
button = Button(side_frame,text="SimpleImputer",fg = 'black' , bg = '#808080' ,command=simple)
button.pack()
button.place(width=138,height=30,x=0,y=160)


#DataPreprocessing
button = Button(side_frame,text="DataPreprocessing",fg = 'black' , bg = '#808080' ,command=preprocessing)
button.pack()
button.place(width=138,height=30,x=0,y=250)

# Button Classification
button = Button(side_frame,text="Classification",fg = 'black' , bg = '#808080' ,command=Classification)
button.pack()
button.place(width=138,height=30,x=0,y=330)

# Button Evaulation
button = Button(side_frame,text="Evaulation",fg = 'black' , bg = '#808080' ,command=Evaulation)
button.pack()
button.place(width=138,height=30,x=0,y=410)

# Button plot (Matplotlip)
button = Button(side_frame,text="Plot",fg = 'black' , bg = '#808080' ,command=plot)
button.pack()
button.place(width=138,height=30,x=0,y=490)

# Frame Show Data
wrapper1 = LabelFrame(window,text="Exel Data",fg = 'black' , bg = '#696969')  
wrapper1.pack(fill="both",expand="yes",padx=40,pady=10 )
wrapper1.place(height=250,width=500,x=170,y=70)

# Return Description Of Data
wrapper2 = LabelFrame(window,text="Description Data",fg = 'black' , bg = '#696969')  
wrapper2.pack(fill="both",expand="yes",padx=40,pady=10)
wrapper2.place(height=130,width=500,x=170,y=400)


# before null data 

null_before_frame = LabelFrame(window,text="Null_Values",fg = 'black' , bg = '#696969')  
null_before_frame.pack(fill="both",expand="yes",padx=40,pady=10 )
null_before_frame.place(height=180,width=80,x=900,y=370)

# After Simle imputer

null_after_frame = LabelFrame(window,text="SimbleImputer",fg = 'black' , bg = '#696969')  
null_after_frame.pack(fill="both",expand="yes",padx=40,pady=10 )
null_after_frame.place(height=180,width=80,x=1100,y=370)

# Frame Plot Data
plot_frame = LabelFrame(window,text="Plot Data",fg = 'black' , bg = '#696969')  
plot_frame.pack(fill="both",expand="yes",padx=40,pady=10 )
plot_frame.place(height=250,width=500,x=750,y=70)

## Treeview Widget 

tv1 = ttk.Treeview(wrapper1)
tv1.place(relheight=1,relwidth=1)

treescrolly = tk.Scrollbar(wrapper1,orient="vertical",command=tv1.yview)     
treescrollx = tk.Scrollbar(wrapper1,orient="horizontal",command=tv1.xview)     
tv1.configure(xscrollcommand=treescrollx.set,yscrollcommand=treescrolly.set)
treescrollx.pack(side="bottom",fill="x")
treescrolly.pack(side="right",fill="y")


## Treeview Widget 2

tv2 = ttk.Treeview(wrapper2)
tv2.place(relheight=1,relwidth=1)

treescrolly = tk.Scrollbar(wrapper2,orient="vertical",command=tv2.yview)     
treescrollx = tk.Scrollbar(wrapper2,orient="horizontal",command=tv2.xview)     
tv2.configure(xscrollcommand=treescrollx.set,yscrollcommand=treescrolly.set)
treescrollx.pack(side="bottom",fill="x")
treescrolly.pack(side="right",fill="y")

## Treeview Widget 3

tv3 = ttk.Treeview(null_before_frame)
tv3.place(relheight=1,relwidth=1)

tv4 = ttk.Treeview(null_after_frame)
tv4.place(relheight=1,relwidth=1)


## tvplot

tvplot = ttk.Treeview(plot_frame)
tvplot.place(relheight=1,relwidth=1)

treescrolly = tk.Scrollbar(plot_frame,orient="vertical",command=tvplot.yview)     
treescrollx = tk.Scrollbar(plot_frame,orient="horizontal",command=tvplot.xview)     
tvplot.configure(xscrollcommand=treescrollx.set,yscrollcommand=treescrolly.set)
treescrollx.pack(side="bottom",fill="x")
treescrolly.pack(side="right",fill="y")



















window.mainloop()