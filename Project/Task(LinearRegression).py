# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:20:01 2023

@author: zead shalaby
"""

import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error



class LinearRegressionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Machine Learning Task 1")
        self.geometry("950x550")
        self.train_test_label = ttk.Label(self, text="Train/Test Split Percentage")
        self.train_test_label.pack()

        self.train_test_slider = ttk.Scale(self, from_=0, to=100, orient="horizontal", length=200, command=self.update_slider_value)
        self.train_test_slider.pack()
        self.train_test_slider.place(x=700,y=80)

        self.slider_value_label = ttk.Label(self, text=f"Slider value: {self.train_test_slider.get()}")
        self.slider_value_label.pack()
        self.slider_value_label.place(x=760,y=110)

        self.lambda_label = ttk.Label(self, text="Lambda (Learning Rate)")
        self.lambda_label.pack()
        self.lambda_label.place(x=739,y=165)

        self.lambda_entry = ttk.Entry(self)
        self.lambda_entry.pack()
        self.lambda_entry.place(x=740,y=200)

        self.error_label = ttk.Label(self, text="Error Metric")
        self.error_label.pack()
        self.error_label.place(x=670,y=275)
        
        self.error_var = tk.StringVar()
        self.mae_radio = ttk.Radiobutton(self, text="MAE", variable=self.error_var, value="mae")
        self.mse_radio = ttk.Radiobutton(self, text="MSE", variable=self.error_var, value="mse")
        self.rmse_radio = ttk.Radiobutton(self, text="RMSE", variable=self.error_var, value="rmse")
        self.mae_radio.pack()
        self.mse_radio.pack()
        self.rmse_radio.pack()
        
        self.mae_radio.place(x=740,y=275)
        self.mse_radio.place(x=790,y=275)
        self.rmse_radio.place(x=840,y=275)

        self.run_button = ttk.Button(self, text="Run Regression", command=self.run_regression)
        self.run_button.pack()
        self.run_button.place(x=740,y=370)
        self.train_error_label = ttk.Label(self, text="Train Error: ")
        self.train_error_label.pack()
        self.train_error_label.place(x=755,y=415)
        self.test_error_label = ttk.Label(self, text="Test Error: ")
        self.test_error_label.pack()
        self.test_error_label.place(x=755,y=450)
        
        self.wrapper1 =ttk.LabelFrame(self,text="Exel Data")  
        self.wrapper1.pack(fill="both",expand="yes",padx=40,pady=10 )
        self.wrapper1.place(height=250,width=500,x=60,y=50)


        self.wrapper2 = ttk.LabelFrame(self,text="Description Data")  
        self.wrapper2.pack(fill="both",expand="yes",padx=40,pady=10)
        self.wrapper2.place(height=130,width=500,x=60,y=350)
    
    
        self.tv1 = ttk.Treeview(self.wrapper1)
        self.tv1.place(relheight=1,relwidth=1)
        self.treescrolly = tk.Scrollbar(self.wrapper1,orient="vertical",command=self.tv1.yview)     
        self.treescrollx = tk.Scrollbar(self.wrapper1,orient="horizontal",command=self.tv1.xview)     
        self.tv1.configure(xscrollcommand=self.treescrollx.set,yscrollcommand=self.treescrolly.set)
        self.treescrollx.pack(side="bottom",fill="x")
        self.treescrolly.pack(side="right",fill="y")


        self.tv2 = ttk.Treeview(self.wrapper2)
        self.tv2.place(relheight=1,relwidth=1)
        self.treescrolly = tk.Scrollbar(self.wrapper2,orient="vertical",command=self.tv2.yview)     
        self.treescrollx = tk.Scrollbar(self.wrapper2,orient="horizontal",command=self.tv2.xview)     
        self.tv2.configure(xscrollcommand=self.treescrollx.set,yscrollcommand=self.treescrolly.set)
        self.treescrollx.pack(side="bottom",fill="x")
        self.treescrolly.pack(side="right",fill="y")
    
        
        
    
    
    
    def update_slider_value(self, value):
        self.slider_value_label.config(text=f"Slider value: {value}")
    def run_regression(self):
        # get values 
        train_test_split = self.train_test_slider.get() / 100
        lamda = float(self.lambda_entry.get())
        error_metric = self.error_var.get()
        # read data from CSV file
        file = filedialog.askopenfilename(filetypes=[('CSV files', '*.csv')])
        df = pd.read_csv(file)
       
        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        print(df)
        self.tv1["column"] = list(df.columns)
        self.tv1["show"]  = "headings"
        for column in self.tv1["column"]:
            self.tv1.heading(column, text=column)
        self.df_rows = df.to_numpy().tolist()
        for row in self.df_rows:
            self.tv1.insert("","end", values=row) 
            
        self.tv2["column"] = list(df.describe().columns)
        self.tv2["show"]  = "headings"
        for column in self.tv2["column"]:
            self.tv2.heading(column, text=column)
        self.df_rows = df.to_numpy().tolist()
        for row in self.df_rows:
            self.tv2.insert("","end", values=row)     
            
        #split data into train and test sets
        split_idx = int(len(x) * train_test_split)
        x_train, y_train = x[:split_idx], y[:split_idx]
        x_test, y_test = x[split_idx:], y[split_idx:]

        model = LinearRegression()
        model.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))

        #predict on train and test data
        y_train_pred = model.predict(x_train.reshape(-1, 1)).flatten()
        y_test_pred = model.predict(x_test.reshape(-1, 1)).flatten()
        #ERRORS
        if error_metric == "mae":
            train_error = mean_absolute_error(y_train, y_train_pred)
            test_error = mean_absolute_error(y_test, y_test_pred)
        elif error_metric == "mse":
            train_error = mean_squared_error(y_train, y_train_pred)
            test_error = mean_squared_error(y_test, y_test_pred)
        elif error_metric == "rmse":
            train_error = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_error = np.sqrt(mean_squared_error(y_test, y_test_pred))

        #update train and test errors
        self.train_error_label.config(text=f"Train Error: {train_error:.2f}")
        self.test_error_label.config(text=f"Test Error: {test_error:.2f}")
        
        
        
        
       
            
        
            
    def clear_data(self):
        self.tv1.delete(*self.tv1.get_children())
        
        
if __name__ == "__main__":
    app = LinearRegressionApp()
    app.mainloop()
