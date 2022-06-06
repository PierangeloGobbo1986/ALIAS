import pandas
import numpy as np
from datetime import datetime
import glob
import os
import os.path
import shutil
import matplotlib.pyplot as plt
from matplotlib import cm
from tkinter import *
from tkinter import scrolledtext
from tkinter import filedialog
from tkinter.filedialog import asksaveasfilename
from sklearn import linear_model
from scipy import optimize
import math
import warnings
#----------------------------------------------------------------------------------------------------------------------#
def Spacer():
    txt.insert(END, "-"*134)
    txt.insert(END, "\n")
    txt.update()
    txt.see("end")

def Create_Output_Folder(new_folder_name):
    try:
        os.mkdir(new_folder_name) # Make new directory

    # Error handling to avoid stopping code
    except OSError:
        txt.insert(END, "Error: folder {} already present.\n".format(new_folder_name))
        txt.update()
        txt.see("end")
        Spacer()
    else:
        txt.insert(END, "Directory {} successfully created.\n".format(new_folder_name))
        txt.update()
        txt.see("end")
        Spacer()

def Find_Experiments(path_filename):
    # Save txt files in new folder with the name of the txt file
    path, filename = os.path.split(path_filename)
    filename_noext, extension = os.path.splitext(filename)
    folder_name = filename_noext
    folder_path = "{}/{}".format(path, folder_name)
    Create_Output_Folder(folder_path) #This is for the experiment


    inputfile = open(path_filename, 'r', encoding='gbk')
    txt.insert(END, "Working directory:\n{}\n\n".format(path))
    txt.insert(END, "File opened: {}\n\n".format(filename))
    txt.update()
    txt.see("end")
    Spacer()

    fileno = -1
    outfile = open(f"{folder_path}/{filename_noext}_{fileno}.txt", "w") #This is the new f-string method of Python 3.6
    txt.insert(END, f"File successfully created: {filename_noext}_{fileno}.txt\n\n")
    txt.update()
    txt.see("end")
    for line in inputfile:
        if not line.strip():
            fileno += 1
            outfile.close()
            outfile = open(f"{folder_path}/{filename_noext}_{fileno}.txt", "w")
            txt.insert(END, f"File successfully created: {filename_noext}_{fileno}.txt\n\n")
            txt.update()
            txt.see("end")
        else:
            outfile.write(line)
    outfile.close()
    inputfile.close()

    txt.insert(END, "Individual measurements successfully detected.\n")
    txt.insert(END, "Detected {} measurements.\n\n".format(fileno+1))
    txt.update()
    txt.see("end")
    return path, filename, filename_noext, extension, folder_path

def Divide_Curve_And_Plot(path_filename):
    path, filename = os.path.split(path_filename)
    filename_noext, extension = os.path.splitext(filename)

    df = pandas.read_table(path_filename, low_memory=False, delim_whitespace=True, names=("Index [#]", "Phase [#]", "Displacement [um]", "Time [s]", "Pos X [um]", "Pos Y [um]", "Pos Z [um]", "Piezo X [um]", "Piezo Y [um]", "Piezo Z [um]", "Force A [uN]", "Force B [uN]", "Gripper [V]", "Voltage A [V]", "Voltage B [V]", "Temperature [oC]"))
    # Clean up txt file to save a proper cvs file
    df = df[~df["Index [#]"].isin(['//'])]  # Drop the rows that contain the comments to keep only the numbers
    df = df.dropna(how='all')  # to drop if all values in the row are nan
    df = df.astype(float)  # Change data from object to float

    # Determine if measurement is done with piezo or step motor
    PosZ_um_col = df.columns.get_loc("Pos Z [um]")  # Get column number of "Pos Z [um]"
    delta_stepmotor = df.iloc[2, PosZ_um_col] - df.iloc[3, PosZ_um_col]

    if delta_stepmotor < 0.005:  # This value was optimised based on the dataset available so far
        txt.insert(END, "Measurement carried out using piezo scanner.\n")
        txt.update()
        txt.see("end")
        # print(delta_stepmotor)
        phase = []
        piezoZ = df["Piezo Z [um]"].to_list()  # Use PiezoZ for determining phase number
        phase_num = 1
        for ind, val in enumerate(piezoZ):  # enumerate loops over list (val) and have an automatic counter (ind).
            try:
                if piezoZ[ind + 1] - piezoZ[ind] >= 0:
                    phase.append(phase_num)
                    phase_num = 1
                else:
                    phase.append(phase_num)
                    phase_num = 0
            except:
                phase.append(phase_num)  # Fixes the last loop where there is not a num+1 to subtract

    else:
        txt.insert(END, "Measurement carried out using stick-slip actuator.\n")
        txt.update()
        txt.see("end")
        # print(delta_stepmotor)
        phase = []
        posZ = df["Pos Z [um]"].to_list()  #Use PosZ for determining phase number
        phase_num = 1
        for ind, val in enumerate(posZ):  # enumerate loops over list (val) and have an automatic counter (ind).
            try:
                if posZ[ind + 1] - posZ[ind] <= 0:
                    phase.append(phase_num)
                    phase_num = 1
                else:
                    phase.append(phase_num)
                    phase_num = 0
            except:
                phase.append(phase_num)

    df["Exp Phase [#]"] = phase  # converts Displacement column into a list and applies function Cycle
    df.to_csv(f"{path}/{filename_noext}.csv", index=False)  # Save all measurements as *.csv, index=False avoids saving index
    txt.insert(END, "Single experiment CSV file saved in working directory: {}.csv\n\n".format(filename_noext))
    txt.update()
    txt.see("end")

    #Plot all measurements
    sub_group = df.groupby(["Exp Phase [#]"]) #Subgroups are 0 (loading) and 1 (unloading)
    #print(sub_group.get_group(0))
    #print(sub_group.get_group(1))
    #print(sub_group)

    #Transform the column subsections in lists for plotting
    Displacement_in = [] #reset lists
    Force_in = []
    Displacement_out = []
    Force_out = []
    Displacement_in = sub_group.get_group(1)["Displacement [um]"].to_list()
    Force_in = sub_group.get_group(1)["Force A [uN]"].to_list()
    try: #necessary if measurement was interrupted before unloading scan - i.e. there is no unloading scan
        Displacement_out = sub_group.get_group(0)["Displacement [um]"].to_list()
        Force_out = sub_group.get_group(0)["Force A [uN]"].to_list()
    except:
        txt.insert(END, f"No unloading scan detected in file: {filename}\n")
        txt.update()
        txt.see("end")
        pass

    plt.figure() #It creates a new figure every time, necessary otherwise it keeps adding to the same plot!
    plt.scatter(Displacement_in, Force_in, s=10, marker="o", color='white', edgecolors='steelblue', label='Loading data')
    try:
        plt.scatter(Displacement_out, Force_out, s=10, marker="o", color='white', edgecolors='orange', label='Unloading data')
    except:
        txt.insert(END, f"No unloading scan detected in file: {filename}\n")
        txt.update()
        txt.see("end")
        pass

    plt.title("{}".format(filename), fontsize=20)
    plt.xlabel("Distance (um)", fontsize=18)
    plt.ylabel("Force (uN)", fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tick_params(direction='in', length=8)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{path}/{filename_noext}.png", bbox_inches='tight', dpi=300)
    plt.close('all') #To close all figures and save memory - max # of figures before warning: 20

def Unit_Transform(df,R):
    #Transform R to m and Pa
    R = R*1e-6

    #Transform force units to N
    df["Force A [N]"] = df["Force A [uN]"]*1e-6 #Force A [N] = Col19
    ForceA_N_col = df.columns.get_loc("Force A [N]") #Get column number
    #print(df[["Phase [#]", "Displacement [um]","Force A [uN]","Displacement [m]","Force A [N]"]]) #Print desired columns

    #Transform time in time from the beginning of the measurement
    Time_s_col = df.columns.get_loc("Time [s]") #Get column number
    df["Corr Time [s]"] = df["Time [s]"]-df.iloc[0,Time_s_col] #Force A [N] = Col10
    Corr_Time_s_col = df.columns.get_loc("Corr Time [s]") #Get column number

    #Transform Displacement, Piezo Z and Pos Z units to m and make them start from 0
    Displacement_um_col = df.columns.get_loc("Displacement [um]") #Get column number
    df["Displacement [m]"] = (df["Displacement [um]"]-df.iloc[0,Displacement_um_col])*1e-6 #Displacement [m] = Col18
    Displacement_m_col = df.columns.get_loc("Displacement [m]") #Get column number
    PiezoZ_um_col = df.columns.get_loc("Piezo Z [um]") #Get column number
    df["Piezo Z [m]"] = (df["Piezo Z [um]"]-df.iloc[0,PiezoZ_um_col])*1e-6 #Transform PiezoZ in m and make it start from 0.
    PiezoZ_m_col = df.columns.get_loc("Piezo Z [m]") #Piezo Z[m] = Col21
    PosZ_um_col = df.columns.get_loc("Pos Z [um]") #Get column number
    df["Pos Z [m]"] = (-df["Pos Z [um]"]+df.iloc[0,PosZ_um_col])*1e-6 #Piezo Z[m] = Col22
    PosZ_m_col = df.columns.get_loc("Pos Z [m]") #Get column number

    return R, ForceA_N_col, Time_s_col, Corr_Time_s_col, Displacement_um_col, Displacement_m_col, PiezoZ_um_col, PiezoZ_m_col, PosZ_um_col, PosZ_m_col

def Contact_Point(df, PosZ_um_col, PosZ_m_col, PiezoZ_m_col,Displacement_m_col, ratio_BL_points, threshold_constant):

    delta_stepmotor = df.iloc[2, PosZ_um_col] - df.iloc[3, PosZ_um_col]
    # print("Delta_stepmotor: {} um".format(delta_stepmotor))

    if delta_stepmotor < 0.005:  # This value was optimised based on the dataset available so far
        Zmov_type = "Piezo scanner"
        txt.insert(END, "Measurement carried out using piezo scanner.\n\n")

        # Get real displacement as (piezoZ [m] - piezoZ [m]0) - (Displacement [m] - Displacement [m]0)
        df["Probe Displacement [m]"] = ((df["Piezo Z [m]"]) - (df.iloc[0, PiezoZ_m_col])) - ((df["Displacement [m]"]) - df.iloc[0, Displacement_m_col])
        Probe_Displacement_m_col = df.columns.get_loc("Probe Displacement [m]")

        # Get contact points for both scans
        forward_scan = pandas.DataFrame(df.groupby(["Exp Phase [#]"]).get_group(1.0))
        # Find mean and standard deviation of first X datapoints of indentation
        forward_scan_BL_mean = forward_scan.iloc[0:int(forward_scan.shape[0] * ratio_BL_points), Probe_Displacement_m_col].mean()  # Baseline on first 1/8 datapoints
        forward_scan_BL_std = forward_scan.iloc[0:int(forward_scan.shape[0] * ratio_BL_points), Probe_Displacement_m_col].std()
        txt.insert(END, f"Probe displacement baseline mean and std: {round(forward_scan_BL_mean, 15)} +/- {round(forward_scan_BL_std, 15)}\n")
        threshold = forward_scan_BL_mean - (forward_scan_BL_std * threshold_constant)  # lower threshold to determine level of noise, 20 was determined empyrically on data available
        txt.insert(END, f"Baseline threshold: {round(threshold, 15)} (threshold constant = {threshold_constant})\n")
        # Find point of minimum
        forward_scan_index_min = forward_scan["Probe Displacement [m]"].idxmin()
        txt.insert(END, "Index of point of contact forward scan: {}\n".format(forward_scan_index_min))
        forward_scan_min = [forward_scan.iloc[forward_scan_index_min, Displacement_m_col], forward_scan.iloc[forward_scan_index_min, Probe_Displacement_m_col]]
        txt.insert(END, f"Point of contact [Displacement, Indentation]: [{round(forward_scan_min[0], 15)}, {round(forward_scan_min[1], 15)}]\n")
        txt.update()
        txt.see("end")
        # If min point is within baseline error => there is no min
        if threshold < forward_scan.iloc[forward_scan_index_min, Probe_Displacement_m_col]:
            txt.insert(END, "No interaction with material detected.\n")
            selection = forward_scan.loc[forward_scan["Probe Displacement [m]"] >= abs(threshold)]
            min_indentation = selection.iloc[0, Probe_Displacement_m_col]
            forward_scan_index_min = forward_scan.loc[forward_scan["Probe Displacement [m]"] == min_indentation].index[0]
            txt.insert(END, "New index of point of contact forward scan: {}\n".format(forward_scan_index_min))
            forward_scan_min = [forward_scan.iloc[forward_scan_index_min, Displacement_m_col], forward_scan.iloc[forward_scan_index_min, Probe_Displacement_m_col]]
            txt.insert(END, f"Point of contact [Displacement, Indentation]: [{round(forward_scan_min[0], 15)}, {round(forward_scan_min[1], 15)}]\n")

        back_scan = pandas.DataFrame(df.groupby(["Exp Phase [#]"]).get_group(0.0))
        back_scan = back_scan.iloc[::-1].reset_index(drop=True)  # Reverse and reindex dataframe
        back_scan_index_min = back_scan["Probe Displacement [m]"].idxmin()
        txt.insert(END, "Index of point of detachment back scan: {}\n".format(back_scan_index_min))
        back_scan_min = [back_scan.iloc[back_scan_index_min, Displacement_m_col], back_scan.iloc[back_scan_index_min, Probe_Displacement_m_col]]
        txt.insert(END, f"Point of contact [Displacement, Indentation]: [{round(back_scan_min[0], 15)}, {round(back_scan_min[1], 15)}]\n")
        txt.update()
        txt.see("end")
        # If min point is within baseline error => there is no min
        if threshold < back_scan.iloc[back_scan_index_min, Probe_Displacement_m_col]:
            txt.insert(END, "No interaction with material detected.\n")
            selection = back_scan.loc[back_scan["Probe Displacement [m]"] >= abs(threshold)]
            min_indentation = selection.iloc[0, Probe_Displacement_m_col]
            back_scan_index_min = back_scan.loc[back_scan["Probe Displacement [m]"] == min_indentation].index[0]
            txt.insert(END, "New index of point of contact back scan: {}\n".format(back_scan_index_min))
            back_scan_min = [back_scan.iloc[back_scan_index_min, Displacement_m_col], back_scan.iloc[back_scan_index_min, Probe_Displacement_m_col]]
            txt.insert(END, f"Point of contact [Displacement, Indentation]: [{round(back_scan_min[0], 15)}, {round(back_scan_min[1], 15)}]\n")

    else:
        Zmov_type = "Stick-slip actuator"
        txt.insert(END, "Measurement carried out using stick-slip actuator.\n\n")

        # Get real displacement as (Pos Z [m] + Pos Z [m]0) - (Displacement [m] - Displacement [m]0)
        df["Probe Displacement [m]"] = ((df["Pos Z [m]"]) - (df.iloc[0, PosZ_m_col])) - ((df["Displacement [m]"]) - df.iloc[0, Displacement_m_col])
        Probe_Displacement_m_col = df.columns.get_loc("Probe Displacement [m]")

        forward_scan = pandas.DataFrame(df.groupby(["Exp Phase [#]"]).get_group(1.0))
        forward_scan_BL_mean = forward_scan.iloc[0:int(forward_scan.shape[0] * ratio_BL_points),Probe_Displacement_m_col].mean()  # Baseline on first 1/8 datapoints
        forward_scan_BL_std = forward_scan.iloc[0:int(forward_scan.shape[0] * ratio_BL_points), Probe_Displacement_m_col].std()
        txt.insert(END, f"Probe displacement baseline mean and std: {round(forward_scan_BL_mean, 15)} +/- {round(forward_scan_BL_std, 15)}\n")
        threshold = forward_scan_BL_mean - (forward_scan_BL_std * threshold_constant)  # lower threshold to determine level of noise, 20 was determined empyrically on data available
        txt.insert(END, f"Baseline threshold: {round(threshold, 15)} (threshold constant = {threshold_constant})\n")
        forward_scan_index_min = forward_scan["Probe Displacement [m]"].idxmin()
        txt.insert(END, "Index of point of contact forward scan: {}\n".format(forward_scan_index_min))
        forward_scan_min = [forward_scan.iloc[forward_scan_index_min, Displacement_m_col], forward_scan.iloc[forward_scan_index_min, Probe_Displacement_m_col]]
        txt.insert(END, f"Point of contact [Displacement, Indentation]: [{round(forward_scan_min[0], 15)}, {round(forward_scan_min[1], 15)}]\n")
        txt.update()
        txt.see("end")
        if threshold < forward_scan.iloc[forward_scan_index_min, Probe_Displacement_m_col]:
            txt.insert(END, "No interaction with material detected.\n")
            selection = forward_scan.loc[forward_scan["Probe Displacement [m]"] >= abs(threshold)]
            min_indentation = selection.iloc[0, Probe_Displacement_m_col]
            forward_scan_index_min = forward_scan.loc[forward_scan["Probe Displacement [m]"] == min_indentation].index[0]
            txt.insert(END, "New index of point of contact forward scan: {}\n".format(forward_scan_index_min))
            forward_scan_min = [forward_scan.iloc[forward_scan_index_min, Displacement_m_col], forward_scan.iloc[forward_scan_index_min, Probe_Displacement_m_col]]
            txt.insert(END, f"Point of contact [Displacement, Indentation]: [{round(forward_scan_min[0], 15)}, {round(forward_scan_min[1], 15)}]\n")

        back_scan = pandas.DataFrame(df.groupby(["Exp Phase [#]"]).get_group(0.0))
        back_scan = back_scan.iloc[::-1].reset_index(drop=True)  # Reverse and reindex dataframe
        back_scan_index_min = back_scan["Probe Displacement [m]"].idxmin()
        txt.insert(END, "Index of point of detachment back scan: {}\n".format(back_scan_index_min))
        back_scan_min = [back_scan.iloc[back_scan_index_min, Displacement_m_col], back_scan.iloc[back_scan_index_min, Probe_Displacement_m_col]]
        txt.insert(END, f"Point of contact [Displacement, Indentation]: [{round(back_scan_min[0], 15)}, {round(back_scan_min[1], 15)}]\n")
        txt.update()
        txt.see("end")
        # If min point is within baseline error => there is no min
        if threshold < back_scan.iloc[back_scan_index_min, Probe_Displacement_m_col]:
            txt.insert(END, "No interaction with material detected.\n")
            selection = back_scan.loc[back_scan["Probe Displacement [m]"] >= abs(threshold)]
            min_indentation = selection.iloc[0, Probe_Displacement_m_col]
            back_scan_index_min = back_scan.loc[back_scan["Probe Displacement [m]"] == min_indentation].index[0]
            txt.insert(END, "New index of point of contact back scan: {}\n".format(back_scan_index_min))
            back_scan_min = [back_scan.iloc[back_scan_index_min, Displacement_m_col], back_scan.iloc[back_scan_index_min, Probe_Displacement_m_col]]
            txt.insert(END, f"Point of contact [Displacement, Indentation]: [{round(back_scan_min[0], 15)}, {round(back_scan_min[1], 15)}]\n")

    txt.insert(END, "\n")
    txt.update()
    txt.see("end")
    return Zmov_type, Probe_Displacement_m_col, forward_scan, back_scan, forward_scan_index_min, forward_scan_min, back_scan_index_min, back_scan_min

def Indentation_Speed(Zmov_type, forward_scan):
    if Zmov_type == "Piezo scanner":
        x_bl = pandas.DataFrame(forward_scan["Corr Time [s]"])
        y_bl = pandas.DataFrame(forward_scan["Piezo Z [m]"])
        regr = linear_model.LinearRegression()  # Check that x adn y have a shape (n, 1)
        regr.fit(x_bl, y_bl)
        indentation_speed = (regr.coef_[0][0])*1e6
    if Zmov_type == "Stick-slip actuator":
        x_bl = pandas.DataFrame(forward_scan["Corr Time [s]"])
        y_bl = pandas.DataFrame(forward_scan["Pos Z [m]"])
        regr = linear_model.LinearRegression()  # Check that x adn y have a shape (n, 1)
        regr.fit(x_bl, y_bl)
        indentation_speed = (regr.coef_[0][0])*1e6
    return indentation_speed

def Baseline_And_Zero(df,forward_scan,forward_scan_index_min,Displacement_m_col,ratio_BL_points):
    #Set indentation with contact point at 0
    df["Indentation [m]"] = df["Displacement [m]"]-forward_scan.iloc[forward_scan_index_min, Displacement_m_col] #Corr Displacement [m] = Col23
    Indentation_m_col = df.columns.get_loc("Indentation [m]")

    numpoint_baseline, _ = df.shape
    #print(numpoint_baseline, y)
    numpoint_baseline = round(numpoint_baseline*ratio_BL_points) #Determine number of points for baseline is 1/8 of datapoints of df
    #Generate baseline dataframe to fit
    baseline = pandas.DataFrame(df.iloc[0:numpoint_baseline])
    #print(baseline)
    x_bl = pandas.DataFrame(baseline["Indentation [m]"])
    y_bl = pandas.DataFrame(baseline["Force A [N]"])
    #print(x_bl.shape)
    #print(y_bl.shape)
    regr = linear_model.LinearRegression() #Check that x adn y have a shape (n, 1)
    regr.fit(x_bl, y_bl)
    #print(regr.coef_[0][0])
    #print(regr.intercept_[0])
    df["Corr Force A [N]"] = df["Force A [N]"]-(regr.intercept_[0]+(regr.coef_[0][0]*df["Indentation [m]"])) #y=a+bx; add new column to df with corrected force values; #Corr Force A [N] = Col20
    CorrForceA_N_col = df.columns.get_loc("Corr Force A [N]")
    #print(df[["Phase [#]", "Displacement [um]","Force A [uN]","Displacement [m]","Force A [N]", "Corr Force A [N]"]]) #Print desired columns

    #Plot graph
    #Add baseline values to df. These are calculated only for loading curve otherwise you get a double line (2 lines overimposed)
    df["Baseline F [N]"] = regr.intercept_[0]+(regr.coef_[0][0]*df.groupby(["Exp Phase [#]"]).get_group(1.0)["Indentation [m]"]) #Baseline F [N] = Col21
    BaselineF_N_col = df.columns.get_loc("Baseline F [N]")

    return baseline, CorrForceA_N_col, BaselineF_N_col, Indentation_m_col

def Max_Indent_And_Max_Force(forward_scan, Indentation_m_col, CorrForceA_N_col):
    #Get Max_Force and Max_Indentation based on last value of the two corresponding columns
    Max_Indent = forward_scan.iloc[-1, Indentation_m_col]
    Max_Force = forward_scan.iloc[-1, CorrForceA_N_col]
    return Max_Indent, Max_Force

def GetXY(df, Zmov_type):
    if Zmov_type == "Piezo scanner":
        PiezoX_um_col = df.columns.get_loc("Piezo X [um]") #Get column number
        PiezoY_um_col = df.columns.get_loc("Piezo Y [um]") #Get column number
        X = df.iloc[0, PiezoX_um_col]
        Y = df.iloc[0, PiezoY_um_col]

    if Zmov_type == "Stick-slip actuator":
        PosX_um_col = df.columns.get_loc("Pos X [um]") #Get column number
        PosY_um_col = df.columns.get_loc("Pos Y [um]") #Get column number
        X = df.iloc[0, PosX_um_col]
        Y = df.iloc[0, PosY_um_col]

    return X, Y


def Hertz_Fitting(df, forward_scan_segment, back_scan_segment, R, forward_scan_index_min, back_scan_index_min, Indentation_m_col, CorrForceA_N_col):

    def Hertz(R, K, x):
        a = ((R / K) * (x)) ** (1 / 3)
        return ((a ** 2) / R)

    #Select starting point for fitting, get Fad, and calculate gamma
    forward_scan = pandas.DataFrame(df.groupby(["Exp Phase [#]"]).get_group(1.0))
    forward_scan = forward_scan.loc[forward_scan.index >= forward_scan_index_min] #Select data for fitting based on contact point
    forward_scan_contactP = forward_scan.iloc[0, Indentation_m_col]*1e6

    # Make curve start from zero indentation and zero force - required by Hertzian model
    forward_scan["Corr Indentation Hertz [m]"] = forward_scan["Indentation [m]"] - forward_scan.iloc[0, Indentation_m_col]
    forward_scan["Corr Force A Hertz [N]"] = forward_scan["Corr Force A [N]"] - forward_scan.iloc[0, CorrForceA_N_col]

    #Fit the entire indentation curve
    forward_scan_x_full_curve = forward_scan["Corr Force A Hertz [N]"].to_list()
    forward_scan_y_full_curve = forward_scan["Corr Indentation Hertz [m]"].to_list()
    custom_Hertz = lambda x, K: Hertz(R, K, x)  # Fix R value
    forward_scan_pars_Hertz_full_curve, forward_scan_cov_Hertz_full_curve = optimize.curve_fit(f=custom_Hertz, xdata=forward_scan_x_full_curve, ydata=forward_scan_y_full_curve)
    forward_scan_cov_Hertz_full_curve = np.sqrt(np.diag(forward_scan_cov_Hertz_full_curve)) #Necessary to convert error matrix output of optimize.curve_fit into list of standard deviations for each parameter
    forward_scan_bulk_modulus_Hertz_full_curve = forward_scan_pars_Hertz_full_curve[0]
    forward_scan_bulk_modulus_error_Hertz_full_curve = forward_scan_cov_Hertz_full_curve[0]
    forward_scan["Indentation Hertz Full Curve [m]"] = forward_scan["Corr Indentation Hertz [m]"]
    forward_scan["Corr Force A Hertz Full Curve [N]"] = forward_scan["Corr Force A Hertz [N]"]
    #Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    forward_scan["Fitting Indentation Hertz Full Curve [m]"] = ((forward_scan["Corr Force A Hertz Full Curve [N]"]/forward_scan_bulk_modulus_Hertz_full_curve)**(2/3))*(1/(R**(1/3)))

    # Create test dataframe for the while loop and cut it to the first set of datapoints
    forward_scan_initial_len = len(forward_scan)
    forward_scan_datapoints_to_fit = forward_scan_segment
    forward_scan_test = forward_scan.loc[forward_scan.index < (forward_scan.index[0] + forward_scan_datapoints_to_fit)]  # Select data for fitting based on threshold value, index was not reset so forward_scan.index[-1] is a high number
    forward_scan_bulk_modulus_Hertz = []
    forward_scan_bulk_modulus_error_Hertz = []
    forward_scan_indentation_Hertz = []
    forward_scan_datapoints_Hertz = []

    while forward_scan_datapoints_to_fit <= forward_scan_initial_len:
        #Create test dataframe for the while loop and cut it to the first set of datapoints
        forward_scan_x = forward_scan_test["Corr Force A Hertz [N]"].to_list()
        forward_scan_y = forward_scan_test["Corr Indentation Hertz [m]"].to_list()

        #Hertz fitting on selected data
        custom_Hertz = lambda x, K: Hertz(R, K, x)  # Fix R value
        forward_scan_pars_Hertz, forward_scan_cov_Hertz = optimize.curve_fit(f=custom_Hertz, xdata=forward_scan_x, ydata=forward_scan_y)
        forward_scan_cov_Hertz = np.sqrt(np.diag(forward_scan_cov_Hertz)) #Necessary to convert error matrix output of optimize.curve_fit into list of standard deviations for each parameter

        forward_scan_bulk_modulus_Hertz.append(forward_scan_pars_Hertz[0])
        forward_scan_bulk_modulus_error_Hertz.append(forward_scan_cov_Hertz[0])
        forward_scan_indentation_Hertz.append(forward_scan_y[-1])
        forward_scan_datapoints_Hertz.append(forward_scan_datapoints_to_fit)

        forward_scan_datapoints_to_fit = forward_scan_datapoints_to_fit + forward_scan_segment
        forward_scan_test = forward_scan.loc[forward_scan.index < (forward_scan.index[0] + forward_scan_datapoints_to_fit)]  # Select data for fitting based on threshold value, index was not reset so forward_scan.index[-1] is a high number

    #Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    forward_scan_index_smallest_error = forward_scan_bulk_modulus_error_Hertz.index(min(forward_scan_bulk_modulus_error_Hertz))
    forward_scan_cov_Hertz_best = forward_scan_bulk_modulus_error_Hertz[forward_scan_index_smallest_error]
    forward_scan_pars_Hertz_best = forward_scan_bulk_modulus_Hertz[forward_scan_index_smallest_error]
    #Redeclare test df with best result and append Indentation and Corr Force columns to forward_scan for storage
    forward_scan_test = forward_scan.loc[forward_scan.index < (forward_scan.index[0] + forward_scan_datapoints_Hertz[forward_scan_index_smallest_error])]
    forward_scan["Selected Indentation Hertz [m]"] = forward_scan_test["Indentation [m]"]
    forward_scan["Selected Corr Force A Hertz [N]"] = forward_scan_test["Corr Force A [N]"]
    forward_scan["Indentation Hertz [m]"] = forward_scan_test["Corr Indentation Hertz [m]"]
    forward_scan["Corr Force A Hertz [N]"] = forward_scan_test["Corr Force A Hertz [N]"]
    forward_scan["Fitting Indentation Hertz [m]"] = ((forward_scan["Corr Force A Hertz [N]"]/forward_scan_pars_Hertz_best)**(2/3))*(1/(R**(1/3)))

    # Extend lists with stored data to match the lenght of forward_scan df otherwise they cannot be added as columns to the df
    forward_scan_indentation_Hertz.extend(["NaN"] * (forward_scan_initial_len - len(forward_scan_indentation_Hertz)))
    forward_scan["Indentation step Hertz [m]"] = forward_scan_indentation_Hertz
    forward_scan_bulk_modulus_Hertz.extend(["NaN"] * (forward_scan_initial_len - len(forward_scan_bulk_modulus_Hertz)))
    forward_scan["K Hertz [Pa]"] = forward_scan_bulk_modulus_Hertz
    # Transform error into logarithm for best plots
    forward_scan_bulk_modulus_error_Hertz = [math.log(x) for x in forward_scan_bulk_modulus_error_Hertz]
    forward_scan_bulk_modulus_error_Hertz.extend(["NaN"] * (forward_scan_initial_len - len(forward_scan_bulk_modulus_error_Hertz)))
    forward_scan["ln (K Error Hertz)"] = forward_scan_bulk_modulus_error_Hertz



    back_scan = pandas.DataFrame(df.groupby(["Exp Phase [#]"]).get_group(0.0))
    back_scan = back_scan.iloc[::-1].reset_index(drop=True) #Reverse and reindex dataframe
    back_scan = back_scan.loc[back_scan.index >= back_scan_index_min] #Select data for fitting based on threshold value
    back_scan_contactP = back_scan.iloc[0, Indentation_m_col]*1e6

    # Make curve start from zero indentation and zero force - required by Hertzian model
    back_scan["Corr Indentation Hertz [m]"] = back_scan["Indentation [m]"] - back_scan.iloc[0, Indentation_m_col]
    back_scan["Corr Force A Hertz [N]"] = back_scan["Corr Force A [N]"] - back_scan.iloc[0, CorrForceA_N_col]

    #Fit the entire indentation curve
    back_scan_x_full_curve = back_scan["Corr Force A Hertz [N]"].to_list()
    back_scan_y_full_curve = back_scan["Corr Indentation Hertz [m]"].to_list()
    custom_Hertz = lambda x, K: Hertz(R, K, x)  # Fix R value
    back_scan_pars_Hertz_full_curve, back_scan_cov_Hertz_full_curve = optimize.curve_fit(f=custom_Hertz, xdata=back_scan_x_full_curve, ydata=back_scan_y_full_curve)
    back_scan_cov_Hertz_full_curve = np.sqrt(np.diag(back_scan_cov_Hertz_full_curve)) #Necessary to convert error matrix output of optimize.curve_fit into list of standard deviations for each parameter
    back_scan_bulk_modulus_Hertz_full_curve = back_scan_pars_Hertz_full_curve[0]
    back_scan_bulk_modulus_error_Hertz_full_curve = back_scan_cov_Hertz_full_curve[0]
    back_scan["Indentation Hertz Full Curve [m]"] = back_scan["Corr Indentation Hertz [m]"]
    back_scan["Corr Force A Hertz Full Curve [N]"] = back_scan["Corr Force A Hertz [N]"]
    #Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    back_scan["Fitting Indentation Hertz Full Curve [m]"] = ((back_scan["Corr Force A Hertz Full Curve [N]"]/back_scan_bulk_modulus_Hertz_full_curve)**(2/3))*(1/(R**(1/3)))

    # Create test dataframe for the while loop and cut it to the first set of datapoints
    back_scan_initial_len = len(back_scan)
    back_scan_datapoints_to_fit = back_scan_segment
    back_scan_test = back_scan.loc[back_scan.index < (back_scan.index[0] + back_scan_datapoints_to_fit)]  # Select data for fitting based on threshold value, index was not reset so forward_scan.index[-1] is a high number
    back_scan_bulk_modulus_Hertz = []
    back_scan_bulk_modulus_error_Hertz = []
    back_scan_indentation_Hertz = []
    back_scan_datapoints_Hertz = []

    while back_scan_datapoints_to_fit <= back_scan_initial_len:
        # Create test dataframe for the while loop and cut it to the first set of datapoints
        back_scan_x = back_scan_test["Corr Force A Hertz [N]"].to_list()
        back_scan_y = back_scan_test["Corr Indentation Hertz [m]"].to_list()

        # Hertz fitting on selected data
        custom_Hertz = lambda x, K: Hertz(R, K, x)  # Fix R value
        back_scan_pars_Hertz, back_scan_cov_Hertz = optimize.curve_fit(f=custom_Hertz, xdata=back_scan_x, ydata=back_scan_y)
        back_scan_cov_Hertz = np.sqrt(np.diag(back_scan_cov_Hertz)) #Necessary to convert error matrix output of optimize.curve_fit into list of standard deviations for each parameter

        back_scan_bulk_modulus_Hertz.append(back_scan_pars_Hertz[0])
        back_scan_bulk_modulus_error_Hertz.append(back_scan_cov_Hertz[0])
        back_scan_indentation_Hertz.append(back_scan_y[-1])
        back_scan_datapoints_Hertz.append(back_scan_datapoints_to_fit)

        back_scan_datapoints_to_fit = back_scan_datapoints_to_fit + back_scan_segment
        back_scan_test = back_scan.loc[back_scan.index < (back_scan.index[0] + back_scan_datapoints_to_fit)]  # Select data for fitting based on threshold value, index was not reset so forward_scan.index[-1] is a high number

    #Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    back_scan_index_smallest_error = back_scan_bulk_modulus_error_Hertz.index(min(back_scan_bulk_modulus_error_Hertz))
    back_scan_cov_Hertz_best = back_scan_bulk_modulus_error_Hertz[back_scan_index_smallest_error]
    back_scan_pars_Hertz_best = back_scan_bulk_modulus_Hertz[back_scan_index_smallest_error]
    #Redeclare test df with best result and append Indentation and Corr Force columns to forward_scan for storage
    back_scan_test = back_scan.loc[back_scan.index < (back_scan.index[0] + back_scan_datapoints_Hertz[back_scan_index_smallest_error])]
    back_scan["Selected Indentation Hertz [m]"] = back_scan_test["Indentation [m]"]
    back_scan["Selected Corr Force A Hertz [N]"] = back_scan_test["Corr Force A [N]"]
    back_scan["Indentation Hertz [m]"] = back_scan_test["Corr Indentation Hertz [m]"]
    back_scan["Corr Force A Hertz [N]"] = back_scan_test["Corr Force A Hertz [N]"]
    back_scan["Fitting Indentation Hertz [m]"] = ((back_scan["Corr Force A Hertz [N]"]/back_scan_pars_Hertz_best)**(2/3))*(1/(R**(1/3)))

    # Extend lists with stored data to match the lenght of forward_scan df otherwise they cannot be added as columns to the df
    back_scan_indentation_Hertz.extend(["NaN"] * (back_scan_initial_len - len(back_scan_indentation_Hertz)))
    back_scan["Indentation step Hertz [m]"] = back_scan_indentation_Hertz
    back_scan_bulk_modulus_Hertz.extend(["NaN"] * (back_scan_initial_len - len(back_scan_bulk_modulus_Hertz)))
    back_scan["K Hertz [Pa]"] = back_scan_bulk_modulus_Hertz
    # Transform error into logarithm for best plots
    back_scan_bulk_modulus_error_Hertz = [math.log(x) for x in back_scan_bulk_modulus_error_Hertz]
    back_scan_bulk_modulus_error_Hertz.extend(["NaN"] * (back_scan_initial_len - len(back_scan_bulk_modulus_error_Hertz)))
    back_scan["ln (K Error Hertz)"] = back_scan_bulk_modulus_error_Hertz

    # Define time
    time_s_col = df.columns.get_loc("Time [s]")  # Get column number
    time_s = df.iloc[0, time_s_col]

    return forward_scan, forward_scan_contactP, back_scan, back_scan_contactP, forward_scan_pars_Hertz_best, forward_scan_cov_Hertz_best, back_scan_pars_Hertz_best, back_scan_cov_Hertz_best, forward_scan_bulk_modulus_Hertz_full_curve, forward_scan_bulk_modulus_error_Hertz_full_curve, back_scan_bulk_modulus_Hertz_full_curve, back_scan_bulk_modulus_error_Hertz_full_curve, time_s

def Hertz_Print_Results(path_filename_noext, folder_path_fitting_restults, filename_noext, df, R, v_sample, Zmov_type, forward_scan, back_scan, forward_scan_contactP, back_scan_contactP, forward_scan_cov_Hertz_best, forward_scan_pars_Hertz_best, back_scan_cov_Hertz_best, back_scan_pars_Hertz_best, indentation_speed, Max_Indent, Max_Force, forward_scan_bulk_modulus_Hertz_full_curve, forward_scan_bulk_modulus_error_Hertz_full_curve, back_scan_bulk_modulus_Hertz_full_curve, back_scan_bulk_modulus_error_Hertz_full_curve, time_s):

    #Print results to screen
    txt.insert(END, "Probe radius: {} um\n".format(R*1e6))
    txt.insert(END, "Elapsed time: {} s\n".format(round(time_s)))

    txt.insert(END, "Poisson's ratio sample: {}\n".format(v_sample))
    txt.insert(END, "Indentation speed: {} um/s\n".format(round(indentation_speed, 3)))
    txt.insert(END, "Max indentation: {} um\n".format(round(Max_Indent*1e6, 3)))
    txt.insert(END, "Max force: {:,} uN\n".format(round(Max_Force*1e6, 3)))
    txt.insert(END, "Contact point: {} um\n".format(str(round(forward_scan_contactP, 3))))
    txt.insert(END, "Separation point: {} um\n\n".format(str(round(back_scan_contactP, 3))))
    txt.update()
    txt.see("end")

    #Calculate and report elastic modulus sample (Es)
    Es_forward_scan_Hertz_full_curve = (3/4)*forward_scan_bulk_modulus_Hertz_full_curve*(1-(v_sample**2))
    error_forward_scan_Hertz_full_curve = (Es_forward_scan_Hertz_full_curve*forward_scan_bulk_modulus_error_Hertz_full_curve)/forward_scan_bulk_modulus_Hertz_full_curve
    Es_forward_scan_Hertz = (3/4)*forward_scan_pars_Hertz_best*(1-(v_sample**2))
    error_forward_scan_Hertz = (Es_forward_scan_Hertz*forward_scan_cov_Hertz_best)/forward_scan_pars_Hertz_best
    txt.insert(END, "K_fc (Hertz, loading scan) = {:,} +/- {:,} Pa\n".format(round(forward_scan_bulk_modulus_Hertz_full_curve, 2), round(forward_scan_bulk_modulus_error_Hertz_full_curve, 2)))
    txt.insert(END, "K_ema (Hertz, loading scan) = {:,} +/- {:,} Pa\n".format(round(forward_scan_pars_Hertz_best, 2), round(forward_scan_cov_Hertz_best, 2)))
    txt.insert(END, "Es_fc (Hertz, loading scan) = {:,} +/- {:,} Pa\n".format(round(Es_forward_scan_Hertz_full_curve, 2), round(error_forward_scan_Hertz_full_curve, 2)))
    txt.insert(END, "Es_ema (Hertz, loading scan) = {:,} +/- {:,} Pa\n\n".format(round(Es_forward_scan_Hertz, 2), round(error_forward_scan_Hertz, 2)))

    Es_back_scan_Hertz_full_curve = (3/4)*back_scan_bulk_modulus_Hertz_full_curve*(1-(v_sample**2))
    error_back_scan_Hertz_full_curve = (Es_back_scan_Hertz_full_curve*back_scan_bulk_modulus_error_Hertz_full_curve)/back_scan_bulk_modulus_Hertz_full_curve
    Es_back_scan_Hertz = (3/4)*back_scan_pars_Hertz_best*(1-(v_sample**2))
    error_back_scan_Hertz = (Es_back_scan_Hertz*back_scan_cov_Hertz_best)/back_scan_pars_Hertz_best
    txt.insert(END, "K_fc (Hertz, unloading scan) = {:,} +/- {:,} Pa\n".format(round(back_scan_bulk_modulus_Hertz_full_curve, 2), round(back_scan_bulk_modulus_error_Hertz_full_curve, 2)))
    txt.insert(END, "K_ema (Hertz, unloading scan) = {:,} +/- {:,} Pa\n".format(round(back_scan_pars_Hertz_best, 2), round(back_scan_cov_Hertz_best, 2)))
    txt.insert(END, "Es_fc (Hertz, unloading scan) = {:,} +/- {:,} Pa\n".format(round(Es_back_scan_Hertz_full_curve, 2), round(error_back_scan_Hertz_full_curve, 2)))
    txt.insert(END, "Es_ema (Hertz, unloading scan) = {:,} +/- {:,} Pa\n".format(round(Es_back_scan_Hertz, 2), round(error_back_scan_Hertz, 2)))
    txt.update()
    txt.see("end")
    Spacer()

    #Save full curve
    df.to_csv(f"{path_filename_noext}_full curve.csv", index=False)

    #Save fitting results for loading curve
    file = open(f"{folder_path_fitting_restults}/{filename_noext}_LS_Hertz fitting results.csv", "w")
    #Because it will be saved as csv, Excell will switch column when it sees a comma, hence all the ","
    file.write("Measurement type:, {}\n".format(Zmov_type))
    file.write("Probe radius:, {}, um\n".format(R*1e6))
    file.write("Poisson's ratio sample:, {}\n".format(v_sample))
    file.write("Indentation speed:, {}, um/s\n".format(round(indentation_speed, 3)))
    file.write("Max indentation:, {}, um\n".format(round(Max_Indent*1e6, 3)))
    file.write("Max force:, {}, uN\n".format(round(Max_Force*1e6, 3)))
    file.write("Contact point:, {}, um\n".format(str(round(forward_scan_contactP, 3))))
    file.write("Separation point:, {}, um\n".format(str(round(back_scan_contactP, 3))))

    file.write("K_fc (Hertz loading scan) =, {}, +/-, {}, Pa\n".format(round(forward_scan_bulk_modulus_Hertz_full_curve, 2), round(forward_scan_bulk_modulus_error_Hertz_full_curve, 2)))
    file.write("K_ema (Hertz loading scan) =, {}, +/-, {}, Pa\n".format(round(forward_scan_pars_Hertz_best, 2), round(forward_scan_cov_Hertz_best, 2)))
    file.write("Es_fc (Hertz loading scan) =, {}, +/-, {}, Pa\n".format(round(Es_forward_scan_Hertz_full_curve, 2), round(error_forward_scan_Hertz_full_curve, 2)))
    file.write("Es_ema (Hertz loading scan) =, {}, +/-, {}, Pa\n".format(round(Es_forward_scan_Hertz, 2), round(error_forward_scan_Hertz, 2)))
    forward_scan.to_csv(file, index=False)
    file.close()

    #Save fitting results for unloading curve
    file = open(f"{folder_path_fitting_restults}/{filename_noext}_US_Hertz fitting results.csv", "w")
    #Because it will be saved as csv, Excell will switch column when it sees a comma, hence all the ","
    file.write("Measurement type:, {}\n".format(Zmov_type))
    file.write("Probe radius:, {}, um\n".format(R*1e6))
    file.write("Poisson's ratio sample:, {}\n".format(v_sample))
    file.write("Indentation speed:, {}, um/s\n".format(round(indentation_speed, 3)))
    file.write("Max indentation:, {}, um\n".format(round(Max_Indent*1e6, 3)))
    file.write("Max force:, {}, uN\n".format(round(Max_Force*1e6, 3)))
    file.write("Contact point:, {}, um\n".format(str(round(forward_scan_contactP, 3))))
    file.write("Separation point:, {}, um\n".format(str(round(back_scan_contactP, 3))))

    file.write("K_fc (Hertz unloading scan) =, {}, +/-, {}, Pa\n".format(round(back_scan_bulk_modulus_Hertz_full_curve, 2), round(back_scan_bulk_modulus_error_Hertz_full_curve, 2)))
    file.write("K_ema (Hertz unloading scan) =, {}, +/-, {}, Pa\n".format(round(back_scan_pars_Hertz_best, 2), round(back_scan_cov_Hertz_best, 2)))
    file.write("Es_fc (Hertz unloading scan) =, {}, +/-, {}, Pa\n".format(round(Es_back_scan_Hertz_full_curve, 2), round(error_back_scan_Hertz_full_curve, 2)))
    file.write("Es_ema (Hertz unloading scan) =, {}, +/-, {}, Pa\n".format(round(Es_back_scan_Hertz, 2), round(error_back_scan_Hertz, 2)))
    back_scan.to_csv(file, index=False)
    file.close()

    #Return data for 3D maps
    sep_point = round(back_scan_contactP, 3)
    Es_LS_fc = round(Es_forward_scan_Hertz_full_curve, 2)
    Es_LS_ema = round(Es_forward_scan_Hertz, 2)
    Es_US_fc = round(Es_back_scan_Hertz_full_curve, 2)
    Es_US_ema = round(Es_back_scan_Hertz, 2)

    return sep_point, Es_LS_fc, Es_LS_ema, Es_US_fc, Es_US_ema

def Hertz_Analyse_Single_CSVFile(path_filename, folder_path_fitting_restults, filename_noext, ratio_BL_points, threshold_constant, forward_scan_segment, back_scan_segment, R, v_sample):

    def Plot_rxc(r, c, title, title_fontsize, xlabel, ylabel, xylabel_fontsize, axes_fontsize, tick_lenght, legend_fontsize):
        ax[r, c].set_title(title, fontsize=title_fontsize)
        ax[r, c].set_xlabel(xlabel, fontsize=xylabel_fontsize)
        ax[r, c].set_ylabel(ylabel, fontsize=xylabel_fontsize)
        ax[r, c].tick_params(direction='in', length=tick_lenght, labelsize=axes_fontsize)
        ax[r, c].legend(fontsize=legend_fontsize, bbox_to_anchor=(0., 1.1, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

    # OPEN FILE AND TRANSFORM UNITS TO N AND M
    df = pandas.read_table(path_filename, sep=',', header=0)
    path_filename_noext, ext = os.path.splitext(path_filename) #Remove ".csv" from filename to avoid weird name for documents
    df = df.astype(float)  #Change data from object to float
    R, ForceA_N_col, Time_s_col, Corr_Time_s_col, Displacement_um_col, Displacement_m_col, PiezoZ_um_col, PiezoZ_m_col, PosZ_um_col, PosZ_m_col = Unit_Transform(df, R)

    # DETERMINE IF MEASUREMENT WAS CARRIED OUT USING PIEZO SCANNER OR STEP MOTOR AND DETERMINE POINT OF CONTACT
    Zmov_type, Probe_Displacement_m_col, forward_scan, back_scan, forward_scan_index_min, forward_scan_min, back_scan_index_min, back_scan_min = Contact_Point(df, PosZ_um_col, PosZ_m_col, PiezoZ_m_col, Displacement_m_col, ratio_BL_points, threshold_constant)

    # DETERMINE SCAN SPEED
    scan_speed = Indentation_Speed(Zmov_type, forward_scan)

    # BASELINE AND ZERO DATA
    baseline, CorrForceA_N_col, BaselineF_N_col, Indentation_m_col = Baseline_And_Zero(df, forward_scan, forward_scan_index_min, Displacement_m_col, ratio_BL_points)

    # PLOT INITIAL ANALYSIS
    if Zmov_type == "Piezo scanner":
        fig1, ax = plt.subplots(2, 2, figsize=(8, 7))
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Piezo Z [m]", color='white', edgecolors='steelblue', label='Piezo Z')
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Displacement [m]", color='white', edgecolors="orange", label='Sample-probe distance')
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Probe Displacement [m]", color='white', edgecolors="firebrick", label='Probe displacement')
        Plot_rxc(0, 0, "", 14, "Time (s)", "Distance (m)", 12, 9, 7, 9)

        df.plot(ax=ax[0, 1], kind='scatter', s=5, x="Piezo Z [m]", y="Displacement [m]", color='white', edgecolors='steelblue', label='Sample-probe distance vs Piezo Z')
        Plot_rxc(0, 1, "", 14, "Piezo Z (m)", "Sample-probe distance (m)", 12, 9, 7, 9)

        df.plot(ax=ax[1, 0], kind='scatter', s=5, x="Piezo Z [m]", y="Force A [N]", color='white', edgecolors='steelblue', label="Piezo Z")
        df.plot(ax=ax[1, 0], kind='scatter', s=5, x="Displacement [m]", y="Force A [N]", color='white', edgecolors="orange", label="Sample-probe distance")
        Plot_rxc(1, 0, "", 14, "Distance (m)", "Force (N)", 12, 9, 7, 9)

        df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="lightblue", label="Loading scan")
        df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="navajowhite", label="Unloading scan")
        df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="steelblue", label="Corr. loading scan")
        df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="orange", label="Corr. unloading scan")
        baseline.plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="gray", label="Datapts used for BL linear fitting")
        df.plot(ax=ax[1, 1], kind='line', style='--', x="Indentation [m]", y="Baseline F [N]", color="black", label="Baseline")
        Plot_rxc(1, 1, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 7)
        plt.tight_layout()
        plt.savefig("{}_Initial analysis.png".format(path_filename_noext), bbox_inches='tight', dpi=300)

    if Zmov_type == "Stick-slip actuator":
        fig1, ax = plt.subplots(2, 2, figsize=(8, 7))
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Pos Z [m]", color='white', edgecolors='steelblue', label="Pos Z")
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Displacement [m]", color='white', edgecolors="orange", label="Sample-probe distance")
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Probe Displacement [m]", color='white', edgecolors="firebrick", label="Probe displacement")
        Plot_rxc(0, 0, "", 14, "Time (s)", "Distance (m)", 12, 9, 7, 9)

        df.plot(ax=ax[0, 1], kind='scatter', s=5, x="Pos Z [m]", y="Displacement [m]", color='white', edgecolors='steelblue', label='Sample-probe distance vs Piezo Z')
        Plot_rxc(0, 1, "", 14, "Piezo Z (m)", "Sample-probe distance (m)", 12, 9, 7, 9)

        df.plot(ax=ax[1, 0], kind='scatter', s=5, x="Pos Z [m]", y="Force A [N]", color='white', edgecolors='steelblue', label="Piezo Z")
        df.plot(ax=ax[1, 0], kind='scatter', s=5, x="Displacement [m]", y="Force A [N]", color='white', edgecolors="orange", label="Sample-probe distance")
        Plot_rxc(1, 0, "", 14, "Distance (m)", "Force (N)", 12, 9, 7, 9)

        df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="lightblue", label="Loading scan")
        df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="navajowhite", label="Unloading scan")
        df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="steelblue", label="Corr. loading scan")
        df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="orange", label="Corr. unloading scan")
        baseline.plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="gray", label="Datapts used for BL linear fitting")
        df.plot(ax=ax[1, 1], kind='line', style='--', x="Indentation [m]", y="Baseline F [N]", color="black", label="Baseline")
        Plot_rxc(1, 1, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 7)
        plt.tight_layout()
        plt.savefig("{}_Initial analysis.png".format(path_filename_noext), bbox_inches='tight', dpi=300)

    # SELECT DATA FOR Hertz FITTING AND PLOT RESULTS
    try:
        forward_scan, forward_scan_contactP, back_scan, back_scan_contactP, forward_scan_pars_Hertz_best, forward_scan_cov_Hertz_best, back_scan_pars_Hertz_best, back_scan_cov_Hertz_best, forward_scan_bulk_modulus_Hertz_full_curve, forward_scan_bulk_modulus_error_Hertz_full_curve, back_scan_bulk_modulus_Hertz_full_curve, back_scan_bulk_modulus_error_Hertz_full_curve, time_s = Hertz_Fitting(df, forward_scan_segment, back_scan_segment, R, forward_scan_index_min, back_scan_index_min, Indentation_m_col, CorrForceA_N_col)
    except ValueError:
        txt.insert(END, f"Error: Datapoints per fitting segment parameter is too high.\n\n")
        Spacer()
        Spacer()
        txt.update()
        txt.see("end")
        return

    # Get individual datapoints to indicate best results in error analysis graphs.
    to_remove = ["NaN"]
    forward_scan_best_Hertz = forward_scan[["Indentation step Hertz [m]", "K Hertz [Pa]", "ln (K Error Hertz)"]]
    forward_scan_best_Hertz = forward_scan_best_Hertz[~forward_scan_best_Hertz["ln (K Error Hertz)"].isin(to_remove)]
    forward_scan_best_Hertz = forward_scan_best_Hertz.loc[forward_scan_best_Hertz["ln (K Error Hertz)"] == forward_scan_best_Hertz["ln (K Error Hertz)"].min()]
    back_scan_best_Hertz = back_scan[["Indentation step Hertz [m]", "K Hertz [Pa]", "ln (K Error Hertz)"]]
    back_scan_best_Hertz = back_scan_best_Hertz[~back_scan_best_Hertz["ln (K Error Hertz)"].isin(to_remove)]
    back_scan_best_Hertz = back_scan_best_Hertz.loc[back_scan_best_Hertz["ln (K Error Hertz)"] == back_scan_best_Hertz["ln (K Error Hertz)"].min()]

    # Plot all data for Hertz
    Fig4, ax = plt.subplots(3, 2, figsize=(8, 7))
    df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[0, 0], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="steelblue", label='Loading scan')
    df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[0, 0], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="orange", label='Unloading scan')
    Plot_rxc(0, 0, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 9)

    forward_scan.plot(ax=ax[0, 1], kind='scatter', x="Indentation Hertz Full Curve [m]", y="Corr Force A Hertz Full Curve [N]", color='white', edgecolors='steelblue', label='Loading data')
    forward_scan.plot(ax=ax[0, 1], kind='line', style="--", x="Fitting Indentation Hertz Full Curve [m]", y="Corr Force A Hertz Full Curve [N]", color='black', label='Hertz fitting')
    back_scan.plot(ax=ax[0, 1], kind='scatter', x="Indentation Hertz Full Curve [m]", y="Corr Force A Hertz Full Curve [N]", color='white', edgecolors='orange', label='Unloading data')
    back_scan.plot(ax=ax[0, 1], kind='line', style="--", x="Fitting Indentation Hertz Full Curve [m]", y="Corr Force A Hertz Full Curve [N]", color='gray', label='Hertz fitting')
    Plot_rxc(0, 1, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 9)

    forward_scan.plot(ax=ax[1, 0], kind='scatter', s=10, x="Indentation step Hertz [m]", y="ln (K Error Hertz)", color='white', edgecolors='steelblue', label='Loading scan Hertz')
    forward_scan_best_Hertz.plot(ax=ax[1, 0], kind='scatter', s=30, x="Indentation step Hertz [m]", y="ln (K Error Hertz)", color='steelblue', edgecolors='steelblue')
    back_scan.plot(ax=ax[1, 0], kind='scatter', s=10, x="Indentation step Hertz [m]", y="ln (K Error Hertz)", color='white', edgecolors='orange', label='Unloading scan Hertz')
    back_scan_best_Hertz.plot(ax=ax[1, 0], kind='scatter', s=30, x="Indentation step Hertz [m]", y="ln (K Error Hertz)", color='orange', edgecolors='orange')
    Plot_rxc(1, 0, "", 14, "Indentation (m)", "ln (K error)", 12, 9, 7, 9)

    forward_scan.plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation step Hertz [m]", y="K Hertz [Pa]", color='white', edgecolors='steelblue', label='Loading scan Hertz')
    forward_scan_best_Hertz.plot(ax=ax[1, 1], kind='scatter', s=30, x="Indentation step Hertz [m]", y="K Hertz [Pa]", color='steelblue', edgecolors='steelblue')
    back_scan.plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation step Hertz [m]", y="K Hertz [Pa]", color='white', edgecolors='orange', label='Unloading scan Hertz')
    back_scan_best_Hertz.plot(ax=ax[1, 1], kind='scatter', s=30, x="Indentation step Hertz [m]", y="K Hertz [Pa]", color='orange', edgecolors='orange')
    Plot_rxc(1, 1, "", 14, "Indentation (m)", "K (Pa)", 12, 9, 7, 9)

    df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[2, 0], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="lightblue", label='Loading scan')
    df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[2, 0], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="navajowhite", label='Unloading scan')
    forward_scan.plot(ax=ax[2, 0], kind='scatter', s=10, x="Selected Indentation Hertz [m]", y="Selected Corr Force A Hertz [N]", color='white', edgecolors='steelblue', label='Sel. loading data')
    back_scan.plot(ax=ax[2, 0], kind='scatter', s=10, x="Selected Indentation Hertz [m]", y="Selected Corr Force A Hertz [N]", color='white', edgecolors='orange', label='Sel. unloading data')
    Plot_rxc(2, 0, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 9)

    forward_scan.plot(ax=ax[2, 1], kind='scatter', x="Corr Indentation Hertz [m]", y="Corr Force A Hertz [N]", color='white', edgecolors='steelblue', label='Loading data')
    forward_scan.plot(ax=ax[2, 1], kind='line', style="--", x="Fitting Indentation Hertz [m]", y="Corr Force A Hertz [N]", color='black', label='Hertz fitting')
    back_scan.plot(ax=ax[2, 1], kind='scatter', x="Corr Indentation Hertz [m]", y="Corr Force A Hertz [N]", color='white', edgecolors='orange', label='Unloading data')
    back_scan.plot(ax=ax[2, 1], kind='line', style="--", x="Fitting Indentation Hertz [m]", y="Corr Force A Hertz [N]", color='gray', label='Hertz fitting')
    Plot_rxc(2, 1, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 9)
    plt.tight_layout()
    plt.savefig(f"{folder_path_fitting_restults}/{filename_noext}_Fitting analysis Hertz.png", bbox_inches='tight', dpi=300)

    plt.close('all') #To close all figures and save memory - max # of figures before warning: 20


    # MAX INDENTATION AND MAX FORCE
    Max_Indent, Max_Force = Max_Indent_And_Max_Force(forward_scan, Indentation_m_col, CorrForceA_N_col)

    # PRINT RESULTS
    sep_point, Es_LS_fc, Es_LS_ema, Es_US_fc, Es_US_ema = Hertz_Print_Results(path_filename_noext, folder_path_fitting_restults, filename_noext, df, R, v_sample, Zmov_type, forward_scan, back_scan, forward_scan_contactP, back_scan_contactP, forward_scan_cov_Hertz_best, forward_scan_pars_Hertz_best, back_scan_cov_Hertz_best, back_scan_pars_Hertz_best, scan_speed, Max_Indent, Max_Force, forward_scan_bulk_modulus_Hertz_full_curve, forward_scan_bulk_modulus_error_Hertz_full_curve, back_scan_bulk_modulus_Hertz_full_curve, back_scan_bulk_modulus_error_Hertz_full_curve, time_s)

    # GET DATA FOR 3D PLOT
    X, Y = GetXY(df, Zmov_type)

    return X, Y, sep_point, Es_LS_fc, Es_LS_ema, Es_US_fc, Es_US_ema


def DMT_Fitting(df, forward_scan_segment, back_scan_segment, R, forward_scan_index_min, back_scan_index_min, Indentation_m_col, CorrForceA_N_col):

    #Define fitting functions
    def DMT(x, R, K, Fad):
        a = ((R/K)*(x+Fad))**(1/3)
        return ((a**2)/R)

    #Select starting point for fitting, get Fad, and calculate gamma
    forward_scan = pandas.DataFrame(df.groupby(["Exp Phase [#]"]).get_group(1.0))
    forward_scan = forward_scan.loc[forward_scan.index >= forward_scan_index_min] #Select data for fitting based on contact point
    forward_scan_Fad = forward_scan.iloc[0, CorrForceA_N_col]
    gamma_forward_scan_DMT = abs(forward_scan_Fad)/(2*math.pi*R)
    forward_scan_contactP = forward_scan.iloc[0, Indentation_m_col]*1e6
    # If Fad >0 set it to 0
    if forward_scan_Fad > 0:
        forward_scan_Fad = 0

    #Fit the entire indentation curve
    forward_scan_x_full_curve = forward_scan["Corr Force A [N]"].to_list()
    forward_scan_y_full_curve = forward_scan["Indentation [m]"].to_list()
    custom_DMT = lambda x, K: DMT(x, R, K, -forward_scan_Fad)  # Fix Fad value
    forward_scan_pars_DMT_full_curve, forward_scan_cov_DMT_full_curve = optimize.curve_fit(f=custom_DMT, xdata=forward_scan_x_full_curve, ydata=forward_scan_y_full_curve)
    forward_scan_cov_DMT_full_curve = np.sqrt(np.diag(forward_scan_cov_DMT_full_curve)) #Necessary to convert error matrix output of optimize.curve_fit into list of standard deviations for each parameter
    forward_scan_bulk_modulus_DMT_full_curve = forward_scan_pars_DMT_full_curve[0]
    forward_scan_bulk_modulus_error_DMT_full_curve = forward_scan_cov_DMT_full_curve[0]
    forward_scan["Indentation DMT Full Curve [m]"] = forward_scan["Indentation [m]"]
    forward_scan["Corr Force A DMT Full Curve [N]"] = forward_scan["Corr Force A [N]"]
    #Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    forward_scan["Fitting Indentation DMT Full Curve [m]"] = ((forward_scan["Corr Force A DMT Full Curve [N]"]-forward_scan_Fad)**(2/3))/((R**(1/3))*(forward_scan_bulk_modulus_DMT_full_curve**(2/3)))

    #Create test dataframe for the while loop and cut it to the first set of datapoints
    forward_scan_initial_len = len(forward_scan)
    forward_scan_datapoints_to_fit = forward_scan_segment
    forward_scan_test = forward_scan.loc[forward_scan.index < (forward_scan.index[0] + forward_scan_datapoints_to_fit)] #Select data for fitting based on threshold value, index was not reset so forward_scan.index[-1] is a high number
    forward_scan_bulk_modulus_DMT = []
    forward_scan_bulk_modulus_error_DMT = []
    forward_scan_indentation_DMT = []
    forward_scan_datapoints_DMT = []

    while forward_scan_datapoints_to_fit <= forward_scan_initial_len:
        forward_scan_x = forward_scan_test["Corr Force A [N]"].to_list()
        forward_scan_y = forward_scan_test["Indentation [m]"].to_list()

        # DMT fitting on selected data
        custom_DMT = lambda x, K: DMT(x, R, K, -forward_scan_Fad)  # Fix Fad value
        forward_scan_pars_DMT, forward_scan_cov_DMT = optimize.curve_fit(f=custom_DMT, xdata=forward_scan_x, ydata=forward_scan_y)
        forward_scan_cov_DMT = np.sqrt(np.diag(forward_scan_cov_DMT)) #Necessary to convert error matrix output of optimize.curve_fit into list of standard deviations for each parameter

        forward_scan_bulk_modulus_DMT.append(forward_scan_pars_DMT[0])
        forward_scan_bulk_modulus_error_DMT.append(forward_scan_cov_DMT[0])
        forward_scan_indentation_DMT.append(forward_scan_y[-1])
        forward_scan_datapoints_DMT.append(forward_scan_datapoints_to_fit)

        forward_scan_datapoints_to_fit = forward_scan_datapoints_to_fit + forward_scan_segment
        forward_scan_test = forward_scan.loc[forward_scan.index < (forward_scan.index[0] + forward_scan_datapoints_to_fit)]  # Select data for fitting based on threshold value, index was not reset so forward_scan.index[-1] is a high number

    #Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    forward_scan_index_smallest_error = forward_scan_bulk_modulus_error_DMT.index(min(forward_scan_bulk_modulus_error_DMT))
    forward_scan_cov_DMT_best = forward_scan_bulk_modulus_error_DMT[forward_scan_index_smallest_error]
    forward_scan_pars_DMT_best = forward_scan_bulk_modulus_DMT[forward_scan_index_smallest_error]
    #Redeclare test df with best result and append Indentation and Corr Force columns to forward_scan for storage
    forward_scan_test = forward_scan.loc[forward_scan.index < (forward_scan.index[0] + forward_scan_datapoints_DMT[forward_scan_index_smallest_error])]
    forward_scan["Indentation DMT [m]"] = forward_scan_test["Indentation [m]"]
    forward_scan["Corr Force A DMT [N]"] = forward_scan_test["Corr Force A [N]"]
    #Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    forward_scan["Fitting Indentation DMT [m]"] = ((forward_scan["Corr Force A DMT [N]"]-forward_scan_Fad)**(2/3))/((R**(1/3))*(forward_scan_pars_DMT_best**(2/3)))

    #Extend lists with stored data to match the lenght of forward_scan df otherwise they cannot be added as columns to the df
    forward_scan_indentation_DMT.extend(["NaN"]*(forward_scan_initial_len-len(forward_scan_indentation_DMT)))
    forward_scan["Indentation step DMT [m]"] = forward_scan_indentation_DMT
    forward_scan_bulk_modulus_DMT.extend(["NaN"]*(forward_scan_initial_len-len(forward_scan_bulk_modulus_DMT)))
    forward_scan["K DMT [Pa]"] = forward_scan_bulk_modulus_DMT
    #Transform error into logarithm for best plots
    forward_scan_bulk_modulus_error_DMT = [math.log(x) for x in forward_scan_bulk_modulus_error_DMT]
    forward_scan_bulk_modulus_error_DMT.extend(["NaN"]*(forward_scan_initial_len-len(forward_scan_bulk_modulus_error_DMT)))
    forward_scan["ln (K Error DMT)"] = forward_scan_bulk_modulus_error_DMT


    #Select starting point for fitting, get Fad, and calculate gamma
    back_scan = pandas.DataFrame(df.groupby(["Exp Phase [#]"]).get_group(0.0))
    back_scan = back_scan.iloc[::-1].reset_index(drop=True) #Reverse and reindex dataframe
    back_scan = back_scan.loc[back_scan.index >= back_scan_index_min] #Select data for fitting based on threshold value
    back_scan_Fad = back_scan.iloc[0, CorrForceA_N_col]
    gamma_back_scan_DMT = abs(back_scan_Fad)/(2*math.pi*R)
    back_scan_contactP = back_scan.iloc[0, Indentation_m_col]*1e6
    # If Fad >0 set it to 0
    if back_scan_Fad > 0:
        back_scan_Fad = 0

    # Make back scan indentation start from 0 for fitting
    back_scan["Indentation Corr [m]"] = back_scan["Indentation [m]"] - back_scan.iloc[0, Indentation_m_col]
    Indentation_Corr_m_col = back_scan.columns.get_loc("Indentation Corr [m]") #25

    #Fit the entire indentation curve
    back_scan_x_full_curve = back_scan["Corr Force A [N]"].to_list()
    back_scan_y_full_curve = back_scan["Indentation Corr [m]"].to_list()
    custom_DMT = lambda x, K: DMT(x, R, K, -back_scan_Fad)  # Fix Fad value
    back_scan_pars_DMT_full_curve, back_scan_cov_DMT_full_curve = optimize.curve_fit(f=custom_DMT, xdata=back_scan_x_full_curve, ydata=back_scan_y_full_curve)
    back_scan_cov_DMT_full_curve = np.sqrt(np.diag(back_scan_cov_DMT_full_curve)) #Necessary to convert error matrix output of optimize.curve_fit into list of standard deviations for each parameter
    back_scan_bulk_modulus_DMT_full_curve = back_scan_pars_DMT_full_curve[0]
    back_scan_bulk_modulus_error_DMT_full_curve = back_scan_cov_DMT_full_curve[0]
    back_scan["Indentation DMT Full Curve [m]"] = back_scan["Indentation [m]"]
    back_scan["Corr Force A DMT Full Curve [N]"] = back_scan["Corr Force A [N]"]
    #Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    back_scan["Fitting Indentation DMT Full Curve [m]"] = ((back_scan["Corr Force A DMT Full Curve [N]"]-back_scan_Fad)**(2/3))/((R**(1/3))*(back_scan_bulk_modulus_DMT_full_curve**(2/3)))

    #Optimise loading curve fitting
    back_scan_initial_len = len(back_scan)
    back_scan_datapoints_to_fit = back_scan_segment
    back_scan_test = back_scan.loc[back_scan.index < (back_scan.index[0] + back_scan_datapoints_to_fit)] #Select data for fitting based on threshold value, index was not reset so forward_scan.index[-1] is a high number
    back_scan_bulk_modulus_DMT = []
    back_scan_bulk_modulus_error_DMT = []
    back_scan_indentation_DMT = []
    back_scan_datapoints_DMT = []

    while back_scan_datapoints_to_fit <= back_scan_initial_len:
        back_scan_x = back_scan_test["Corr Force A [N]"].to_list()
        back_scan_y = back_scan_test["Indentation Corr [m]"].to_list()

        # DMT fitting on selected data
        custom_DMT = lambda x, K: DMT(x, R, K, -back_scan_Fad)  # Fix Fad value
        back_scan_pars_DMT, back_scan_cov_DMT = optimize.curve_fit(f=custom_DMT, xdata=back_scan_x, ydata=back_scan_y)
        back_scan_cov_DMT = np.sqrt(np.diag(back_scan_cov_DMT)) #Necessary to convert error matrix output of optimize.curve_fit into list of standard deviations for each parameter

        back_scan_bulk_modulus_DMT.append(back_scan_pars_DMT[0])
        back_scan_bulk_modulus_error_DMT.append(back_scan_cov_DMT[0])
        back_scan_indentation_DMT.append(back_scan_y[-1])
        back_scan_datapoints_DMT.append(back_scan_datapoints_to_fit)

        back_scan_datapoints_to_fit = back_scan_datapoints_to_fit + back_scan_segment
        back_scan_test = back_scan.loc[back_scan.index < (back_scan.index[0] + back_scan_datapoints_to_fit)]  # Select data for fitting based on threshold value, index was not reset so forward_scan.index[-1] is a high number

    #Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    back_scan_index_smallest_error = back_scan_bulk_modulus_error_DMT.index(min(back_scan_bulk_modulus_error_DMT))
    back_scan_cov_DMT_best = back_scan_bulk_modulus_error_DMT[back_scan_index_smallest_error]
    back_scan_pars_DMT_best = back_scan_bulk_modulus_DMT[back_scan_index_smallest_error]
    #Redeclare test df with best result and append Indentation and Corr Force columns to forward_scan for storage
    back_scan_test = back_scan.loc[back_scan.index < (back_scan.index[0] + back_scan_datapoints_DMT[back_scan_index_smallest_error])]
    #Store selcted indentaion for data selection graph and corrected indentation used for fitting
    back_scan["Indentation DMT [m]"] = back_scan_test["Indentation [m]"]
    back_scan["Indentation Corr DMT [m]"] = back_scan_test["Indentation Corr [m]"]
    back_scan["Corr Force A DMT [N]"] = back_scan_test["Corr Force A [N]"]
    #Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    back_scan["Fitting Indentation DMT [m]"] = ((back_scan["Corr Force A DMT [N]"]-back_scan_Fad)**(2/3))/((R**(1/3))*(back_scan_pars_DMT_best**(2/3)))

    #Extend lists with stored data to match the lenght of forward_scan df otherwise they cannot be added as columns to the df
    back_scan_indentation_DMT.extend(["NaN"]*(back_scan_initial_len-len(back_scan_indentation_DMT)))
    back_scan["Indentation step DMT [m]"] = back_scan_indentation_DMT
    back_scan_bulk_modulus_DMT.extend(["NaN"]*(back_scan_initial_len-len(back_scan_bulk_modulus_DMT)))
    back_scan["K DMT [Pa]"] = back_scan_bulk_modulus_DMT
    #Transform error into logarithm for best plots
    back_scan_bulk_modulus_error_DMT = [math.log(x) for x in back_scan_bulk_modulus_error_DMT]
    back_scan_bulk_modulus_error_DMT.extend(["NaN"]*(back_scan_initial_len-len(back_scan_bulk_modulus_error_DMT)))
    back_scan["ln (K Error DMT)"] = back_scan_bulk_modulus_error_DMT

    # Define time
    time_s_col = df.columns.get_loc("Time [s]")  # Get column number
    time_s = df.iloc[0, time_s_col]

    return forward_scan, forward_scan_Fad, gamma_forward_scan_DMT, forward_scan_contactP, back_scan, back_scan_Fad, gamma_back_scan_DMT, back_scan_contactP, forward_scan_pars_DMT_best, forward_scan_cov_DMT_best, back_scan_pars_DMT_best, back_scan_cov_DMT_best, forward_scan_bulk_modulus_DMT_full_curve, forward_scan_bulk_modulus_error_DMT_full_curve, back_scan_bulk_modulus_DMT_full_curve, back_scan_bulk_modulus_error_DMT_full_curve, time_s

def DMT_Print_Results(path_filename_noext, folder_path_fitting_restults, filename_noext, df, R, v_sample, Zmov_type, forward_scan, back_scan, forward_scan_Fad, forward_scan_contactP, back_scan_Fad, back_scan_contactP, forward_scan_cov_DMT_best, forward_scan_pars_DMT_best, gamma_forward_scan_DMT, back_scan_cov_DMT_best, back_scan_pars_DMT_best, gamma_back_scan_DMT, indentation_speed, Max_Indent, Max_Force, forward_scan_bulk_modulus_DMT_full_curve, forward_scan_bulk_modulus_error_DMT_full_curve, back_scan_bulk_modulus_DMT_full_curve, back_scan_bulk_modulus_error_DMT_full_curve, time_s):

    #Print results to screen
    txt.insert(END, "Probe radius: {} um\n".format(R*1e6))
    txt.insert(END, "Elapsed time: {} s\n".format(round(time_s)))

    txt.insert(END, "Poisson's ratio sample: {}\n".format(v_sample))
    txt.insert(END, "Indentation speed: {} um/s\n".format(round(indentation_speed, 3)))
    txt.insert(END, "Max indentation: {} um\n".format(round(Max_Indent*1e6, 3)))
    txt.insert(END, "Max force: {:,} uN\n".format(round(Max_Force*1e6, 3)))

    txt.insert(END, "Contact point: {} um\n".format(str(round(forward_scan_contactP, 3))))
    txt.insert(END, "Attractive force: {} uN\n".format(str(round(forward_scan_Fad*1e6, 3))))

    txt.insert(END, "Separation point: {} um\n".format(str(round(back_scan_contactP, 3))))
    txt.insert(END, "Adhesion force: {} uN\n\n".format(str(round(back_scan_Fad*1e6, 3))))
    txt.update()
    txt.see("end")

    #Calculate and report elastic modulus sample (Es)
    Es_forward_scan_DMT_full_curve = (3/4)*forward_scan_bulk_modulus_DMT_full_curve*(1-(v_sample**2))
    error_forward_scan_DMT_full_curve = (Es_forward_scan_DMT_full_curve*forward_scan_bulk_modulus_error_DMT_full_curve)/forward_scan_bulk_modulus_DMT_full_curve
    Es_forward_scan_DMT = (3/4)*forward_scan_pars_DMT_best*(1-(v_sample**2))
    error_forward_scan_DMT = (Es_forward_scan_DMT*forward_scan_cov_DMT_best)/forward_scan_pars_DMT_best
    txt.insert(END, "K_fc (DMT, loading scan) = {:,} +/- {:,} Pa\n".format(round(forward_scan_bulk_modulus_DMT_full_curve, 2), round(forward_scan_bulk_modulus_error_DMT_full_curve, 2)))
    txt.insert(END, "K_ema (DMT, loading scan) = {:,} +/- {:,} Pa\n".format(round(forward_scan_pars_DMT_best, 2), round(forward_scan_cov_DMT_best, 2)))
    txt.insert(END, "Es_fc (DMT, loading scan) = {:,} +/- {:,} Pa\n".format(round(Es_forward_scan_DMT_full_curve, 2), round(error_forward_scan_DMT_full_curve, 2)))
    txt.insert(END, "Es_ema (DMT, loading scan) = {:,} +/- {:,} Pa\n".format(round(Es_forward_scan_DMT, 2), round(error_forward_scan_DMT, 2)))
    txt.insert(END, "Interfacial energy (DMT, loading scan): {} mJ m^-2\n\n".format(round(gamma_forward_scan_DMT*1000, 2)))

    Es_back_scan_DMT_full_curve = (3/4)*back_scan_bulk_modulus_DMT_full_curve*(1-(v_sample**2))
    error_back_scan_DMT_full_curve = (Es_back_scan_DMT_full_curve*back_scan_bulk_modulus_error_DMT_full_curve)/back_scan_bulk_modulus_DMT_full_curve
    Es_back_scan_DMT = (3/4)*back_scan_pars_DMT_best*(1-(v_sample**2))
    error_back_scan_DMT = (Es_back_scan_DMT*back_scan_cov_DMT_best)/back_scan_pars_DMT_best
    txt.insert(END, "K_fc (DMT, unloading scan) = {:,} +/- {:,} Pa\n".format(round(back_scan_bulk_modulus_DMT_full_curve, 2), round(back_scan_bulk_modulus_error_DMT_full_curve, 2)))
    txt.insert(END, "K_ema (DMT, unloading scan) = {:,} +/- {:,} Pa\n".format(round(back_scan_pars_DMT_best, 2), round(back_scan_cov_DMT_best, 2)))
    txt.insert(END, "Es_fc (DMT, unloading scan) = {:,} +/- {:,} Pa\n".format(round(Es_back_scan_DMT_full_curve, 2), round(error_back_scan_DMT_full_curve, 2)))
    txt.insert(END, "Es_ema (DMT, unloading scan) = {:,} +/- {:,} Pa\n".format(round(Es_back_scan_DMT, 2), round(error_back_scan_DMT, 2)))
    txt.insert(END, "Interfacial energy (DMT, unloading scan): {} mJ m^-2\n\n".format(round(gamma_back_scan_DMT*1000, 2)))

    forward_Tabor_param_DMT = ((R*(gamma_forward_scan_DMT**2))/((Es_forward_scan_DMT**2)*((0.333*1e-9)**3)))**(1/3)
    back_Tabor_param_DMT = ((R*(gamma_back_scan_DMT**2))/((Es_back_scan_DMT**2)*((0.333*1e-9)**3)))**(1/3)
    txt.insert(END, "u (DMT, loading scan) = {}\n".format(round(forward_Tabor_param_DMT,2)))
    txt.insert(END, "u (DMT, unloading scan) = {}\n".format(round(back_Tabor_param_DMT,2)))
    txt.update()
    txt.see("end")
    Spacer()

    #Save full curve
    df.to_csv(f"{path_filename_noext}_full curve.csv", index=False)

    #Save fitting results for loading curve
    file = open(f"{folder_path_fitting_restults}/{filename_noext}_LS_DMT fitting results.csv", "w")
    #Because it will be saved as csv, Excell will switch column when it sees a comma, hence all the ","
    file.write("Measurement type:, {}\n".format(Zmov_type))
    file.write("Probe radius:, {}, um\n".format(R*1e6))
    file.write("Poisson's ratio sample:, {}\n".format(v_sample))
    file.write("Indentation speed:, {}, um/s\n".format(round(indentation_speed, 3)))
    file.write("Max indentation:, {}, um\n".format(round(Max_Indent*1e6, 3)))
    file.write("Max force:, {}, uN\n".format(round(Max_Force*1e6, 3)))
    file.write("Contact point:, {}, um\n".format(str(round(forward_scan_contactP, 3))))
    file.write("Attractive force:, {}, uN\n".format(str(round(forward_scan_Fad*1e6, 3))))
    file.write("Separation point:, {}, um\n".format(str(round(back_scan_contactP, 3))))
    file.write("Adhesion force:, {}, uN\n\n".format(str(round(back_scan_Fad*1e6, 3))))

    file.write("K_fc (DMT loading scan) =, {}, +/-, {}, Pa\n".format(round(forward_scan_bulk_modulus_DMT_full_curve, 2), round(forward_scan_bulk_modulus_error_DMT_full_curve, 2)))
    file.write("K_ema (DMT loading scan) =, {}, +/-, {}, Pa\n".format(round(forward_scan_pars_DMT_best, 2), round(forward_scan_cov_DMT_best, 2)))
    file.write("Es_fc (DMT loading scan) =, {}, +/-, {}, Pa\n".format(round(Es_forward_scan_DMT_full_curve, 2), round(error_forward_scan_DMT_full_curve, 2)))
    file.write("Es_ema (DMT loading scan) =, {}, +/-, {}, Pa\n".format(round(Es_forward_scan_DMT, 2), round(error_forward_scan_DMT, 2)))
    file.write("Interfacial energy (DMT loading scan):, {}, mJ m^-2\n".format(round(gamma_forward_scan_DMT*1000, 2)))

    file.write("u (DMT loading scan):, {}\n".format(round(forward_Tabor_param_DMT, 2)))
    forward_scan.to_csv(file, index=False)
    file.close()

    #Save fitting results for unloading curve
    file = open(f"{folder_path_fitting_restults}/{filename_noext}_US_DMT fitting results.csv", "w")
    #Because it will be saved as csv, Excell will switch column when it sees a comma, hence all the ","
    file.write("Measurement type:, {}\n".format(Zmov_type))
    file.write("Probe radius:, {}, um\n".format(R*1e6))
    file.write("Poisson's ratio sample:, {}\n".format(v_sample))
    file.write("Indentation speed:, {}, um/s\n".format(round(indentation_speed, 3)))
    file.write("Max indentation:, {}, um\n".format(round(Max_Indent*1e6, 3)))
    file.write("Max force:, {}, uN\n".format(round(Max_Force*1e6, 3)))
    file.write("Contact point:, {}, um\n".format(str(round(forward_scan_contactP, 3))))
    file.write("Attractive force:, {}, uN\n".format(str(round(forward_scan_Fad*1e6, 3))))
    file.write("Separation point:, {}, um\n".format(str(round(back_scan_contactP, 3))))
    file.write("Adhesion force:, {}, uN\n\n".format(str(round(back_scan_Fad*1e6, 3))))

    file.write("K_fc (DMT unloading scan) =, {}, +/-, {}, Pa\n".format(round(back_scan_bulk_modulus_DMT_full_curve, 2), round(back_scan_bulk_modulus_error_DMT_full_curve, 2)))
    file.write("K_ema (DMT unloading scan) =, {}, +/-, {}, Pa\n".format(round(back_scan_pars_DMT_best, 2), round(back_scan_cov_DMT_best, 2)))
    file.write("Es_fc (DMT unloading scan) =, {}, +/-, {}, Pa\n".format(round(Es_back_scan_DMT_full_curve, 2), round(error_back_scan_DMT_full_curve, 2)))
    file.write("Es_ema (DMT unloading scan) =, {}, +/-, {}, Pa\n".format(round(Es_back_scan_DMT, 2), round(error_back_scan_DMT, 2)))
    file.write("Interfacial energy (DMT unloading scan):, {}, mJ m^-2\n".format(round(gamma_back_scan_DMT*1000, 2)))

    file.write("u (DMT unloading scan):, {}\n".format(round(back_Tabor_param_DMT, 2)))
    back_scan.to_csv(file, index=False)

    file.close()

    #Return data for 3D maps
    attr_force = round(forward_scan_Fad * 1e6, 3)
    adhes_force = round(back_scan_Fad * 1e6, 3)
    sep_point = round(back_scan_contactP, 3)
    Es_LS_fc = round(Es_forward_scan_DMT_full_curve, 2)
    Es_LS_ema = round(Es_forward_scan_DMT, 2)
    Es_US_fc = round(Es_back_scan_DMT_full_curve, 2)
    Es_US_ema = round(Es_back_scan_DMT, 2)
    interf_en_LS = round(forward_Tabor_param_DMT, 2)
    interf_en_US = round(back_Tabor_param_DMT, 2)

    return attr_force, adhes_force, sep_point, Es_LS_fc, Es_LS_ema, Es_US_fc, Es_US_ema, interf_en_LS, interf_en_US

def DMT_Analyse_Single_CSVFile(path_filename, folder_path_fitting_restults, filename_noext, ratio_BL_points, threshold_constant, forward_scan_segment, back_scan_segment, R, v_sample):

    def Plot_rxc(r, c, title, title_fontsize, xlabel, ylabel, xylabel_fontsize, axes_fontsize, tick_lenght, legend_fontsize):
        ax[r, c].set_title(title, fontsize=title_fontsize)
        ax[r, c].set_xlabel(xlabel, fontsize=xylabel_fontsize)
        ax[r, c].set_ylabel(ylabel, fontsize=xylabel_fontsize)
        ax[r, c].tick_params(direction='in', length=tick_lenght, labelsize=axes_fontsize)
        ax[r, c].legend(fontsize=legend_fontsize, bbox_to_anchor=(0., 1.1, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

    # OPEN FILE AND TRANSFORM UNITS TO N AND M
    df = pandas.read_table(path_filename, sep=',', header=0)
    path_filename_noext, ext = os.path.splitext(path_filename) #Remove ".csv" from filename to avoid weird name for documents
    df = df.astype(float)  #Change data from object to float
    R, ForceA_N_col, Time_s_col, Corr_Time_s_col, Displacement_um_col, Displacement_m_col, PiezoZ_um_col, PiezoZ_m_col, PosZ_um_col, PosZ_m_col = Unit_Transform(df, R)

    # DETERMINE IF MEASUREMENT WAS CARRIED OUT USING PIEZO SCANNER OR STEP MOTOR AND DETERMINE POINT OF CONTACT
    Zmov_type, Probe_Displacement_m_col, forward_scan, back_scan, forward_scan_index_min, forward_scan_min, back_scan_index_min, back_scan_min = Contact_Point(df, PosZ_um_col, PosZ_m_col, PiezoZ_m_col, Displacement_m_col, ratio_BL_points, threshold_constant)

    # DETERMINE SCAN SPEED
    scan_speed = Indentation_Speed(Zmov_type, forward_scan)

    # BASELINE AND ZERO DATA
    baseline, CorrForceA_N_col, BaselineF_N_col, Indentation_m_col = Baseline_And_Zero(df, forward_scan, forward_scan_index_min, Displacement_m_col, ratio_BL_points)

    # PLOT INITIAL ANALYSIS
    if Zmov_type == "Piezo scanner":
        fig1, ax = plt.subplots(2, 2, figsize=(8, 7))
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Piezo Z [m]", color='white', edgecolors='steelblue', label='Piezo Z')
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Displacement [m]", color='white', edgecolors="orange", label='Sample-probe distance')
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Probe Displacement [m]", color='white', edgecolors="firebrick", label='Probe displacement')
        Plot_rxc(0, 0, "", 14, "Time (s)", "Distance (m)", 12, 9, 7, 9)

        df.plot(ax=ax[0, 1], kind='scatter', s=5, x="Piezo Z [m]", y="Displacement [m]", color='white', edgecolors='steelblue', label='Sample-probe distance vs Piezo Z')
        Plot_rxc(0, 1, "", 14, "Piezo Z (m)", "Sample-probe distance (m)", 12, 9, 7, 9)

        df.plot(ax=ax[1, 0], kind='scatter', s=5, x="Piezo Z [m]", y="Force A [N]", color='white', edgecolors='steelblue', label="Piezo Z")
        df.plot(ax=ax[1, 0], kind='scatter', s=5, x="Displacement [m]", y="Force A [N]", color='white', edgecolors="orange", label="Sample-probe distance")
        Plot_rxc(1, 0, "", 14, "Distance (m)", "Force (N)", 12, 9, 7, 9)

        df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="lightblue", label="Loading scan")
        df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="navajowhite", label="Unloading scan")
        df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="steelblue", label="Corr. loading scan")
        df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="orange", label="Corr. unloading scan")
        baseline.plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="gray", label="Datapts used for BL linear fitting")
        df.plot(ax=ax[1, 1], kind='line', style='--', x="Indentation [m]", y="Baseline F [N]", color="black", label="Baseline")
        Plot_rxc(1, 1, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 7)
        plt.tight_layout()
        plt.savefig("{}_Initial analysis.png".format(path_filename_noext), bbox_inches='tight', dpi=300)

    if Zmov_type == "Stick-slip actuator":
        fig1, ax = plt.subplots(2, 2, figsize=(8, 7))
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Pos Z [m]", color='white', edgecolors='steelblue', label="Pos Z")
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Displacement [m]", color='white', edgecolors="orange", label="Sample-probe distance")
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Probe Displacement [m]", color='white', edgecolors="firebrick", label="Probe displacement")
        Plot_rxc(0, 0, "", 14, "Time (s)", "Distance (m)", 12, 9, 7, 9)

        df.plot(ax=ax[0, 1], kind='scatter', s=5, x="Pos Z [m]", y="Displacement [m]", color='white', edgecolors='steelblue', label='Sample-probe distance vs Piezo Z')
        Plot_rxc(0, 1, "", 14, "Piezo Z (m)", "Sample-probe distance (m)", 12, 9, 7, 9)

        df.plot(ax=ax[1, 0], kind='scatter', s=5, x="Pos Z [m]", y="Force A [N]", color='white', edgecolors='steelblue', label="Piezo Z")
        df.plot(ax=ax[1, 0], kind='scatter', s=5, x="Displacement [m]", y="Force A [N]", color='white', edgecolors="orange", label="Sample-probe distance")
        Plot_rxc(1, 0, "", 14, "Distance (m)", "Force (N)", 12, 9, 7, 9)

        df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="lightblue", label="Loading scan")
        df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="navajowhite", label="Unloading scan")
        df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="steelblue", label="Corr. loading scan")
        df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="orange", label="Corr. unloading scan")
        baseline.plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="gray", label="Datapts used for BL linear fitting")
        df.plot(ax=ax[1, 1], kind='line', style='--', x="Indentation [m]", y="Baseline F [N]", color="black", label="Baseline")
        Plot_rxc(1, 1, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 7)
        plt.tight_layout()
        plt.savefig("{}_Initial analysis.png".format(path_filename_noext), bbox_inches='tight', dpi=300)

    # SELECT DATA FOR DMT FITTING AND PLOT RESULTS
    try:
        forward_scan, forward_scan_Fad, gamma_forward_scan_DMT, forward_scan_contactP, back_scan, back_scan_Fad, gamma_back_scan_DMT, back_scan_contactP, forward_scan_pars_DMT_best, forward_scan_cov_DMT_best, back_scan_pars_DMT_best, back_scan_cov_DMT_best, forward_scan_bulk_modulus_DMT_full_curve, forward_scan_bulk_modulus_error_DMT_full_curve, back_scan_bulk_modulus_DMT_full_curve, back_scan_bulk_modulus_error_DMT_full_curve, time_s = DMT_Fitting(df, forward_scan_segment, back_scan_segment, R, forward_scan_index_min, back_scan_index_min, Indentation_m_col, CorrForceA_N_col)
    except ValueError:
        txt.insert(END, f"Error: Datapoints per fitting segment parameter is too high.\n\n")
        Spacer()
        Spacer()
        txt.update()
        txt.see("end")
        return

    # Get individual datapoints to indicate best results in error analysis graphs.
    to_remove = ["NaN"]
    forward_scan_best_DMT = forward_scan[["Indentation step DMT [m]", "K DMT [Pa]", "ln (K Error DMT)"]]
    forward_scan_best_DMT = forward_scan_best_DMT[~forward_scan_best_DMT["ln (K Error DMT)"].isin(to_remove)]
    forward_scan_best_DMT = forward_scan_best_DMT.loc[forward_scan_best_DMT["ln (K Error DMT)"] == forward_scan_best_DMT["ln (K Error DMT)"].min()]
    back_scan_best_DMT = back_scan[["Indentation step DMT [m]", "K DMT [Pa]", "ln (K Error DMT)"]]
    back_scan_best_DMT = back_scan_best_DMT[~back_scan_best_DMT["ln (K Error DMT)"].isin(to_remove)]
    back_scan_best_DMT = back_scan_best_DMT.loc[back_scan_best_DMT["ln (K Error DMT)"] == back_scan_best_DMT["ln (K Error DMT)"].min()]

    # Plot all data for DMT
    Fig4, ax = plt.subplots(3, 2, figsize=(8, 7))
    df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[0, 0], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="steelblue", label='Loading scan')
    df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[0, 0], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="orange", label='Unloading scan')
    Plot_rxc(0, 0, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 9)

    forward_scan.plot(ax=ax[0, 1], kind='scatter', x="Indentation DMT Full Curve [m]", y="Corr Force A DMT Full Curve [N]", color='white', edgecolors='steelblue', label='Loading data')
    forward_scan.plot(ax=ax[0, 1], kind='line', style="--", x="Fitting Indentation DMT Full Curve [m]", y="Corr Force A DMT Full Curve [N]", color='black', label='DMT fitting')
    back_scan.plot(ax=ax[0, 1], kind='scatter', x="Indentation DMT Full Curve [m]", y="Corr Force A DMT Full Curve [N]", color='white', edgecolors='orange', label='Unloading data')
    back_scan.plot(ax=ax[0, 1], kind='line', style="--", x="Fitting Indentation DMT Full Curve [m]", y="Corr Force A DMT Full Curve [N]", color='gray', label='DMT fitting')
    Plot_rxc(0, 1, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 9)

    forward_scan.plot(ax=ax[1, 0], kind='scatter', s=10, x="Indentation step DMT [m]", y="ln (K Error DMT)", color='white', edgecolors='steelblue', label='Loading scan DMT')
    forward_scan_best_DMT.plot(ax=ax[1, 0], kind='scatter', s=30, x="Indentation step DMT [m]", y="ln (K Error DMT)", color='steelblue', edgecolors='steelblue')
    back_scan.plot(ax=ax[1, 0], kind='scatter', s=10, x="Indentation step DMT [m]", y="ln (K Error DMT)", color='white', edgecolors='orange', label='Unloading scan DMT')
    back_scan_best_DMT.plot(ax=ax[1, 0], kind='scatter', s=30, x="Indentation step DMT [m]", y="ln (K Error DMT)", color='orange', edgecolors='orange')
    Plot_rxc(1, 0, "", 14, "Indentation (m)", "ln (K error)", 12, 9, 7, 9)

    forward_scan.plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation step DMT [m]", y="K DMT [Pa]", color='white', edgecolors='steelblue', label='Loading scan DMT')
    forward_scan_best_DMT.plot(ax=ax[1, 1], kind='scatter', s=30, x="Indentation step DMT [m]", y="K DMT [Pa]", color='steelblue', edgecolors='steelblue')
    back_scan.plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation step DMT [m]", y="K DMT [Pa]", color='white', edgecolors='orange', label='Unloading scan DMT')
    back_scan_best_DMT.plot(ax=ax[1, 1], kind='scatter', s=30, x="Indentation step DMT [m]", y="K DMT [Pa]", color='orange', edgecolors='orange')
    Plot_rxc(1, 1, "", 14, "Indentation (m)", "K (Pa)", 12, 9, 7, 9)

    df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[2, 0], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="lightblue", label='Loading scan')
    df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[2, 0], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="navajowhite", label='Unloading scan')
    forward_scan.plot(ax=ax[2, 0], kind='scatter', s=10, x="Indentation DMT [m]", y="Corr Force A DMT [N]", color='white', edgecolors='steelblue', label='Sel. loading data')
    back_scan.plot(ax=ax[2, 0], kind='scatter', s=10, x="Indentation DMT [m]", y="Corr Force A DMT [N]", color='white', edgecolors='orange', label='Sel. unloading data')
    Plot_rxc(2, 0, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 9)

    forward_scan.plot(ax=ax[2, 1], kind='scatter', x="Indentation DMT [m]", y="Corr Force A DMT [N]", color='white', edgecolors='steelblue', label='Loading data')
    forward_scan.plot(ax=ax[2, 1], kind='line', style="--", x="Fitting Indentation DMT [m]", y="Corr Force A [N]", color='black', label='DMT fitting')
    back_scan.plot(ax=ax[2, 1], kind='scatter', x="Indentation Corr DMT [m]", y="Corr Force A DMT [N]", color='white', edgecolors='orange', label='Unloading data')
    back_scan.plot(ax=ax[2, 1], kind='line', style="--", x="Fitting Indentation DMT [m]", y="Corr Force A [N]", color='gray', label='DMT fitting')
    Plot_rxc(2, 1, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 9)
    plt.tight_layout()
    plt.savefig(f"{folder_path_fitting_restults}/{filename_noext}_Fitting analysis DMT.png", bbox_inches='tight', dpi=300)

    plt.close('all') #To close all figures and save memory - max # of figures before warning: 20


    # MAX INDENTATION AND MAX FORCE
    Max_Indent, Max_Force = Max_Indent_And_Max_Force(forward_scan, Indentation_m_col, CorrForceA_N_col)

    # PRINT RESULTS
    attr_force, adhes_force, sep_point, Es_LS_fc, Es_LS_ema, Es_US_fc, Es_US_ema, interf_en_LS, interf_en_US = DMT_Print_Results(path_filename_noext, folder_path_fitting_restults, filename_noext, df, R, v_sample, Zmov_type, forward_scan, back_scan, forward_scan_Fad, forward_scan_contactP, back_scan_Fad, back_scan_contactP, forward_scan_cov_DMT_best, forward_scan_pars_DMT_best, gamma_forward_scan_DMT, back_scan_cov_DMT_best, back_scan_pars_DMT_best, gamma_back_scan_DMT, scan_speed, Max_Indent, Max_Force, forward_scan_bulk_modulus_DMT_full_curve, forward_scan_bulk_modulus_error_DMT_full_curve, back_scan_bulk_modulus_DMT_full_curve, back_scan_bulk_modulus_error_DMT_full_curve, time_s)

    # GET DATA FOR 3D PLOT
    X, Y = GetXY(df, Zmov_type)

    return X, Y, attr_force, adhes_force, sep_point, Es_LS_fc, Es_LS_ema, Es_US_fc, Es_US_ema, interf_en_LS, interf_en_US


def PT_Fitting(df, forward_scan_segment, back_scan_segment, R, forward_scan_index_min, back_scan_index_min, Indentation_m_col, CorrForceA_N_col):

    #Define fitting functions
    def PT(x, alpha, dc, a, R, Fad):
        S = -2.160*(alpha**0.019) + 2.7531*(alpha**0.064) +0.073*(alpha**1.919)
        beta = 0.516*(alpha**4) - 0.683*(alpha**3) + 0.253*(alpha**2) + 0.429*alpha
        return -dc+(((a**2)/R)*(((alpha+(1+(x/Fad))**(1/2))/(1+alpha))**(4/3) - S*((alpha+(1+(x/Fad))**(1/2))/(1+alpha))**((2*beta)/3)))


    #Select starting point for fitting, get Fad, and calculate gamma
    forward_scan = pandas.DataFrame(df.groupby(["Exp Phase [#]"]).get_group(1.0))
    forward_scan = forward_scan.loc[forward_scan.index >= forward_scan_index_min] #Select data for fitting based on contact point
    forward_scan_Fad = forward_scan.iloc[0, CorrForceA_N_col]
    forward_scan_contactP = forward_scan.iloc[0, Indentation_m_col]*1e6
    #Fad cannot be 0 as it is the denominator in the equation.
    #The fitting uses -forward_scan_Fad therefore if Fad>0 that needs to be converted to -Fad.
    if forward_scan_Fad > 0:
        forward_scan_Fad = -forward_scan_Fad

    #Fit the entire indentation curve
    forward_scan_x_full_curve = forward_scan["Corr Force A [N]"].to_list()
    forward_scan_y_full_curve = forward_scan["Indentation [m]"].to_list()
    custom_PT = lambda x, alpha, dc, a: PT(x, alpha, dc, a, R, -forward_scan_Fad)  # Fix Fad value
    forward_scan_pars_PT_full_curve, forward_scan_cov_PT_full_curve = optimize.curve_fit(f=custom_PT, xdata=forward_scan_x_full_curve, ydata=forward_scan_y_full_curve)
    forward_scan_cov_PT_full_curve = np.sqrt(np.diag(forward_scan_cov_PT_full_curve)) #Necessary to convert error matrix output of optimize.curve_fit into list of standard deviations for each parameter

    forward_scan_alpha_PT_full_curve = forward_scan_pars_PT_full_curve[0]
    forward_scan_alpha_error_PT_full_curve = forward_scan_cov_PT_full_curve[0]
    forward_scan_dc_PT_full_curve = forward_scan_pars_PT_full_curve[1]
    forward_scan_dc_error_PT_full_curve = forward_scan_cov_PT_full_curve[1]
    forward_scan_a_PT_full_curve = forward_scan_pars_PT_full_curve[2]
    forward_scan_a_error_PT_full_curve = forward_scan_cov_PT_full_curve[2]
    forward_scan_a_calc_full_curve = -0.451 * (forward_scan_pars_PT_full_curve[0] ** 4) + 1.417 * (forward_scan_pars_PT_full_curve[0] ** 3) - 1.365 * (forward_scan_pars_PT_full_curve[0] ** 2) + 0.950 * forward_scan_pars_PT_full_curve[0] + 1.264
    forward_scan_Fad_calc_full_curve = 0.267 * (forward_scan_pars_PT_full_curve[0] ** 2) - 0.767 * forward_scan_pars_PT_full_curve[0] + 2.000
    forward_scan_K_PT_full_curve = ((forward_scan_a_calc_full_curve / forward_scan_pars_PT_full_curve[2]) ** 3) * (-forward_scan_Fad / forward_scan_Fad_calc_full_curve) * R
    forward_scan_K_error_PT_full_curve = (forward_scan_K_PT_full_curve * forward_scan_cov_PT_full_curve[0] / forward_scan_pars_PT_full_curve[0])
    forward_scan["Indentation PT Full Curve [m]"] = forward_scan["Indentation [m]"]
    forward_scan["Corr Force A PT Full Curve [N]"] = forward_scan["Corr Force A [N]"]
    #Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    S = -2.160 * (forward_scan_alpha_PT_full_curve ** 0.019) + 2.7531 * (forward_scan_alpha_PT_full_curve ** 0.064) + 0.073 * (forward_scan_alpha_PT_full_curve ** 1.919)
    beta = 0.516 * (forward_scan_alpha_PT_full_curve ** 4) - 0.683 * (forward_scan_alpha_PT_full_curve ** 3) + 0.253 * (forward_scan_alpha_PT_full_curve ** 2) + 0.429 * forward_scan_alpha_PT_full_curve
    forward_scan["Fitting Indentation PT Full Curve [m]"] = -forward_scan_dc_PT_full_curve+(((forward_scan_a_PT_full_curve ** 2) / R) * (((forward_scan_alpha_PT_full_curve + (1 + (forward_scan["Corr Force A [N]"] / (-forward_scan_Fad))) ** (1 / 2)) / (1 + forward_scan_alpha_PT_full_curve)) ** (4 / 3) - S * ((forward_scan_alpha_PT_full_curve + (1 + (forward_scan["Corr Force A [N]"] / (-forward_scan_Fad))) ** (1 / 2)) / (1 + forward_scan_alpha_PT_full_curve)) ** ((2 * beta) / 3)))
    gamma_forward_scan_PT_full_curve = -forward_scan_Fad / (forward_scan_Fad_calc_full_curve*R*math.pi)
    #forward_scan_alpha_PT_full_curve, forward_scan_alpha_error_PT_full_curve, forward_scan_dc_PT_full_curve, forward_scan_dc_error_PT_full_curve, forward_scan_a_PT_full_curve, forward_scan_a_error_PT_full_curve, forward_scan_K_PT_full_curve, forward_scan_K_error_PT_full_curve, gamma_forward_scan_PT

    #Optimise loading curve fitting
    forward_scan_initial_len = len(forward_scan)
    forward_scan_datapoints_to_fit = forward_scan_segment
    forward_scan_test = forward_scan.loc[forward_scan.index < (forward_scan.index[0] + forward_scan_datapoints_to_fit)] #Select data for fitting based on threshold value, index was not reset so forward_scan.index[-1] is a high number
    forward_scan_bulk_modulus_PT = []
    forward_scan_bulk_modulus_error_PT = []
    forward_scan_alpha_PT = []
    forward_scan_alpha_error_PT = []
    forward_scan_a_PT = []
    forward_scan_a_error_PT = []
    forward_scan_dc_PT = []
    forward_scan_dc_error_PT = []
    forward_scan_indentation_PT = []
    forward_scan_datapoints_PT = []

    while forward_scan_datapoints_to_fit <= forward_scan_initial_len:
        forward_scan_x = forward_scan_test["Corr Force A [N]"].to_list()
        forward_scan_y = forward_scan_test["Indentation [m]"].to_list()

        # PT fitting on selected data
        custom_PT = lambda x, alpha, dc, a: PT(x, alpha, dc, a, R, -forward_scan_Fad)  # Fix Fad value
        forward_scan_pars_PT, forward_scan_cov_PT = optimize.curve_fit(f=custom_PT, xdata=forward_scan_x, ydata=forward_scan_y)
        forward_scan_cov_PT = np.sqrt(np.diag(forward_scan_cov_PT)) #Necessary to convert error matrix output of optimize.curve_fit into list of standard deviations for each parameter

        #Calculate bulk moduls and error from values of alpha and in every look to generate 2 arrays
        forward_scan_alpha_PT.append(forward_scan_pars_PT[0])
        forward_scan_alpha_error_PT.append(forward_scan_cov_PT[0])

        forward_scan_dc_PT.append(forward_scan_pars_PT[1])
        forward_scan_dc_error_PT.append(forward_scan_cov_PT[1])

        forward_scan_a_PT.append(forward_scan_pars_PT[2])
        forward_scan_a_error_PT.append(forward_scan_cov_PT[2])

        forward_scan_a_calc = -0.451 * (forward_scan_pars_PT[0] ** 4) + 1.417 * (forward_scan_pars_PT[0] ** 3) - 1.365 * (forward_scan_pars_PT[0] ** 2) + 0.950 * forward_scan_pars_PT[0] + 1.264
        forward_scan_Fad_calc = 0.267 * (forward_scan_pars_PT[0] ** 2) - 0.767 * forward_scan_pars_PT[0] + 2.000
        forward_scan_K_PT = ((forward_scan_a_calc / forward_scan_pars_PT[2]) ** 3) * (-forward_scan_Fad / forward_scan_Fad_calc) * R
        forward_scan_bulk_modulus_PT.append(forward_scan_K_PT)
        K_error_forward_scan_PT = (forward_scan_K_PT * forward_scan_cov_PT[0] / forward_scan_pars_PT[0])
        forward_scan_bulk_modulus_error_PT.append(K_error_forward_scan_PT)

        forward_scan_indentation_PT.append(forward_scan_y[-1])
        forward_scan_datapoints_PT.append(forward_scan_datapoints_to_fit)

        forward_scan_datapoints_to_fit = forward_scan_datapoints_to_fit + forward_scan_segment
        forward_scan_test = forward_scan.loc[forward_scan.index < (forward_scan.index[0] + forward_scan_datapoints_to_fit)]  # Select data for fitting based on threshold value, index was not reset so forward_scan.index[-1] is a high number

    # Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    forward_scan_index_smallest_error = forward_scan_alpha_error_PT.index(min(forward_scan_alpha_error_PT))
    forward_scan_alpha_error_PT_best = forward_scan_alpha_error_PT[forward_scan_index_smallest_error]
    forward_scan_alpha_PT_best = forward_scan_alpha_PT[forward_scan_index_smallest_error]
    forward_scan_dc_error_PT_best = forward_scan_dc_error_PT[forward_scan_index_smallest_error]
    forward_scan_dc_PT_best = forward_scan_dc_PT[forward_scan_index_smallest_error]
    forward_scan_a_error_PT_best = forward_scan_a_error_PT[forward_scan_index_smallest_error]
    forward_scan_a_PT_best = forward_scan_a_PT[forward_scan_index_smallest_error]
    forward_scan_K_error_PT_best = forward_scan_bulk_modulus_error_PT[forward_scan_index_smallest_error]
    forward_scan_K_PT_best = forward_scan_bulk_modulus_PT[forward_scan_index_smallest_error]

    #Redeclare test df with best result and append Indentation and Corr Force columns to forward_scan for storage
    forward_scan_test = forward_scan.loc[forward_scan.index < (forward_scan.index[0] + forward_scan_datapoints_PT[forward_scan_index_smallest_error])]
    forward_scan["Indentation PT [m]"] = forward_scan_test["Indentation [m]"]
    forward_scan["Corr Force A PT [N]"] = forward_scan_test["Corr Force A [N]"]
    #Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    S = -2.160 * (forward_scan_alpha_PT_best ** 0.019) + 2.7531 * (forward_scan_alpha_PT_best ** 0.064) + 0.073 * (forward_scan_alpha_PT_best ** 1.919)
    beta = 0.516 * (forward_scan_alpha_PT_best ** 4) - 0.683 * (forward_scan_alpha_PT_best ** 3) + 0.253 * (forward_scan_alpha_PT_best ** 2) + 0.429 * forward_scan_alpha_PT_best
    forward_scan["Fitting Indentation PT [m]"] = -forward_scan_dc_PT_best+(((forward_scan_a_PT_best ** 2) / R) * (((forward_scan_alpha_PT_best + (1 + (forward_scan["Corr Force A [N]"] / (-forward_scan_Fad))) ** (1 / 2)) / (1 + forward_scan_alpha_PT_best)) ** (4 / 3) - S * ((forward_scan_alpha_PT_best + (1 + (forward_scan["Corr Force A [N]"] / (-forward_scan_Fad))) ** (1 / 2)) / (1 + forward_scan_alpha_PT_best)) ** ((2 * beta) / 3)))

    forward_scan_a_calc = -0.451 * (forward_scan_alpha_PT_best ** 4) + 1.417 * (forward_scan_alpha_PT_best ** 3) - 1.365 * (forward_scan_alpha_PT_best ** 2) + 0.950 * forward_scan_alpha_PT_best + 1.264
    forward_scan_Fad_calc = 0.267 * (forward_scan_alpha_PT_best ** 2) - 0.767 * forward_scan_alpha_PT_best + 2.000
    forward_scan_K_PT = ((forward_scan_a_calc / forward_scan_a_PT_best) ** 3) * (-forward_scan_Fad / forward_scan_Fad_calc) * R
    gamma_forward_scan_PT = -forward_scan_Fad / (forward_scan_Fad_calc*R*math.pi)


    #Extend lists with stored data to match the lenght of forward_scan df otherwise they cannot be added as columns to the df
    forward_scan_indentation_PT.extend(["NaN"]*(forward_scan_initial_len-len(forward_scan_indentation_PT)))
    forward_scan["Indentation step PT [m]"] = forward_scan_indentation_PT
    forward_scan_bulk_modulus_PT.extend(["NaN"]*(forward_scan_initial_len-len(forward_scan_bulk_modulus_PT)))
    forward_scan["K PT [Pa]"] = forward_scan_bulk_modulus_PT
    #Transform error into logarithm for best plots
    forward_scan_bulk_modulus_error_PT = [math.log(x) for x in forward_scan_bulk_modulus_error_PT]
    forward_scan_bulk_modulus_error_PT.extend(["NaN"]*(forward_scan_initial_len-len(forward_scan_bulk_modulus_error_PT)))
    forward_scan["ln (K Error PT)"] = forward_scan_bulk_modulus_error_PT

    forward_scan_alpha_PT.extend(["NaN"]*(forward_scan_initial_len-len(forward_scan_alpha_PT)))
    forward_scan["Alpha PT"] = forward_scan_alpha_PT
    #Transform error into logarithm for best plots
    forward_scan_alpha_error_PT = [math.log(x) for x in forward_scan_alpha_error_PT]
    forward_scan_alpha_error_PT.extend(["NaN"]*(forward_scan_initial_len-len(forward_scan_alpha_error_PT)))
    forward_scan["ln (Alpha Error PT)"] = forward_scan_alpha_error_PT


    #Select starting point for fitting, get Fad, and calculate gamma
    back_scan = pandas.DataFrame(df.groupby(["Exp Phase [#]"]).get_group(0.0))
    back_scan = back_scan.iloc[::-1].reset_index(drop=True) #Reverse and reindex dataframe
    back_scan = back_scan.loc[back_scan.index >= back_scan_index_min] #Select data for fitting based on threshold value
    back_scan_Fad = back_scan.iloc[0, CorrForceA_N_col]
    back_scan_contactP = back_scan.iloc[0, Indentation_m_col]*1e6

    # Fad cannot be 0 as is the denominator in the equation.
    # The fitting uses -forward_scan_Fad therefore if Fad>0 that needs to be converted to -Fad.
    if back_scan_Fad > 0:
        back_scan_Fad = -back_scan_Fad

    #Fit the entire indentation curve
    back_scan_x_full_curve = back_scan["Corr Force A [N]"].to_list()
    back_scan_y_full_curve = back_scan["Indentation [m]"].to_list()
    custom_PT = lambda x, alpha, dc, a: PT(x, alpha, dc, a, R, -back_scan_Fad)  # Fix Fad value
    back_scan_pars_PT_full_curve, back_scan_cov_PT_full_curve = optimize.curve_fit(f=custom_PT, xdata=back_scan_x_full_curve, ydata=back_scan_y_full_curve)
    back_scan_cov_PT_full_curve = np.sqrt(np.diag(back_scan_cov_PT_full_curve)) #Necessary to convert error matrix output of optimize.curve_fit into list of standard deviations for each parameter
    back_scan_alpha_PT_full_curve = back_scan_pars_PT_full_curve[0]
    back_scan_alpha_error_PT_full_curve = back_scan_cov_PT_full_curve[0]
    back_scan_dc_PT_full_curve = back_scan_pars_PT_full_curve[1]
    back_scan_dc_error_PT_full_curve = back_scan_cov_PT_full_curve[1]
    back_scan_a_PT_full_curve = back_scan_pars_PT_full_curve[2]
    back_scan_a_error_PT_full_curve = back_scan_cov_PT_full_curve[2]
    back_scan_a_calc_full_curve = -0.451 * (back_scan_pars_PT_full_curve[0] ** 4) + 1.417 * (back_scan_pars_PT_full_curve[0] ** 3) - 1.365 * (back_scan_pars_PT_full_curve[0] ** 2) + 0.950 * back_scan_pars_PT_full_curve[0] + 1.264
    back_scan_Fad_calc_full_curve = 0.267 * (back_scan_pars_PT_full_curve[0] ** 2) - 0.767 * back_scan_pars_PT_full_curve[0] + 2.000
    back_scan_K_PT_full_curve = ((back_scan_a_calc_full_curve / back_scan_pars_PT_full_curve[2]) ** 3) * (-back_scan_Fad / back_scan_Fad_calc_full_curve) * R
    back_scan_K_error_PT_full_curve = (back_scan_K_PT_full_curve * back_scan_cov_PT_full_curve[0] / back_scan_pars_PT_full_curve[0])
    back_scan["Indentation PT Full Curve [m]"] = back_scan["Indentation [m]"]
    back_scan["Corr Force A PT Full Curve [N]"] = back_scan["Corr Force A [N]"]
    #Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    S = -2.160 * (back_scan_alpha_PT_full_curve ** 0.019) + 2.7531 * (back_scan_alpha_PT_full_curve ** 0.064) + 0.073 * (back_scan_alpha_PT_full_curve ** 1.919)
    beta = 0.516 * (back_scan_alpha_PT_full_curve ** 4) - 0.683 * (back_scan_alpha_PT_full_curve ** 3) + 0.253 * (back_scan_alpha_PT_full_curve ** 2) + 0.429 * back_scan_alpha_PT_full_curve
    back_scan["Fitting Indentation PT Full Curve [m]"] = -back_scan_dc_PT_full_curve+(((back_scan_a_PT_full_curve ** 2) / R) * (((back_scan_alpha_PT_full_curve + (1 + (back_scan["Corr Force A [N]"] / (-back_scan_Fad))) ** (1 / 2)) / (1 + back_scan_alpha_PT_full_curve)) ** (4 / 3) - S * ((back_scan_alpha_PT_full_curve + (1 + (back_scan["Corr Force A [N]"] / (-back_scan_Fad))) ** (1 / 2)) / (1 + back_scan_alpha_PT_full_curve)) ** ((2 * beta) / 3)))
    gamma_back_scan_PT_full_curve = -back_scan_Fad / (back_scan_Fad_calc_full_curve*R*math.pi)

    #Optimise loading curve fitting
    back_scan_initial_len = len(back_scan)
    back_scan_datapoints_to_fit = back_scan_segment
    back_scan_test = back_scan.loc[back_scan.index < (back_scan.index[0] + back_scan_datapoints_to_fit)] #Select data for fitting based on threshold value, index was not reset so forward_scan.index[-1] is a high number
    back_scan_bulk_modulus_PT = []
    back_scan_bulk_modulus_error_PT = []
    back_scan_alpha_PT = []
    back_scan_alpha_error_PT = []
    back_scan_a_PT = []
    back_scan_a_error_PT = []
    back_scan_dc_PT = []
    back_scan_dc_error_PT = []
    back_scan_indentation_PT = []
    back_scan_datapoints_PT = []

    while back_scan_datapoints_to_fit <= back_scan_initial_len:
        back_scan_x = back_scan_test["Corr Force A [N]"].to_list()
        back_scan_y = back_scan_test["Indentation [m]"].to_list()

        # PT fitting on selected data
        custom_PT = lambda x, alpha, dc, a: PT(x, alpha, dc, a, R, -back_scan_Fad)  # Fix Fad value
        back_scan_pars_PT, back_scan_cov_PT = optimize.curve_fit(f=custom_PT, xdata=back_scan_x, ydata=back_scan_y)
        back_scan_cov_PT = np.sqrt(np.diag(back_scan_cov_PT))  # Necessary to convert error matrix output of optimize.curve_fit into list of standard deviations for each parameter

        # Calculate bulk moduls and error from values of alpha and in every look to generate 2 arrays
        back_scan_alpha_PT.append(back_scan_pars_PT[0])
        back_scan_alpha_error_PT.append(back_scan_cov_PT[0])

        back_scan_dc_PT.append(back_scan_pars_PT[1])
        back_scan_dc_error_PT.append(back_scan_cov_PT[1])

        back_scan_a_PT.append(back_scan_pars_PT[2])
        back_scan_a_error_PT.append(back_scan_cov_PT[2])

        back_scan_a_calc = -0.451 * (back_scan_pars_PT[0] ** 4) + 1.417 * (back_scan_pars_PT[0] ** 3) - 1.365 * (back_scan_pars_PT[0] ** 2) + 0.950 * back_scan_pars_PT[0] + 1.264
        back_scan_Fad_calc = 0.267 * (back_scan_pars_PT[0] ** 2) - 0.767 * back_scan_pars_PT[0] + 2.000
        back_scan_K_PT = ((back_scan_a_calc / back_scan_pars_PT[2]) ** 3) * (-back_scan_Fad / back_scan_Fad_calc) * R
        back_scan_bulk_modulus_PT.append(back_scan_K_PT)
        K_error_back_scan_PT = (back_scan_K_PT * back_scan_cov_PT[0] / back_scan_pars_PT[0])
        back_scan_bulk_modulus_error_PT.append(K_error_back_scan_PT)

        back_scan_indentation_PT.append(back_scan_y[-1])
        back_scan_datapoints_PT.append(back_scan_datapoints_to_fit)

        back_scan_datapoints_to_fit = back_scan_datapoints_to_fit + back_scan_segment
        back_scan_test = back_scan.loc[back_scan.index < (back_scan.index[0] + back_scan_datapoints_to_fit)]  # Select data for fitting based on threshold value, index was not reset so forward_scan.index[-1] is a high number

    # Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    back_scan_index_smallest_error = back_scan_alpha_error_PT.index(min(back_scan_alpha_error_PT))
    back_scan_alpha_error_PT_best = back_scan_alpha_error_PT[back_scan_index_smallest_error]
    back_scan_alpha_PT_best = back_scan_alpha_PT[back_scan_index_smallest_error]
    back_scan_dc_error_PT_best = back_scan_dc_error_PT[back_scan_index_smallest_error]
    back_scan_dc_PT_best = back_scan_dc_PT[back_scan_index_smallest_error]
    back_scan_a_error_PT_best = back_scan_a_error_PT[back_scan_index_smallest_error]
    back_scan_a_PT_best = back_scan_a_PT[back_scan_index_smallest_error]
    back_scan_K_error_PT_best = back_scan_bulk_modulus_error_PT[back_scan_index_smallest_error]
    back_scan_K_PT_best = back_scan_bulk_modulus_PT[back_scan_index_smallest_error]

    # Redeclare test df with best result and append Indentation and Corr Force columns to forward_scan for storage
    back_scan_test = back_scan.loc[back_scan.index < (back_scan.index[0] + back_scan_datapoints_PT[back_scan_index_smallest_error])]
    back_scan["Indentation PT [m]"] = back_scan_test["Indentation [m]"]
    back_scan["Corr Force A PT [N]"] = back_scan_test["Corr Force A [N]"]
    # Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    S = -2.160 * (back_scan_alpha_PT_best ** 0.019) + 2.7531 * (back_scan_alpha_PT_best ** 0.064) + 0.073 * (back_scan_alpha_PT_best ** 1.919)
    beta = 0.516 * (back_scan_alpha_PT_best ** 4) - 0.683 * (back_scan_alpha_PT_best ** 3) + 0.253 * (back_scan_alpha_PT_best ** 2) + 0.429 * back_scan_alpha_PT_best
    back_scan["Fitting Indentation PT [m]"] = -back_scan_dc_PT_best + (((back_scan_a_PT_best ** 2) / R) * (((back_scan_alpha_PT_best + (1 + (back_scan["Corr Force A [N]"] / (-back_scan_Fad))) ** (1 / 2)) / (1 + back_scan_alpha_PT_best)) ** (4 / 3) - S * ((back_scan_alpha_PT_best + (1 + (back_scan["Corr Force A [N]"] / (-back_scan_Fad))) ** (1 / 2)) / (1 + back_scan_alpha_PT_best)) ** ((2 * beta) / 3)))

    back_scan_a_calc = -0.451 * (back_scan_alpha_PT_best ** 4) + 1.417 * (back_scan_alpha_PT_best ** 3) - 1.365 * (back_scan_alpha_PT_best ** 2) + 0.950 * back_scan_alpha_PT_best + 1.264
    back_scan_Fad_calc = 0.267 * (back_scan_alpha_PT_best ** 2) - 0.767 * back_scan_alpha_PT_best + 2.000
    back_scan_K_PT = ((back_scan_a_calc / back_scan_a_PT_best) ** 3) * (-back_scan_Fad / back_scan_Fad_calc) * R
    gamma_back_scan_PT = -back_scan_Fad / (back_scan_Fad_calc * R * math.pi)

    # Extend lists with stored data to match the lenght of forward_scan df otherwise they cannot be added as columns to the df
    back_scan_indentation_PT.extend(["NaN"] * (back_scan_initial_len - len(back_scan_indentation_PT)))
    back_scan["Indentation step PT [m]"] = back_scan_indentation_PT
    back_scan_bulk_modulus_PT.extend(["NaN"] * (back_scan_initial_len - len(back_scan_bulk_modulus_PT)))
    back_scan["K PT [Pa]"] = back_scan_bulk_modulus_PT
    # Transform error into logarithm for best plots
    back_scan_bulk_modulus_error_PT = [math.log(x) for x in back_scan_bulk_modulus_error_PT]
    back_scan_bulk_modulus_error_PT.extend(["NaN"] * (back_scan_initial_len - len(back_scan_bulk_modulus_error_PT)))
    back_scan["ln (K Error PT)"] = back_scan_bulk_modulus_error_PT

    back_scan_alpha_PT.extend(["NaN"] * (back_scan_initial_len - len(back_scan_alpha_PT)))
    back_scan["Alpha PT"] = back_scan_alpha_PT
    # Transform error into logarithm for best plots
    back_scan_alpha_error_PT = [math.log(x) for x in back_scan_alpha_error_PT]
    back_scan_alpha_error_PT.extend(["NaN"] * (back_scan_initial_len - len(back_scan_alpha_error_PT)))
    back_scan["ln (Alpha Error PT)"] = back_scan_alpha_error_PT

    # Define time
    time_s_col = df.columns.get_loc("Time [s]")  # Get column number
    time_s = df.iloc[0, time_s_col]

    return forward_scan, forward_scan_Fad, forward_scan_contactP, back_scan, back_scan_Fad, back_scan_contactP, gamma_forward_scan_PT, gamma_back_scan_PT, forward_scan_alpha_PT_best, forward_scan_alpha_error_PT_best, forward_scan_dc_PT_best, forward_scan_dc_error_PT_best, forward_scan_a_PT_best, forward_scan_a_error_PT_best, forward_scan_K_PT_best, forward_scan_K_error_PT_best, back_scan_alpha_PT_best, back_scan_alpha_error_PT_best, back_scan_dc_PT_best, back_scan_dc_error_PT_best, back_scan_a_PT_best, back_scan_a_error_PT_best, back_scan_K_error_PT_best, back_scan_K_PT_best, forward_scan_alpha_PT_full_curve, forward_scan_alpha_error_PT_full_curve, forward_scan_dc_PT_full_curve, forward_scan_dc_error_PT_full_curve, forward_scan_a_PT_full_curve, forward_scan_a_error_PT_full_curve, forward_scan_K_PT_full_curve, forward_scan_K_error_PT_full_curve, gamma_forward_scan_PT_full_curve, back_scan_alpha_PT_full_curve, back_scan_alpha_error_PT_full_curve, back_scan_dc_PT_full_curve, back_scan_dc_error_PT_full_curve, back_scan_a_PT_full_curve, back_scan_a_error_PT_full_curve, back_scan_K_PT_full_curve, back_scan_K_error_PT_full_curve, gamma_back_scan_PT_full_curve, time_s

def PT_Print_Results(path_filename_noext, folder_path_fitting_restults, filename_noext, df, R, v_sample, Zmov_type, forward_scan, back_scan, forward_scan_Fad, forward_scan_contactP, back_scan_Fad, back_scan_contactP, scan_speed, Max_Indent, Max_Force, gamma_forward_scan_PT, gamma_back_scan_PT, forward_scan_dc_PT_best, forward_scan_dc_error_PT_best, back_scan_dc_PT_best, back_scan_dc_error_PT_best, forward_scan_K_PT_best, forward_scan_K_error_PT_best, back_scan_K_error_PT_best, back_scan_K_PT_best, forward_scan_alpha_PT_full_curve, forward_scan_alpha_error_PT_full_curve, forward_scan_dc_PT_full_curve, forward_scan_dc_error_PT_full_curve, forward_scan_a_PT_full_curve, forward_scan_a_error_PT_full_curve, forward_scan_K_PT_full_curve, forward_scan_K_error_PT_full_curve, gamma_forward_scan_PT_full_curve, back_scan_alpha_PT_full_curve, back_scan_alpha_error_PT_full_curve, back_scan_dc_PT_full_curve, back_scan_dc_error_PT_full_curve, back_scan_a_PT_full_curve, back_scan_a_error_PT_full_curve, back_scan_K_PT_full_curve, back_scan_K_error_PT_full_curve, gamma_back_scan_PT_full_curve, time_s):

    #Print results to screen
    txt.insert(END, "Probe radius: {} um\n".format(R*1e6))
    txt.insert(END, "Elapsed time: {} s\n".format(round(time_s)))

    txt.insert(END, "Poisson's ratio sample: {}\n".format(v_sample))
    txt.insert(END, "Indentation speed: {} um/s\n".format(round(scan_speed, 3)))
    txt.insert(END, "Max indentation: {} um\n".format(round(Max_Indent*1e6, 3)))
    txt.insert(END, "Max force: {:,} uN\n".format(round(Max_Force*1e6, 3)))

    txt.insert(END, "Contact point: {} um\n".format(str(round(forward_scan_contactP, 3))))
    txt.insert(END, "Attractive force: {} uN\n".format(str(round(forward_scan_Fad*1e6, 3))))
    txt.insert(END, "PZI_fc (loading scan): {} um\n".format(np.round(forward_scan_dc_PT_full_curve*1e6, 3)))
    txt.insert(END, "PZI_ema (loading scan): {} um\n".format(np.round(forward_scan_dc_PT_best*1e6, 3)))

    txt.insert(END, "Separation point: {} um\n".format(str(round(back_scan_contactP, 3))))
    txt.insert(END, "Adhesion force: {} uN\n".format(str(round(back_scan_Fad*1e6, 3))))
    txt.insert(END, "PZI_fc (unloading scan): {} um\n".format(np.round(back_scan_dc_PT_full_curve*1e6, 3)))
    txt.insert(END, "PZI_ema (unloading scan): {} um\n\n".format(np.round(back_scan_dc_PT_best*1e6, 3)))
    txt.update()
    txt.see("end")

    #Calculate and report elastic modulus sample (Es)
    Es_forward_scan_PT_full_curve = (3/4)*forward_scan_K_PT_full_curve*(1-(v_sample**2))
    Es_error_forward_scan_PT_full_curve = (Es_forward_scan_PT_full_curve*forward_scan_K_error_PT_full_curve)/forward_scan_K_PT_full_curve
    Es_forward_scan_PT = (3/4)*forward_scan_K_PT_best*(1-(v_sample**2))
    Es_error_forward_scan_PT = (Es_forward_scan_PT*forward_scan_K_error_PT_best)/forward_scan_K_PT_best
    txt.insert(END, "K_fc (PT, loading scan) = {:,} +/- {:,} Pa\n".format(round(forward_scan_K_PT_full_curve, 2), round(forward_scan_K_error_PT_full_curve, 2)))
    txt.insert(END, "K_ema (PT, loading scan) = {:,} +/- {:,} Pa\n".format(round(forward_scan_K_PT_best, 2), round(forward_scan_K_error_PT_best, 2)))
    txt.insert(END, "Es_fc (PT, loading scan) = {:,} +/- {:,} Pa\n".format(round(Es_forward_scan_PT_full_curve, 2), round(Es_error_forward_scan_PT_full_curve, 2)))
    txt.insert(END, "Es_ema (PT, loading scan) = {:,} +/- {:,} Pa\n".format(round(Es_forward_scan_PT, 2), round(Es_error_forward_scan_PT, 2)))
    txt.insert(END, "Interfacial energy_fc (PT, loading scan): {} mJ m^-2\n".format(round(gamma_forward_scan_PT_full_curve*1000, 2)))
    txt.insert(END, "Interfacial energy_ema (PT, loading scan): {} mJ m^-2\n\n".format(round(gamma_forward_scan_PT*1000, 2)))

    Es_back_scan_PT_full_curve = (3/4)*back_scan_K_PT_full_curve*(1-(v_sample**2))
    Es_error_back_scan_PT_full_curve = (Es_back_scan_PT_full_curve*back_scan_K_error_PT_full_curve)/back_scan_K_PT_full_curve
    Es_back_scan_PT = (3/4)*back_scan_K_PT_best*(1-(v_sample**2))
    Es_error_back_scan_PT = (Es_back_scan_PT*back_scan_K_error_PT_best)/back_scan_K_PT_best
    txt.insert(END, "K_fc (PT, unloading scan) = {:,} +/- {:,} Pa\n".format(round(back_scan_K_PT_full_curve, 2), round(back_scan_K_error_PT_full_curve, 2)))
    txt.insert(END, "K_ema (PT, unloading scan) = {:,} +/- {:,} Pa\n".format(round(back_scan_K_PT_best, 2), round(back_scan_K_error_PT_best, 2)))
    txt.insert(END, "Es_fc (PT, unloading scan) = {:,} +/- {:,} Pa\n".format(round(Es_back_scan_PT_full_curve, 2), round(Es_error_back_scan_PT_full_curve, 2)))
    txt.insert(END, "Es_ema (PT, unloading scan) = {:,} +/- {:,} Pa\n".format(round(Es_back_scan_PT, 2), round(Es_error_back_scan_PT, 2)))
    txt.insert(END, "Interfacial energy_fc (PT, unloading scan): {} mJ m^-2\n".format(round(gamma_back_scan_PT_full_curve*1000, 2)))
    txt.insert(END, "Interfacial energy_ema (PT, unloading scan): {} mJ m^-2\n\n".format(round(gamma_back_scan_PT*1000, 2)))


    forward_Tabor_param_PT = ((R*(gamma_forward_scan_PT**2))/((Es_forward_scan_PT**2)*((0.333*1e-9)**3)))**(1/3)
    back_Tabor_param_PT = ((R*(gamma_back_scan_PT**2))/((Es_back_scan_PT**2)*((0.333*1e-9)**3)))**(1/3)
    txt.insert(END, "u (PT, loading scan) = {}\n".format(round(forward_Tabor_param_PT,2)))
    txt.insert(END, "u (PT, unloading scan) = {}\n".format(round(back_Tabor_param_PT,2)))
    txt.update()
    txt.see("end")
    Spacer()

    #Save full curve
    df.to_csv(f"{path_filename_noext}_full curve.csv", index=False)

    #Save fitting results for loading curve
    file = open(f"{folder_path_fitting_restults}/{filename_noext}_LS_PT fitting results.csv", "w")
    #Because it will be saved as csv, Excell will switch column when it sees a comma, hence all the ","
    file.write("Measurement type:, {}\n".format(Zmov_type))
    file.write("Probe radius:, {}, um\n".format(R*1e6))
    file.write("Poisson's ratio sample:, {}\n".format(v_sample))
    file.write("Indentation speed:, {}, um/s\n".format(round(scan_speed, 3)))
    file.write("Max indentation:, {}, um\n".format(round(Max_Indent*1e6, 3)))
    file.write("Max force:, {}, uN\n".format(round(Max_Force*1e6, 3)))
    file.write("Contact point:, {}, um\n".format(str(round(forward_scan_contactP, 3))))
    file.write("Attractive force:, {}, uN\n".format(str(round(forward_scan_Fad*1e6, 3))))
    file.write("PZI_fc (loading scan):, {}, um\n".format(np.round(forward_scan_dc_PT_full_curve*1e6, 3)))
    file.write("PZI_ema (loading scan):, {}, um\n".format(np.round(forward_scan_dc_PT_best*1e6, 3)))
    file.write("Separation point:, {}, um\n".format(str(round(back_scan_contactP, 3))))
    file.write("Adhesion force:, {}, uN\n".format(str(round(back_scan_Fad*1e6, 3))))
    file.write("PZI_fc (unloading scan):, {}, um\n".format(np.round(back_scan_dc_PT_full_curve*1e6, 3)))
    file.write("PZI_ema (unloading scan):, {}, um\n\n".format(np.round(back_scan_dc_PT_best*1e6, 3)))

    file.write("K_fc (PT loading scan) =, {}, +/-, {}, Pa\n".format(round(forward_scan_K_PT_full_curve, 2), round(forward_scan_K_error_PT_full_curve, 2)))
    file.write("K_ema (PT loading scan) =, {}, +/-, {}, Pa\n".format(round(forward_scan_K_PT_best, 2), round(forward_scan_K_error_PT_best, 2)))
    file.write("Es_fc (PT loading scan) =, {}, +/-, {}, Pa\n".format(round(Es_forward_scan_PT_full_curve, 2), round(Es_error_forward_scan_PT_full_curve, 2)))
    file.write("Es_ema (PT loading scan) =, {}, +/-, {}, Pa\n".format(round(Es_forward_scan_PT, 2), round(Es_error_forward_scan_PT, 2)))
    file.write("Interfacial energy_fc (PT loading scan):, {}, mJ m^-2\n".format(round(gamma_forward_scan_PT_full_curve*1000, 2)))
    file.write("Interfacial energy_ema (PT loading scan):, {}, mJ m^-2\n\n".format(round(gamma_forward_scan_PT*1000, 2)))

    file.write("u (PT loading scan):, {}\n\n".format(round(forward_Tabor_param_PT, 2)))
    forward_scan.to_csv(file, index=False)
    file.close()

    #Save fitting results for unloading curve
    file = open(f"{folder_path_fitting_restults}/{filename_noext}_US_PT fitting results.csv", "w")
    #Because it will be saved as csv, Excell will switch column when it sees a comma, hence all the ","
    file.write("Measurement type:, {}\n".format(Zmov_type))
    file.write("Probe radius:, {}, um\n".format(R*1e6))
    file.write("Poisson's ratio sample:, {}\n".format(v_sample))
    file.write("Indentation speed:, {}, um/s\n".format(round(scan_speed, 3)))
    file.write("Max indentation:, {}, um\n".format(round(Max_Indent*1e6, 3)))
    file.write("Max force:, {}, uN\n".format(round(Max_Force*1e6, 3)))
    file.write("Contact point:, {}, um\n".format(str(round(forward_scan_contactP, 3))))
    file.write("Attractive force:, {}, uN\n".format(str(round(forward_scan_Fad*1e6, 3))))
    file.write("PZI_fc (loading scan):, {}, um\n".format(np.round(forward_scan_dc_PT_full_curve*1e6, 3)))
    file.write("PZI_ema (loading scan):, {}, um\n".format(np.round(forward_scan_dc_PT_best*1e6, 3)))
    file.write("Separation point:, {}, um\n".format(str(round(back_scan_contactP, 3))))
    file.write("Adhesion force:, {}, uN\n".format(str(round(back_scan_Fad*1e6, 3))))
    file.write("PZI_fc (unloading scan):, {}, um\n".format(np.round(back_scan_dc_PT_full_curve*1e6, 3)))
    file.write("PZI_ema (unloading scan):, {}, um\n\n".format(np.round(back_scan_dc_PT_best*1e6, 3)))

    file.write("K_fc (PT unloading scan) =, {}, +/-, {}, Pa\n".format(round(back_scan_K_PT_full_curve, 2), round(back_scan_K_error_PT_full_curve, 2)))
    file.write("K_ema (PT unloading scan) =, {}, +/-, {}, Pa\n".format(round(back_scan_K_PT_best, 2), round(back_scan_K_error_PT_best, 2)))
    file.write("Es_fc (PT unloading scan) =, {}, +/-, {}, Pa\n".format(round(Es_back_scan_PT_full_curve, 2), round(Es_error_back_scan_PT_full_curve, 2)))
    file.write("Es_ema (PT unloading scan) =, {}, +/-, {}, Pa\n".format(round(Es_back_scan_PT, 2), round(Es_error_back_scan_PT, 2)))
    file.write("Interfacial energy_fc (PT unloading scan):, {}, mJ m^-2\n".format(round(gamma_back_scan_PT_full_curve*1000, 2)))
    file.write("Interfacial energy_ema (PT unloading scan):, {}, mJ m^-2\n\n".format(round(gamma_back_scan_PT*1000, 2)))

    file.write("u (PT unloading scan):, {} \n\n".format(round(back_Tabor_param_PT, 2)))

    back_scan.to_csv(file, index=False)

    file.close()

    #Return data for 3D maps
    attr_force = round(forward_scan_Fad * 1e6, 3)
    PZI_LS = np.round(forward_scan_dc_PT_best*1e6, 3)
    adhes_force = round(back_scan_Fad * 1e6, 3)
    sep_point = round(back_scan_contactP, 3)
    PZI_US = np.round(back_scan_dc_PT_best*1e6, 3)
    Es_LS_fc = round(Es_forward_scan_PT_full_curve, 2)
    Es_LS_ema = round(Es_forward_scan_PT, 2)
    Es_US_fc = round(Es_back_scan_PT_full_curve, 2)
    Es_US_ema = round(Es_back_scan_PT, 2)
    interf_en_LS = round(gamma_forward_scan_PT, 2)
    interf_en_US = round(gamma_back_scan_PT, 2)

    return attr_force, PZI_LS, adhes_force, sep_point, PZI_US, Es_LS_fc, Es_LS_ema, Es_US_fc, Es_US_ema, interf_en_LS, interf_en_US

def PT_Analyse_Single_CSVFile(path_filename, folder_path_fitting_restults, filename_noext, ratio_BL_points, threshold_constant, forward_scan_segment, back_scan_segment, R, v_sample):

    def Plot_rxc(r, c, title, title_fontsize, xlabel, ylabel, xylabel_fontsize, axes_fontsize, tick_lenght, legend_fontsize):
        ax[r, c].set_title(title, fontsize=title_fontsize)
        ax[r, c].set_xlabel(xlabel, fontsize=xylabel_fontsize)
        ax[r, c].set_ylabel(ylabel, fontsize=xylabel_fontsize)
        ax[r, c].tick_params(direction='in', length=tick_lenght, labelsize=axes_fontsize)
        ax[r, c].legend(fontsize=legend_fontsize, bbox_to_anchor=(0., 1.1, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

    # OPEN FILE AND TRANSFORM UNITS TO N AND M
    df = pandas.read_table("{}".format(path_filename), sep=',', header=0)
    path_filename_noext, ext = os.path.splitext(path_filename) #Remove ".csv" from filename to avoid weird name for documents
    df = df.astype(float)  #Change data from object to float
    R, ForceA_N_col, Time_s_col, Corr_Time_s_col, Displacement_um_col, Displacement_m_col, PiezoZ_um_col, PiezoZ_m_col, \
    PosZ_um_col, PosZ_m_col = Unit_Transform(df, R)
    # ----------------------------------------------------------------------------------------------------------------------#
    # DETERMINE IF MEASUREMENT WAS CARRIED OUT USING PIEZO SCANNER OR STEP MOTOR AND DETERMINE POINT OF CONTACT

    Zmov_type, Probe_Displacement_m_col, forward_scan, back_scan, forward_scan_index_min, forward_scan_min, back_scan_index_min, back_scan_min = Contact_Point(df, PosZ_um_col, PosZ_m_col, PiezoZ_m_col, Displacement_m_col, ratio_BL_points, threshold_constant)
    # ----------------------------------------------------------------------------------------------------------------------#
    # DETERMINE SCAN SPEED

    scan_speed = Indentation_Speed(Zmov_type, forward_scan)
    # ----------------------------------------------------------------------------------------------------------------------#
    # BASELINE AND ZERO DATA

    baseline, CorrForceA_N_col, BaselineF_N_col, Indentation_m_col = Baseline_And_Zero(df, forward_scan, forward_scan_index_min, Displacement_m_col, ratio_BL_points)
    # ----------------------------------------------------------------------------------------------------------------------#
    # PLOT INITIAL ANALYSIS

    if Zmov_type == "Piezo scanner":
        fig1, ax = plt.subplots(2, 2, figsize=(8, 7))
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Piezo Z [m]", color='white', edgecolors='steelblue', label='Piezo Z')
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Displacement [m]", color='white', edgecolors="orange", label='Sample-probe distance')
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Probe Displacement [m]", color='white', edgecolors="firebrick", label='Probe displacement')
        Plot_rxc(0, 0, "", 14, "Time (s)", "Distance (m)", 12, 9, 7, 9)

        df.plot(ax=ax[0, 1], kind='scatter', s=5, x="Piezo Z [m]", y="Displacement [m]", color='white', edgecolors='steelblue', label='Sample-probe distance vs Piezo Z')
        Plot_rxc(0, 1, "", 14, "Piezo Z (m)", "Sample-probe distance (m)", 12, 9, 7, 9)

        df.plot(ax=ax[1, 0], kind='scatter', s=5, x="Piezo Z [m]", y="Force A [N]", color='white', edgecolors='steelblue', label="Piezo Z")
        df.plot(ax=ax[1, 0], kind='scatter', s=5, x="Displacement [m]", y="Force A [N]", color='white', edgecolors="orange", label="Sample-probe distance")
        Plot_rxc(1, 0, "", 14, "Distance (m)", "Force (N)", 12, 9, 7, 9)

        df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="lightblue", label="Loading scan")
        df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="navajowhite", label="Unloading scan")
        df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="steelblue", label="Corr. loading scan")
        df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="orange", label="Corr. unloading scan")
        baseline.plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="gray", label="Datapts used for BL linear fitting")
        df.plot(ax=ax[1, 1], kind='line', style='--', x="Indentation [m]", y="Baseline F [N]", color="black", label="Baseline")
        Plot_rxc(1, 1, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 7)
        plt.tight_layout()
        plt.savefig("{}_Initial analysis.png".format(path_filename_noext), bbox_inches='tight', dpi=300)

    if Zmov_type == "Stick-slip actuator":
        fig1, ax = plt.subplots(2, 2, figsize=(8, 7))
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Pos Z [m]", color='white', edgecolors='steelblue', label="Pos Z")
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Displacement [m]", color='white', edgecolors="orange", label="Sample-probe distance")
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Probe Displacement [m]", color='white', edgecolors="firebrick", label="Probe displacement")
        Plot_rxc(0, 0, "", 14, "Time (s)", "Distance (m)", 12, 9, 7, 9)

        df.plot(ax=ax[0, 1], kind='scatter', s=5, x="Pos Z [m]", y="Displacement [m]", color='white', edgecolors='steelblue', label='Sample-probe distance vs Piezo Z')
        Plot_rxc(0, 1, "", 14, "Piezo Z (m)", "Sample-probe distance (m)", 12, 9, 7, 9)

        df.plot(ax=ax[1, 0], kind='scatter', s=5, x="Pos Z [m]", y="Force A [N]", color='white', edgecolors='steelblue', label="Piezo Z")
        df.plot(ax=ax[1, 0], kind='scatter', s=5, x="Displacement [m]", y="Force A [N]", color='white', edgecolors="orange", label="Sample-probe distance")
        Plot_rxc(1, 0, "", 14, "Distance (m)", "Force (N)", 12, 9, 7, 9)

        df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="lightblue", label="Loading scan")
        df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="navajowhite", label="Unloading scan")
        df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="steelblue", label="Corr. loading scan")
        df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="orange", label="Corr. unloading scan")
        baseline.plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="gray", label="Datapts used for BL linear fitting")
        df.plot(ax=ax[1, 1], kind='line', style='--', x="Indentation [m]", y="Baseline F [N]", color="black", label="Baseline")
        Plot_rxc(1, 1, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 7)
        plt.tight_layout()
        plt.savefig("{}_Initial analysis.png".format(path_filename_noext), bbox_inches='tight', dpi=300)
    # ----------------------------------------------------------------------------------------------------------------------#
    # SELECT DATA FOR DMT AND PT FITTING AND PLOT RESULTS
    try:
        forward_scan, forward_scan_Fad, forward_scan_contactP, back_scan, back_scan_Fad, back_scan_contactP, gamma_forward_scan_PT, gamma_back_scan_PT, forward_scan_alpha_PT_best, forward_scan_alpha_error_PT_best, forward_scan_dc_PT_best, forward_scan_dc_error_PT_best, forward_scan_a_PT_best, forward_scan_a_error_PT_best, forward_scan_K_PT_best, forward_scan_K_error_PT_best, back_scan_alpha_PT_best, back_scan_alpha_error_PT_best, back_scan_dc_PT_best, back_scan_dc_error_PT_best, back_scan_a_PT_best, back_scan_a_error_PT_best, back_scan_K_error_PT_best, back_scan_K_PT_best, forward_scan_alpha_PT_full_curve, forward_scan_alpha_error_PT_full_curve, forward_scan_dc_PT_full_curve, forward_scan_dc_error_PT_full_curve, forward_scan_a_PT_full_curve, forward_scan_a_error_PT_full_curve, forward_scan_K_PT_full_curve, forward_scan_K_error_PT_full_curve, gamma_forward_scan_PT_full_curve, back_scan_alpha_PT_full_curve, back_scan_alpha_error_PT_full_curve, back_scan_dc_PT_full_curve, back_scan_dc_error_PT_full_curve, back_scan_a_PT_full_curve, back_scan_a_error_PT_full_curve, back_scan_K_PT_full_curve, back_scan_K_error_PT_full_curve, gamma_back_scan_PT_full_curve, time_s = PT_Fitting(df, forward_scan_segment, back_scan_segment, R, forward_scan_index_min, back_scan_index_min, Indentation_m_col, CorrForceA_N_col)
    except ValueError:
        txt.insert(END, f"Error: Datapoints per fitting segment parameter is too high.\n\n")
        Spacer()
        Spacer()
        txt.update()
        txt.see("end")
        return

    # Get individual datapoints to indicate best results in error analysis graphs.
    to_remove = ["NaN"]
    forward_scan_best_PT = forward_scan[["Indentation step PT [m]", "K PT [Pa]", "ln (K Error PT)", "Alpha PT", "ln (Alpha Error PT)"]]
    forward_scan_best_PT = forward_scan_best_PT[~forward_scan_best_PT["ln (Alpha Error PT)"].isin(to_remove)]
    forward_scan_best_PT = forward_scan_best_PT.loc[forward_scan_best_PT["ln (Alpha Error PT)"] == forward_scan_best_PT["ln (Alpha Error PT)"].min()]
    back_scan_best_PT = back_scan[["Indentation step PT [m]", "K PT [Pa]", "ln (K Error PT)", "Alpha PT", "ln (Alpha Error PT)"]]
    back_scan_best_PT = back_scan_best_PT[~back_scan_best_PT["ln (Alpha Error PT)"].isin(to_remove)]
    back_scan_best_PT = back_scan_best_PT.loc[back_scan_best_PT["ln (Alpha Error PT)"] == back_scan_best_PT["ln (Alpha Error PT)"].min()]

    # Plot all data for PT
    fig5, ax = plt.subplots(3, 2, figsize=(8, 7))
    df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[0, 0], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="steelblue", label='Loading scan')
    df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[0, 0], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="orange", label='Unloading scan')
    Plot_rxc(0, 0, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 9)

    forward_scan.plot(ax=ax[0, 1], kind='scatter', x="Indentation PT Full Curve [m]", y="Corr Force A PT Full Curve [N]", color='white', edgecolors='steelblue', label='Loading data')
    forward_scan.plot(ax=ax[0, 1], kind='line', style="--", x="Fitting Indentation PT Full Curve [m]", y="Corr Force A PT Full Curve [N]", color='black', label='PT fitting')
    back_scan.plot(ax=ax[0, 1], kind='scatter', x="Indentation PT Full Curve [m]", y="Corr Force A PT Full Curve [N]", color='white', edgecolors='orange', label='Unloading data')
    back_scan.plot(ax=ax[0, 1], kind='line', style="--", x="Fitting Indentation PT Full Curve [m]", y="Corr Force A PT Full Curve [N]", color='gray', label='PT fitting')
    Plot_rxc(0, 1, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 9)

    forward_scan.plot(ax=ax[1, 0], kind='scatter', s=10, x="Indentation step PT [m]", y="ln (Alpha Error PT)", color='white', edgecolors='steelblue', label='Loading scan PT')
    forward_scan_best_PT.plot(ax=ax[1, 0], kind='scatter', s=30, x="Indentation step PT [m]", y="ln (Alpha Error PT)", color='steelblue', edgecolors='steelblue')
    back_scan.plot(ax=ax[1, 0], kind='scatter', s=10, x="Indentation step PT [m]", y="ln (Alpha Error PT)", color='white', edgecolors='orange', label='Unloading scan PT')
    back_scan_best_PT.plot(ax=ax[1, 0], kind='scatter', s=30, x="Indentation step PT [m]", y="ln (Alpha Error PT)", color='orange', edgecolors='orange')
    Plot_rxc(1, 0, "", 14, "Indentation (m)", "ln (Alpha error)", 12, 9, 7, 9)

    forward_scan.plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation step PT [m]", y="Alpha PT", color='white', edgecolors='steelblue', label='Loading scan PT')
    forward_scan_best_PT.plot(ax=ax[1, 1], kind='scatter', s=30, x="Indentation step PT [m]", y="Alpha PT", color='steelblue', edgecolors='steelblue')
    back_scan.plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation step PT [m]", y="Alpha PT", color='white', edgecolors='orange', label='Unloading scan PT')
    back_scan_best_PT.plot(ax=ax[1, 1], kind='scatter', s=30, x="Indentation step PT [m]", y="Alpha PT", color='orange', edgecolors='orange')
    Plot_rxc(1, 1, "", 14, "Indentation (m)", "Alpha", 12, 9, 7, 9)

    df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[2, 0], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="lightblue", label='Loading scan')
    df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[2, 0], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="navajowhite", label='Unloading scan')
    forward_scan.plot(ax=ax[2, 0], kind='scatter', s=10, x="Indentation PT [m]", y="Corr Force A PT [N]", color='white', edgecolors='steelblue', label='Sel. loading data')
    back_scan.plot(ax=ax[2, 0], kind='scatter', s=10, x="Indentation PT [m]", y="Corr Force A PT [N]", color='white', edgecolors='orange', label='Sel. unloading data')
    Plot_rxc(2, 0, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 9)

    forward_scan.plot(ax=ax[2, 1], kind='scatter', x="Indentation PT [m]", y="Corr Force A PT [N]", color='white', edgecolors='steelblue', label='Loading data')
    forward_scan.plot(ax=ax[2, 1], kind='line', style="--", x="Fitting Indentation PT [m]", y="Corr Force A PT [N]", color='black', label='PT fitting')
    back_scan.plot(ax=ax[2, 1], kind='scatter', x="Indentation PT [m]", y="Corr Force A PT [N]", color='white', edgecolors='orange', label='Unloading data')
    back_scan.plot(ax=ax[2, 1], kind='line', style="--", x="Fitting Indentation PT [m]", y="Corr Force A PT [N]", color='gray', label='PT fitting')
    Plot_rxc(2, 1, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 9)
    plt.tight_layout()
    plt.savefig(f"{folder_path_fitting_restults}/{filename_noext}_Fitting analysis PT.png", bbox_inches='tight', dpi=300)

    plt.close('all') #To close all figures and save memory - max # of figures before warning: 20

    # MAX INDENTATION AND MAX FORCE
    Max_Indent, Max_Force = Max_Indent_And_Max_Force(forward_scan, Indentation_m_col, CorrForceA_N_col)

    # PRINT RESULTS
    attr_force, PZI_LS, adhes_force, sep_point, PZI_US, Es_LS_fc, Es_LS_ema, Es_US_fc, Es_US_ema, interf_en_LS, interf_en_US = PT_Print_Results(path_filename_noext, folder_path_fitting_restults, filename_noext, df, R, v_sample, Zmov_type, forward_scan, back_scan, forward_scan_Fad, forward_scan_contactP, back_scan_Fad, back_scan_contactP, scan_speed, Max_Indent, Max_Force, gamma_forward_scan_PT, gamma_back_scan_PT, forward_scan_dc_PT_best, forward_scan_dc_error_PT_best, back_scan_dc_PT_best, back_scan_dc_error_PT_best, forward_scan_K_PT_best, forward_scan_K_error_PT_best, back_scan_K_error_PT_best, back_scan_K_PT_best, forward_scan_alpha_PT_full_curve, forward_scan_alpha_error_PT_full_curve, forward_scan_dc_PT_full_curve, forward_scan_dc_error_PT_full_curve, forward_scan_a_PT_full_curve, forward_scan_a_error_PT_full_curve, forward_scan_K_PT_full_curve, forward_scan_K_error_PT_full_curve, gamma_forward_scan_PT_full_curve, back_scan_alpha_PT_full_curve, back_scan_alpha_error_PT_full_curve, back_scan_dc_PT_full_curve, back_scan_dc_error_PT_full_curve, back_scan_a_PT_full_curve, back_scan_a_error_PT_full_curve, back_scan_K_PT_full_curve, back_scan_K_error_PT_full_curve, gamma_back_scan_PT_full_curve, time_s)

    # GET DATA FOR 3D PLOT
    X, Y = GetXY(df, Zmov_type)

    return X, Y, attr_force, PZI_LS, adhes_force, sep_point, PZI_US, Es_LS_fc, Es_LS_ema, Es_US_fc, Es_US_ema, interf_en_LS, interf_en_US


def JKR_Fitting(df, forward_scan_segment, back_scan_segment, R, forward_scan_index_min, back_scan_index_min, Indentation_m_col, CorrForceA_N_col):

    #Define fitting functions
    def JKR(x, R, K, Fad, dc):
        a = ((R/K)*((((Fad)**(1/2))+((x+Fad)**(1/2)))**2))**(1/3)
        return ((a**2)/R)-((4/3)*((a*Fad)/(R*K))**(1/2))-dc #The addition of -dc improves teh fitting by a lot. indentation = d - dc = equation

    #Select starting point for fitting, get Fad, and calculate gamma
    forward_scan = pandas.DataFrame(df.groupby(["Exp Phase [#]"]).get_group(1.0))
    forward_scan = forward_scan.loc[forward_scan.index >= forward_scan_index_min] #Select data for fitting based on contact point
    forward_scan_Fad = forward_scan.iloc[0, CorrForceA_N_col]
    gamma_forward_scan_JKR = (2*abs(forward_scan_Fad))/(3*math.pi*R)
    forward_scan_contactP = forward_scan.iloc[0, Indentation_m_col]*1e6
    if forward_scan_Fad > 0:
        forward_scan_Fad = 0

    #Calculate Point of Zero Indentation
    forward_scan_force_at_zero_indentation = (8 / 9) * forward_scan_Fad
    forward_scan_PZI_df = forward_scan.loc[forward_scan["Corr Force A [N]"] >= forward_scan_force_at_zero_indentation]
    forward_scan_PZI = round(forward_scan_PZI_df.iloc[0, Indentation_m_col]*1e6, 3)

    #Fit the entire indentation curve
    forward_scan_x_full_curve = forward_scan["Corr Force A [N]"].to_list()
    forward_scan_y_full_curve = forward_scan["Indentation [m]"].to_list()
    custom_JKR = lambda x, K, dc: JKR(x, R, K, -forward_scan_Fad, dc)  # Fix Fad value
    forward_scan_pars_JKR_full_curve, forward_scan_cov_JKR_full_curve = optimize.curve_fit(f=custom_JKR, xdata=forward_scan_x_full_curve, ydata=forward_scan_y_full_curve)
    forward_scan_cov_JKR_full_curve = np.sqrt(np.diag(forward_scan_cov_JKR_full_curve)) #Necessary to convert error matrix output of optimize.curve_fit into list of standard deviations for each parameter
    forward_scan_bulk_modulus_JKR_full_curve = forward_scan_pars_JKR_full_curve[0]
    forward_scan_bulk_modulus_error_JKR_full_curve = forward_scan_cov_JKR_full_curve[0]
    forward_scan_dc_JKR_full_curve = forward_scan_pars_JKR_full_curve[1]
    forward_scan_dc_error_JKR_full_curve = forward_scan_cov_JKR_full_curve[1]
    forward_scan["Indentation JKR Full Curve [m]"] = forward_scan["Indentation [m]"]
    forward_scan["Corr Force A JKR Full Curve [N]"] = forward_scan["Corr Force A [N]"]
    #Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    forward_scan["a_full_curve"] = ((R/forward_scan_bulk_modulus_JKR_full_curve)*((((-forward_scan_Fad)**(1/2))+((forward_scan["Corr Force A JKR Full Curve [N]"]-forward_scan_Fad)**(1/2)))**2))**(1/3)
    forward_scan["Fitting Indentation JKR Full Curve [m]"] = ((forward_scan["a_full_curve"]**2)/R)-((4/3)*((forward_scan["a_full_curve"]*(-forward_scan_Fad))/(R*forward_scan_bulk_modulus_JKR_full_curve))**(1/2))-forward_scan_dc_JKR_full_curve

    #Optimise loading curve fitting
    forward_scan_initial_len = len(forward_scan)
    forward_scan_datapoints_to_fit = forward_scan_segment
    forward_scan_test = forward_scan.loc[forward_scan.index < (forward_scan.index[0] + forward_scan_datapoints_to_fit)] #Select data for fitting based on threshold value, index was not reset so forward_scan.index[-1] is a high number
    forward_scan_bulk_modulus_JKR = []
    forward_scan_bulk_modulus_error_JKR = []
    forward_scan_dc_JKR = []
    forward_scan_dc_error_JKR = []
    forward_scan_indentation_JKR = []
    forward_scan_datapoints_JKR = []

    while forward_scan_datapoints_to_fit <= forward_scan_initial_len:
        forward_scan_x = forward_scan_test["Corr Force A [N]"].to_list()
        forward_scan_y = forward_scan_test["Indentation [m]"].to_list()

        # JKR fitting on selected data
        # If Fad >0 set it to 0
        custom_JKR = lambda x, K, dc: JKR(x, R, K, -forward_scan_Fad, dc)  # Fix Fad value
        forward_scan_pars_JKR, forward_scan_cov_JKR = optimize.curve_fit(f=custom_JKR, xdata=forward_scan_x, ydata=forward_scan_y)
        forward_scan_cov_JKR = np.sqrt(np.diag(forward_scan_cov_JKR)) #Necessary to convert error matrix output of optimize.curve_fit into list of standard deviations for each parameter

        forward_scan_bulk_modulus_JKR.append(forward_scan_pars_JKR[0])
        forward_scan_bulk_modulus_error_JKR.append(forward_scan_cov_JKR[0])
        forward_scan_dc_JKR.append(forward_scan_pars_JKR[1])
        forward_scan_dc_error_JKR.append(forward_scan_cov_JKR[1])
        forward_scan_indentation_JKR.append(forward_scan_y[-1])
        forward_scan_datapoints_JKR.append(forward_scan_datapoints_to_fit)

        forward_scan_datapoints_to_fit = forward_scan_datapoints_to_fit + forward_scan_segment
        forward_scan_test = forward_scan.loc[forward_scan.index < (forward_scan.index[0] + forward_scan_datapoints_to_fit)]  # Select data for fitting based on threshold value, index was not reset so forward_scan.index[-1] is a high number

    #Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    forward_scan_index_smallest_error = forward_scan_bulk_modulus_error_JKR.index(min(forward_scan_bulk_modulus_error_JKR))
    forward_scan_Kerror_JKR_best = forward_scan_bulk_modulus_error_JKR[forward_scan_index_smallest_error]
    forward_scan_K_JKR_best = forward_scan_bulk_modulus_JKR[forward_scan_index_smallest_error]
    forward_scan_dcerror_JKR_best = forward_scan_dc_error_JKR[forward_scan_index_smallest_error]
    forward_scan_dc_JKR_best = forward_scan_dc_JKR[forward_scan_index_smallest_error]
    #Redeclare test df with best result and append Indentation and Corr Force columns to forward_scan for storage
    forward_scan_test = forward_scan.loc[forward_scan.index < (forward_scan.index[0] + forward_scan_datapoints_JKR[forward_scan_index_smallest_error])]
    forward_scan["Indentation JKR [m]"] = forward_scan_test["Indentation [m]"]
    forward_scan["Corr Force A JKR [N]"] = forward_scan_test["Corr Force A [N]"]
    #Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    forward_scan["a"] = ((R/forward_scan_K_JKR_best)*((((-forward_scan_Fad)**(1/2))+((forward_scan["Corr Force A JKR [N]"]-forward_scan_Fad)**(1/2)))**2))**(1/3)
    forward_scan["Fitting Indentation JKR [m]"] = ((forward_scan["a"]**2)/R)-((4/3)*((forward_scan["a"]*(-forward_scan_Fad))/(R*forward_scan_K_JKR_best))**(1/2))-forward_scan_dc_JKR_best

    #Extend lists with stored data to match the lenght of forward_scan df otherwise they cannot be added as columns to the df
    forward_scan_indentation_JKR.extend(["NaN"]*(forward_scan_initial_len-len(forward_scan_indentation_JKR)))
    forward_scan["Indentation step JKR [m]"] = forward_scan_indentation_JKR
    forward_scan_bulk_modulus_JKR.extend(["NaN"]*(forward_scan_initial_len-len(forward_scan_bulk_modulus_JKR)))
    forward_scan["K JKR [Pa]"] = forward_scan_bulk_modulus_JKR
    #Transform error into logarithm for best plots
    forward_scan_bulk_modulus_error_JKR = [math.log(x) for x in forward_scan_bulk_modulus_error_JKR]
    forward_scan_bulk_modulus_error_JKR.extend(["NaN"]*(forward_scan_initial_len-len(forward_scan_bulk_modulus_error_JKR)))
    forward_scan["ln (K Error JKR)"] = forward_scan_bulk_modulus_error_JKR




    # Select starting point for fitting, get Fad, and calculate gamma
    back_scan = pandas.DataFrame(df.groupby(["Exp Phase [#]"]).get_group(0.0))
    back_scan = back_scan.iloc[::-1].reset_index(drop=True) #Reverse and reindex dataframe
    back_scan = back_scan.loc[back_scan.index >= back_scan_index_min] #Select data for fitting based on threshold value
    back_scan_Fad = back_scan.iloc[0, CorrForceA_N_col]
    gamma_back_scan_JKR = (2 * abs(back_scan_Fad)) / (3 * math.pi * R)
    back_scan_contactP = back_scan.iloc[0, Indentation_m_col]*1e6
    # If Fad >0 set it to 0
    if back_scan_Fad > 0:
        back_scan_Fad = 0

    #Calculate Point of Zero Indentation
    back_scan_force_at_zero_indentation = (8 / 9) * back_scan_Fad
    back_scan_PZI_df = back_scan.loc[back_scan["Corr Force A [N]"] >= back_scan_force_at_zero_indentation]
    back_scan_PZI = round(back_scan_PZI_df.iloc[0, Indentation_m_col]*1e6, 3)

    #Fit the entire indentation curve
    back_scan_x_full_curve = back_scan["Corr Force A [N]"].to_list()
    back_scan_y_full_curve = back_scan["Indentation [m]"].to_list()
    custom_JKR = lambda x, K, dc: JKR(x, R, K, -back_scan_Fad, dc)  # Fix Fad value
    back_scan_pars_JKR_full_curve, back_scan_cov_JKR_full_curve = optimize.curve_fit(f=custom_JKR, xdata=back_scan_x_full_curve, ydata=back_scan_y_full_curve)
    back_scan_cov_JKR_full_curve = np.sqrt(np.diag(back_scan_cov_JKR_full_curve)) #Necessary to convert error matrix output of optimize.curve_fit into list of standard deviations for each parameter
    back_scan_bulk_modulus_JKR_full_curve = back_scan_pars_JKR_full_curve[0]
    back_scan_bulk_modulus_error_JKR_full_curve = back_scan_cov_JKR_full_curve[0]
    back_scan_dc_JKR_full_curve = back_scan_pars_JKR_full_curve[1]
    back_scan_dc_error_JKR_full_curve = back_scan_cov_JKR_full_curve[1]
    back_scan["Indentation JKR Full Curve [m]"] = back_scan["Indentation [m]"]
    back_scan["Corr Force A JKR Full Curve [N]"] = back_scan["Corr Force A [N]"]
    #Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    back_scan["a_full_curve"] = ((R/back_scan_bulk_modulus_JKR_full_curve)*((((-back_scan_Fad)**(1/2))+((back_scan["Corr Force A JKR Full Curve [N]"]-back_scan_Fad)**(1/2)))**2))**(1/3)
    back_scan["Fitting Indentation JKR Full Curve [m]"] = ((back_scan["a_full_curve"]**2)/R)-((4/3)*((back_scan["a_full_curve"]*(-back_scan_Fad))/(R*back_scan_bulk_modulus_JKR_full_curve))**(1/2))-back_scan_dc_JKR_full_curve

    #Optimise loading curve fitting
    back_scan_initial_len = len(back_scan)
    back_scan_datapoints_to_fit = back_scan_segment
    back_scan_test = back_scan.loc[back_scan.index < (back_scan.index[0] + back_scan_datapoints_to_fit)] #Select data for fitting based on threshold value, index was not reset so forward_scan.index[-1] is a high number
    back_scan_bulk_modulus_JKR = []
    back_scan_bulk_modulus_error_JKR = []
    back_scan_dc_JKR = []
    back_scan_dc_error_JKR = []
    back_scan_indentation_JKR = []
    back_scan_datapoints_JKR = []

    while back_scan_datapoints_to_fit <= back_scan_initial_len:
        back_scan_x = back_scan_test["Corr Force A [N]"].to_list()
        back_scan_y = back_scan_test["Indentation [m]"].to_list()

        # JKR fitting on selected data
        custom_JKR = lambda x, K, dc: JKR(x, R, K, -back_scan_Fad, dc)  # Fix Fad value
        back_scan_pars_JKR, back_scan_cov_JKR = optimize.curve_fit(f=custom_JKR, xdata=back_scan_x, ydata=back_scan_y)
        back_scan_cov_JKR = np.sqrt(np.diag(back_scan_cov_JKR)) #Necessary to convert error matrix output of optimize.curve_fit into list of standard deviations for each parameter

        back_scan_bulk_modulus_JKR.append(back_scan_pars_JKR[0])
        back_scan_bulk_modulus_error_JKR.append(back_scan_cov_JKR[0])
        back_scan_dc_JKR.append(back_scan_pars_JKR[1])
        back_scan_dc_error_JKR.append(back_scan_cov_JKR[1])
        back_scan_indentation_JKR.append(back_scan_y[-1])
        back_scan_datapoints_JKR.append(back_scan_datapoints_to_fit)

        back_scan_datapoints_to_fit = back_scan_datapoints_to_fit + back_scan_segment
        back_scan_test = back_scan.loc[back_scan.index < (back_scan.index[0] + back_scan_datapoints_to_fit)]  # Select data for fitting based on threshold value, index was not reset so forward_scan.index[-1] is a high number

    # Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    back_scan_index_smallest_error = back_scan_bulk_modulus_error_JKR.index(min(back_scan_bulk_modulus_error_JKR))
    back_scan_Kerror_JKR_best = back_scan_bulk_modulus_error_JKR[back_scan_index_smallest_error]
    back_scan_K_JKR_best = back_scan_bulk_modulus_JKR[back_scan_index_smallest_error]
    back_scan_dcerror_JKR_best = back_scan_dc_error_JKR[back_scan_index_smallest_error]
    back_scan_dc_JKR_best = back_scan_dc_JKR[back_scan_index_smallest_error]
    # Redeclare test df with best result and append Indentation and Corr Force columns to forward_scan for storage
    back_scan_test = back_scan.loc[back_scan.index < (back_scan.index[0] + back_scan_datapoints_JKR[back_scan_index_smallest_error])]
    back_scan["Indentation JKR [m]"] = back_scan_test["Indentation [m]"]
    back_scan["Corr Force A JKR [N]"] = back_scan_test["Corr Force A [N]"]
    #Using the found bulk modulus K calculate the indentation values from the values of Corr Force
    back_scan["a"] = ((R/back_scan_K_JKR_best)*((((-back_scan_Fad)**(1/2))+((back_scan["Corr Force A JKR [N]"]-back_scan_Fad)**(1/2)))**2))**(1/3)
    back_scan["Fitting Indentation JKR [m]"] = ((back_scan["a"]**2)/R)-((4/3)*((back_scan["a"]*(-back_scan_Fad))/(R*back_scan_K_JKR_best))**(1/2))-back_scan_dc_JKR_best

    #Extend lists with stored data to match the lenght of forward_scan df otherwise they cannot be added as columns to the df
    back_scan_indentation_JKR.extend(["NaN"]*(back_scan_initial_len-len(back_scan_indentation_JKR)))
    back_scan["Indentation step JKR [m]"] = back_scan_indentation_JKR
    back_scan_bulk_modulus_JKR.extend(["NaN"]*(back_scan_initial_len-len(back_scan_bulk_modulus_JKR)))
    back_scan["K JKR [Pa]"] = back_scan_bulk_modulus_JKR
    #Transform error into logarithm for best plots
    back_scan_bulk_modulus_error_JKR = [math.log(x) for x in back_scan_bulk_modulus_error_JKR]
    back_scan_bulk_modulus_error_JKR.extend(["NaN"]*(back_scan_initial_len-len(back_scan_bulk_modulus_error_JKR)))
    back_scan["ln (K Error JKR)"] = back_scan_bulk_modulus_error_JKR

    # Define time
    time_s_col = df.columns.get_loc("Time [s]")  # Get column number
    time_s = df.iloc[0, time_s_col]

    return forward_scan, forward_scan_Fad, gamma_forward_scan_JKR, forward_scan_contactP, back_scan, back_scan_Fad, gamma_back_scan_JKR, back_scan_contactP, forward_scan_K_JKR_best, forward_scan_Kerror_JKR_best, forward_scan_dc_JKR_best, forward_scan_dcerror_JKR_best, back_scan_K_JKR_best, back_scan_Kerror_JKR_best, back_scan_dc_JKR_best, back_scan_dcerror_JKR_best, forward_scan_PZI, back_scan_PZI, forward_scan_bulk_modulus_JKR_full_curve, forward_scan_bulk_modulus_error_JKR_full_curve, forward_scan_dc_JKR_full_curve, forward_scan_dc_error_JKR_full_curve, back_scan_bulk_modulus_JKR_full_curve, back_scan_bulk_modulus_error_JKR_full_curve, back_scan_dc_JKR_full_curve, back_scan_dc_error_JKR_full_curve, time_s

def JKR_Print_Results(path_filename_noext, folder_path_fitting_restults, filename_noext, df, R, v_sample, Zmov_type, forward_scan, back_scan, forward_scan_Fad, forward_scan_contactP, back_scan_Fad, back_scan_contactP, forward_scan_K_JKR_best, forward_scan_Kerror_JKR_best, forward_scan_dc_JKR_best, forward_scan_dcerror_JKR_best, back_scan_K_JKR_best, back_scan_Kerror_JKR_best, back_scan_dc_JKR_best, back_scan_dcerror_JKR_best, gamma_forward_scan_JKR, gamma_back_scan_JKR, indentation_speed, Max_Indent, Max_Force, forward_scan_PZI, back_scan_PZI, forward_scan_bulk_modulus_JKR_full_curve, forward_scan_bulk_modulus_error_JKR_full_curve, forward_scan_dc_JKR_full_curve, forward_scan_dc_error_JKR_full_curve, back_scan_bulk_modulus_JKR_full_curve, back_scan_bulk_modulus_error_JKR_full_curve, back_scan_dc_JKR_full_curve, back_scan_dc_error_JKR_full_curve, time_s):

    #Print results to screen
    txt.insert(END, "Probe radius: {} um\n".format(R*1e6))
    txt.insert(END, "Elapsed time: {} s\n".format(round(time_s)))

    txt.insert(END, "Poisson's ratio sample: {}\n".format(v_sample))
    txt.insert(END, "Indentation speed: {} um/s\n".format(round(indentation_speed, 3)))
    txt.insert(END, "Max indentation: {} um\n".format(round(Max_Indent*1e6, 3)))
    txt.insert(END, "Max force: {:,} uN\n".format(round(Max_Force*1e6, 3)))

    txt.insert(END, "Contact point: {} um\n".format(str(round(forward_scan_contactP, 3))))
    txt.insert(END, "Calculated contact point (fc): {} um\n".format(str(round(forward_scan_dc_JKR_full_curve*1e6, 3))))
    txt.insert(END, "Calculated contact point (ema): {} um\n".format(str(round(forward_scan_dc_JKR_best*1e6, 3))))
    txt.insert(END, "Attractive force: {} uN\n".format(str(round(forward_scan_Fad*1e6, 3))))
    txt.insert(END, "PZI (loading scan): {} um\n".format(forward_scan_PZI))

    txt.insert(END, "Separation point: {} um\n".format(str(round(back_scan_contactP, 3))))
    txt.insert(END, "Calculated separation point (fc): {} um\n".format(str(round(back_scan_dc_JKR_full_curve*1e6, 3))))
    txt.insert(END, "Calculated separation point (ema): {} um\n".format(str(round(back_scan_dc_JKR_best*1e6, 3))))
    txt.insert(END, "Adhesion force: {} uN\n".format(str(round(back_scan_Fad*1e6, 3))))
    txt.insert(END, "PZI (unloading scan): {} um\n\n".format(back_scan_PZI))
    txt.update()
    txt.see("end")

    #forward_scan_bulk_modulus_JKR_full_curve, forward_scan_bulk_modulus_error_JKR_full_curve, forward_scan_dc_JKR_full_curve, forward_scan_dc_error_JKR_full_curve, back_scan_bulk_modulus_JKR_full_curve, back_scan_bulk_modulus_error_JKR_full_curve, back_scan_dc_JKR_full_curve, back_scan_dc_error_JKR_full_curve

    # Calculate and report elastic modulus sample (Es) from full curve ("_fc")
    # Calculate and report elastic modulus sample (Es) from error minimisation algorithm ("_ema")
    Es_forward_scan_JKR_full_curve = (3 / 4) * forward_scan_bulk_modulus_JKR_full_curve * (1 - (v_sample ** 2))
    error_forward_scan_JKR_full_curve = (Es_forward_scan_JKR_full_curve * forward_scan_bulk_modulus_error_JKR_full_curve) / forward_scan_bulk_modulus_JKR_full_curve
    Es_forward_scan_JKR = (3 / 4) * forward_scan_K_JKR_best * (1 - (v_sample ** 2))
    error_forward_scan_JKR = (Es_forward_scan_JKR * forward_scan_Kerror_JKR_best) / forward_scan_K_JKR_best
    txt.insert(END, "K_fc (JKR, loading scan) = {:,} +/- {:,} Pa\n".format(round(forward_scan_bulk_modulus_JKR_full_curve, 2), round(forward_scan_bulk_modulus_error_JKR_full_curve, 2)))
    txt.insert(END, "K_ema (JKR, loading scan) = {:,} +/- {:,} Pa\n".format(round(forward_scan_K_JKR_best, 2), round(forward_scan_Kerror_JKR_best, 2)))
    txt.insert(END, "Es_fc (JKR, loading scan) = {:,} +/- {:,} Pa\n".format(round(Es_forward_scan_JKR_full_curve, 2), round(error_forward_scan_JKR_full_curve, 2)))
    txt.insert(END, "Es_ema (JKR, loading scan) = {:,} +/- {:,} Pa\n".format(round(Es_forward_scan_JKR, 2), round(error_forward_scan_JKR, 2)))
    txt.insert(END, "Interfacial energy (JKR, loading scan): {} mJ m^-2\n\n".format(round(gamma_forward_scan_JKR * 1000, 2)))

    # Calculate and report elastic modulus sample (Es) from full curve ("_fc")
    # Calculate and report elastic modulus sample (Es) from error minimisation algorithm ("_ma")
    Es_back_scan_JKR_full_curve = (3 / 4) * back_scan_bulk_modulus_JKR_full_curve * (1 - (v_sample ** 2))
    error_back_scan_JKR_full_curve = (Es_back_scan_JKR_full_curve * back_scan_bulk_modulus_error_JKR_full_curve) / back_scan_bulk_modulus_JKR_full_curve
    Es_back_scan_JKR = (3 / 4) * back_scan_K_JKR_best * (1 - (v_sample ** 2))
    error_back_scan_JKR = (Es_back_scan_JKR * back_scan_Kerror_JKR_best) / back_scan_K_JKR_best
    txt.insert(END, "K_fc (JKR, unloading scan) = {:,} +/- {:,} Pa\n".format(round(back_scan_bulk_modulus_JKR_full_curve, 2), round(back_scan_bulk_modulus_error_JKR_full_curve, 2)))
    txt.insert(END, "K_ema (JKR, unloading scan) = {:,} +/- {:,} Pa\n".format(round(back_scan_K_JKR_best, 2), round(back_scan_Kerror_JKR_best, 2)))
    txt.insert(END, "Es_fc (JKR, unloading scan) = {:,} +/- {:,} Pa\n".format(round(Es_back_scan_JKR_full_curve, 2), round(error_back_scan_JKR_full_curve, 2)))
    txt.insert(END, "Es_ema (JKR, unloading scan) = {:,} +/- {:,} Pa\n".format(round(Es_back_scan_JKR, 2), round(error_back_scan_JKR, 2)))
    txt.insert(END, "Interfacial energy (JKR, unloading scan): {} mJ m^-2\n\n".format(round(gamma_back_scan_JKR * 1000, 2)))

    forward_Tabor_param_JKR = ((R * (gamma_forward_scan_JKR ** 2)) / ((Es_forward_scan_JKR ** 2) * ((0.333 * 1e-9) ** 3))) ** (1 / 3)
    back_Tabor_param_JKR = ((R * (gamma_back_scan_JKR ** 2)) / ((Es_back_scan_JKR ** 2) * ((0.333 * 1e-9) ** 3))) ** (1 / 3)
    txt.insert(END, "u (JKR, loading scan) = {}\n".format(round(forward_Tabor_param_JKR, 2)))
    txt.insert(END, "u (JKR, unloading scan) = {}\n".format(round(back_Tabor_param_JKR, 2)))
    txt.update()
    txt.see("end")
    Spacer()

    # Save full curve
    df.to_csv(f"{path_filename_noext}_full curve.csv", index=False)

    # Save fitting results for loading curve
    file = open(f"{folder_path_fitting_restults}/{filename_noext}_LS_JKR fitting results.csv", "w")
    # Because it will be saved as csv, Excell will switch column when it sees a comma, hence all the ","
    file.write("Measurement type:, {}\n".format(Zmov_type))
    file.write("Probe radius:, {}, um\n".format(R * 1e6))
    file.write("Poisson's ratio sample:, {}\n".format(v_sample))
    file.write("Indentation speed:, {}, um/s\n".format(round(indentation_speed, 3)))
    file.write("Max indentation:, {}, um\n".format(round(Max_Indent * 1e6, 3)))
    file.write("Max force:, {}, uN\n".format(round(Max_Force * 1e6, 3)))
    file.write("Contact point:, {}, um\n".format(str(round(forward_scan_contactP, 3))))
    file.write("Calculated contact point (fc):, {}, um\n".format(str(round(forward_scan_dc_JKR_full_curve*1e6, 3))))
    file.write("Calculated contact point (ema):, {}, um\n".format(str(round(forward_scan_dc_JKR_best*1e6, 3))))
    file.write("Attractive force:, {}, uN\n".format(str(round(forward_scan_Fad * 1e6, 3))))
    file.write("PZI (loading scan):, {}, um\n".format(forward_scan_PZI))
    file.write("Separation point:, {}, um\n".format(str(round(back_scan_contactP, 3))))
    file.write("Calculated separation point (fc):, {}, um\n".format(str(round(back_scan_dc_JKR_full_curve*1e6, 3))))
    file.write("Calculated separation point (ema):, {}, um\n".format(str(round(back_scan_dc_JKR_best*1e6, 3))))
    file.write("Adhesion force:, {}, uN\n".format(str(round(back_scan_Fad * 1e6, 3))))
    file.write("PZI (unloading scan):, {}, um\n\n".format(back_scan_PZI))

    #Report elastic modulus sample (Es) from full curve (_fc)
    #Report elastic modulus sample (Es) from error minimisation algorithm (_ema)
    file.write("K_fc (JKR loading scan) =, {}, +/-, {}, Pa\n".format(round(forward_scan_bulk_modulus_JKR_full_curve, 2), round(forward_scan_bulk_modulus_error_JKR_full_curve, 2)))
    file.write("K_ema (JKR loading scan) =, {}, +/-, {}, Pa\n".format(round(forward_scan_K_JKR_best, 2), round(forward_scan_Kerror_JKR_best, 2)))
    file.write("Es_fc (JKR loading scan) =, {}, +/-, {}, Pa\n".format(round(Es_forward_scan_JKR_full_curve, 2), round(error_forward_scan_JKR_full_curve, 2)))
    file.write("Es_ema (JKR loading scan) =, {}, +/-, {}, Pa\n".format(round(Es_forward_scan_JKR, 2), round(error_forward_scan_JKR, 2)))
    file.write("Interfacial energy (JKR loading scan):, {}, mJ m^-2\n\n".format(round(gamma_forward_scan_JKR * 1000, 2)))

    file.write("u (JKR loading scan):, {}\n\n".format(round(forward_Tabor_param_JKR, 2)))
    forward_scan.to_csv(file, index=False)
    file.close()

    # Save fitting results for unloading curve
    file = open(f"{folder_path_fitting_restults}/{filename_noext}_US_JKR fitting results.csv", "w")
    # Because it will be saved as csv, Excell will switch column when it sees a comma, hence all the ","
    file.write("Measurement type:, {}\n".format(Zmov_type))
    file.write("Probe radius:, {}, um\n".format(R * 1e6))
    file.write("Poisson's ratio sample:, {}\n".format(v_sample))
    file.write("Indentation speed:, {}, um/s\n".format(round(indentation_speed, 3)))
    file.write("Max indentation:, {}, um\n".format(round(Max_Indent * 1e6, 3)))
    file.write("Max force:, {}, uN\n".format(round(Max_Force * 1e6, 3)))
    file.write("Contact point:, {}, um\n".format(str(round(forward_scan_contactP, 3))))
    file.write("Calculated contact point (fc):, {}, um\n".format(str(round(forward_scan_dc_JKR_full_curve*1e6, 3))))
    file.write("Calculated contact point (ema):, {}, um\n".format(str(round(forward_scan_dc_JKR_best*1e6, 3))))
    file.write("Attractive force:, {}, uN\n".format(str(round(forward_scan_Fad * 1e6, 3))))
    file.write("PZI (loading scan):, {}, um,\n".format(forward_scan_PZI))
    file.write("Separation point:, {}, um\n".format(str(round(back_scan_contactP, 3))))
    file.write("Calculated separation point (fc):, {}, um\n".format(str(round(back_scan_dc_JKR_full_curve*1e6, 3))))
    file.write("Calculated separation point (ema):, {}, um\n".format(str(round(back_scan_dc_JKR_best*1e6, 3))))
    file.write("Adhesion force:, {}, uN\n".format(str(round(back_scan_Fad * 1e6, 3))))
    file.write("PZI (unloading scan):, {}, um\n\n".format(back_scan_PZI))

    #Report elastic modulus sample (Es) from full curve (_fc)
    #Report elastic modulus sample (Es) from error minimisation algorithm
    file.write("K_fc (JKR loading scan) =, {}, +/-, {}, Pa\n".format(round(back_scan_bulk_modulus_JKR_full_curve, 2), round(back_scan_bulk_modulus_error_JKR_full_curve, 2)))
    file.write("K_ema (JKR unloading scan) =, {}, +/-, {}, Pa\n".format(round(back_scan_K_JKR_best, 2), round(back_scan_Kerror_JKR_best, 2)))
    file.write("Es_fc (JKR loading scan) =, {}, +/-, {}, Pa\n".format(round(Es_back_scan_JKR_full_curve, 2), round(error_back_scan_JKR_full_curve, 2)))
    file.write("Es_ema (JKR unloading scan) =, {}, +/-, {}, Pa\n".format(round(Es_back_scan_JKR, 2), round(error_back_scan_JKR, 2)))
    file.write("Interfacial energy (JKR unloading scan):, {}, mJ m^-2\n\n".format(round(gamma_back_scan_JKR * 1000, 2)))

    file.write("u (JKR unloading scan):, {} \n\n".format(round(back_Tabor_param_JKR, 2)))
    back_scan.to_csv(file, index=False)

    file.close()

    #Return data for 3D maps
    attr_force = round(forward_scan_Fad * 1e6, 3)
    PZI_LS = forward_scan_PZI
    adhes_force = round(back_scan_Fad * 1e6, 3)
    sep_point = round(back_scan_contactP, 3)
    PZI_US = back_scan_PZI
    Es_LS_fc = round(Es_forward_scan_JKR_full_curve, 2)
    Es_LS_ema = round(Es_forward_scan_JKR, 2)
    Es_US_fc = round(Es_back_scan_JKR_full_curve, 2)
    Es_US_ema = round(Es_back_scan_JKR, 2)
    interf_en_LS = round(gamma_forward_scan_JKR, 2)
    interf_en_US = round(gamma_back_scan_JKR, 2)

    return attr_force, PZI_LS, adhes_force, sep_point, PZI_US, Es_LS_fc, Es_LS_ema, Es_US_fc, Es_US_ema, interf_en_LS, interf_en_US

def JKR_Analyse_Single_CSVFile(path_filename, folder_path_fitting_restults, filename_noext, ratio_BL_points, threshold_constant, forward_scan_segment, back_scan_segment, R, v_sample):

    def Plot_rxc(r, c, title, title_fontsize, xlabel, ylabel, xylabel_fontsize, axes_fontsize, tick_lenght, legend_fontsize):
        ax[r, c].set_title(title, fontsize=title_fontsize)
        ax[r, c].set_xlabel(xlabel, fontsize=xylabel_fontsize)
        ax[r, c].set_ylabel(ylabel, fontsize=xylabel_fontsize)
        ax[r, c].tick_params(direction='in', length=tick_lenght, labelsize=axes_fontsize)
        ax[r, c].legend(fontsize=legend_fontsize, bbox_to_anchor=(0., 1.1, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

    # OPEN FILE AND TRANSFORM UNITS TO N AND M
    df = pandas.read_table("{}".format(path_filename), sep=',', header=0)
    path_filename_noext, ext = os.path.splitext(path_filename) #Remove ".csv" from filename to avoid weird name for documents
    df = df.astype(float)  #Change data from object to float
    R, ForceA_N_col, Time_s_col, Corr_Time_s_col, Displacement_um_col, Displacement_m_col, PiezoZ_um_col, PiezoZ_m_col, \
    PosZ_um_col, PosZ_m_col = Unit_Transform(df, R)
    # ----------------------------------------------------------------------------------------------------------------------#
    # DETERMINE IF MEASUREMENT WAS CARRIED OUT USING PIEZO SCANNER OR STEP MOTOR AND DETERMINE POINT OF CONTACT

    Zmov_type, Probe_Displacement_m_col, forward_scan, back_scan, forward_scan_index_min, forward_scan_min, back_scan_index_min, back_scan_min = Contact_Point(df, PosZ_um_col, PosZ_m_col, PiezoZ_m_col, Displacement_m_col, ratio_BL_points, threshold_constant)

    # ----------------------------------------------------------------------------------------------------------------------#
    # DETERMINE SCAN SPEED

    scan_speed = Indentation_Speed(Zmov_type, forward_scan)
    # ----------------------------------------------------------------------------------------------------------------------#
    # BASELINE AND ZERO DATA

    baseline, CorrForceA_N_col, BaselineF_N_col, Indentation_m_col = Baseline_And_Zero(df, forward_scan, forward_scan_index_min, Displacement_m_col, ratio_BL_points)
    # ----------------------------------------------------------------------------------------------------------------------#
    # PLOT INITIAL ANALYSIS

    if Zmov_type == "Piezo scanner":
        fig1, ax = plt.subplots(2, 2, figsize=(8, 7))
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Piezo Z [m]", color='white', edgecolors='steelblue', label='Piezo Z')
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Displacement [m]", color='white', edgecolors="orange", label='Sample-probe distance')
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Probe Displacement [m]", color='white', edgecolors="firebrick", label='Probe displacement')
        Plot_rxc(0, 0, "", 14, "Time (s)", "Distance (m)", 12, 9, 7, 9)

        df.plot(ax=ax[0, 1], kind='scatter', s=5, x="Piezo Z [m]", y="Displacement [m]", color='white', edgecolors='steelblue', label='Sample-probe distance vs Piezo Z')
        Plot_rxc(0, 1, "", 14, "Piezo Z (m)", "Sample-probe distance (m)", 12, 9, 7, 9)

        df.plot(ax=ax[1, 0], kind='scatter', s=5, x="Piezo Z [m]", y="Force A [N]", color='white', edgecolors='steelblue', label="Piezo Z")
        df.plot(ax=ax[1, 0], kind='scatter', s=5, x="Displacement [m]", y="Force A [N]", color='white', edgecolors="orange", label="Sample-probe distance")
        Plot_rxc(1, 0, "", 14, "Distance (m)", "Force (N)", 12, 9, 7, 9)

        df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="lightblue", label="Loading scan")
        df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="navajowhite", label="Unloading scan")
        df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="steelblue", label="Corr. loading scan")
        df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="orange", label="Corr. unloading scan")
        baseline.plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="gray", label="Datapts used for BL linear fitting")
        df.plot(ax=ax[1, 1], kind='line', style='--', x="Indentation [m]", y="Baseline F [N]", color="black", label="Baseline")
        Plot_rxc(1, 1, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 7)
        plt.tight_layout()
        plt.savefig("{}_Initial analysis.png".format(path_filename_noext), bbox_inches='tight', dpi=300)

    if Zmov_type == "Stick-slip actuator":
        fig1, ax = plt.subplots(2, 2, figsize=(8, 7))
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Pos Z [m]", color='white', edgecolors='steelblue', label="Pos Z")
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Displacement [m]", color='white', edgecolors="orange", label="Sample-probe distance")
        df.plot(ax=ax[0, 0], kind='scatter', s=5, x="Corr Time [s]", y="Probe Displacement [m]", color='white', edgecolors="firebrick", label="Probe displacement")
        Plot_rxc(0, 0, "", 14, "Time (s)", "Distance (m)", 12, 9, 7, 9)

        df.plot(ax=ax[0, 1], kind='scatter', s=5, x="Pos Z [m]", y="Displacement [m]", color='white', edgecolors='steelblue', label='Sample-probe distance vs Piezo Z')
        Plot_rxc(0, 1, "", 14, "Piezo Z (m)", "Sample-probe distance (m)", 12, 9, 7, 9)

        df.plot(ax=ax[1, 0], kind='scatter', s=5, x="Pos Z [m]", y="Force A [N]", color='white', edgecolors='steelblue', label="Piezo Z")
        df.plot(ax=ax[1, 0], kind='scatter', s=5, x="Displacement [m]", y="Force A [N]", color='white', edgecolors="orange", label="Sample-probe distance")
        Plot_rxc(1, 0, "", 14, "Distance (m)", "Force (N)", 12, 9, 7, 9)

        df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="lightblue", label="Loading scan")
        df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="navajowhite", label="Unloading scan")
        df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="steelblue", label="Corr. loading scan")
        df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="orange", label="Corr. unloading scan")
        baseline.plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation [m]", y="Force A [N]", color='white', edgecolors="gray", label="Datapts used for BL linear fitting")
        df.plot(ax=ax[1, 1], kind='line', style='--', x="Indentation [m]", y="Baseline F [N]", color="black", label="Baseline")
        Plot_rxc(1, 1, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 7)
        plt.tight_layout()
        plt.savefig("{}_Initial analysis.png".format(path_filename_noext), bbox_inches='tight', dpi=300)
    # ----------------------------------------------------------------------------------------------------------------------#
    # SELECT DATA FOR JKR FITTING AND PLOT RESULTS
    try:
        forward_scan, forward_scan_Fad, gamma_forward_scan_JKR, forward_scan_contactP, back_scan, back_scan_Fad, gamma_back_scan_JKR, back_scan_contactP, forward_scan_K_JKR_best, forward_scan_Kerror_JKR_best, forward_scan_dc_JKR_best, forward_scan_dcerror_JKR_best, back_scan_K_JKR_best, back_scan_Kerror_JKR_best, back_scan_dc_JKR_best, back_scan_dcerror_JKR_best, forward_scan_PZI, back_scan_PZI, forward_scan_bulk_modulus_JKR_full_curve, forward_scan_bulk_modulus_error_JKR_full_curve, forward_scan_dc_JKR_full_curve, forward_scan_dc_error_JKR_full_curve, back_scan_bulk_modulus_JKR_full_curve, back_scan_bulk_modulus_error_JKR_full_curve, back_scan_dc_JKR_full_curve, back_scan_dc_error_JKR_full_curve, time_s = JKR_Fitting(df, forward_scan_segment, back_scan_segment, R, forward_scan_index_min, back_scan_index_min, Indentation_m_col, CorrForceA_N_col)
    except ValueError:
        txt.insert(END, f"Error: Datapoints per fitting segment parameter is too high.\n\n")
        Spacer()
        Spacer()
        txt.update()
        txt.see("end")
        return

    # Get individual datapoints to indicate best results in error analysis graphs.
    to_remove = ["NaN"]
    forward_scan_best_JKR = forward_scan[["Indentation step JKR [m]", "K JKR [Pa]", "ln (K Error JKR)"]]
    forward_scan_best_JKR = forward_scan_best_JKR[~forward_scan_best_JKR["ln (K Error JKR)"].isin(to_remove)]
    forward_scan_best_JKR = forward_scan_best_JKR.loc[forward_scan_best_JKR["ln (K Error JKR)"] == forward_scan_best_JKR["ln (K Error JKR)"].min()]
    back_scan_best_JKR = back_scan[["Indentation step JKR [m]", "K JKR [Pa]", "ln (K Error JKR)"]]
    back_scan_best_JKR = back_scan_best_JKR[~back_scan_best_JKR["ln (K Error JKR)"].isin(to_remove)]
    back_scan_best_JKR = back_scan_best_JKR.loc[back_scan_best_JKR["ln (K Error JKR)"] == back_scan_best_JKR["ln (K Error JKR)"].min()]

    # Plot all data for JKR
    fig5, ax = plt.subplots(3, 2, figsize=(8, 7))
    df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[0, 0], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="steelblue", label='Loading scan')
    df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[0, 0], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="orange", label='Unloading scan')
    Plot_rxc(0, 0, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 9)

    forward_scan.plot(ax=ax[0, 1], kind='scatter', x="Indentation JKR Full Curve [m]", y="Corr Force A JKR Full Curve [N]", color='white', edgecolors='steelblue', label='Loading data')
    forward_scan.plot(ax=ax[0, 1], kind='line', style="--", x="Fitting Indentation JKR Full Curve [m]", y="Corr Force A JKR Full Curve [N]", color='black', label='JKR fitting')
    back_scan.plot(ax=ax[0, 1], kind='scatter', x="Indentation JKR Full Curve [m]", y="Corr Force A JKR Full Curve [N]", color='white', edgecolors='orange', label='Unloading data')
    back_scan.plot(ax=ax[0, 1], kind='line', style="--", x="Fitting Indentation JKR Full Curve [m]", y="Corr Force A JKR Full Curve [N]", color='gray', label='JKR fitting')
    Plot_rxc(0, 1, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 9)

    forward_scan.plot(ax=ax[1, 0], kind='scatter', s=10, x="Indentation step JKR [m]", y="ln (K Error JKR)", color='white', edgecolors='steelblue', label='Loading scan JKR')
    forward_scan_best_JKR.plot(ax=ax[1, 0], kind='scatter', s=30, x="Indentation step JKR [m]", y="ln (K Error JKR)", color='steelblue', edgecolors='steelblue')
    back_scan.plot(ax=ax[1, 0], kind='scatter', s=10, x="Indentation step JKR [m]", y="ln (K Error JKR)", color='white', edgecolors='orange', label='Unloading scan JKR')
    back_scan_best_JKR.plot(ax=ax[1, 0], kind='scatter', s=30, x="Indentation step JKR [m]", y="ln (K Error JKR)", color='orange', edgecolors='orange')
    Plot_rxc(1, 0, "", 14, "Indentation (m)", "ln (K error)", 12, 9, 7, 9)

    forward_scan.plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation step JKR [m]", y="K JKR [Pa]", color='white', edgecolors='steelblue', label='Loading scan JKR')
    forward_scan_best_JKR.plot(ax=ax[1, 1], kind='scatter', s=30, x="Indentation step JKR [m]", y="K JKR [Pa]", color='steelblue', edgecolors='steelblue')
    back_scan.plot(ax=ax[1, 1], kind='scatter', s=10, x="Indentation step JKR [m]", y="K JKR [Pa]", color='white', edgecolors='orange', label='Unloading scan JKR')
    back_scan_best_JKR.plot(ax=ax[1, 1], kind='scatter', s=30, x="Indentation step JKR [m]", y="K JKR [Pa]", color='orange', edgecolors='orange')
    Plot_rxc(1, 1, "", 14, "Indentation (m)", "K (Pa)", 12, 9, 7, 9)

    df.groupby(["Exp Phase [#]"]).get_group(1.0).plot(ax=ax[2, 0], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="lightblue", label='Loading scan')
    df.groupby(["Exp Phase [#]"]).get_group(0.0).plot(ax=ax[2, 0], kind='scatter', s=10, x="Indentation [m]", y="Corr Force A [N]", color='white', edgecolors="navajowhite", label='Unloading scan')
    forward_scan.plot(ax=ax[2, 0], kind='scatter', s=10, x="Indentation JKR [m]", y="Corr Force A JKR [N]", color='white', edgecolors='steelblue', label='Sel. loading data')
    back_scan.plot(ax=ax[2, 0], kind='scatter', s=10, x="Indentation JKR [m]", y="Corr Force A JKR [N]", color='white', edgecolors='orange', label='Sel. unloading data')
    Plot_rxc(2, 0, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 9)

    forward_scan.plot(ax=ax[2, 1], kind='scatter', x="Indentation JKR [m]", y="Corr Force A JKR [N]", color='white', edgecolors='steelblue', label='Loading data')
    forward_scan.plot(ax=ax[2, 1], kind='line', style="--", x="Fitting Indentation JKR [m]", y="Corr Force A JKR [N]", color='black', label='JKR fitting')
    back_scan.plot(ax=ax[2, 1], kind='scatter', x="Indentation JKR [m]", y="Corr Force A JKR [N]", color='white', edgecolors='orange', label='Unloading data')
    back_scan.plot(ax=ax[2, 1], kind='line', style="--", x="Fitting Indentation JKR [m]", y="Corr Force A JKR [N]", color='gray', label='JKR fitting')
    Plot_rxc(2, 1, "", 14, "Indentation (m)", "Force (N)", 12, 9, 7, 9)
    plt.tight_layout()
    plt.savefig(f"{folder_path_fitting_restults}/{filename_noext}_Fitting analysis JKR.png", bbox_inches='tight', dpi=300)

    plt.close('all') #To close all figures and save memory - max # of figures before warning: 20

    # MAX INDENTATION AND MAX FORCE
    Max_Indent, Max_Force = Max_Indent_And_Max_Force(forward_scan, Indentation_m_col, CorrForceA_N_col)

    # PRINT RESULTS
    attr_force, PZI_LS, adhes_force, sep_point, PZI_US, Es_LS_fc, Es_LS_ema, Es_US_fc, Es_US_ema, interf_en_LS, interf_en_US = JKR_Print_Results(path_filename_noext, folder_path_fitting_restults, filename_noext, df, R, v_sample, Zmov_type, forward_scan, back_scan, forward_scan_Fad, forward_scan_contactP, back_scan_Fad, back_scan_contactP, forward_scan_K_JKR_best, forward_scan_Kerror_JKR_best, forward_scan_dc_JKR_best, forward_scan_dcerror_JKR_best, back_scan_K_JKR_best, back_scan_Kerror_JKR_best, back_scan_dc_JKR_best, back_scan_dcerror_JKR_best, gamma_forward_scan_JKR, gamma_back_scan_JKR, scan_speed, Max_Indent, Max_Force, forward_scan_PZI, back_scan_PZI, forward_scan_bulk_modulus_JKR_full_curve, forward_scan_bulk_modulus_error_JKR_full_curve, forward_scan_dc_JKR_full_curve, forward_scan_dc_error_JKR_full_curve, back_scan_bulk_modulus_JKR_full_curve, back_scan_bulk_modulus_error_JKR_full_curve, back_scan_dc_JKR_full_curve, back_scan_dc_error_JKR_full_curve, time_s)

    # GET DATA FOR 3D PLOT
    X, Y = GetXY(df, Zmov_type)

    return X, Y, attr_force, PZI_LS, adhes_force, sep_point, PZI_US, Es_LS_fc, Es_LS_ema, Es_US_fc, Es_US_ema, interf_en_LS, interf_en_US

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#FUNCITONS FOR 3D MAPS

def Divide_Array_And_Plot(path_filename):
    path, filename = os.path.split(path_filename)
    filename_noext, extension = os.path.splitext(filename)

    df = pandas.read_table(path_filename, low_memory=False, delim_whitespace=True, names=("Index [#]", "Phase [#]", "Displacement [um]", "Time [s]", "Pos X [um]", "Pos Y [um]", "Pos Z [um]", "Piezo X [um]", "Piezo Y [um]", "Piezo Z [um]", "Force A [uN]", "Force B [uN]", "Gripper [V]", "Voltage A [V]", "Voltage B [V]", "Temperature [oC]"))
    # Clean up txt file to save a proper cvs file
    df = df[~df["Index [#]"].isin(['//'])]  # Drop the rows that contain the comments to keep only the numbers
    df = df.dropna(how='all')  # to drop if all values in the row are nan
    df = df.astype(float)  # Change data from object to float
    # Determine if measurement is done with piezo or step motor
    PosZ_um_col = df.columns.get_loc("Pos Z [um]")  # Get column number of "Pos Z [um]"
    delta_stepmotor = df.iloc[2, PosZ_um_col] - df.iloc[3, PosZ_um_col]

    if delta_stepmotor < 0.005:  # This value was optimised based on the dataset available so far
        txt.insert(END, "Measurement carried out using piezo scanner.\n")
        txt.update()
        txt.see("end")

        #Determine number of measurements
        num_mes = int(df["Index [#]"].max() + 1)
        txt.insert(END, f"Number of measurements found: {str(num_mes)}\n")
        txt.update()
        txt.see("end")

        grouped = df.groupby(["Index [#]"]) #Group full database by measurement#

        for num in range(num_mes):
            txt.insert(END, f"{num+1}) Plotting file {filename_noext}_{str(num+1)}...\n")
            txt.update()
            txt.see("end")

            group = pandas.DataFrame(grouped.get_group(num)) #Get a single experiement
            phase = []
            piezoZ = group["Piezo Z [um]"].to_list()  #Use PiezoZ for determining phase number
            phase_num = 1
            for ind, val in enumerate(piezoZ):  #enumerate loops over list (val) and have an automatic counter (ind).
                try:
                    if piezoZ[ind + 1] - piezoZ[ind] >= 0:
                        phase.append(phase_num)
                        phase_num = 1

                    else:
                        phase.append(phase_num)
                        phase_num = 0
                except:
                    phase.append(phase_num) #Fixes the last loop where there is not a num+1 to subtract


            group["Exp Phase [#]"] = phase  # converts Displacement column into a list and applies function Cycle
            group.to_csv(f"{path}/{filename_noext}_{str(num+1)}.csv", index=False)  #Save all measurements as *.csv, index=False avoids saving index

            #Plot all measurements
            sub_group = group.groupby(["Exp Phase [#]"]) #Subgroups are 0 (loading) and 1 (unloading)
            #print(sub_group.get_group(0))
            #print(sub_group.get_group(1))
            #print(sub_group)

            #Transform the column subsections in lists for plotting
            Displacement_in = [] #reset lists
            Force_in = []
            Displacement_out = []
            Force_out = []
            Displacement_in = sub_group.get_group(1)["Displacement [um]"].to_list()
            Force_in = sub_group.get_group(1)["Force A [uN]"].to_list()
            try: #necessary if measurement was interrupted before unloading scan - i.e. there is no unloading scan
                Displacement_out = sub_group.get_group(0)["Displacement [um]"].to_list()
                Force_out = sub_group.get_group(0)["Force A [uN]"].to_list()
            except:
                txt.insert(END, f"No unloading scan detected in file: {filename}\n")
                txt.update()
                txt.see("end")
                pass

            plt.figure() #It creates a new figure every time, necessary otherwise it keeps adding to the same plot!
            plt.scatter(Displacement_in, Force_in, s=10, marker="o", color='white', edgecolors='steelblue', label='Loading data')
            try:
                plt.scatter(Displacement_out, Force_out, s=10, marker="o", color='white', edgecolors='orange', label='Unloading data')
            except:
                txt.insert(END, f"No unloading scan detected in file: {filename}\n")
                txt.update()
                txt.see("end")
                pass

            plt.title("Measurement_{}".format(str(num)), fontsize=20)
            plt.xlabel("Distance (um)", fontsize=18)
            plt.ylabel("Force (uN)", fontsize=18)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tick_params(direction='in', length=8)
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{path}/{filename_noext}_{str(num+1)}.png", bbox_inches='tight', dpi=300)
            plt.close('all') #To close all figures and save memory - max # of figures before warning: 20
            txt.insert(END, f"    Plotting file {filename_noext}_{str(num+1)}...Done!\n\n")
            txt.update()
            txt.see("end")


    else:
        txt.insert(END, "Measurement carried out using stick-slip actuator.\n")
        txt.update()
        txt.see("end")

        #Determine number of measurements
        num_mes = int(df["Index [#]"].max() + 1)
        txt.insert(END, f"Number of measurements found: {str(num_mes)}\n\n")
        txt.update()
        txt.see("end")

        grouped = df.groupby(["Index [#]"]) #Group full database by measurement#

        for num in range(num_mes):
            txt.insert(END, f"{num+1}) Plotting file {filename_noext}_{str(num+1)}...\n")
            txt.update()
            txt.see("end")

            group = pandas.DataFrame(grouped.get_group(num)) #Get a single experiement
            phase = []
            posZ = group["Pos Z [um]"].to_list()  #Use PiezoZ for determining phase number
            phase_num = 1
            for ind, val in enumerate(posZ):  #enumerate loops over list (val) and have an automatic counter (ind).
                try:
                    if posZ[ind + 1] - posZ[ind] <= 0:
                        phase.append(phase_num)
                        phase_num = 1

                    else:
                        phase.append(phase_num)
                        phase_num = 0
                except:
                    phase.append(phase_num) #Fixes the last loop where there is not a num+1 to subtract


            group["Exp Phase [#]"] = phase  # converts Displacement column into a list and applies function Cycle
            group.to_csv(f"{path}/{filename_noext}_{str(num+1)}.csv", index=False)  #Save all measurements as *.csv, index=False avoids saving index

            #Plot all measurements
            sub_group = group.groupby(["Exp Phase [#]"]) #Subgroups are 0 (loading) and 1 (unloading)
            #print(sub_group.get_group(0))
            #print(sub_group.get_group(1))
            #print(sub_group)

            #Transform the column subsections in lists for plotting
            Displacement_in = [] #reset lists
            Force_in = []
            Displacement_out = []
            Force_out = []
            Displacement_in = sub_group.get_group(1)["Displacement [um]"].to_list()
            Force_in = sub_group.get_group(1)["Force A [uN]"].to_list()
            try: #necessary if measurement was interrupted before unloading scan - i.e. there is no unloading scan
                Displacement_out = sub_group.get_group(0)["Displacement [um]"].to_list()
                Force_out = sub_group.get_group(0)["Force A [uN]"].to_list()
            except:
                txt.insert(END, f"No unloading scan detected in file: {filename}\n")
                txt.update()
                txt.see("end")
                pass

            plt.figure() #It creates a new figure every time, necessary otherwise it keeps adding to the same plot!
            plt.scatter(Displacement_in, Force_in, s=10, marker="o", color='white', edgecolors='steelblue', label='Loading data')
            try:
                plt.scatter(Displacement_out, Force_out, s=10, marker="o", color='white', edgecolors='orange', label='Unloading data')
            except:
                txt.insert(END, f"No unloading scan detected in file: {filename}\n")
                txt.update()
                txt.see("end")
                pass

            plt.title("Measurement_{}".format(str(num)), fontsize=20)
            plt.xlabel("Distance (um)", fontsize=18)
            plt.ylabel("Force (uN)", fontsize=18)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tick_params(direction='in', length=8)
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{path}/{filename_noext}_{str(num+1)}.png", bbox_inches='tight', dpi=300)
            plt.close('all') #To close all figures and save memory - max # of figures before warning: 20
            txt.insert(END, f"    Plotting file {filename_noext}_{str(num+1)}...Done!\n\n")
            txt.update()
            txt.see("end")

def Generate3DPlot(data_list, folder_path_fitting_restults, filename_element, Z_axis_label, Z_axis): #Takes in input a 2D list [[Exp_number, X, Y, Z], [Exp_number, X, Y, Z], ...]

    def myround(x, base = 2.5):
        return  round(x/base) * base

    sorted_list = sorted(data_list) #Sort list based on first element which is the experiemnt number. This ensures that datapoint in correct order for 3D plotting

    #Zero X and Y coordinates based on first datapoint and generate the 3 lists (X, Y, Z) for 3D plotting
    list_corr = []
    num = []
    X = []
    Y = []
    Z = []
    n = len(sorted_list)
    m = len(sorted_list[0])
    for i in range(n):
        #list_corr.append([sorted_list[i][0], myround(sorted_list[i][1] - sorted_list[0][1], 0), myround(sorted_list[i][2] - sorted_list[0][2], 0), sorted_list[i][m - 1]])
        list_corr.append([sorted_list[i][0], round(sorted_list[i][1] - sorted_list[0][1], 0), round(sorted_list[i][2] - sorted_list[0][2], 0), sorted_list[i][m - 1]])
        for j in range(m):
            if j == 0:
                num.append(list_corr[i][j])
            if j == 1:
                X.append(list_corr[i][j])
            if j == 2:
                Y.append(list_corr[i][j])
            if j == 3:
                Z.append(list_corr[i][j])
    txt.insert(END, f"List of experiment numbers (check): {num[0:10]}\n")
    txt.insert(END, f"List of X (check): {X[0:10]}\n")
    txt.insert(END, f"List of Y (check): {Y[0:10]}\n")
    txt.insert(END, f"List of Z (check): {Z[0:10]}\n\n")
    txt.update()
    txt.see("end")

    #Copy X, Y, Z to xyz dataframe and save it to csv
    xyz = pandas.DataFrame()
    xyz["X"] = X
    xyz["Y"] = Y
    xyz["Z"] = Z
    xyz.to_csv(f"{folder_path_fitting_restults}/{filename_element}_3D plot data {Z_axis}.csv", index=False)

    try:
        #Transform lists into numpy arrays
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)

        #Determine shape of array and size/area of 3D surface
        container_x = []
        for x in X:
            if x not in container_x:
                container_x.append(x)

        container_y = []
        for y in Y:
            if y not in container_y:
                container_y.append(y)

        #Reshape all to be 2d array for plotting
        X = X.reshape(len(container_x), len(container_y))
        Y = Y.reshape(len(container_x), len(container_y))
        Z = Z.reshape(len(container_x), len(container_y))

        #Generate figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 8))#make two subplot

        #first graph - 2d map
        axes[0].set_xlabel('X position (um)', fontsize=20)#Set x aixs title and font size
        axes[0].set_ylabel('Y position (um)', fontsize=20)#Set y aixs title
        p1 = axes[0].pcolor(X, Y, Z, cmap=cm.viridis,shading='nearest')#For colormaps see: https://matplotlib.org/stable/tutorials/colors/colormaps.html
        axes[0].tick_params(labelsize=16, direction='in', length=8)#Adjust axis ticks and label font size
        cbar_ax = fig.add_axes([0.81, 0.11, 0.04, 0.77])  #Adjust size and position of colorbar [left, bottom, width, height]
        cb = fig.colorbar(p1, shrink=0.5, cax=cbar_ax)#Add colorbar to plot; it is an independent object shared to both graphs
        cb.ax.tick_params(labelsize=16)#Adjust colorbar label font size

        #second graph - 3d map
        axes[1].axis('off')#orignial placeholder delete 2d axis
        plt.subplots_adjust(right = 1.6)
        ax3d_l = fig.add_subplot(1, 2, 2, projection='3d')#add 3d plot 1, 2, 2 indicate the position of the plot
        ax3d_l.set_xlabel('X position (um)', fontsize=20, labelpad=10)#set x aixs title
        ax3d_l.set_ylabel('Y position (um)', fontsize=20, labelpad=10)#set y aixs title
        ax3d_l.set_zlabel(Z_axis_label, fontsize=20, labelpad=10)#set z aixs title
        ax3d_l.plot_surface(X, Y, Z, cmap=cm.viridis)#plot3d graph
        ax3d_l.tick_params(labelsize=16, direction='in', length=15)#Set axis font size, unfortunately matplot lib ignores the other parameters because designed for 2D plots.

        #Generate filename and save plot
        #path, filename = os.path.split(file)
        #filename_noext = filename[:-4]
        #filename_elements = filename_noext.split("_")
        #path_filename_date = f"{path}/{filename_elements[0]}_{filename_elements[1]}"
        #plt.savefig(f"{path_filename_date}_3D plots {Z_axis}.png", bbox_inches='tight', dpi=300)#save figure. please change directory to your saving directory
        plt.savefig(f"{folder_path_fitting_restults}/{filename_element}_3D plots {Z_axis}.png", bbox_inches='tight', dpi=300)#save figure. please change directory to your saving directory
    except:
        txt.insert(END, f"3D plotting of {Z_axis} failed. Data saved to csv file: check XY coordinates. \n")
        txt.update()
        txt.see("end")
        Spacer()
        pass

def Find_Diplacement_For_Threshold_Map(path_filename, ratio_BL_points, thres_F):
    #Open df
    path, filename = os.path.split(path_filename)
    df = pandas.read_table(path_filename, sep=',', header=0)
    df = df.astype(float)  #Change data from object to float

    #DETECT WHETHER THE MEASUREMENT WAS DONE USING PIEZO SCANNER OR STICK-SLIP ACTUATOR
    PosZ_um_col = df.columns.get_loc("Pos Z [um]") #Get column number
    delta_stepmotor = df.iloc[2, PosZ_um_col] - df.iloc[3, PosZ_um_col]
    if delta_stepmotor < 0.005:  # This value was optimised based on the dataset available so far
        Zmov_type = "Piezo scanner"
        txt.insert(END, "Measurement carried out using piezo scanner.\n")
    else:
        Zmov_type = "Stick-slip actuator"
        txt.insert(END, "Measurement carried out using stick-slip actuator.\n")

    # BASELINE
    numpoint_baseline, _ = df.shape
    numpoint_baseline = round(numpoint_baseline*ratio_BL_points) #Determine number of points for baseline is 1/8 of datapoints of df
    baseline = pandas.DataFrame(df.iloc[0:numpoint_baseline])
    x_bl = pandas.DataFrame(baseline["Displacement [um]"])
    y_bl = pandas.DataFrame(baseline["Force A [uN]"])
    regr = linear_model.LinearRegression() #Check that x adn y have a shape (n, 1)
    regr.fit(x_bl, y_bl)
    df["Corr Force A [uN]"] = df["Force A [uN]"]-(regr.intercept_[0]+(regr.coef_[0][0]*df["Displacement [um]"])) #y=a+bx; add new column to df with corrected force values; #Corr Force A [N] = Col20
    CorrForceA_uN_col = df.columns.get_loc("Corr Force A [uN]")
    Displacement_um_col = df.columns.get_loc("Displacement [um]")


    # FIND CONTACT AND DETACHMENT POINTS
    # Find mean and standard deviation of first X datapoints of indentation
    forward_scan = pandas.DataFrame(df.groupby(["Exp Phase [#]"]).get_group(1.0))
    back_scan = pandas.DataFrame(df.groupby(["Exp Phase [#]"]).get_group(0.0))

    forward_scan_BL_mean = forward_scan.iloc[0:int(forward_scan.shape[0] * ratio_BL_points), CorrForceA_uN_col].mean()  # Baseline on first 1/8 datapoints
    forward_scan_BL_std = forward_scan.iloc[0:int(forward_scan.shape[0] * ratio_BL_points), CorrForceA_uN_col].std()
    txt.insert(END, f"Baseline Force (uN) mean and std: {round(forward_scan_BL_mean,5)} +/- {round(forward_scan_BL_std,5)}\n")
    threshold = forward_scan_BL_mean - (forward_scan_BL_std * 20)  # lower threshold to determine level of noise, 20 was determined empyrically on data available
    txt.insert(END, f"Baseline Force (uN) threshold: {round(threshold,5)}\n")
    #Find point of minimum
    forward_scan_index_min = forward_scan["Corr Force A [uN]"].idxmin()
    txt.insert(END, "Index of point of contact forward scan: {}\n".format(forward_scan_index_min))
    txt.update()
    txt.see("end")
    # If min point is within baseline error => there is no min
    if threshold < forward_scan.iloc[forward_scan_index_min, CorrForceA_uN_col]:
        txt.insert(END, "No interaction with material detected.\n\n")
        selection = forward_scan.loc[forward_scan["Corr Force A [uN]"] >= abs(threshold)]
        min_indentation = selection.iloc[0, CorrForceA_uN_col]
        forward_scan_index_min = forward_scan.loc[forward_scan["Corr Force A [uN]"] == min_indentation].index[0]
        txt.insert(END, "New index of point of contact forward scan: {}\n".format(forward_scan_index_min))

    back_scan = back_scan.iloc[::-1].reset_index(drop=True)  # Reverse and reindex dataframe
    back_scan_index_min = back_scan["Corr Force A [uN]"].idxmin()
    txt.insert(END, "Index of point of detachment back scan: {}\n\n".format(back_scan_index_min))
    txt.update()
    txt.see("end")
    # If min point is within baseline error => there is no min
    if threshold < back_scan.iloc[back_scan_index_min, CorrForceA_uN_col]:
        txt.insert(END, "No interaction with material detected.\n\n")
        selection = back_scan.loc[back_scan["Corr Force A [uN]"] >= abs(threshold)]
        min_indentation = selection.iloc[0, CorrForceA_uN_col]
        back_scan_index_min = back_scan.loc[back_scan["Corr Force A [uN]"] == min_indentation].index[0]
        txt.insert(END, "New index of point of detachment back scan: {}\n\n".format(back_scan_index_min))

    #Select only part of curves after contact and detachment points
    forward_scan = forward_scan[forward_scan.index >= forward_scan_index_min]
    back_scan = back_scan[back_scan.index >= back_scan_index_min]

    #Make curve start from zero indentation and zero force
    forward_scan["Corr Force A [uN]"] = forward_scan["Corr Force A [uN]"]-forward_scan.iloc[0, CorrForceA_uN_col]
    forward_scan["Corr Displacement [um]"] = forward_scan["Displacement [um]"]-forward_scan.iloc[0, Displacement_um_col]
    back_scan["Corr Force A [uN]"] = back_scan["Corr Force A [uN]"]-back_scan.iloc[0, CorrForceA_uN_col]
    back_scan["Corr Displacement [um]"] = back_scan["Displacement [um]"]-back_scan.iloc[0, Displacement_um_col]
    CorrDisplacement_um_col = forward_scan.columns.get_loc("Corr Displacement [um]")

    #Find displacement at thres_F
    forward_scan = forward_scan[forward_scan["Corr Force A [uN]"] <= thres_F]
    displacement_LS = forward_scan.iloc[forward_scan.shape[0]-1, CorrDisplacement_um_col]
    back_scan = back_scan[back_scan["Corr Force A [uN]"] <= thres_F]

    try:
        displacement_US = back_scan.iloc[forward_scan.shape[0]-1, CorrDisplacement_um_col]
    except:
        displacement_US = displacement_LS

    #Find X,Y coordinates
    X, Y = GetXY(df, Zmov_type)

    return X, Y, displacement_LS, displacement_US

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#RUN SOFTWARE
root  = Tk()
root.wm_title("Gobbo Group – A.L.I.A.S.") #A.L.I.A.S. = A Lovely Indentation Analysis System
root.geometry("1250x680")


#SET VARIABLES TO BE USED IN OTHER BUTTONS
#NOTE: Tkinter button cannot return a variable. To modify these variables inside the button's function and make them available outside
# the button's function you need to declare them outside the function and then declare them as global inside the function.
path_filename = ""
path = ""
filename = ""
filename_noext = ""
extension = ""
folder_path = ""
path_foldername = ""

year_now = datetime.now().strftime('%Y')
month_now = datetime.now().strftime("%B")  # returns the full name of the month as a string

def File_Extractor_b1():
    dir = filedialog.askdirectory(initialdir="/Users/pierangelogobbo/Dropbox/@Documents/@My Research/@My papers/Micro-indentation/Esperimental data", title="Select a folder")
    # to handle Cancel button
    if dir == "":
        return


    txt.insert(END, f"Working directory: {dir}\n\n")
    txt.update()
    txt.see("end")


    #Find all folders
    folder_list = glob.glob(f"{dir}/*/")

    counter_found = 0
    for folder_path_name in folder_list:
        elements = folder_path_name.split("/")
        folder_name = elements[len(elements)-2]
        txt.insert(END, f"Extracting from: {folder_name}\n")
        txt.update()
        txt.see("end")

        found = 0
        for root, dirs, files in os.walk(folder_path_name, topdown=False):
            for name in files:
                # print(os.path.join(root, name))
                if name == "data.txt":
                    if os.path.isfile(f"{dir}/{folder_name}.txt") == True:
                        shutil.copy(f"{os.path.join(root, name)}", f"{dir}/{folder_name}_{counter_found+1}.txt")
                    else:
                        shutil.copy(f"{os.path.join(root, name)}", f"{dir}/{folder_name}.txt")
                    txt.insert(END, f"File found: {name}.\n")
                    txt.update()
                    txt.see("end")
                    counter_found = counter_found+1
                    found = 1

        if found == 0:
            txt.insert(END, f"Error: No file data.txt found.\n")
            txt.update()
            txt.see("end")

        txt.insert(END, f"Moving {folder_name} to backup folder 'Original data'\n")
        txt.insert(END, f"Operation successful!\n\n")
        txt.update()
        txt.see("end")
        shutil.copytree(f"{folder_path_name[:-1]}", f"{dir}/Original data/{folder_name}")
        shutil.rmtree(f"{folder_path_name[:-1]}")

    txt.insert(END, f"Found {counter_found} data.txt files in {len(folder_list)} experiment folders.\n\n")
    txt.update()
    txt.see("end")
    Spacer()
    Spacer()
    txt.insert(END, "END OF PROGRAM.\n".format(year_now))
    Spacer()
    Spacer()
    txt.update()
    txt.see("end")

def Gen_Single_Exp_b2():
    #SET GLOBAL VARIABLE TO BE USED IN OTHER BUTTONS
    global path_filename #Global to reassing value adn have it available outside the button's function.
    global path
    global filename
    global filename_noext
    global extension
    global folder_path

    #Ask to open file and get filename and path
    path_filename = filedialog.askopenfilename(initialdir = "/Users/pierangelogobbo/Dropbox/@Documents/@My Research/Lab useful docs/FemtoTools Nanoindenter/Indentation_Curve_Analysis/Input files", title = "Select a file", filetypes = (("Text files", "*.txt"),("All files","*.*")))

    try:
        path, filename, filename_noext, extension, folder_path = Find_Experiments(path_filename)
    except:
        return #to handle Cancel button

    Spacer()
    Spacer()
    txt.insert(END, "END OF PROGRAM.\n".format(year_now))
    Spacer()
    Spacer()
    txt.update()
    txt.see("end")

def Divide_And_Plot_All_Exp_b3():
    #SET GLOBAL VARIABLE TO BE USED IN OTHER BUTTONS
    global path_filename #Global to reassing value adn have it available outside the button's function.
    global path
    global filename
    global filename_noext
    global extension

    path, filename = os.path.split(path_filename)

    #OPEN FILE
    path_foldername = filedialog.askdirectory(initialdir="/Users/pierangelogobbo/Dropbox/@Documents/@My Research/Lab useful docs/FemtoTools Nanoindenter/Indentation_Curve_Analysis/Input files", title="Select a folder")
    # to handle Cancel button
    if path_foldername == "":
        return

    #Find all *.txt files AND count them
    paths_filenames = sorted(glob.glob("{}/*.txt".format(path_foldername)))  # Get all .csv files and put them in alphabetical order

    txt.insert(END, f"Converting *.txt files to *.csv files and plotting the experiments...\n")
    txt.update()
    txt.see("end")
    txt.insert(END, "Number of files to process: {}\n\n\n".format(len(paths_filenames)))
    txt.update()
    txt.see("end")

    for path_filename in paths_filenames:
        try:
            Divide_Curve_And_Plot(path_filename)
        except:
            path, filename = os.path.split(path_filename)
            txt.insert(END, f"Error: Impossible to analise {filename}\n\n")
            txt.update()
            txt.see("end")

    Spacer()
    Spacer()
    txt.insert(END, "END OF PROGRAM.\n".format(year_now))
    Spacer()
    Spacer()
    txt.update()
    txt.see("end")

def FitCurves():

    txt.delete(1.0, END) #Clear dialog window

    global path_foldername
    model = clicked.get()

    #PARAMETERS DEFINITION
    R = (float(e1.get()))/2
    v_sample = float(e2.get())
    ratio_BL_points = float(e3.get())
    threshold_constant = float(e4.get())
    forward_scan_segment = float(e5.get())
    back_scan_segment = float(e6.get())

    #Ask to open folder and get folder name and path
    path_filenames = filedialog.askopenfilenames(initialdir="/Users/pierangelogobbo/Dropbox/@Documents/@My Research/Lab useful docs/FemtoTools Nanoindenter/Indentation_Curve_Analysis/Input files", title="Select all files to process", filetypes = [("CSV files", "*.csv")])

    # to handle Cancel button
    if path_filenames == "":
        return

    working_dir_path, working_dir_filename = os.path.split(path_filenames[0])
    txt.insert(END, "Working directory:\n{}\n\n".format(working_dir_path))
    txt.update()
    txt.see("end")
    Spacer()

    txt.insert(END, "Number of selected files: {}\n".format(len(path_filenames)))
    txt.update()
    txt.see("end")

    #Generate list of files to be analised
    file_list = []
    for i, path_filename in enumerate(path_filenames):
        if ("curve" not in path_filename) and ("results" not in path_filename):
            file_list.append(path_filename)
    txt.insert(END, "Number of files to analyse: {}\n\n".format(len(file_list)))

    # Create a folder for the fitting results
    filename_noext, extension = os.path.splitext(working_dir_filename)
    filename_elements = filename_noext.split("_")
    folder_path_fitting_restults = working_dir_path + f"/{filename_elements[0]}_Fitting Results"
    Create_Output_Folder(folder_path_fitting_restults)

    warning_list = []
    error_list = []

    if model == "Hertz":
        Spacer()
        Spacer()
        txt.insert(END, "INITIATING FITTING WITH HERTZ MODEL...\n")
        txt.update()
        txt.see("end")
        Spacer()

        #Get all paths and filenames and memorise them, then show all filenames found.
        for i, path_filename in enumerate(file_list):
            path, filename = os.path.split(path_filename)  # splits path and file name
            filename_noext, extension = os.path.splitext(filename)
            txt.insert(END, "{}. Analysisng {}\n\n".format(i+1, filename))
            txt.update()
            txt.see("end")
            try:
                with warnings.catch_warnings(record=True) as w:
                    # Cause all warnings to always be triggered.
                    warnings.simplefilter("always")
                    Hertz_Analyse_Single_CSVFile(path_filename, folder_path_fitting_restults, filename_noext, ratio_BL_points, threshold_constant, forward_scan_segment, back_scan_segment, R, v_sample)
                    if len(w) > 0:
                        warning_list.append(f"Warning detected in file: {filename}\n")
            except:
                txt.insert(END, f"Error in file {filename}.\n\n")
                txt.update()
                txt.see("end")
                error_list.append(f"Error in file {filename}.\n")
                pass
        Spacer()
        Spacer()
        if len(warning_list) == 0:
            txt.insert(END, "No warnings to report.\n")
            txt.update()
            txt.see("end")
        else:
            for warning in warning_list:
                txt.insert(END, warning)
                txt.update()
                txt.see("end")
        if len(error_list) == 0:
            txt.insert(END, "No errors to report.\n")
            txt.update()
            txt.see("end")
        else:
            for error in error_list:
                txt.insert(END, error)
                txt.update()
                txt.see("end")
        Spacer()
        Spacer()
        txt.insert(END, "FITTING WITH HERTZ MODEL DONE.\n".format(year_now))
        Spacer()
        Spacer()
        txt.update()
        txt.see("end")

    if model == "DMT":
        Spacer()
        Spacer()
        txt.insert(END, "INITIATING FITTING WITH DMT MODEL...\n")
        txt.update()
        txt.see("end")
        Spacer()

        #Get all paths and filenames and memorise them, then show all filenames found.
        for i, path_filename in enumerate(file_list):
            path, filename = os.path.split(path_filename)  # splits path and file name
            filename_noext, extension = os.path.splitext(filename)
            txt.insert(END, "{}. Analysisng {}\n\n".format(i+1, filename))
            txt.update()
            txt.see("end")
            try:
                with warnings.catch_warnings(record=True) as w:
                    # Cause all warnings to always be triggered.
                    warnings.simplefilter("always")
                    DMT_Analyse_Single_CSVFile(path_filename, folder_path_fitting_restults, filename_noext, ratio_BL_points, threshold_constant, forward_scan_segment, back_scan_segment, R, v_sample)
                    if len(w) > 0:
                        warning_list.append(f"Warning detected in file: {filename}\n")
            except:
                txt.insert(END, f"Error in file {filename}.\n\n")
                txt.update()
                txt.see("end")
                error_list.append(f"Error in file {filename}.\n")
                pass
        Spacer()
        Spacer()
        if len(warning_list) == 0:
            txt.insert(END, "No warnings to report.\n")
            txt.update()
            txt.see("end")
        else:
            for warning in warning_list:
                txt.insert(END, warning)
                txt.update()
                txt.see("end")
        if len(error_list) == 0:
            txt.insert(END, "No errors to report.\n")
            txt.update()
            txt.see("end")
        else:
            for error in error_list:
                txt.insert(END, error)
                txt.update()
                txt.see("end")
        Spacer()
        Spacer()
        txt.insert(END, "FITTING WITH DMT MODEL DONE.\n".format(year_now))
        Spacer()
        Spacer()
        txt.update()
        txt.see("end")

    if model == "PT":
        Spacer()
        Spacer()
        txt.insert(END, "INITIATING FITTING WITH PT MODEL...\n")
        txt.update()
        txt.see("end")
        Spacer()

        #Get all paths and filenames and memorise them, then show all filenames found.
        for i, path_filename in enumerate(file_list):
            path, filename = os.path.split(path_filename)  # splits path and file name
            filename_noext, extension = os.path.splitext(filename)
            txt.insert(END, "{}. Analysisng {}\n\n".format(i+1, filename))
            txt.update()
            txt.see("end")
            try:
                with warnings.catch_warnings(record=True) as w:
                    # Cause all warnings to always be triggered.
                    warnings.simplefilter("always")
                    PT_Analyse_Single_CSVFile(path_filename, folder_path_fitting_restults, filename_noext, ratio_BL_points, threshold_constant, forward_scan_segment, back_scan_segment, R, v_sample)
                    if len(w) > 0:
                        warning_list.append(f"Warning detected in file: {filename}\n")
            except:
                txt.insert(END, f"Error in file {filename}.\n\n")
                txt.update()
                txt.see("end")
                error_list.append(f"Error in file {filename}.\n")
                pass
        Spacer()
        Spacer()
        if len(warning_list) == 0:
            txt.insert(END, "No warnings to report.\n")
            txt.update()
            txt.see("end")
        else:
            for warning in warning_list:
                txt.insert(END, warning)
                txt.update()
                txt.see("end")
        if len(error_list) == 0:
            txt.insert(END, "No errors to report.\n")
            txt.update()
            txt.see("end")
        else:
            for error in error_list:
                txt.insert(END, error)
                txt.update()
                txt.see("end")
        Spacer()
        Spacer()
        txt.insert(END, "FITTING WITH PT MODEL DONE.\n".format(year_now))
        Spacer()
        Spacer()
        txt.update()
        txt.see("end")

    if model == "JKR":
        Spacer()
        Spacer()
        txt.insert(END, "INITIATING FITTING WITH JKR MODEL...\n")
        txt.update()
        txt.see("end")
        Spacer()

        #Get all paths and filenames and memorise them, then show all filenames found.
        for i, path_filename in enumerate(file_list):
            path, filename = os.path.split(path_filename)  # splits path and file name
            filename_noext, extension = os.path.splitext(filename)
            txt.insert(END, "{}. Analysisng {}\n\n".format(i+1, filename))
            txt.update()
            txt.see("end")
            try:
                with warnings.catch_warnings(record=True) as w:
                    # Cause all warnings to always be triggered.
                    warnings.simplefilter("always")
                    JKR_Analyse_Single_CSVFile(path_filename, folder_path_fitting_restults, filename_noext, ratio_BL_points, threshold_constant, forward_scan_segment, back_scan_segment, R, v_sample)
                    if len(w) > 0:
                        warning_list.append(f"Warning detected in file: {filename}\n")
            except:
                txt.insert(END, f"Error in file {filename}.\n\n")
                txt.update()
                txt.see("end")
                error_list.append(f"Error in file {filename}.\n")
                pass
        Spacer()
        Spacer()
        if len(warning_list) == 0:
            txt.insert(END, "No warnings to report.\n")
            txt.update()
            txt.see("end")
        else:
            for warning in warning_list:
                txt.insert(END, warning)
                txt.update()
                txt.see("end")
        if len(error_list) == 0:
            txt.insert(END, "No errors to report.\n")
            txt.update()
            txt.see("end")
        else:
            for error in error_list:
                txt.insert(END, error)
                txt.update()
                txt.see("end")
        Spacer()
        Spacer()
        txt.insert(END, "FITTING WITH JKR MODEL DONE.\n".format(year_now))
        Spacer()
        Spacer()
        txt.update()
        txt.see("end")

    #Save report in dialogue window as txt file
    file = open(f"{folder_path_fitting_restults}/{filename_noext}_report.txt", mode='w')
    report = txt.get(1.0, END)
    file.write(report)
    file.close()

def Divide_Array_And_Plot_All_Exp_b4():
    #SET GLOBAL VARIABLE TO BE USED IN OTHER BUTTONS
    global path_filename #Global to reassing value adn have it available outside the button's function.
    global path
    global filename
    global filename_noext
    global extension

    path, filename = os.path.split(path_filename)

    #OPEN FILE
    path_filename = filedialog.askopenfilename(initialdir = "/Users/pierangelogobbo/Dropbox/@Documents/@My Research/Lab useful docs/FemtoTools Nanoindenter/Indentation_Curve_Analysis/Input files", title = "Select a file", filetypes = (("Text files", "*.txt"),("All files","*.*")))

    # to handle Cancel button
    if path_filename == "":
        return


    txt.insert(END, f"Converting *.txt files to *.csv files and plotting the experiments...\n")
    txt.update()
    txt.see("end")

    try:
        Divide_Array_And_Plot(path_filename)
    except:
        path, filename = os.path.split(path_filename)
        txt.insert(END, f"Error: Impossible to analise {filename}\n\n")
        txt.update()
        txt.see("end")

    Spacer()
    Spacer()
    txt.insert(END, "END OF PROGRAM.\n".format(year_now))
    Spacer()
    Spacer()
    txt.update()
    txt.see("end")

def Generate3DMaps_b5():

    txt.delete(1.0, END) #Clear dialog window

    global path_foldername
    model = clicked2.get()

    #PARAMETERS DEFINITION
    R = (float(e1.get()))/2
    v_sample = float(e2.get())
    ratio_BL_points = float(e3.get())
    threshold_constant = float(e4.get())
    forward_scan_segment = float(e5.get())
    back_scan_segment = float(e6.get())

    #Ask to open files and get folder name and path
    path_filenames = filedialog.askopenfilenames(initialdir="/Users/pierangelogobbo/Dropbox/@Documents/@My Research/Lab useful docs/FemtoTools Nanoindenter/Indentation_Curve_Analysis/Input files", title="Select all files to process", filetypes = [("CSV files", "*.csv")])

    # to handle Cancel button
    if path_filenames == "":
        return

    working_dir_path, working_dir_filename = os.path.split(path_filenames[0])
    txt.insert(END, "Working directory:\n{}\n\n".format(working_dir_path))
    txt.update()
    txt.see("end")
    Spacer()

    txt.insert(END, "Number of selected files: {}\n".format(len(path_filenames)))
    txt.update()
    txt.see("end")

    #Generate list of files to be analised
    file_list = []
    for i, path_filename in enumerate(path_filenames):
        if ("curve" not in path_filename) and ("results" not in path_filename):
            file_list.append(path_filename)
    txt.insert(END, "Number of files to analyse: {}\n\n".format(len(file_list)))

    # Create a folder for the fitting results
    filename_noext, extension = os.path.splitext(working_dir_filename)
    filename_elements = filename_noext.split("_")
    folder_path_fitting_restults = working_dir_path + f"/{filename_elements[0]}_Fitting Results"
    Create_Output_Folder(folder_path_fitting_restults)
    folder_path_fitting_3Dplots = working_dir_path + f"/{filename_elements[0]}_3D Plots"
    Create_Output_Folder(folder_path_fitting_3Dplots)


    warning_list = []
    error_list = []

    if model == "Hertz":
        Spacer()
        Spacer()
        txt.insert(END, "INITIATING GENERATION OF 3D MAPS WITH HERTZ MODEL...\n")
        txt.update()
        txt.see("end")
        Spacer()

        #Generate all the lists required for the 3D maps
        sep_point_list = []
        Es_LS_fc_list = []
        Es_LS_ema_list = []
        Es_US_fc_list = []
        Es_US_ema_list = []

        for i, path_filename in enumerate(file_list):
            # Get all paths and filenames and memorise them, then show all filenames found.
            path, filename = os.path.split(path_filename)  # splits path and file name
            txt.insert(END, "{}. Analysisng {}\n\n".format(i+1, filename))
            txt.update()
            txt.see("end")

            #Get the experiemnt number from the file name to order the list later and allow 3D plotting
            filename_noext, extension = os.path.splitext(filename)
            filename_elements = filename_noext.split("_")
            filename_num = int(filename_elements[2])

            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always") # Cause all warnings to always be triggered.
                    X, Y, sep_point, Es_LS_fc, Es_LS_ema, Es_US_fc, Es_US_ema = Hertz_Analyse_Single_CSVFile(path_filename, folder_path_fitting_restults, filename_noext, ratio_BL_points, threshold_constant, forward_scan_segment, back_scan_segment, R, v_sample)
                    if len(w) > 0:
                        warning_list.append(f"Warning detected in file: {filename}\n")
            except:
                #Open the dataframe
                df = pandas.read_table("{}".format(path_filename), sep=',', header=0)
                #path_filename_noext, ext = os.path.splitext(path_filename)  # Remove ".csv" from filename to avoid weird name for documents
                df = df.astype(float)  # Change data from object to float
                whatever, ForceA_N_col, Time_s_col, Corr_Time_s_col, Displacement_um_col, Displacement_m_col, PiezoZ_um_col, PiezoZ_m_col, PosZ_um_col, PosZ_m_col = Unit_Transform(df, R)

                #Understand if the measurement was carried out using the piezo-scanner or teh stick-slip actuator
                delta_stepmotor = df.iloc[2, PosZ_um_col] - df.iloc[3, PosZ_um_col]
                if delta_stepmotor < 0.005:  # This value was optimised based on the dataset available so far
                    Zmov_type = "Piezo scanner"
                    txt.insert(END, "Measurement carried out using piezo scanner.\n\n")
                else:
                    Zmov_type = "Stick-slip actuator"
                    txt.insert(END, "Measurement carried out using stick-slip actuator.\n\n")

                #Find X and Y and set the Z variable to 1
                X, Y = GetXY(df, Zmov_type)
                sep_point, Es_LS_fc, Es_LS_ema, Es_US_fc, Es_US_ema = 1, 1, 1, 1, 1
                txt.insert(END, f"Error in file {filename}. Z axis value set to 1.\n\n")
                txt.update()
                txt.see("end")
                error_list.append(f"Error in file {filename}.\n")
                pass
            sep_point_list.append([filename_num, X, Y, sep_point])
            Es_LS_fc_list.append([filename_num, X, Y, Es_LS_fc])
            Es_LS_ema_list.append([filename_num, X, Y, Es_LS_ema])
            Es_US_fc_list.append([filename_num, X, Y, Es_US_fc])
            Es_US_ema_list.append([filename_num, X, Y, Es_US_ema])

        #Generate the 3D maps
        Generate3DPlot(Es_LS_fc_list, folder_path_fitting_3Dplots, filename_elements[0], "Es (Pa)", "Es LS_fc")
        Generate3DPlot(Es_LS_ema_list, folder_path_fitting_3Dplots, filename_elements[0], "Es (Pa)", "Es LS_ema")
        Generate3DPlot(Es_US_fc_list, folder_path_fitting_3Dplots, filename_elements[0], "Es (Pa)", "Es US_fc")
        Generate3DPlot(Es_US_ema_list, folder_path_fitting_3Dplots, filename_elements[0], "Es (Pa)", "Es US_ema")
        Generate3DPlot(sep_point_list, folder_path_fitting_3Dplots, filename_elements[0], "Separation point (um)", "SepPoint")

        Spacer()
        Spacer()
        if len(warning_list) == 0:
            txt.insert(END, "No warnings to report.\n")
            txt.update()
            txt.see("end")
        else:
            for warning in warning_list:
                txt.insert(END, warning)
                txt.update()
                txt.see("end")
        if len(error_list) == 0:
            txt.insert(END, "No errors to report.\n")
            txt.update()
            txt.see("end")
        else:
            for error in error_list:
                txt.insert(END, error)
                txt.update()
                txt.see("end")
        Spacer()
        Spacer()
        txt.insert(END, "GENERATION OF 3D MAPS WITH HERTZ MODEL DONE.\n".format(year_now))
        Spacer()
        Spacer()
        txt.update()
        txt.see("end")

    if model == "DMT":
        Spacer()
        Spacer()
        txt.insert(END, "INITIATING GENERATION OF 3D MAPS WITH DMT MODEL...\n")
        txt.update()
        txt.see("end")
        Spacer()

        #Generate all the lists required for the 3D maps
        attr_force_list = []
        adhes_force_list = []
        sep_point_list = []
        Es_LS_fc_list = []
        Es_LS_ema_list = []
        Es_US_fc_list = []
        Es_US_ema_list = []
        interf_en_LS_list = []
        interf_en_US_list = []

        for i, path_filename in enumerate(file_list):
            # Get all paths and filenames and memorise them, then show all filenames found.
            path, filename = os.path.split(path_filename)  # splits path and file name
            txt.insert(END, "{}. Analysisng {}\n\n".format(i+1, filename))
            txt.update()
            txt.see("end")

            #Get the experiemnt number from the file name to order the list later and allow 3D plotting
            filename_noext, extension = os.path.splitext(filename)
            filename_elements = filename_noext.split("_")
            filename_num = int(filename_elements[2])

            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always") # Cause all warnings to always be triggered.
                    X, Y, attr_force, adhes_force, sep_point, Es_LS_fc, Es_LS_ema, Es_US_fc, Es_US_ema, interf_en_LS, interf_en_US = DMT_Analyse_Single_CSVFile(path_filename, folder_path_fitting_restults, filename_noext, ratio_BL_points, threshold_constant, forward_scan_segment, back_scan_segment, R, v_sample)
                    if len(w) > 0:
                        warning_list.append(f"Warning detected in file: {filename}\n")

            except :
                #Open the dataframe
                df = pandas.read_table("{}".format(path_filename), sep=',', header=0)
                #path_filename_noext, ext = os.path.splitext(path_filename)  # Remove ".csv" from filename to avoid weird name for documents
                df = df.astype(float)  # Change data from object to float
                whatever, ForceA_N_col, Time_s_col, Corr_Time_s_col, Displacement_um_col, Displacement_m_col, PiezoZ_um_col, PiezoZ_m_col, PosZ_um_col, PosZ_m_col = Unit_Transform(df, R)

                #Understand if the measurement was carried out using the piezo-scanner or teh stick-slip actuator
                delta_stepmotor = df.iloc[2, PosZ_um_col] - df.iloc[3, PosZ_um_col]
                if delta_stepmotor < 0.005:  # This value was optimised based on the dataset available so far
                    Zmov_type = "Piezo scanner"
                    txt.insert(END, "Measurement carried out using piezo scanner.\n\n")
                else:
                    Zmov_type = "Stick-slip actuator"
                    txt.insert(END, "Measurement carried out using stick-slip actuator.\n\n")

                #Find X and Y and set the Z variable to 1
                X, Y = GetXY(df, Zmov_type)
                attr_force, adhes_force, sep_point, Es_LS_fc, Es_LS_ema, Es_US_fc, Es_US_ema, interf_en_LS, interf_en_US = 1, 1, 1, 1, 1, 1, 1, 1, 1
                txt.insert(END, f"Error in file {filename}. Z axis value set to 1.\n\n")
                txt.update()
                txt.see("end")
                error_list.append(f"Error in file {filename}.\n")
                pass
            attr_force_list.append([filename_num, X, Y, attr_force])
            adhes_force_list.append([filename_num, X, Y, adhes_force])
            sep_point_list.append([filename_num, X, Y, sep_point])
            Es_LS_fc_list.append([filename_num, X, Y, Es_LS_fc])
            Es_LS_ema_list.append([filename_num, X, Y, Es_LS_ema])
            Es_US_fc_list.append([filename_num, X, Y, Es_US_fc])
            Es_US_ema_list.append([filename_num, X, Y, Es_US_ema])
            interf_en_LS_list.append([filename_num, X, Y, interf_en_LS])
            interf_en_US_list.append([filename_num, X, Y, interf_en_US])

        #Generate the 3D maps
        Generate3DPlot(Es_LS_fc_list, folder_path_fitting_3Dplots, filename_elements[0], "Es (Pa)", "Es LS_fc")
        Generate3DPlot(Es_LS_ema_list, folder_path_fitting_3Dplots, filename_elements[0], "Es (Pa)", "Es LS_ema")
        Generate3DPlot(Es_US_fc_list, folder_path_fitting_3Dplots, filename_elements[0], "Es (Pa)", "Es US_fc")
        Generate3DPlot(Es_US_ema_list, folder_path_fitting_3Dplots, filename_elements[0], "Es (Pa)", "Es US_ema")
        Generate3DPlot(attr_force_list, folder_path_fitting_3Dplots, filename_elements[0], "Attractive Force (uN)", "AttrForce")
        Generate3DPlot(adhes_force_list, folder_path_fitting_3Dplots, filename_elements[0], "Adhesive Force (uN)", "AdhesForce")
        Generate3DPlot(sep_point_list, folder_path_fitting_3Dplots, filename_elements[0], "Separation point (um)", "SepPoint")
        Generate3DPlot(interf_en_LS_list, folder_path_fitting_3Dplots, filename_elements[0], "Interfacial Energy (mJ m^-2)", "Interf E LS")
        Generate3DPlot(interf_en_US_list, folder_path_fitting_3Dplots, filename_elements[0], "Interfacial Energy (mJ m^-2)", "Interf E US")

        Spacer()
        Spacer()
        if len(warning_list) == 0:
            txt.insert(END, "No warnings to report.\n")
            txt.update()
            txt.see("end")
        else:
            for warning in warning_list:
                txt.insert(END, warning)
                txt.update()
                txt.see("end")
        if len(error_list) == 0:
            txt.insert(END, "No errors to report.\n")
            txt.update()
            txt.see("end")
        else:
            for error in error_list:
                txt.insert(END, error)
                txt.update()
                txt.see("end")
        Spacer()
        Spacer()
        txt.insert(END, "GENERATION OF 3D MAPS WITH DMT MODEL DONE.\n".format(year_now))
        Spacer()
        Spacer()
        txt.update()
        txt.see("end")

    if model == "PT":
        Spacer()
        Spacer()
        txt.insert(END, "INITIATING GENERATION OF 3D MAPS WITH PT MODEL...\n")
        txt.update()
        txt.see("end")
        Spacer()

        #Generate all the lists required for the 3D maps
        attr_force_list = []
        PZI_LS_list = []
        adhes_force_list = []
        sep_point_list = []
        PZI_US_list = []
        Es_LS_fc_list = []
        Es_LS_ema_list = []
        Es_US_fc_list = []
        Es_US_ema_list = []
        interf_en_LS_list = []
        interf_en_US_list = []

        for i, path_filename in enumerate(file_list):
            # Get all paths and filenames and memorise them, then show all filenames found.
            path, filename = os.path.split(path_filename)  # splits path and file name
            txt.insert(END, "{}. Analysisng {}\n\n".format(i+1, filename))
            txt.update()
            txt.see("end")

            #Get the experiemnt number from the file name to order the list later and allow 3D plotting
            filename_noext, extension = os.path.splitext(filename)
            filename_elements = filename_noext.split("_")
            filename_num = int(filename_elements[2])

            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always") # Cause all warnings to always be triggered.
                    X, Y, attr_force, PZI_LS, adhes_force, sep_point, PZI_US, Es_LS_fc, Es_LS_ema, Es_US_fc, Es_US_ema, interf_en_LS, interf_en_US = PT_Analyse_Single_CSVFile(path_filename, folder_path_fitting_restults, filename_noext, ratio_BL_points, threshold_constant, forward_scan_segment, back_scan_segment, R, v_sample)
                    if len(w) > 0:
                        warning_list.append(f"Warning detected in file: {filename}\n")
            except:
                #Open the dataframe
                df = pandas.read_table("{}".format(path_filename), sep=',', header=0)
                #path_filename_noext, ext = os.path.splitext(path_filename)  # Remove ".csv" from filename to avoid weird name for documents
                df = df.astype(float)  # Change data from object to float
                whatever, ForceA_N_col, Time_s_col, Corr_Time_s_col, Displacement_um_col, Displacement_m_col, PiezoZ_um_col, PiezoZ_m_col, PosZ_um_col, PosZ_m_col = Unit_Transform(df, R)

                #Understand if the measurement was carried out using the piezo-scanner or teh stick-slip actuator
                delta_stepmotor = df.iloc[2, PosZ_um_col] - df.iloc[3, PosZ_um_col]
                if delta_stepmotor < 0.005:  # This value was optimised based on the dataset available so far
                    Zmov_type = "Piezo scanner"
                    txt.insert(END, "Measurement carried out using piezo scanner.\n\n")
                else:
                    Zmov_type = "Stick-slip actuator"
                    txt.insert(END, "Measurement carried out using stick-slip actuator.\n\n")

                #Find X and Y and set the Z variable to 1
                X, Y = GetXY(df, Zmov_type)
                attr_force, PZI_LS, adhes_force, sep_point, PZI_US, Es_LS_fc, Es_LS_ema, Es_US_fc, Es_US_ema, interf_en_LS, interf_en_US = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
                txt.insert(END, f"Error in file {filename}. Z axis value set to 1.\n\n")
                txt.update()
                txt.see("end")
                error_list.append(f"Error in file {filename}.\n")
                pass
            attr_force_list.append([filename_num, X, Y, attr_force])
            PZI_LS_list.append([filename_num, X, Y, PZI_LS])
            adhes_force_list.append([filename_num, X, Y, adhes_force])
            sep_point_list.append([filename_num, X, Y, sep_point])
            PZI_US_list.append([filename_num, X, Y, PZI_US])
            Es_LS_fc_list.append([filename_num, X, Y, Es_LS_fc])
            Es_LS_ema_list.append([filename_num, X, Y, Es_LS_ema])
            Es_US_fc_list.append([filename_num, X, Y, Es_US_fc])
            Es_US_ema_list.append([filename_num, X, Y, Es_US_ema])
            interf_en_LS_list.append([filename_num, X, Y, interf_en_LS])
            interf_en_US_list.append([filename_num, X, Y, interf_en_US])

        #Generate the 3D maps
        Generate3DPlot(Es_LS_fc_list, folder_path_fitting_3Dplots, filename_elements[0], "Es (Pa)", "Es LS_fc")
        Generate3DPlot(Es_LS_ema_list, folder_path_fitting_3Dplots, filename_elements[0], "Es (Pa)", "Es LS_ema")
        Generate3DPlot(Es_US_fc_list, folder_path_fitting_3Dplots, filename_elements[0], "Es (Pa)", "Es US_fc")
        Generate3DPlot(Es_US_ema_list, folder_path_fitting_3Dplots, filename_elements[0], "Es (Pa)", "Es US_ema")
        Generate3DPlot(attr_force_list, folder_path_fitting_3Dplots, filename_elements[0], "Attractive Force (uN)", "AttrForce")
        Generate3DPlot(adhes_force_list, folder_path_fitting_3Dplots, filename_elements[0], "Adhesive Force (uN)", "AdhesForce")
        Generate3DPlot(sep_point_list, folder_path_fitting_3Dplots, filename_elements[0], "Separation point (um)", "SepPoint")
        Generate3DPlot(interf_en_LS_list, folder_path_fitting_3Dplots, filename_elements[0], "Interfacial Energy (mJ m^-2)", "Interf E LS")
        Generate3DPlot(interf_en_US_list, folder_path_fitting_3Dplots, filename_elements[0], "Interfacial Energy (mJ m^-2)", "Interf E US")
        Generate3DPlot(PZI_LS_list, folder_path_fitting_3Dplots, filename_elements[0], "PZI (um)", "PZI LS")
        Generate3DPlot(PZI_US_list, folder_path_fitting_3Dplots, filename_elements[0], "PZI (um)", "PZI US")

        Spacer()
        Spacer()
        if len(warning_list) == 0:
            txt.insert(END, "No warnings to report.\n")
            txt.update()
            txt.see("end")
        else:
            for warning in warning_list:
                txt.insert(END, warning)
                txt.update()
                txt.see("end")
        if len(error_list) == 0:
            txt.insert(END, "No errors to report.\n")
            txt.update()
            txt.see("end")
        else:
            for error in error_list:
                txt.insert(END, error)
                txt.update()
                txt.see("end")
        Spacer()
        Spacer()
        txt.insert(END, "GENERATION OF 3D MAPS WITH PT MODEL DONE.\n".format(year_now))
        Spacer()
        Spacer()
        txt.update()
        txt.see("end")

    if model == "JKR":
        Spacer()
        Spacer()
        txt.insert(END, "INITIATING GENERATION OF 3D MAPS WITH JKR MODEL...\n")
        txt.update()
        txt.see("end")
        Spacer()

        #Generate all the lists required for the 3D maps
        attr_force_list = []
        PZI_LS_list = []
        adhes_force_list = []
        sep_point_list = []
        PZI_US_list = []
        Es_LS_fc_list = []
        Es_LS_ema_list = []
        Es_US_fc_list = []
        Es_US_ema_list = []
        interf_en_LS_list = []
        interf_en_US_list = []

        for i, path_filename in enumerate(file_list):
            #Get all paths and filenames and memorise them, then show all filenames found.
            path, filename = os.path.split(path_filename)
            txt.insert(END, "{}. Analysisng {}\n\n".format(i+1, filename))
            txt.update()
            txt.see("end")

            #Get the experiemnt number from the file name to order the list later and allow 3D plotting
            filename_noext, extension = os.path.splitext(filename)
            filename_elements = filename_noext.split("_")
            filename_num = int(filename_elements[2])

            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always") # Cause all warnings to always be triggered.
                    X, Y, attr_force, PZI_LS, adhes_force, sep_point, PZI_US, Es_LS_fc, Es_LS_ema, Es_US_fc, Es_US_ema, interf_en_LS, interf_en_US = JKR_Analyse_Single_CSVFile(path_filename, folder_path_fitting_restults, filename_noext, ratio_BL_points, threshold_constant, forward_scan_segment, back_scan_segment, R, v_sample)
                    if len(w) > 0:
                        warning_list.append(f"Warning detected in file: {filename}\n")
            except:
                #Open the dataframe
                df = pandas.read_table("{}".format(path_filename), sep=',', header=0)
                #path_filename_noext, ext = os.path.splitext(path_filename)  # Remove ".csv" from filename to avoid weird name for documents
                df = df.astype(float)  # Change data from object to float
                whatever, ForceA_N_col, Time_s_col, Corr_Time_s_col, Displacement_um_col, Displacement_m_col, PiezoZ_um_col, PiezoZ_m_col, PosZ_um_col, PosZ_m_col = Unit_Transform(df, R)

                #Understand if the measurement was carried out using the piezo-scanner or teh stick-slip actuator
                delta_stepmotor = df.iloc[2, PosZ_um_col] - df.iloc[3, PosZ_um_col]
                if delta_stepmotor < 0.005:  # This value was optimised based on the dataset available so far
                    Zmov_type = "Piezo scanner"
                    txt.insert(END, "Measurement carried out using piezo scanner.\n\n")
                else:
                    Zmov_type = "Stick-slip actuator"
                    txt.insert(END, "Measurement carried out using stick-slip actuator.\n\n")

                #Find X and Y and set the Z variable to 1
                X, Y = GetXY(df, Zmov_type)
                attr_force, PZI_LS, adhes_force, sep_point, PZI_US, Es_LS_fc, Es_LS_ema, Es_US_fc, Es_US_ema, interf_en_LS, interf_en_US = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
                txt.insert(END, f"Error in file {filename}. Z axis value set to 1.\n\n")
                txt.update()
                txt.see("end")
                error_list.append(f"Error in file {filename}.\n")
                pass

            attr_force_list.append([filename_num, X, Y, attr_force])
            PZI_LS_list.append([filename_num, X, Y, PZI_LS])
            adhes_force_list.append([filename_num, X, Y, adhes_force])
            sep_point_list.append([filename_num, X, Y, sep_point])
            PZI_US_list.append([filename_num, X, Y, PZI_US])
            Es_LS_fc_list.append([filename_num, X, Y, Es_LS_fc])
            Es_LS_ema_list.append([filename_num, X, Y, Es_LS_ema])
            Es_US_fc_list.append([filename_num, X, Y, Es_US_fc])
            Es_US_ema_list.append([filename_num, X, Y, Es_US_ema])
            interf_en_LS_list.append([filename_num, X, Y, interf_en_LS])
            interf_en_US_list.append([filename_num, X, Y, interf_en_US])

        #Generate the 3D maps
        Generate3DPlot(Es_LS_fc_list, folder_path_fitting_3Dplots, filename_elements[0], "Es (Pa)", "Es LS_fc")
        Generate3DPlot(Es_LS_ema_list, folder_path_fitting_3Dplots, filename_elements[0], "Es (Pa)", "Es LS_ema")
        Generate3DPlot(Es_US_fc_list, folder_path_fitting_3Dplots, filename_elements[0], "Es (Pa)", "Es US_fc")
        Generate3DPlot(Es_US_ema_list, folder_path_fitting_3Dplots, filename_elements[0], "Es (Pa)", "Es US_ema")
        Generate3DPlot(attr_force_list, folder_path_fitting_3Dplots, filename_elements[0], "Attractive Force (uN)", "AttrForce")
        Generate3DPlot(adhes_force_list, folder_path_fitting_3Dplots, filename_elements[0], "Adhesive Force (uN)", "AdhesForce")
        Generate3DPlot(sep_point_list, folder_path_fitting_3Dplots, filename_elements[0], "Separation point (um)", "SepPoint")
        Generate3DPlot(interf_en_LS_list, folder_path_fitting_3Dplots, filename_elements[0], "Interfacial Energy (mJ m^-2)", "Interf E LS")
        Generate3DPlot(interf_en_US_list, folder_path_fitting_3Dplots, filename_elements[0], "Interfacial Energy (mJ m^-2)", "Interf E US")
        Generate3DPlot(PZI_LS_list, folder_path_fitting_3Dplots, filename_elements[0], "PZI (um)", "PZI LS")
        Generate3DPlot(PZI_US_list, folder_path_fitting_3Dplots, filename_elements[0], "PZI (um)", "PZI US")

        Spacer()
        Spacer()
        if len(warning_list) == 0:
            txt.insert(END, "No warnings to report.\n")
            txt.update()
            txt.see("end")
        else:
            for warning in warning_list:
                txt.insert(END, warning)
                txt.update()
                txt.see("end")
        if len(error_list) == 0:
            txt.insert(END, "No errors to report.\n")
            txt.update()
            txt.see("end")
        else:
            for error in error_list:
                txt.insert(END, error)
                txt.update()
                txt.see("end")
        Spacer()
        Spacer()
        txt.insert(END, "GENERATION OF 3D MAPS WITH JKR MODEL DONE.\n".format(year_now))
        Spacer()
        Spacer()
        txt.update()
        txt.see("end")

    #Save report in dialogue window as txt file inside folder with 3D plots
    file = open(f"{folder_path_fitting_3Dplots}/{filename_noext}_3D plotting report.txt", mode='w')
    report = txt.get(1.0, END)
    file.write(report)
    file.close()

def ThresholdMap_b6():

    txt.delete(1.0, END) #Clear dialog window

    global path_foldername

    #PARAMETERS DEFINITION
    ratio_BL_points = float(e3.get())
    thres_F = float(e7.get())

    # Ask to open files and get folder name and path
    path_filenames = filedialog.askopenfilenames(initialdir="/Users/pierangelogobbo/Dropbox/@Documents/@My Research/Lab useful docs/FemtoTools Nanoindenter/Indentation_Curve_Analysis/Input files", title="Select all files to process", filetypes=[("CSV files", "*.csv")])
    if path_filenames == "": #to handle Cancel button
        return

    working_dir_path, working_dir_filename = os.path.split(path_filenames[0])
    txt.insert(END, "Working directory:\n{}\n\n".format(working_dir_path))
    txt.update()
    txt.see("end")
    Spacer()

    txt.insert(END, "Number of selected files: {}\n".format(len(path_filenames)))
    txt.update()
    txt.see("end")

    # Generate list of files to be analysed
    file_list = []
    for i, path_filename in enumerate(path_filenames):
        if ("curve" not in path_filename) and ("results" not in path_filename):
            file_list.append(path_filename)
    txt.insert(END, "Number of files to analyse: {}\n\n".format(len(file_list)))

    # Create a folder for the fitting results
    filename_noext, extension = os.path.splitext(working_dir_filename)
    filename_elements = filename_noext.split("_")
    folder_path_threshold_map = working_dir_path + f"/{filename_elements[0]}_Threshold Map"
    Create_Output_Folder(folder_path_threshold_map)

    Spacer()
    Spacer()
    txt.insert(END, "GENERATION OF THRESHOLD MAP....\n".format(year_now))
    Spacer()
    Spacer()
    txt.update()
    txt.see("end")

    # Generate all the lists required for the 3D map
    displacement_LS_list = []
    displacement_US_list = []
    error_list = []
    for i, path_filename in enumerate(file_list):
        try:
            # Get all paths and filenames and memorise them, then show all filenames found.
            path, filename = os.path.split(path_filename)  # splits path and file name
            txt.insert(END, "{}. Analysisng {}\n\n".format(i + 1, filename))
            txt.update()
            txt.see("end")
            #Get the experiemnt number from the file name to order the list later and allow 3D plotting
            filename_noext, extension = os.path.splitext(filename)
            filename_elements = filename_noext.split("_")
            filename_num = int(filename_elements[2])

            X, Y, displacement_LS, displacement_US = Find_Diplacement_For_Threshold_Map(path_filename, ratio_BL_points, thres_F)
        except:
            # Open the dataframe
            df = pandas.read_table("{}".format(path_filename), sep=',', header=0)
            df = df.astype(float)  # Change data from object to float
            PosZ_um_col = df.columns.get_loc("Pos Z [um]")

            # Understand if the measurement was carried out using the piezo-scanner or teh stick-slip actuator
            delta_stepmotor = df.iloc[2, PosZ_um_col] - df.iloc[3, PosZ_um_col]
            if delta_stepmotor < 0.005:  # This value was optimised based on the dataset available so far
                Zmov_type = "Piezo scanner"
                txt.insert(END, "Measurement carried out using piezo scanner.\n\n")
            else:
                Zmov_type = "Stick-slip actuator"
                txt.insert(END, "Measurement carried out using stick-slip actuator.\n\n")

            if Zmov_type == "Piezo scanner":
                PiezoX_um_col = df.columns.get_loc("Piezo X [um]")  # Get column number
                PiezoY_um_col = df.columns.get_loc("Piezo Y [um]")  # Get column number
                X = df.iloc[0, PiezoX_um_col]
                Y = df.iloc[0, PiezoY_um_col]

            if Zmov_type == "Stick-slip actuator":
                PosX_um_col = df.columns.get_loc("Pos X [um]")  # Get column number
                PosY_um_col = df.columns.get_loc("Pos Y [um]")  # Get column number
                X = df.iloc[0, PosX_um_col]
                Y = df.iloc[0, PosY_um_col]

            displacement_LS, displacement_US = 1, 1
            txt.insert(END, f"Error in file {filename}. Z axis value set to 1.\n\n")
            txt.update()
            txt.see("end")
            error_list.append(f"Error in file {filename}.\n")
            pass
        displacement_LS_list.append([filename_num, X, Y, displacement_LS])
        displacement_US_list.append([filename_num, X, Y, displacement_US])

    #Generate the 3D maps
    Generate3DPlot(displacement_LS_list, folder_path_threshold_map, filename_elements[0], "Displacement (um)", "threshold map LS")
    Generate3DPlot(displacement_US_list, folder_path_threshold_map, filename_elements[0], "Displacement (um)", "threshold map US")

    Spacer()
    Spacer()
    if len(error_list) == 0:
        txt.insert(END, "No errors to report.\n")
        txt.update()
        txt.see("end")
    else:
        for error in error_list:
            txt.insert(END, error)
            txt.update()
            txt.see("end")
    Spacer()
    Spacer()
    txt.insert(END, "GENERATION OF THRESHOLD MAP DONE.\n".format(year_now))
    Spacer()
    Spacer()
    txt.update()
    txt.see("end")

def SaveReport_b7():
    filename = asksaveasfilename(defaultextension=".txt", filetypes = [("Text files", "*.txt")], title = "Save report as *.txt file", initialfile = "Report_")

    try:
        file = open (filename, mode = 'w')
        report = txt.get(1.0, END)
    except:
        return #to handle Cancel button

    file.write(report)
    file.close()

def ClearDW_b8():
    txt.delete(1.0, END)

l1 = Label(root, text="A.L.I.A.S.: A Lovely Indentation Analysis System ", font='Helvetica 24 bold', fg = "SteelBlue4").grid(row = 0, column = 0, sticky = W, padx = 375, pady = 2)


#PARAMETERS DEFINITION
l2 = Label(root, text="Extract and reorganise original data:", font='Helvetica 18 bold', fg = "SteelBlue4").grid(row = 1, column = 0, sticky = W, padx = 5, pady = 2)
b1 = Button(root, text="Exrtact files", command=File_Extractor_b1).grid(row = 2, column = 0, sticky = W, padx = 5, pady = 2)

l3 = Label(root, text="Experimental and fitting parameters:", font='Helvetica 18 bold', fg = "SteelBlue4").grid(row = 3, column = 0, sticky = W, padx = 5, pady = 2)
l4 = Label(root, text="Probe's diameter (um):").grid(row = 4, column = 0, sticky = W, padx = 5, pady = 2)
e1 = Entry(root, width=5)
e1.insert(END, "290")
e1.grid(row = 4, column = 0, sticky = W, padx = 350, pady = 2)

l5 = Label(root, text="Sample's Poisson's ratio:").grid(row = 5, column = 0, sticky = W, padx = 5, pady = 2)
e2 = Entry(root, width=5)
e2.insert(END, "0.5") # Poisson's ratio of sample (v_PDMS 0.45-0.5; v_polystyrene 0.34)
e2.grid(row = 5, column = 0, sticky = W, padx = 350, pady = 2)

l6 = Label(root, text="Baseline data points (default: first 1/8 of datapoints):").grid(row = 6, column = 0, sticky = W, padx = 5, pady = 2)
e3 = Entry(root, width=5)
e3.insert(END, "0.125") #Number of points for baseline is 1/8 of datapoints of df
e3.grid(row = 6, column = 0, sticky = W, padx = 350, pady = 2)

l7 = Label(root, text="Contact point threshold constant (*st.dev.):").grid(row = 7, column = 0, sticky = W, padx = 5, pady = 2)
e4 = Entry(root, width=5)
e4.insert(END, "30") #Number of points for baseline is 1/8 of datapoints of df
e4.grid(row = 7, column = 0, sticky = W, padx = 350, pady = 2)

l8 = Label(root, text="Loading scan - Datapoints per fitting segment:").grid(row = 8, column = 0, sticky = W, padx = 5, pady = 2)
e5 = Entry(root, width=5)
e5.insert(END, "5") #Fitting segment
e5.grid(row = 8, column = 0, sticky = W, padx = 350, pady = 2)

l9 = Label(root, text="Unloading scan - Datapoints per fitting segment:").grid(row = 9, column = 0, sticky = W, padx = 5, pady = 2)
e6 = Entry(root, width=5)
e6.insert(END, "5") #Fitting segment
e6.grid(row = 9, column = 0, sticky = W, padx = 350, pady = 2)




l10 = Label(root, text="Analysis of individual indentation curves:", font='Helvetica 18 bold', fg = "SteelBlue4").grid(row = 11, column = 0, sticky = W, padx = 5, pady = 2)
b2 = Button(root, text="Find curves", command=Gen_Single_Exp_b2).grid(row = 12, column = 0, sticky = W, padx = 50, pady = 2)
b3 = Button(root, text="Txt to Csv and plot curves", command=Divide_And_Plot_All_Exp_b3).grid(row = 12, column = 0, sticky = W, padx = 210, pady = 2)

l11 = Label(root, text="Select fitting model:").grid(row = 13, column = 0, sticky = W, padx = 5, pady = 2)
options = ["Hertz","DMT", "PT", "JKR"]
clicked = StringVar()
clicked.set(options[3])
dm1 = OptionMenu(root, clicked, *options).grid(row = 13, column = 0, sticky = W, padx = 150, pady = 2)
b4 = Button(root, text="Fit curves!", command=FitCurves).grid(row = 13, column = 0, sticky = W, padx = 290, pady = 2)





l12 = Label(root, text="Analysis of an array of indentation curves:", font='Helvetica 18 bold', fg = "SteelBlue4").grid(row = 16, column = 0, sticky = W, padx = 5, pady = 2)
b5 = Button(root, text="Find arrays", command=Gen_Single_Exp_b2).grid(row = 17, column = 0, sticky = W, padx = 50, pady = 2)
b6 = Button(root, text="Txt to Csv and plot curves", command=Divide_Array_And_Plot_All_Exp_b4).grid(row = 17, column = 0, sticky = W, padx = 210, pady = 2)

l13 = Label(root, text="Select fitting model:").grid(row = 18, column = 0, sticky = W, padx = 5, pady = 2)
options2 = ["Hertz","DMT", "PT", "JKR",]
clicked2 = StringVar()
clicked2.set(options[3])
dm2 = OptionMenu(root, clicked2, *options2).grid(row = 18, column = 0, sticky = W, padx = 150, pady = 2)
b7 = Button(root, text="Generate 3D maps!", command=Generate3DMaps_b5).grid(row = 18, column = 0, sticky = W, padx = 250, pady = 2)

l14 = Label(root, text="Force threshold (uN):").grid(row = 19, column = 0, sticky = W, padx = 5, pady = 2)
e7 = Entry(root, width=5)
e7.insert(END, "10")
e7.grid(row = 19, column = 0, sticky = W, padx = 150, pady = 2)
b8 = Button(root, text="Threshold map!", command=ThresholdMap_b6).grid(row = 19, column = 0, sticky = W, padx = 273, pady = 2)


#Create and write inside a dialog box
l15 = Label(root, text="Dialog window:", font='Helvetica 18 bold', fg = "SteelBlue4").grid(row = 1, column = 0, sticky = W, padx = 450, pady = 2)
txt = scrolledtext.ScrolledText(root, height=37, width=95)
txt.configure(font=("TkDefaultFont", 12, "normal"))
txt.grid(row=2, column = 0, rowspan = 17, sticky=W, padx = 450) #W=allign to left
txt.see("end")


b9 = Button(root, text="Save report as *.txt file", command=SaveReport_b7).grid(row = 19, column = 0, sticky = W, padx = 1040, pady = 10)
b10 = Button(root, text="Clear dialog window", command=ClearDW_b8).grid(row = 19, column = 0, sticky = W, padx = 870, pady = 10)



root.mainloop()