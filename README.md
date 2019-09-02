# IRPSolver
A Python file used for solving large scaled IRP. It is used in Eric's dissertation.

# Brief User Guide

0. This guide is not aimed for a commercial user, but for markers and researchers (if there is any). 

1. This is developed in IDLE Python 3.7.4, 32-bit, debugging by print(). 

2. Several packages need to be installed in python first, including: numpy, csv, geopy, subprocess, matplotlib
Some more packages may be required. Please refer to the error message, if there is any. The relevant error type is ‘ModuleNotFoundError’. 

3. Data file of retailers, LKH solver (LKH-2.exe) should be under the same directory as python file. 

4. Data file of retailer should be a CSV file, with account number as first column, latitude as second column, longitude as third, and demand as fourth. Second row should the information of depot, followed by rows of retailer information. 

5. The author fails to find an automatic way to close the pop-up windows of LKH solver. If operating system is Windows 10, right click on the solver’s icon in taskbar, and click ‘Close all windows’. 

6. Close the windows of figure to continue the program, if variable ‘draw_route’ is set to ‘True’. 


Queries are welcomed.  :)
