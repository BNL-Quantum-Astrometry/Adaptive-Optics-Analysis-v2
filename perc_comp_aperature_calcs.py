import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from scipy import special
from matplotlib.cm import ScalarMappable
import seaborn as sns



# DATA PROCESSING


# Take text file cenx, ceny, radius data for each frame and convert into arrays:
# generates a 1D array for each of the 3 data points at every percentage, creates a 2d array 
# for each of these data points where rows are different frames and columns are the different percent captures
def process_text_files(opt_cenx_txt_paths, opt_ceny_txt_paths, opt_radius_txt_paths):
    cenx_array_appended=[]
    ceny_array_appended=[]
    radius_array_appended=[]
    for file_path in opt_cenx_txt_paths:
        with open(file_path, 'r') as file:
            data = file.read()
            cenx_array_1d = np.fromstring(data, sep='\n')
            cenx_array_appended.append(cenx_array_1d)
    for file_path in opt_ceny_txt_paths:
        with open(file_path, 'r') as file:
            data = file.read()
            ceny_array_1d = np.fromstring(data, sep='\n')
            ceny_array_appended.append(ceny_array_1d)
    for file_path in opt_radius_txt_paths:
        with open(file_path, 'r') as file:
            data = file.read()
            radius_array_1d = np.fromstring(data, sep='\n')
            radius_array_appended.append(radius_array_1d)

    cenx_array_2d = np.column_stack(cenx_array_appended)
    ceny_array_2d = np.column_stack(ceny_array_appended)
    radius_array_2d = np.column_stack(radius_array_appended)

    return cenx_array_2d, ceny_array_2d, radius_array_2d


# ANALYSIS


# 2d histograms of chagne in center for change in percent capture (10-25, 25,50, 50-75, 75-90) 
# set up generically so you can call function multiple times for each comparison
# columns are in order of increasing percent, i.e. column_1 = 0 would mean the column of 10 percent values for the given datapoint
# Spherical symmetry
def delta_cen_with_delta_perc_2dhist(cenx_array_2d, ceny_array_2d, column_1, column_2, percent_1, percent_2, output_path, output_file): #delta is (index_2 - index_1)
    cenx_for_comp_1 = cenx_array_2d[:,column_1]
    cenx_for_comp_2 = cenx_array_2d[:,column_2]
    ceny_for_comp_1 = ceny_array_2d[:,column_1]
    ceny_for_comp_2 = ceny_array_2d[:,column_2]

    delta_x = cenx_for_comp_2 - cenx_for_comp_1
    delta_y = ceny_for_comp_2 - ceny_for_comp_1

    plt.figure(figsize=(10, 6))
    plt.hist2d(delta_x, delta_y, bins=[np.arange(np.min(delta_x),np.max(delta_x),.25),np.arange(np.min(delta_y),np.max(delta_y), .25)], cmap='gist_heat_r',  norm = mcolors.Normalize(vmin=0, vmax=6))
    plt.colorbar(label='Frequency')
    plt.title(f'ΔCenter ({percent_1}% -> {percent_2}%)')
    plt.xlabel('ΔX (pixels)')
    plt.ylabel('ΔY (pixels)')
    plt.grid(True)

    std_dev_x = np.std(delta_x)
    std_dev_y = np.std(delta_y)
    range_x = np.ptp(delta_x)
    range_y = np.ptp(delta_y)
    avg_x = np.mean(delta_x)
    avg_y = np.mean(delta_y)
    
    plt.text(0.05, 0.95, f'ΔX Std Dev: {std_dev_x:.2f}, ΔY Std Dev: {std_dev_y:.2f}\nΔX Range: {range_x:.2f}, ΔY Range: {range_y:.2f}\n Avg Δ: ({avg_x:.1f},{avg_y:.1f})\nBin Size = (0.25,0.25)',
             transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    plt.savefig(os.path.join(output_path, output_file), format='jpg', dpi=300)
    plt.close()

# Plot all radii for every frame at every percentage in a scatter plot then fit a function
# Central Concentration
def roc_of_radii_with_percent(radius_array_2d, start_column, end_column, output_path, output_file):
    percs = [5,10,15,25,35,50,60,75,90]
    x = np.float64(percs)

    def exponential_func(x, a, b):
        return a * np.exp(b * x)

    popt_array = np.zeros((radius_array_2d.shape[0], 2))    
    for i in range(radius_array_2d.shape[0]):
        y = radius_array_2d[i, start_column:end_column]
        try:
            # Fit exponential
            popt, _ = curve_fit(exponential_func, x, y)
            # Store the best-fit parameters
            popt_array[i] = popt[0]
            # Plot the scatter
            plt.scatter(x, y, alpha=0.25)
        except ValueError as e:
            print(f"Skipping row {i+1} due to mismatched shapes: {e}")

    # Calculate avg y vals for each x
    y_avg = np.mean(radius_array_2d[:, start_column:end_column], axis=0)
    # Fit exponential to average data
    popt_avg = curve_fit(exponential_func, x, y_avg, p0=[1,.06])[0]

    # Extend fitted curve
    x_extended = np.linspace(x.min()-5, x.max()+10, 1000)
    y_extended = exponential_func(x_extended, *popt_avg)
    plt.plot(x_extended, y_extended, color='red')

    equation_avg = f'$f(x) = {popt_avg[0]:.5f}  e^{{{popt_avg[1]:.2f}  x}}$'
    derivative_equation_avg = f'$\\frac{{df(x)}}{{dx}} = {popt_avg[0]*popt_avg[1]:.5f}  e^{{{popt_avg[1]:.2f}  x}}$'
    plt.text(0.05, 0.9, equation_avg, transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.05, 0.825, derivative_equation_avg, transform=plt.gca().transAxes, fontsize=10)

    plt.xticks([0,20,40,60,80,100])
    plt.yticks(np.arange(0,220,10))
    plt.xlabel('Percent Capture (%)')
    plt.ylabel('Radius (Pixels)')

    plt.yscale('log')
    plt.title('Aperture Radius as a Functon of % Capture')
    plt.savefig(os.path.join(output_path, output_file), format='jpg', dpi=300)
    plt.close()



## SUBPLOTS



def delta_cen_with_delta_perc_2dhist_subplots(cenx_array_2d, ceny_array_2d, columns, percents, output_path, output_file):
    global_delta_x_min, global_delta_x_max, global_delta_y_min, global_delta_y_max = float('inf'), float('-inf'), float('inf'), float('-inf')

    for column_pair in columns:
        column_1, column_2 = column_pair
        delta_x = cenx_array_2d[:, column_2] - cenx_array_2d[:, column_1]
        delta_y = ceny_array_2d[:, column_2] - ceny_array_2d[:, column_1]

        global_delta_x_min = min(global_delta_x_min, np.min(delta_x))
        global_delta_x_max = max(global_delta_x_max, np.max(delta_x))
        global_delta_y_min = min(global_delta_y_min, np.min(delta_y))
        global_delta_y_max = max(global_delta_y_max, np.max(delta_y))

    # Consistent bins
    bins_x = np.arange(global_delta_x_min, global_delta_x_max, .5)
    bins_y = np.arange(global_delta_y_min, global_delta_y_max, .5)

    fig, axs = plt.subplots(2, 4, figsize=(20, 12), sharex=True, sharey=True, constrained_layout=True)
    axs = axs.flatten()  

    norm = mcolors.Normalize(vmin=0, vmax=6)
    cmap = plt.get_cmap('gist_heat_r') 

    for i, ax in enumerate(axs):
        column_1, column_2 = columns[i]
        percent_1, percent_2 = percents[i]
        
        delta_x = cenx_array_2d[:, column_2] - cenx_array_2d[:, column_1]
        delta_y = ceny_array_2d[:, column_2] - ceny_array_2d[:, column_1]

        ax.hist2d(delta_x, delta_y, bins=[bins_x, bins_y], cmap='gist_heat_r', norm=mcolors.Normalize(vmin=0, vmax=6))
        if  0<=i<=3:
            ax.set_title(f'ΔCenter ({percent_1}% -> {percent_2}%)')
        ax.grid(True)

        std_dev_x = np.std(delta_x)
        std_dev_y = np.std(delta_y)
        avg_x = np.mean(delta_x)
        avg_y = np.mean(delta_y)

        ax.text(0.05, 0.95, f'ΔX Std Dev: {std_dev_x:.2f}, ΔY Std Dev: {std_dev_y:.2f}\n Avg Δ: ({avg_x:.1f},{avg_y:.1f})',
                transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))

    # Shared labels
    fig.text(0.03, 0.01, 'ΔX (pixels)', ha='center', va='center')
    fig.text(0.015, 0.035, 'ΔY (pixels)', ha='center', va='center', rotation='vertical')

    fig.text(0.00, 0.75, 'Collimator Camera', va='center', rotation='vertical', fontsize=12)
    fig.text(0.00, 0.25, 'Guide Camera (Control)', va='center', rotation='vertical', fontsize=12)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Frequency')

    plt.savefig(os.path.join(output_path, output_file), format='jpg', dpi=300)
    plt.close()


def roc_of_radii_with_percent_subplots_lin(radius_array_2d, start_column_1, end_column_1, start_column_2, end_column_2, output_path, output_file):
    percs = [5, 10, 15, 25, 35, 50, 60, 75, 90]
    x = np.float64(percs)
    
    def exponential_func(x, a, b):
        return a * np.exp(b * x)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True) 

    subplot_titles = ['Guide Camera (Control)', 'Collimator Camera']


    # Subplotting
    for ax, start_column, end_column, title in zip(axs, [start_column_2, start_column_1], [end_column_2, end_column_1], subplot_titles):
        popt_array = np.zeros((radius_array_2d.shape[0], 2))
        for i in range(radius_array_2d.shape[0]):
            y = radius_array_2d[i, start_column:end_column]
            try:
                popt, _ = curve_fit(exponential_func, x, y)
                popt_array[i] = popt
                ax.scatter(x, y, alpha=0.25)
            except ValueError as e:
                print(f"Skipping row {i+1} due to mismatched shapes: {e}")
        
        y_avg = np.mean(radius_array_2d[:, start_column:end_column], axis=0)
        popt_avg, _ = curve_fit(exponential_func, x, y_avg, p0=[1, 0.06], maxfev=1000000)
        x_extended = np.linspace(x.min()-5, x.max()+10, 1000)
        y_extended = exponential_func(x_extended, *popt_avg)
        ax.plot(x_extended, y_extended, color='red')
        equation_avg = f'$f(x) = {popt_avg[0]:.5f} e^{{{popt_avg[1]:.2f} x}}$'
        derivative_equation_avg = f'$\\frac{{df(x)}}{{dx}} = {popt_avg[0]*popt_avg[1]:.5f} e^{{{popt_avg[1]:.2f} x}}$'
        ax.text(0.05, 0.9, equation_avg, transform=ax.transAxes, fontsize=10)
        ax.text(0.05, 0.825, derivative_equation_avg, transform=ax.transAxes, fontsize=10)

        ax.set_title(title)

    # Shared axis labels
    fig.supxlabel('Percent Capture (%)')
    fig.supylabel('Radius (Pixels)')
    
    # Shared title
    fig.suptitle('Aperture Radius as a Function of % Capture', fontsize=16)

    plt.savefig(os.path.join(output_path, output_file), format='jpg', dpi=300)
    plt.close()


def roc_of_radii_with_percent_subplots_log(radius_array_2d, start_column_1, end_column_1, start_column_2, end_column_2, output_path, output_file):
    blue_color = (0.121, 0.466, 0.705)
    red_color = (0.839, 0.153, 0.157)
    percs = [5, 10, 15, 25, 35, 50, 60, 75, 90]
    x = np.float64(percs)
    

    def exponential_func(x, a, b):
        return a * np.exp(b * x)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True, constrained_layout=True) 
    subplot_titles = ['Control', 'AO Corrected']

    # Subplotting
    for ax, start_column, end_column, title in zip(axs, [start_column_2, start_column_1], [end_column_2, end_column_1], subplot_titles):
        data_to_plot = [radius_array_2d[:, i] for i in range(start_column, end_column)]
        
        # Violin plots
        parts = ax.violinplot(data_to_plot, positions=x, widths=4)
        for pc in parts['bodies']:
            pc.set_facecolor(blue_color) 

        y_avg = np.mean(radius_array_2d[:, start_column:end_column], axis=0)
        try:
            popt_avg, _ = curve_fit(exponential_func, x, y_avg, p0=[1, 0.06], maxfev=1000000)
            x_extended = np.linspace(x.min()-10, x.max()+10, 1000)
            y_extended = exponential_func(x_extended, *popt_avg)
            ax.plot(x_extended, y_extended, color=red_color, alpha=.5)
        except Exception as e:
            print(f"Curve fitting error: {e}")
        ax.set_title(title, fontsize=20)
        ax.set_xticks([0,10,20,30,40,50,60,70,80,90,100])
        ax.set_xlim(0,100)
        ax.tick_params(axis='both', which='major', labelsize=15, length=10, width=2)
        ax.tick_params(axis='y', which='minor', length=8, width=2)
        equation_avg = f'$f(x) = {popt_avg[0]:.5f} e^{{{popt_avg[1]:.2f} x}}$'
        ax.text(0.05, 0.825, equation_avg, transform=ax.transAxes, fontsize=15)


    ax.set_yscale('log')


    plt.savefig(os.path.join(output_path, output_file), format='svg', dpi=750, bbox_inches='tight')



## PERC COMP JOINT PLOTS ##
    


def delta_cen_with_delta_perc_jointplot(cenx_array_2d, ceny_array_2d, gc_column_i, gc_column_f, cc_column_i, cc_column_f, percent_1, percent_2, output_path, output_file):
    def compute_deltas(cenx_array_2d, ceny_array_2d, column_i, column_f):
        delta_x = cenx_array_2d[:, column_f] - cenx_array_2d[:, column_i]
        delta_y = ceny_array_2d[:, column_f] - ceny_array_2d[:, column_i]
        return delta_x, delta_y

    # Calc deltas
    gc_delta_x, gc_delta_y = compute_deltas(cenx_array_2d, ceny_array_2d, gc_column_i, gc_column_f)
    cc_delta_x, cc_delta_y = compute_deltas(cenx_array_2d, ceny_array_2d, cc_column_i, cc_column_f)

    gc_delta_x_abs, gc_delta_y_abs = np.absolute(gc_delta_x), np.absolute(gc_delta_y)
    cc_delta_x_abs, cc_delta_y_abs = np.absolute(cc_delta_x), np.absolute(cc_delta_y)

    # DataFrame for each delta
    df_gc = pd.DataFrame({'Delta X': gc_delta_x_abs, 'Delta Y': gc_delta_y_abs, 'Dataset': 'Control'})
    df_cc = pd.DataFrame({'Delta X': cc_delta_x_abs, 'Delta Y': cc_delta_y_abs, 'Dataset': 'AO Corrected'})

    combined_df = pd.concat([df_gc, df_cc])

    g = sns.JointGrid(data=combined_df, x='Delta X', y='Delta Y', space=0)

    blue_color = (0.121, 0.466, 0.705)
    red_color = (0.839, 0.153, 0.157)

    bin_width_x = 0.5  
    bin_width_y = 0.5

    bins_x = np.arange(start=combined_df['Delta X'].min(), stop=combined_df['Delta X'].max() + bin_width_x, step=bin_width_x)
    bins_y = np.arange(start=combined_df['Delta Y'].min(), stop=combined_df['Delta Y'].max() + bin_width_y, step=bin_width_y)

    sns.scatterplot(data=combined_df, x='Delta X', y='Delta Y', hue='Dataset', ax=g.ax_joint, palette={'Control': blue_color, 'AO Corrected': red_color}, legend=False)

    sns.histplot(df_gc['Delta X'], kde=False, bins=bins_x, ax=g.ax_marg_x, alpha=0.75, element="step", fill=False, color=blue_color, label='Control')
    sns.histplot(df_cc['Delta X'], kde=False, bins=bins_x, ax=g.ax_marg_x, alpha=0.75, element="step", fill=False, color=red_color, label='AO Corrected')
    sns.histplot(df_gc, y='Delta Y', kde=False, bins=bins_y, ax=g.ax_marg_y, alpha=0.75, element="step", fill=False, color=blue_color)
    sns.histplot(df_cc, y='Delta Y', kde=False, bins=bins_y, ax=g.ax_marg_y, alpha=0.75, element="step", fill=False, color=red_color)
    g.ax_marg_x.legend(loc='upper right', bbox_to_anchor=(1, 0), fontsize=25)

    max_range = max(combined_df['Delta X'].max(), combined_df['Delta Y'].max())

    g.ax_joint.set_xlabel('ΔX (pixels)', fontsize=25) 
    g.ax_joint.set_ylabel('ΔY (pixels)', fontsize=25)  

    g.figure.suptitle(f'ΔCenter: {percent_1}% -> {percent_2}% Capture', fontsize=30, y=1.01)
    g.figure.set_size_inches(10,10)
    g.ax_joint.set_xlim(0, max_range)
    g.ax_joint.set_ylim(0, max_range)
    g.ax_joint.set_aspect('equal')

    plt.savefig(os.path.join(output_path, output_file), format='svg', dpi=750, bbox_inches='tight')
    plt.close()

## FRAME COMP JOINT PLOTS
    
def single_perc_cen_jointplot(cenx_array_2d, ceny_array_2d, gc_column, cc_column, percent, output_path, output_file):
    def compute_centered_centers(cenx_array_2d, ceny_array_2d, column):
        center_x = cenx_array_2d[:, column] - np.mean(cenx_array_2d[:, column])
        center_y = ceny_array_2d[:, column] - np.mean(ceny_array_2d[:, column])
        return center_x, center_y

    gc_center_x, gc_center_y = compute_centered_centers(cenx_array_2d, ceny_array_2d, gc_column)
    cc_center_x, cc_center_y = compute_centered_centers(cenx_array_2d, ceny_array_2d, cc_column)

    df_gc = pd.DataFrame({'Centered X': gc_center_x, 'Centered Y': gc_center_y, 'Dataset': 'Control'})
    df_cc = pd.DataFrame({'Centered X': cc_center_x, 'Centered Y': cc_center_y, 'Dataset': 'AO Corrected'})

    combined_df = pd.concat([df_gc, df_cc])

    g = sns.JointGrid(data=combined_df, x='Centered X', y='Centered Y', space=0)

    # Contours
    def plot_contours(data, color, ax):
        x = data['Centered X']
        y = data['Centered Y']
        kde = gaussian_kde([x, y])
        # Define the grid
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        zz = kde(np.vstack([xx.flatten(), yy.flatten()]))
    
        
        ax.contour(xx, yy, zz.reshape(xx.shape), levels=10, cmap=color, alpha=0.8)

    plot_contours(df_gc, 'Blues', g.ax_joint)
    plot_contours(df_cc, 'Reds', g.ax_joint)

    bin_width_x = 0.5
    bin_width_y = 0.5
    bins_x = np.arange(combined_df['Centered X'].min(), combined_df['Centered X'].max() + bin_width_x, step=bin_width_x)
    bins_y = np.arange(combined_df['Centered Y'].min(), combined_df['Centered Y'].max() + bin_width_y, step=bin_width_y)

    blue_color = (0.121, 0.466, 0.705)
    red_color = (0.839, 0.153, 0.157)
    sns.histplot(data=df_gc, x='Centered X', bins=bins_x, ax=g.ax_marg_x, element="step", fill=False, alpha=.75, color=blue_color, label='Control')
    sns.histplot(data=df_cc, x='Centered X', bins=bins_x, ax=g.ax_marg_x, element="step", fill=False, alpha=.75, color=red_color, label='AO Corrected')
    sns.histplot(data=df_gc, y='Centered Y', bins=bins_y, ax=g.ax_marg_y, element="step", fill=False, alpha=.75, color=blue_color)
    sns.histplot(data=df_cc, y='Centered Y', bins=bins_y, ax=g.ax_marg_y, element="step", fill=False, alpha=.75, color=red_color)
    g.ax_marg_x.legend(loc='upper right', bbox_to_anchor=(1, 0), fontsize=25)

    max_range = max(combined_df['Centered X'].max(), combined_df['Centered Y'].max())

    g.ax_joint.set_xlabel('X Relative to Mean (Pixels)', fontsize=25)
    g.ax_joint.set_ylabel('Y Relative to Mean (Pixels)', fontsize=25)
    g.figure.suptitle(f'Optimal Centers for {percent}% Capture', fontsize=30, y=1.01)
    g.figure.set_size_inches(10, 10)
    g.ax_joint.set_xlim(-max_range, max_range)
    g.ax_joint.set_ylim(-max_range, max_range)
    g.ax_joint.set_aspect('equal')

    plt.savefig(os.path.join(output_path, output_file), format='svg', dpi=750, bbox_inches='tight')
    plt.close()



def plt_opt_cen_hist_ind(cenx_array_2d, ceny_array_2d, column, type, target_percentage, output_path):
    filename = f"{target_percentage}perc_{type}_opt_cen_hist_prog.svg"
    x_centers = cenx_array_2d[:, column]  - np.mean(cenx_array_2d[:, column])
    y_centers = ceny_array_2d[:, column]  - np.mean(ceny_array_2d[:,column])

    bin_range_x = np.max(np.abs(y_centers))
    bin_range_y = np.max(np.abs(y_centers))
    bins_x = np.arange(-bin_range_x, bin_range_x + 0.5, 0.5)
    bins_y = np.arange(-bin_range_y, bin_range_y + 0.5, 0.5)

    plt.figure(figsize=(10, 5))
    plt.hist2d(x_centers, y_centers, bins=[bins_x, bins_y], cmap='gist_heat_r', norm=mcolors.Normalize(vmin=0, vmax=8))
    plt.colorbar(label='Frequency')
    plt.title(f'{type} Control Optimal Centers for {target_percentage}% Capture', fontsize=20)
    plt.xlabel('X (Pixel)', fontsize=20)
    plt.ylabel('Y (Pixel)', fontsize=20)
    plt.grid(True)
    plt.axhline(0, color='gray', linestyle='--') 
    plt.axvline(0, color='gray', linestyle='--')  
    plt.tick_params(axis='both', which='major', labelsize=15, length=6, width=2, ) 

    std_dev_x = np.std(x_centers)
    std_dev_y = np.std(y_centers)
    range_x = np.ptp(x_centers)
    range_y = np.ptp(y_centers)

    plt.text(0.05, 0.95, f'X Std Dev: {std_dev_x:.2f}, Y Std Dev: {std_dev_y:.2f}\nX Range: {range_x:.2f}, Y Range: {range_y:.2f}',
             transform=plt.gca().transAxes, 
             fontsize=15, verticalalignment='top', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    plt.gca().set_aspect('equal')
    plt.savefig(os.path.join(output_path, filename), format='svg', dpi=750)




def main(opt_cenx_txt_paths, opt_ceny_txt_paths, opt_radius_txt_paths):
    cenx_array_2d, ceny_array_2d, radius_array_2d = process_text_files(opt_cenx_txt_paths, opt_ceny_txt_paths, opt_radius_txt_paths)
    columns = [(0,1), (1,2), (2,3), (3,4), (5,6), (6,7), (7,8), (8,9)]
    percents = [(10,25), (25,50), (50,75), (75,90), (10,25), (25,50), (50,75), (75,90)]
    delta_cen_with_delta_perc_2dhist(cenx_array_2d,ceny_array_2d, 0, 1, 10, 25, output_path, 'cc_delta_center_10_25.jpg')
    delta_cen_with_delta_perc_2dhist(cenx_array_2d,ceny_array_2d, 1, 2, 25, 50, output_path, 'cc_delta_center_25_50.jpg')
    delta_cen_with_delta_perc_2dhist(cenx_array_2d,ceny_array_2d, 2, 3, 50, 75, output_path, 'cc_delta_center_50_75.jpg')
    delta_cen_with_delta_perc_2dhist(cenx_array_2d,ceny_array_2d, 3, 4, 75, 90, output_path, 'cc_delta_center_75_90.jpg')
    delta_cen_with_delta_perc_2dhist(cenx_array_2d,ceny_array_2d, 5, 6, 10, 25, output_path, 'gc_delta_center_10_25.jpg')
    delta_cen_with_delta_perc_2dhist(cenx_array_2d,ceny_array_2d, 6, 7, 25, 50, output_path, 'gc_delta_center_25_50.jpg')
    delta_cen_with_delta_perc_2dhist(cenx_array_2d,ceny_array_2d, 7, 8, 50, 75, output_path, 'gc_delta_center_50_75.jpg')
    delta_cen_with_delta_perc_2dhist(cenx_array_2d,ceny_array_2d, 8, 9, 75, 90, output_path, 'gc_delta_center_75_90.jpg')
    delta_cen_with_delta_perc_2dhist_subplots(cenx_array_2d, ceny_array_2d, columns, percents, output_path, 'delta_cen_all.jpg')

    roc_of_radii_with_percent(radius_array_2d, 0, 9, output_path, 'cc_roc_radii_func_fit.jpg')
    roc_of_radii_with_percent(radius_array_2d, 9, 18, output_path, 'gc_roc_radii_func_fit.jpg')

    delta_cen_with_delta_perc_2dhist_subplots(cenx_array_2d, ceny_array_2d, columns, percents, output_path, 'delta_cen_all.jpg')
    roc_of_radii_with_percent_subplots_lin(radius_array_2d, 0, 9, 9, 18, output_path, 'lin_roc_radii_func_fit_all.jpg')
    roc_of_radii_with_percent_subplots_log(radius_array_2d, 0, 9, 9, 18, output_path, 'log_roc_radii_func_fit_all.svg')

    delta_cen_with_delta_perc_jointplot(cenx_array_2d, ceny_array_2d, 5, 6, 0, 1, 10, 25, output_path, 'delta_cen_gc_cc_10-25.svg')
    delta_cen_with_delta_perc_jointplot(cenx_array_2d, ceny_array_2d, 6, 7, 1, 2, 25, 50, output_path, 'delta_cen_gc_cc_25-50.svg')
    delta_cen_with_delta_perc_jointplot(cenx_array_2d, ceny_array_2d, 7, 8, 2, 3, 50, 75, output_path, 'delta_cen_gc_cc_50-75.svg')
    delta_cen_with_delta_perc_jointplot(cenx_array_2d, ceny_array_2d, 8, 9, 3, 4, 75, 90, output_path, 'delta_cen_gc_cc_75-90.svg')

    single_perc_cen_jointplot(cenx_array_2d, ceny_array_2d, 5, 0, 10, output_path, '10perc_cen_jointplot.svg')
    single_perc_cen_jointplot(cenx_array_2d, ceny_array_2d, 6, 1, 25, output_path, '25perc_cen_jointplot.svg')
    single_perc_cen_jointplot(cenx_array_2d, ceny_array_2d, 7, 2, 50, output_path, '50perc_cen_jointplot.svg')
    single_perc_cen_jointplot(cenx_array_2d, ceny_array_2d, 8, 3, 75, output_path, '75perc_cen_jointplot.svg')
    single_perc_cen_jointplot(cenx_array_2d, ceny_array_2d, 9, 4, 90, output_path, '90perc_cen_jointplot.svg')



# cc are index 0-4, gc are index 5-9
if __name__ == "__main__":
    output_path = '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Poster_Data/Part_2'
    opt_cenx_txt_paths = ['/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/10perc_LABcc_opt_cenx_list.txt',
                          '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/25perc_LABcc_opt_cenx_list.txt',
                          '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/50perc_LABcc_opt_cenx_list.txt',
                          '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/75perc_LABcc_opt_cenx_list.txt',
                          '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/90perc_LABcc_opt_cenx_list.txt',
                          '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/10perc_LABgc_opt_cenx_list.txt',
                          '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/25perc_LABgc_opt_cenx_list.txt',
                          '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/50perc_LABgc_opt_cenx_list.txt',
                          '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/75perc_LABgc_opt_cenx_list.txt',
                          '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/90perc_LABgc_opt_cenx_list.txt']

    opt_ceny_txt_paths = ['/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/10perc_LABcc_opt_ceny_list.txt',
                          '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/25perc_LABcc_opt_ceny_list.txt',
                          '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/50perc_LABcc_opt_ceny_list.txt',
                          '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/75perc_LABcc_opt_ceny_list.txt',
                          '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/90perc_LABcc_opt_ceny_list.txt',
                          '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/10perc_LABgc_opt_ceny_list.txt',
                          '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/25perc_LABgc_opt_ceny_list.txt',
                          '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/50perc_LABgc_opt_ceny_list.txt',
                          '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/75perc_LABgc_opt_ceny_list.txt',
                          '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/90perc_LABgc_opt_ceny_list.txt']

    opt_radius_txt_paths = ['/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/5perc_LABcc_opt_radii_list.txt',
                            '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/10perc_LABcc_opt_radii_list.txt',
                            '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/15perc_LABcc_opt_radii_list.txt',
                            '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/25perc_LABcc_opt_radii_list.txt',
                            '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/35perc_LABcc_opt_radii_list.txt',
                            '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/50perc_LABcc_opt_radii_list.txt',
                            '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/60perc_LABcc_opt_radii_list.txt',
                            '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/75perc_LABcc_opt_radii_list.txt',
                            '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/90perc_LABcc_opt_radii_list.txt',
                            '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/5perc_LABgc_opt_radii_list.txt',
                            '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/10perc_LABgc_opt_radii_list.txt',
                            '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/15perc_LABgc_opt_radii_list.txt',
                            '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/25perc_LABgc_opt_radii_list.txt',
                            '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/35perc_LABgc_opt_radii_list.txt',
                            '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/50perc_LABgc_opt_radii_list.txt',
                            '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/60perc_LABgc_opt_radii_list.txt',
                            '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/75perc_LABgc_opt_radii_list.txt',
                            '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Aperature_Analysis/Frame_Comp_Aperature_Analysis/90perc_LABgc_opt_radii_list.txt']
    main(opt_cenx_txt_paths, opt_ceny_txt_paths, opt_radius_txt_paths)  