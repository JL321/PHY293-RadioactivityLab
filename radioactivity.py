import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import torch

eps_ratio = 1.19
bi_l = np.log(2)/(19.71*60)
pb_l = np.log(2)/(26.8*60)
# Cited from https://pubmed.ncbi.nlm.nih.gov/1917488/

def find_operating_voltage(file_name=None):
    file_names = []
    counts_pms = [] # Counts per millisecond (plotted against voltage as x axis)
    base_name = "Cs137_"
    for i in range(9):
        file_names.append(base_name+str(600+50*i)+"V.csv")

    data_path = "/home/jamesl/Downloads/RadioactivityData1/"
    plt.title("Scatter plot of counts/20s")
    plt.xlabel("Voltage used")
    plt.ylabel("count/20s")
    for i in range(len(file_names)):
        file_names[i] = data_path+file_names[i]

    for k, path in enumerate(file_names):
        df = pd.read_csv(path)
        list_conversion = df.values.tolist()
        local_countpms = []
        '''
        for i in range(1, len(list_conversion)):
            count_dif = list_conversion[i][1]-list_conversion[i-1][1]
            time_dif = list_conversion[i][0]-list_conversion[i-1][0]
            local_countpms.append(count_dif/time_dif)
        '''
        for i in range(len(list_conversion)):
            local_countpms.append(list_conversion[i][1])
        volt_dict = {str(600+50*k): local_countpms}
        counts_pms.append(volt_dict)
        x_val = 600+50*k
        local_countpms = np.array(local_countpms)
        mean_pms = np.mean(local_countpms)
        plt.scatter(x_val, mean_pms, 10)
        std = np.sqrt(np.sum(np.square(local_countpms-mean_pms)/local_countpms.shape[0]))
        plt.errorbar(x_val, mean_pms, yerr=std, fmt='', color='black')

    plt.plot(marker='o')
    plt.show()

def long_term_products():

    # Find linear regression model for long lived isotopes
    file_path = "/home/jamesl/Downloads/airsample_dust_3hr_newtime.csv"
    df = pd.read_csv(file_path)
    list_conversion = df.values.tolist()
    break_index = 0
    for i in range(len(list_conversion)):
        if list_conversion[i][1] > 1800:
            break_index = i
            print("Break index: {} Length: {}".format(break_index, len(list_conversion)))
            break

    average_bcount = 10.6108108108108
    print("Columns: {}".format(df.columns))
    x = np.array(df['time'].values.tolist()[break_index:])
    x = np.expand_dims(x, 1)
    y = np.array(df['count'].values.tolist()[break_index:])-average_bcount
    y = np.expand_dims(y, 1)
    reg = LinearRegression().fit(x, y)
    m = reg.coef_
    b = reg.intercept_
    m = np.squeeze(m)
    b = np.squeeze(b)
    x = np.squeeze(x)
    y = np.squeeze(y)
    y_std = np.sqrt(np.sum(np.square(y-np.mean(y))/(x.shape[0]-1)))

    print("b: {} m: {}".format(b, m))
    print("Chi squared: {}".format(np.sum(np.square((y-(m*x+b))/y_std))))

    graph_error(x, y, linF, [m, b])
    plt.show()
    graph_residual(x, y, m*x+b)
    plt.show()

def linF(x, m, b):
    return m*x+b

def find_model_params(epochs=5000, batch_size=16):

    # Find optimal model parameters for filtered data
    file_path = "/home/jamesl/Downloads/airsample_dust_3hr_newtime.csv"
    df = pd.read_csv(file_path)
    list_conversion = df.values.tolist()
    break_index = 0
    for i in range(len(list_conversion)):
        if list_conversion[i][1] > 1800:
            break_index = i
            print("Break index: {} Length: {}".format(break_index, len(list_conversion)))
            break
    plot_x = np.array(df['time'].values.tolist())
    plot_y = np.array(df['count'].values.tolist())
    average_bcount = 10.6108108108108
    opt_m = -0.012000306389773823
    opt_b = 112.98577

    x = plot_x
    # We don't subtract average_bcount as the background noise should be accounted for in the linear model
    y = plot_y-(x*opt_m+opt_b)

    x = np.expand_dims(x, 1)
    y = np.expand_dims(y, 1)
    device = torch.device("cuda:0")

    # Code for gradient descent
    dtype = torch.float
    input = torch.from_numpy(x).double().to(device)
    label = torch.from_numpy(y).double().to(device)
    Ainit = torch.tensor(np.random.normal(1), device=device, dtype=torch.double, requires_grad=True)
    R = torch.tensor(np.random.normal(1), device=device, dtype=torch.double, requires_grad=True)
    print("init A: {}".format(Ainit))
    opt = torch.optim.Adam([Ainit, R], lr=1e-3)
    data_idx = 0
    end_index = 0
    access_index = np.arange(x.shape[0])
    for i in range(epochs):
        while (data_idx < x.shape[0]):
            if data_idx+batch_size >= x.shape[0]:
                end_idx = x.shape[0]
            else:
                end_idx = data_idx+batch_size
            x_batch = input[access_index[data_idx:end_idx]]
            y_batch = label[access_index[data_idx:end_idx]]

            t1 = (1+eps_ratio*(bi_l/(bi_l-pb_l)))*torch.exp(-pb_l*x_batch)
            t2 = eps_ratio*(R-(bi_l/(bi_l-pb_l)))*torch.exp(-bi_l*x_batch)
            y_pred = torch.multiply(t1+t2, Ainit)
            loss = torch.mean(torch.square(y_batch-y_pred))
            loss.backward()

            opt.step()
            opt.zero_grad()
            data_idx += batch_size
        data_idx = 0
        access_idx = np.random.permutation(x.shape[0])
        if i%100 == 0:
            print("Epoch {} loss: {}".format(i+1, loss))
    print("Ainit and R: {}, {}".format(Ainit, R))
    Ainit_p = Ainit.cpu().detach().numpy()
    R_p = R.cpu().detach().numpy()
    print("Numpy variants: {}, {}".format(Ainit_p, R_p))
    plt.plot(np.squeeze(x), np.squeeze(model(x, Ainit_p, R_p)), c='r')
    plt.xlabel("Counting duration (seconds)")
    plt.ylabel("Counts (per 20 seconds)")
    plt.legend("Dataset", "Regression plot")
    plt.show()

def model(x, Ainit, R):
    # Note: Optimized values:
    # Ainit = 15.0357, R = 14.1844
    # Ainit = 15.643596747491209, R = 12.690181063858889 (Noise corrected)
    # Full Data (using linear model for correction)
    # Ainit = 3.8064464923782184, R = 13.73394018773772
    t1 = (1 + eps_ratio * (bi_l / (bi_l - pb_l))) * np.exp(-pb_l * x)
    t2 = eps_ratio * (R - (bi_l / (bi_l - pb_l))) * np.exp(-bi_l * x)
    y_pred = (t1 + t2)*Ainit
    return y_pred

def combined_model(x, Ainit, R):
    average_bcount = 10.6108108108108
    opt_m = -0.012000306389773823
    opt_b = 112.98577
    t1 = (1 + eps_ratio * (bi_l / (bi_l - pb_l))) * np.exp(-pb_l * x)
    t2 = eps_ratio * (R - (bi_l / (bi_l - pb_l))) * np.exp(-bi_l * x)
    y_pred = (t1 + t2) * Ainit
    return y_pred+average_bcount+(opt_b+opt_m*x)

def comparison():
    file_path = "/home/jamesl/Downloads/airsample_dust_3hr_newtime.csv"
    df = pd.read_csv(file_path)
    list_conversion = df.values.tolist()

    average_bcount = 10.6108108108108
    opt_m = -0.012000306389773823
    opt_b = 112.98577
    plot_x = np.array(df['time'].values.tolist())
    plot_y = np.array(df['count'].values.tolist())
    plt.scatter(plot_y, combined_model(plot_x, 3.8064464923782184, 13.73394018773772), s=3)

    plot_model = np.expand_dims(combined_model(plot_x, 3.8064464923782184, 13.73394018773772), 1)
    plot_data = np.expand_dims(plot_y, 1)
    reg = LinearRegression(fit_intercept=False).fit(plot_data, plot_model)
    m = np.squeeze(reg.coef_)
    plot_model = np.squeeze(plot_model)
    print("M: {}".format(m))
    plt.plot(plot_y, m*plot_y, label="Line of Best Fit", c='orange')
    var_y = np.sum(np.square(plot_model-np.mean(plot_model))/(plot_y.shape[0]-1))
    chi_square = np.sum(np.square(plot_model-(m*plot_y))/var_y)
    print("Chi square of compared fit: {}".format(chi_square))
    print("Variance: {}".format(var_y))
    print("Sum of MSE: {}".format(np.sum(np.square(plot_model-(m*plot_y)))))

    #plt.plot(plot_x, combined_model(plot_x, 3.8064464923782184, 13.73394018773772), label='Model', c='orange')
    plt.legend(loc='upper right')
    plt.xlabel("Counts (per 20 seconds) - Data")
    plt.ylabel("Counts (per 20 seconds) - Model")
    plt.title("Model Counts vs Observed Counts")
    plt.show()

def display():
    file_path = "/home/jamesl/Downloads/airsample_dust_3hr_newtime.csv"
    df = pd.read_csv(file_path)
    list_conversion = df.values.tolist()
    break_index = 0
    for i in range(len(list_conversion)):
        if list_conversion[i][1] > 1800:
            break_index = i
            print("Break index: {} Length: {}".format(break_index, len(list_conversion)))
            break
    x = np.array(df['time'].values.tolist())
    y = np.array(df['count'].values.tolist())

    average_bcount = 10.6108108108108
    opt_m = -0.012000306389773823
    opt_b = 112.98577
    y = y - (opt_m*x+opt_b)
    plot_x = np.array(df['time'].values.tolist())
    plot_y = np.array(df['count'].values.tolist())
    y_std = np.sqrt(np.sum(np.square(y-np.mean(y))/(x.shape[0]-1)))
    print("Chi squared: {}".format(np.sum(np.square((y-model(plot_x, 3.8064464923782184, 13.73394018773772))/y_std))))

    graph_error(x, y, model, [3.8064464923782184, 13.73394018773772])
    graph_residual(x, y, model(plot_x, 3.8064464923782184, 13.73394018773772))

def graph_residual(x, label, y_model):
    x = np.expand_dims(x, 1)
    res = y_model-label
    res = np.expand_dims(res, 1)
    reg = LinearRegression().fit(x, res)
    m = np.squeeze(reg.coef_)
    b = np.squeeze(reg.intercept_)
    print("Slope m: {}".format(m))
    x = np.squeeze(x)
    res = np.squeeze(res)
    plt.scatter(x, res, s=2)
    plt.plot(x, m*x+b, c='red', label='Residual Line')
    plt.legend(loc='upper right')
    plt.xlabel("Time (in seconds)")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()

def graph_error(x, y, y_model, model_args):
    x_bar = np.array([x[0] + i * (x[x.shape[0] - 1] - x[0]) / 5 for i in range(6)])
    error_std = np.sqrt(np.sum(np.square(y - y_model(x, *model_args)) / (y.shape[0] - 1)))
    print("Error std: {}".format(error_std))
    plt.scatter(x, y, s=2)
    plt.plot(x, y_model(x, *model_args), c='orange', label="Regression model")
    plt.errorbar(x_bar, y_model(x_bar, *model_args), yerr=error_std, fmt='.', c='black')

    plt.xlabel("Time (seconds)")
    plt.ylabel("Counts (per interval of 20s)")
    plt.title("Counts (20s intervals) over time")
    plt.show()

if __name__ == '__main__':
    comparison()



