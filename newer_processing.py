import pandas as pd
import matplotlib.pyplot as plt
import pdb
from sklearn.linear_model import LinearRegression
from sklearn import metrics, model_selection, tree


df = pd.read_csv('forestfires.csv')

month_order = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
day_order = ["mon","tue","wed","thu","fri","sat","sun"]

df["month"] = pd.Categorical(df["month"], categories=month_order, ordered=True)
df["day"] = pd.Categorical(df["day"], categories=day_order, ordered=True)

input_col_names = ['temp', 'RH', 'wind']
data = {}
for group, group_data in df.groupby(by=['X', 'Y', 'month']):
    data[group] = group_data[input_col_names].mean()
print(data)

output_data = []
for (x,y,mon) in data:
    prev_mon_idx = month_order.index(mon) - 1
    prev_mon = month_order[prev_mon_idx]
    cur_values = data[(x,y,mon)]
    if (x,y,prev_mon) not in data:
        continue
    prev_values = data[(x,y,prev_mon)]
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            key = (x+dx, y+dy, prev_mon)
            if key in data:
                neighbors.append(data[key]) 
    if len(neighbors) == 0:
        continue
    neighbor_avg_values = pd.concat(neighbors, axis=1).mean(axis=1)
    location_month_data = pd.concat([cur_values, prev_values, neighbor_avg_values])
    output_data.append([x, y, mon] + location_month_data.values.tolist())

output_col_names = ['x', 'y', 'mon'] + [f'target_{col}' for col in input_col_names] + [f'prev_{col}' for col in input_col_names] + [f'prev_avg_neighbor_{col}' for col in input_col_names]
output_df = pd.DataFrame(output_data, columns=output_col_names)
print(output_df)
pdb.set_trace()

#scatter to show that the grid is not uniform
plt.scatter(df["Y"], df["X"])
plt.show()
            


'''Input feature 
- Precipitation and temp on the previous day
- Precipitation 

cur_temp, cur_rain, prev_temp, prev_rain, pre_neighbor_temp, prev_neighbor_rain, wind??

Loop over every row of the go data frame, and then make selections  '''


#models to try: descicion/forest tree, XGBoost...