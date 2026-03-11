Forest Fire Simulator

Our project simulates forest fire patterns in certain weather conditions which are generated from a random tree regressor model that was trained on a public dataset. 
Our model takes input features such as temperature, wind, precipitation etc., and predicts the next set of target features--wind, temperature, humidity, precipitation. We then pass these features into our "blocks" which represent a region within our visualization. Each blocks weather influences how likely it is to be set on fire. Once each is given its respective characteristics, we crete a pygame visualization to show how the fire--and weather--will be simulated over time.

To run this module, make sure you pip install **hkb_diamondsquare**. 

How to use:
Run the program, go to the main() function if you want to change the seed or framerate
Start a fire by left click a block
Put out a fire by right click a block
Start a rain by press 1, press multiple times to make the rain heavy enough to put out all the fire
Boost wind speed by press 2
Increase temperature by press 3
Reset by press r
