Forest Fire Simulator

Our project simulates forest fire patterns in certain weather conditions which are generated from a random tree regressor model that was trained on a public dataset. 
Our model takes input features such as temperature, wind, precipitation etc., and predicts the next set of target features--wind, temperature, humidity, precipitation. We then pass these features into our "blocks" which represent a region within our visualization. Each blocks weather influences how likely it is to be set on fire, which is controlled by function _______. Once each is given its respective characteristics, we crete a pygame visualization to show how the fire--and weather--will be simulated over time.
