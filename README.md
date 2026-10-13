# Spike-Lab! 

Non-navigation
- Input: one-hot color, one-hot shape
- Output: position (x,y) 
- Time course: Color, delay, shape, delay, go, output 
- Hypothesis: no grid cells 

Navigation (static) 
- Input: one-hot color, one-hot shape, current location (x,y)
- Output: velocity vector (straight path) 
- Hypothesis: no grid cells 

Navigation 
- Input: current (x,y), velocity (varies with t) 
- Output: next/current (x,y) (varies with t) 
- Hypothesis: grid cells 
- Essentially copy existing 

Retrain the trained 3 on tasks 1 and 2
- Are the grids maintained? 
- Are they useful for 1 and 2?
