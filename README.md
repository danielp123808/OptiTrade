Description:

OptiTrade: OptiTrade is a systems engineering tool designed to resolve the payload-capacity versus flight-range conflict in multirotor drone development. The engine implements an optimization algorithm based on NSGA-II principles to evaluate hundreds of hardware configurations against real-world aerodynamic and powertrain constraints. By mapping non-dominated solutions across a cargo spectrum of 500g to 3,000g, OptiTrade provides an engineering trade chart that isolates the most efficient component sizing for specific mission profiles.

Steps to use:

git clone https://github.com

cd uav-multi-objective-optimizer

pip install numpy pandas matplotlib

python uav_optimizer.py
