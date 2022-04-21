# DeepKoopmanWithControl
Deep Koopman Operator with Control for Nonlinear Systems

Paper: https://arxiv.org/abs/2202.08004

## Prediction
<img src="Prediction.png"  width="1000"/>
<img src="PredictionResults.png" width="1000">

## Control
<img src="Control.png" width = "1000">
<img src="FrankaStar.png" width = "1000">

## Requirement
``` python 
pytorch, gym
```
## Environment
For gym environment, you should replace the gym env file with files in folder ./gym_env/

All the environments:
``` python 
"DampingPendulum"
"DoublePendulum"
"Franka"
"Pendulum-v1"
"MountainCarContinous-v0"
"CartPole-v1"
```

## Example
To train the network, you can just run 
``` python 
cd train/
python Learn_koopman_with_KlinearEig.py
```
To evaluate the prediction performance, you can utilize the notebooks in folder prediction/

To evaluate the control performance, you can utilize the notebooks in folder control/