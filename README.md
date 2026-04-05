# Purpose:
Compare performance of conventional CBFs vs. CBFs learned via Gaussian Processes. 
Incorporate "smart polling" into learning-based GP-CBFs, intelligently selecting a next point to explore by choosing a nearby point of high covariance.

https://arxiv.org/pdf/2106.11899
# 4/05: Zero-noise CBF vs noisy CBF vs learned-CBF under varying simulated sensor noise conditions.
<img width="250" height="200" alt="image" src="https://github.com/user-attachments/assets/04d7d6f4-5843-4a32-8320-63e42ae6191a" />
<img width="250" height="200" alt="image" src="https://github.com/user-attachments/assets/0eedc0af-3398-4176-883a-8b3c963453f2" />





# 3/24: Conventional CBF performance under noise:
<img width="548" height="451" alt="image" src="https://github.com/user-attachments/assets/173dd130-b97f-491b-bbad-eecf73851c47" />


# Stage 4 (GIBO)

<img width="485" height="385" alt="image" src="https://github.com/user-attachments/assets/fb720dd6-a6e3-4f89-a8f0-f8ccafc29f86" />

Locally "Trapped"


<img width="495" height="391" alt="image" src="https://github.com/user-attachments/assets/e36f2e7b-d980-40d0-9426-0313bc64243e" />


# Stage 1: Basic GP regression implementation (Done)
GP regression on 1D function.
Implement experiments with at least 3 kernels (SE, Matern kernels).
For each kernel, fit hyperparameters, then compare the posterior mean and predictive variance.

Deliverables: 
 - Plot is under GP_Example_Plots. ( GP_Example_Plots/Stage1_Plots.png ) 
 - We saw strongest performance from Linear + Matern (measured by lowest loss). This makes sense as the linear kernel accounts for the linear bias in our underlying function.
![alt text](GP_Example_Plots/Stage1_Plots.png)

Discuss the effect of kernel smoothness.
 - matern 5/2 is twice differentiable, while Squared exponential is infinitely differentiable. In the image below, I cranked up the rate parameter of the initial linear scale. This shows that Matern 5/2 was much noisier around the edges as compared to the SE kernel. This seems to imply that infinitely differentiable kernels will have less noise and be more smooth.
 - Another Note: we do see that final loss of Matern 5/2 is slightly higher, indicating worse performance. 
 ![alt text](image.png)


# NOTES
 - GIBO tends to get Stuck in local regions. We can shift this by changing the step size that theta t+1 is updated by, but this seemed to be a recurring issue.


Stage 4:
 - interestingly, if we have more points at start we dont try to poll tghere

 
03/24 Deliverable
Safe navigation of robot with obstacles, (get familiar with signed distance functions)
 - implement learning cbf
 - compare performance with Azra's weighted acq funciton, gibo
