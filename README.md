Tasks
https://docs.gpytorch.ai/en/stable/index.htm
Stage 0: Setup (DONE)

Create a repo.
Pick one GP library -> Chose GPyTorch

# Stage 4
<img width="985" height="785" alt="image" src="https://github.com/user-attachments/assets/fb720dd6-a6e3-4f89-a8f0-f8ccafc29f86" />

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

# Takeaways:
Stage 2 Takeaways:
 - In the posterior mean function, only the kernel is dependent on the input variables, and to get gradient we just differentiate kernel.
 - Accomplish gradient calc by running torch.backwards() on our model at different test_x.


Stage 4:
 - interestinglu, if we have more points at start we dont try to poll tghere
