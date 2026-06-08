
-- Initial Conditions (Fixed for all Acquisition Functions) --
@ dt=0.04, end_time=11
a = 0.2
b = 1.0
lengthscale_prior = GammaPrior(a, b) # (a,b) : lengthscale = a/b
outputscale_prior = GammaPrior(2.0, .17) # (a,b) : outputscale = a/b

-- GIBO --
 - Total Runtime (s): 38.9055
 - CBF Violations:    0
 - Min h value (deg): 6.022
 - Sum of Posterior Covariance: 68.25805007848507



-- Composite (Lambda = 0.2) --
 - Total Runtime (s): 59.6986
 - CBF Violations:    0
 - Min h value (deg): 11.766
 - Sum of Posterior Covariance: 68.3383922334335

-- Composite (Lambda = 0.8) --
 - Total Runtime (s): 61.8361
 - CBF Violations:    0
 - Min h value (deg): 10.538
 - Sum of Posterior Covariance: 67.90198212181937

-- Random QP (Baseline) --
 - Total Runtime (s): 3.0582
 - CBF Violations:    3
 - Min h value (deg): -2.820
 - Sum of Posterior Covariance: 73.14073541976286

-- Composite (Lambda = 0.95) --
 - Total Runtime (s): 107.4989
 - CBF Violations:    0
 - Min h value (deg): 6.693
 - Sum of Posterior Covariance: 68.16441918244723

-- Adaptive Lambda Function-- 
 - Total Runtime (s): 73.1554
 - CBF Violations:    0
 - Min h value (deg): 9.154
 - Sum of Posterior Covariance: 66.68580784568525

-- SOCP -- 
Total Runtime (s): 64.5590
CBF Violations:    0
Min h value (deg): 6.370
Sum of Posterior Covariance: 70.01258699664356