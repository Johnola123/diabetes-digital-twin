# Screen recording: demonstrate the hard professional decisions

1. Show repository tree, synthetic data model, and README limitations.
2. Open `src/dtwin/generate.py`: explain the latent severity variable, treatment feedback, visit process and shared potential-outcome shocks.
3. Open `src/dtwin/pipeline.py`: explain Bronze/Silver/Gold and strict lagged feature generation.
4. Run DEV workflow, show checks and output folders.
5. In dashboard EDA, show differential initial severity and observation frequency.
6. In estimator comparison, show conventional association vs MSM and two g-computation variants.
7. In diagnostics, inspect effective sample size and extreme propensity weights; explain bias/variance implications.
8. In monitoring sensitivity, compare weighted and unweighted g-computation; explain why inference is uncertain.
9. In patient twin, choose patient; show never/always trajectories and hidden synthetic truth.
10. In validation view, show that outcome prediction accuracy is not causal effect accuracy.
11. Run tests and environment checks; start three Docker services and open all ports.
12. In GitHub Actions, manually choose an environment; capture workflow and downloadable artifact if run succeeds.
13. End with defensible limitations; do not describe simulated findings as clinical results.

Do not claim a command or cloud workflow succeeded unless you show its actual successful output.
