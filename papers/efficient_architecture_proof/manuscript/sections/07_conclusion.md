# 6. Conclusion

We presented LAHR (Latent Adaptive Hierarchical Reasoning), an architecture that
combines Mixture-of-Depths adaptive computation, latent-space reasoning, and
differentiable memory retrieval. Through a full 2³ factorial ablation study at
small scale (~20M parameters), we investigated whether these components provide
complementary benefits.

Our key findings from the pilot study are:

1. **Throughput improvement from MoD**: Mixture-of-Depths successfully provides 13%
   throughput improvement through sparse computation, with routing efficiency matching
   the target 12.5% capacity.

2. **Early-stage parity**: At 100 training steps, baseline and LAHR variants show
   similar training loss (~10.4), with no significant benefit from additional
   components at this stage.

3. **Architectural overhead**: The full LAHR model has 17% lower throughput than
   baseline due to latent reasoning iterations, creating a trade-off that may only
   pay off with longer training where the reasoning capacity provides quality gains.

This work represents an exploratory study with acknowledged limitations in scale
and statistical power. We release our implementation to enable future work on
efficient reasoning architectures.

**Importantly, our pilot results are negative**: the full LAHR model underperforms
the baseline at this scale and training duration. We cannot claim that combining
these techniques provides benefits. The architecture requires further validation
at longer training horizons and larger scales before any positive claims can be made.

We release our implementation and ablation infrastructure to enable future work.
The negative results presented here should inform expectations for similar architectural
combination studies.

---

## Code Availability

Code is available at: [REPOSITORY URL]

## Acknowledgments

[TO BE ADDED]

## References

[1] Raposo, D., et al. (2024). Mixture-of-Depths: Dynamically allocating compute
    in transformer-based language models.

[2] Hao, Y., et al. (2024). Training Large Language Models to Reason in a
    Continuous Latent Space. (COCONUT)

[3] Wu, Y., et al. (2022). Memorizing Transformers.

[4] Graves, A. (2016). Adaptive Computation Time for Recurrent Neural Networks.

[5] Dehghani, M., et al. (2018). Universal Transformers.

[6] Eldan, R., & Li, Y. (2023). TinyStories: How Small Can Language Models Be
    and Still Speak Coherent English?

[7] Fedus, W., et al. (2021). Switch Transformers: Scaling to Trillion Parameter
    Models with Simple and Efficient Sparsity.
