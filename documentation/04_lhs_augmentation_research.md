# Phase 4: LHS Data Augmentation — Research & Documentation

**Date**: 2026-03-18  
**Author**: Vaiebhav (via Antigravity)

## The Problem

The TCAD dataset has extremely sparse coverage of the 5D input space:
- 8 distinct pressures
- 10 distinct O2 flows  
- 10 distinct N2 flows
- 10 distinct temperatures
- 10 distinct times

This means the data lives on at most 8000 unique (P, O2, N2, T) "recipes" with no intermediate values. A standard regression model will overfit to these specific corners.

## What is LHS?

**Latin Hypercube Sampling** is a stratified Design-of-Experiments technique:
1. Divide each parameter's range into N equal-probability intervals
2. Select exactly one sample per interval per dimension
3. This guarantees near-uniform coverage — no clustering, no gaps

Compared to random sampling: LHS achieves the same coverage with ~10x fewer samples.

## Research Findings

### Key Papers / Approaches

1. **LHS for PINN collocation points**: Recent work (2024-2025) uses LHS to place collocation points densely near boundaries/interfaces where concentration gradients are steepest. This is relevant to Stage 2 (PINN) but not Stage 1 (thickness model).

2. **Physics-informed data augmentation**: Use calibrated physics models to generate synthetic observations at unseen conditions. The ML model then trains on real + synthetic data.

3. **Adaptive LHS**: Generates initial samples via standard LHS, then adds more samples in high-error regions iteratively. Particularly useful for non-linear response surfaces.

## Proposed Approach for This Project

### Strategy: Deal-Grove Interpolation + LHS

1. **Calibrate** the Deal-Grove model to all 79,181 observed thickness values
2. **Generate** N=5000 LHS samples within the bounding box of observed parameters
3. **Predict** thickness at each LHS point using calibrated Deal-Grove
4. **Add noise** proportional to the observed residual distribution (so synthetic data has realistic variance)
5. **Combine** original + augmented data for model training

### Why This Works

- The Deal-Grove model captures the fundamental physics (diffusion + reaction kinetics)
- LHS fills the gaps between the sparse 8×10×10×10×10 grid
- We only interpolate (never extrapolate beyond observed bounds)
- The grey-box model's job is to learn the residual between Deal-Grove and reality — augmented data helps it see smoother input landscapes

### What This Does NOT Do

- Does NOT create new spatial concentration profiles (that would require running TCAD)
- Does NOT extrapolate beyond the observed parameter ranges
- Does NOT replace the need for more real TCAD simulations at novel conditions

## Implementation

See `lhs_augmentation.py` for the full implementation.
