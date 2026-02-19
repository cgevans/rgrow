# SDC: Scaffolded DNA Computers

SDC models simulate the systems in Stérin, T., Eshra, A., Adio, J., Evans, C. G. & Woods, D. A Thermodynamically Favoured Molecular Computer: Robust, Fast, Renewable, Scalable <https://doi.org/10.1101/2025.07.16.664196> (2025).

There are currently two models in rgrow:

- `SDC`: a 'standard' TAM-like kinetic model with strand/tile attachment and detachment based on stochastic chemical kinetics.
- `SDC1DBindReplace`: the abstract, bind-replace model in S3.3 of the paper SI.

## SDC

- `SDC` models strand interactions at an attachment/detachment level, like the kTAM.  On attaching, a strand is assumed to make all possible bonds with its neighbors and the scaffold, and detaches by simultaneously breaking all these bonds.  
- `SDC` implements optional event-level strand concentration depletion via event-skipping: rates for non-depleted attachments are used for choosing events, but events are skipped based on depletion.  This works efficiently when compute strands are in reasonable excess: for example, if compute strand concentrations are 10 times the scaffold concentration, then at most, 10% of events would be skipped.

## SDC1DBindReplace

