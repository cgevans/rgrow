\(
    \gdef\kfh{\hat{k}_\text{f}}
    \gdef\kf{k_\text{f}}
    \gdef\ratt{r_\text{att}}
    \gdef\rdet{r_\text{det}}
    \gdef\gmc{G_\text{mc}}
    \gdef\gse{G_\text{se}}
    \gdef\conc#1{\left[#1\right]}
    \gdef\dgse{\Delta G^\circ_\text{se}}
\)

# Kinetic Tile Assembly Model (kTAM)

The kinetic Tile Assembly Model, along with the abstract Tile Assembly Model, were introduced in Winfree 1998[@winfree_simulations_1998]; Evans and Winfree 2017 is a modern review on the model[@evans_physical_2017].

\[
\begin{align*}
    \gmc &\equiv - \log \left( \conc{t} / u_0 \right) + \alpha \\
    \gse &\equiv - \dgse / R T
\end{align*}
\]

\[ 
    \begin{align*}
    \ratt &= \kf u_0 e^{-\gmc + \alpha} \\
    \rdet &= \kf u_0 e^{-b \gse + \alpha} 
    \end{align*}
\]

Rgrow implements the 


- Zero-strength attachments events are ignored (equivalent to the TODO_FIND_THIS option in xgrow).
- Fission events, which are not addressed in the kTAM definition, may be handled in several different ways (see Fission below).

## Fission

Fission occurs when a detachment event would split the assembly into multiple disconnected fragments. The `fission` option controls how this is handled:

- **NoFission** (aliases: `off`, `no-fission`):  
  Any detachment event that would result in fission is prohibited.  Detailed balance is preserved.  This is similar to 'no' in Xgrow.

- **JustDetach** (aliases: `just-detach`, `surface`):  
  Allow detachment; all resulting fragments remain on the canvas, even if disconnected.  Detailed balance is preserved.

- **KeepSeeded** (aliases: `on`, `keep-seeded`):  
  After fission, keep only fragments containing a seed tile. Remove all other fragments.  Detailed balance is violated.  This is similar to 'on' or 'chunk' in Xgrow, depending on the chunk settings.

- **KeepLargest** (alias: `keep-largest`):  
  After fission, keep only the largest fragment (by tile count). Remove all others.  Detailed balance is violated.

- **KeepWeighted** (alias: `keep-weighted`):  
  After fission, randomly keep one fragment with probability proportional to its size. Remove all others.  Detailed balance is violated.

