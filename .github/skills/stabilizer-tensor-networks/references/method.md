# STN Method Reference (arXiv:2403.08724)

Distilled equations and update rules from Masot-Llima & Garcia-Saez, *Stabilizer Tensor
Networks*, PRL 133, 230601 (2024). Equation numbers match the arXiv v2 paper.

## Core representation

A state is a **stabilizer basis** plus a **coefficient (amplitude) state**:

- Basis $\mathcal B(\mathcal S,\mathcal D)$: $n$ stabilizer generators $s_i$ and $n$
  destabilizer generators $d_i$, with $\{d_i,s_i\}=0$, $[d_i,d_j]=0$, $[d_i,s_j]=0$ for
  $i\neq j$. Stored as the $2n\times(2n+1)$ boolean tableau (Aaronson–Gottesman,
  arXiv:quant-ph/0406196). This is exactly a stim tableau.
- $\hat d_{\hat i}=d_1^{i_1}\cdots d_n^{i_n}$ for a bitstring $\hat i=(i_1\dots i_n)$.
- Basis states $\{\hat d_{\hat i}|\psi_{\mathcal S}\rangle\}$ are orthonormal and span
  $\mathcal H_n$ (Lemma 1).

$$|\psi\rangle=\sum_{i=0}^{2^n-1}\nu_i\,\hat d_{\hat i}\,|\psi_{\mathcal S}\rangle,\qquad
\sum_i|\nu_i|^2=1.\tag{2}$$

Encode the amplitudes as an $n$-qubit MPS $|\nu\rangle=\sum_i\nu_i|i\rangle$. Bond
dimension $\chi$ is the resource. Initial $|0\rangle^{\otimes n}$ ⇒ identity tableau
($s_i=Z_i,d_i=X_i$) and $|\nu\rangle=|0\dots0\rangle$ ($\chi=1$).

## Action of (de)stabilizers on |nu> (Eqs. 14–16)

Multiplying by a destabilizer shifts the basis index; multiplying by a stabilizer adds a
sign. On the coefficient state these are exactly Pauli $X$ / $Z$:

$$\delta_{\hat d}\,|\psi\rangle \;\Leftrightarrow\; X_{\hat d}\,|\nu\rangle,\qquad
\sigma_{\hat s}\,|\psi\rangle \;\Leftrightarrow\; Z_{\hat s}\,|\nu\rangle.\tag{16}$$

Any operator decomposes as $\mathcal U=\sum_i \phi_i\,\delta_{\hat d_i}\sigma_{\hat s_i}$
(Eq. 17), since $\mathcal S\cup\mathcal D$ generate $\mathcal P_n$.

## Pauli → basis decomposition (needed for every non-Clifford step)

Given a single-qubit Pauli $P$ (the rotation axis $X_q$, $Y_q$, or $Z_q$), write it in the
current basis as $P=\alpha\,\delta_{\hat d}\,\sigma_{\hat s}$:

- The X-part vector $\hat d$ (which destabilizers) and Z-part vector $\hat s$ (which
  stabilizers) are recovered from **symplectic (commutation) products** with the conjugate
  generators. Because $d_i$ anticommutes only with $s_i$: $\hat d_i=\langle s_i,P\rangle$
  and $\hat s_i=\langle d_i,P\rangle$, where $\langle A,B\rangle=0$ if $[A,B]=0$ and $1$ if
  $\{A,B\}=0$ (boolean inner product of the Pauli bit-vectors).
- $\alpha$ is the phase fixing $P=\alpha\,\hat d\,\hat s$ (may force writing $XZ=-iY$).
- In stim: build $P$ as a `stim.PauliString`, use the tableau (`stim.Tableau` /
  `TableauSimulator.current_inverse_tableau()`) to map $P$ onto the generator basis; the
  resulting X/Z support gives $\hat d,\hat s$ and the sign gives $\alpha$.

## Live Pepsy shortcut: conjugate the physical operator

Pepsy stores the same basis information as a Clifford unitary $C$, so the representation is
exactly $|\psi\rangle=C|p\rangle$. For a physical Pauli $P$, call
`STNState.frame_pauli(P)` to obtain

$$M=C^\dagger P C=\alpha\prod_j P_j.$$

This signed Pauli is the executable form of the basis decomposition above. The current
implementation does not reconstruct $\hat d,\hat s$ or the $I_x,I_y,I_z$ masks during
normal evolution:

- $\exp(-i\theta P/2)|\psi\rangle=C\exp(-i\theta M/2)|p\rangle$;
- $\langle\psi|P|\psi\rangle=\langle p|M|p\rangle/\langle p|p\rangle$;
- $(I+mP)|\psi\rangle/2=C(I+mM)|p\rangle/2$ for fixed-basis collapse.

The equations below remain the derivation and an independent sign/phase cross-check. For
execution, prefer the direct stim conjugation and the exact `c I + coef M` local/sub-MPO
builders.

## Update rule 1 — Clifford gate G

Conjugate the basis by $G$ (standard tableau update, Appendix D / stim). The coefficient
state is unchanged:

$$G|\psi\rangle=\sum_i\nu_i\,\tilde d_i|\psi_{\tilde{\mathcal S}}\rangle,\qquad
G|\nu\rangle=|\nu\rangle.\tag{3,4}$$

⇒ Clifford gates are **free** (no $\chi$ growth).

## Update rule 2 — Non-Clifford gate (Lemma 2)

For a two-term unitary
$\mathcal U=\phi_1\delta_{\hat d_1}\sigma_{\hat s_1}+\phi_2\delta_{\hat d_2}\sigma_{\hat s_2}$
(this covers every single-qubit rotation $RX,RY,RZ$), the action on $|\nu\rangle$ is a
Clifford Pauli update followed by a single multi-qubit rotation:

$$\mathcal U|\psi\rangle=\mathcal R_{X^{I_x}Y^{I_y}Z^{I_z}}(2\theta)\,|\nu\rangle,\qquad
\mathcal R_{P}(2\theta)=\cos\theta\,I-i\sin\theta\,P,\tag{19,27}$$

with $\theta=\arccos(\mathrm{Re}\,\phi_1)$ and axis masks (Eq. 20):

$$I_y=(\hat d_1+\hat d_2)\circ_h(\hat s_1+\hat s_2),\quad
I_x=(\hat d_1+\hat d_2)+I_y,\quad
I_z=(\hat s_1+\hat s_2)+I_y,$$

where $+$ is bitwise XOR and $\circ_h$ is elementwise (Hadamard) product. $I_x$ marks pure-X
qubits, $I_z$ pure-Z, $I_y$ the shared (Y) qubits. Apply the leftover Clifford factor
$\delta_1\sigma_1$ first (as $X/Z$ on $|\nu\rangle$). Track the sign $\delta_1\cdot\sigma_2$
to get the rotation angle sign right.

For a single-qubit axis $P=\alpha\,\delta_{\hat d}\sigma_{\hat s}$: apply $X_{\hat d}Z_{\hat s}$,
then $\mathcal R(2\theta)$ with masks from $\hat d,\hat s$ via Eq. 33 (see measurement).

### Corollary 2.1 — free non-Clifford ops
If the rotation axis (after the Clifford part) is a **single basis generator**
$d_i s_i$, then on $|\nu\rangle$ it is a **local** single-qubit rotation $R_X(-\theta)$ —
no $\chi$ increase. Example: with basis $s_i=X_i,d_i=Z_i$, each $T_i=\cos\frac\pi8 I-i\sin\frac\pi8\,d_i$
is local, so $|T\rangle^{\otimes n}$ is a trivial $\chi=1$ MPS (Eqs. 11–12), even though its
pseudo-stabilizer rank is maximal $\tilde\xi=2^n$.

## Update rule 3 — Measurement of observable O (Lemma 3)

Decompose $\mathcal O=\alpha\,\delta_{\hat n}\sigma_{\hat m}$. Expectation:

$$\langle\mathcal O\rangle=\alpha\,\langle\nu|X_{\hat n}Z_{\hat m}|\nu\rangle.\tag{29}$$

Pick outcome $m\in\{+,-\}$ with $p_+=(1+\langle\mathcal O\rangle)/2,\ p_-=1-p_+$. Let $k$ be
the position of the first 1 in $\hat n$. Update the basis to $\mathcal B'(\mathcal S',\mathcal D')$
(tableau measurement update). The projection $\tfrac{I\pm\mathcal O}{2}$ on $|\nu\rangle$ is
a **non-unitary rotation then a projector**:

$$\tfrac{I\pm\mathcal O}{2}|\nu\rangle=P_k\,\tilde{\mathcal R}\,|\nu\rangle,\qquad
\tilde{\mathcal R}=\tfrac12 I\pm\alpha\,(-i)^{|I_y|}\,X^{I_x}Y^{I_y}Z^{I_z},\tag{32,40}$$

with masks (Eq. 33)

$$I_y=\hat n\circ_h\hat m,\quad I_x=\hat n+I_y,\quad I_z=\hat m+I_y,$$

$P_k=|0\rangle\langle0|_k$ the projector on qubit $k$, and renormalization
$\mathcal N=\sqrt{(1\pm\langle\mathcal O\rangle)/2}$ (Eq. 43). The central one-qubit
non-unitary of the CNOT cascade is (Eq. 44):

$$\tilde{\mathcal R}_{\text{1q}}=\tfrac12\begin{pmatrix}1&\pm\alpha(-i)^{|I_y|}\\
\pm\alpha(-i)^{|I_y|}&1\end{pmatrix}.$$

When $\hat n=0$ (measuring a stabilizer) the basis is unchanged and there is no projector.

### Measurement policies in Pepsy

`MpsStabOptimizer` deliberately exposes two equivalent collapse representations:

- **Fixed basis (default):** leave $C$ unchanged and apply $(I+mM)/2$ directly to $|p\rangle$.
  Use a local 2x2 projector for one-site support or a windowed bond-dimension-2 sub-MPO for
  multi-site support, then normalize at the tracked canonical centre.
- **Basis updating (`absorb_basis=True`):** choose a Clifford $V$ with
  $VMV^\dagger=sZ_k$, apply $V$ to $|p\rangle$, replace $C$ by $CV^\dagger$, and project
  site $k$ onto the required computational value. Before projection,
  $(CV^\dagger)(V|p\rangle)=C|p\rangle$; afterward the pivot is disentangled. The
  localizer uses a median support pivot and merges nearer support sites first.

Reset and magic-state injection use the basis-updating path because they need a reusable,
disentangled ancilla.

## Paper/reference CNOT-cascade rotation

A multi-qubit Pauli rotation is realized by: basis-change single-qubit gates ($X\to Y,Z$ per
$I_x,I_y,I_z$), a CNOT cascade collecting parity onto the innermost affected qubit, one
central single-qubit rotation ($R_X$ for Lemma 2, or $\tilde{\mathcal R}_{\text{1q}}$ for
Lemma 3), then the reverse CNOT cascade and basis changes. The paper's implementation uses
**two cascades centered on the middle qubit** (Fig. 4). Centering on the innermost qubit and
adapting the TN geometry to circuit connectivity limits $\chi$ growth.

This construction explains the paper's bond-growth bounds and is useful for cross-checking.
The live Pepsy rotation/projector path does **not** emit this cascade: it represents
$cI+\mathrm{coef}\,M$ directly as an exact bond-dimension-2 MPO on the true support window
and applies it with quimb `gate_with_submpo_`. The basis-updating measurement does use a
one-way Clifford localizer (axis changes plus a CNOT ladder), because it must move the
measured information onto one coefficient site before absorbing the basis change.

## Resource / cost facts
- Free operations: all Clifford gates + non-Clifford ops satisfying Corollary 2.1.
- $\chi$ growth per single non-Clifford rotation: worst case $\chi'\le4\chi$ (MPS-local
  bond), $\chi'\le16\chi$ when the rotation spans far-apart qubits (SWAP overhead, Schmidt
  rank 4). Non-MPS geometries matching connectivity can reach the $4\chi$ bound. Empirically
  (Fig. 2) the average per-T-gate increase is $\log_2\chi'\sim2.46$ and does not grow with
  $n$.
- Pseudo-stabilizer rank $\tilde\xi$ = number of non-zero $\nu_i$; upper-bounds the true
  stabilizer rank $\xi$.

## Worked example (Appendix C, 5 qubits)

$\mathcal U=\tfrac{\sqrt3}{2}\delta_{\hat d_1}\sigma_{\hat s_1}+\tfrac12\delta_{\hat d_2}\sigma_{\hat s_2}$
with
$\hat d_1=(1,1,0,0,0),\ \hat d_2=(1,0,0,1,0),\ \hat s_1=(0,0,0,1,0),\ \hat s_2=(0,0,1,0,0)$,
so $\hat d_2\hat d_1=(0,1,0,1,0)$, $\hat s_2\hat s_1=(0,0,1,1,0)$, and $\hat d_1\cdot\hat s_2=1$.
Then

$$\mathcal U|\nu\rangle=\big(\cos\tfrac\pi6+i\sin\tfrac\pi6\,X_1Y_3Z_2\big)\,X_0X_1Z_3\,|\nu\rangle,$$

which fits Eq. 6: a Clifford $X_0X_1Z_3$ update followed by a 3-qubit rotation (axes $X$ on
1, $Y$ on 3, $Z$ on 2), implemented with the CNOT cascade of Fig. 4.

## Appendix D — tableau update rules (use stim; reference only)

Row-wise updates on the $2n\times(2n+1)$ tableau, $\oplus$ = XOR:
- **CNOT** (control $a$, target $b$), each row $i$:
  $r_i \mathrel{\oplus}= x_{ia}z_{ib}(x_{ib}\oplus z_{ia}\oplus1)$;
  $x_{ib}\mathrel{\oplus}=x_{ia}$; $z_{ia}\mathrel{\oplus}=z_{ib}$ (Eq. 53).
- **Hadamard** on $a$: $r_i\mathrel{\oplus}=x_{ia}z_{ia}$; swap $x_{ia}\leftrightarrow z_{ia}$ (Eq. 54).
- **Phase (S)** on $a$: $r_i\mathrel{\oplus}=x_{ia}z_{ia}$; $z_{ia}\mathrel{\oplus}=x_{ia}$ (Eq. 55).
- **Z-measurement** on $a$: if it commutes with all stabilizers, `rowsum` into an aux row to
  read the deterministic outcome; else outcome is random, pick an anticommuting row $i$, do
  `rowsum(i,h)` for the other anticommuting rows, then set stabilizer $i=Z_a$ and store the
  old stabilizer as destabilizer $i-n$.
- **rowsum(a,b)**: $a\leftarrow a+b$ with the $g(x_1,z_1,x_2,z_2)$ phase bookkeeping of
  Eqs. 56–57. X/Y measurements = Clifford basis change + Z-measurement.

Prefer stim's `TableauSimulator.measure` / `.h/.s/.cnot` and `stim.Tableau` over
re-deriving these by hand.
