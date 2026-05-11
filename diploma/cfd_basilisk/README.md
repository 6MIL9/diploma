# Real CFD route: Basilisk rising bubble

The original `cfd_data/rising_bubble.h5` is a CFD result, not something the PINN script can regenerate. The paper points to ALPACA for data generation; for learning and reproducing the Hysing rising-bubble benchmark, Basilisk is a compact starting point because it already ships an official `src/test/rising.c` case.

Sources:

- Basilisk official rising-bubble test: https://basilisk.fr/src/test/rising.c
- OpenFOAM `interFoam` docs if you prefer OpenFOAM later: https://doc.openfoam.com/2306/tools/processing/solvers/rtm/multiphase/interFoam/
- Original PINN repository: https://github.com/aaronbuhendwa/twophasePINN

## 1. Install Basilisk

Follow the official Basilisk installation instructions for your OS:

```text
https://basilisk.fr/src/INSTALL
```

After installation, `qcc` should be available:

```bash
qcc --version
```

On this machine Basilisk was found at `/Users/erklyavlin/repos/basilisk/src/qcc`. If `qcc` is not in `PATH`, either use the full path or add it for the current shell:

```bash
export PATH="/Users/erklyavlin/repos/basilisk/src:$PATH"
```

## 2. Run the benchmark

From this directory:

```bash
qcc -O2 -Wall rising_bubble.c -o rising_bubble -lm
./rising_bubble > log.txt
```

Fast smoke run:

```bash
qcc -O2 -Wall -DLEVEL=5 -DSNAPSHOT_DT=0.5 -DT_END=1.0 rising_bubble.c -o rising_bubble_test -lm
./rising_bubble_test > log-test.txt
```

This writes:

- `log.txt`: bubble volume error, center position, rise velocity
- `snapshot-*.tsv`: grid snapshots with phase fraction, velocity and pressure
- `interface-final.dat`: final interface facets

Change the bubble radius:

```bash
qcc -O2 -Wall -DRADIUS=0.20 rising_bubble.c -o rising_bubble -lm
./rising_bubble > log-r020.txt
```

Increase resolution:

```bash
qcc -O2 -Wall -DLEVEL=9 rising_bubble.c -o rising_bubble -lm
```

`LEVEL=8` is the starter resolution. Each +1 level roughly doubles cells in each direction.

## 3. Convert snapshots to HDF5

```bash
../../.venv/bin/python convert_snapshots_to_h5.py \
  --input-dir . \
  --output ../cfd_data/rising_bubble_basilisk_r020.h5
```

By default the converter writes `pressure` in the same absolute gauge as `cfd_data/rising_bubble.h5`, so downstream notebooks do not need manual pressure correction. If you need diagnostics, you can still store raw Basilisk pressure:

```bash
../../.venv/bin/python convert_snapshots_to_h5.py \
  --input-dir . \
  --output ../cfd_data/rising_bubble_basilisk_raw.h5 \
  --pressure-mode raw
```

For direct validation against an existing reference with the same radius and shape, use frame-mean alignment:

```bash
../../.venv/bin/python convert_snapshots_to_h5.py \
  --input-dir . \
  --output ../cfd_data/rising_bubble_basilisk_reference_gauge.h5 \
  --pressure-mode reference-mean \
  --reference-path ../cfd_data/rising_bubble.h5
```

Then train the PINN on the CFD output:

```bash
cd ..
../.venv/bin/python train_rising_bubble.py \
  --preset smoke \
  --data-path cfd_data/rising_bubble_basilisk_r020.h5
```

## Notes

The Basilisk reference case uses a rotated half-domain: vertical coordinate `x in [0, 2]` and horizontal coordinate `y in [0, 0.5]`. The converter mirrors it into the same orientation as your HDF5/PINN data: `X in [-0.5, 0.5]`, `Y in [0, 2]`.

The paper's original data used ALPACA, so Basilisk will not match byte-for-byte. The learning target is the CFD workflow: define physical parameters, run the multiphase solver, validate center/velocity/volume, export fields, then feed them into the PINN.
