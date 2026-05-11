# Rising bubble workflow

The original `cfd_data/rising_bubble.h5` is not produced by the PINN code. It is a CFD result generated with ALPACA and then used as training/reference data for the inverse PINN problem.

From section 3.2.3 of Buhendwa, Adami and Adams:

- domain: `(x, y) in [-0.5, 0.5] x [-1.0, 1.0]`; the provided HDF5 stores this as `y in [0, 2]`
- time interval: `t in [0, 3]`
- initial bubble radius: `R = 0.25`
- densities: `rho_1 = 100`, `rho_2 = 1000`
- viscosities: `mu_1 = 1`, `mu_2 = 10`
- surface tension: `sigma = 24.5`
- gravity: `g = [0, -0.98]`
- north/south boundaries: no slip
- east/west boundaries: periodic
- north pressure: ambient pressure

The local file has this schema:

```text
X          (256,)
Y          (512,)
time       (151,)
levelset   (151, 256, 512)
density    (151, 256, 512)
pressure   (151, 256, 512)
velocityX  (151, 256, 512)
velocityY  (151, 256, 512)
```

For a real CFD workflow, start with the Basilisk case in `cfd_basilisk/`. It runs the Hysing rising-bubble benchmark with a real incompressible two-phase Navier-Stokes solver and exports snapshots that can be converted to the same HDF5 schema.

```bash
cd cfd_basilisk
qcc -O2 -Wall -DRADIUS=0.20 rising_bubble.c -o rising_bubble -lm
./rising_bubble > log-r020.txt

../../.venv/bin/python convert_snapshots_to_h5.py \
  --input-dir . \
  --output ../cfd_data/rising_bubble_basilisk_r020.h5
```

The converter writes pressure in the same absolute gauge as `rising_bubble.h5` by default. Use `--pressure-mode raw` only when you explicitly want unshifted Basilisk pressure for debugging.

Then train:

```bash
cd ..
../.venv/bin/python train_rising_bubble.py \
  --preset smoke \
  --data-path cfd_data/rising_bubble_basilisk_r020.h5
```

The paper's original data used ALPACA, so Basilisk results are not expected to match byte-for-byte. They are, however, a real CFD simulation of the same benchmark family and are a good way to learn the full loop.
