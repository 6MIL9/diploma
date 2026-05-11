// Hysing et al. rising-bubble benchmark, close to the case used in
// Buhendwa, Adami and Adams, section 3.2.3.
//
// Run with Basilisk:
//   qcc -O2 -Wall rising_bubble.c -o rising_bubble -lm
//   ./rising_bubble > log.txt
//
// Change radius/resolution without editing the file:
//   qcc -O2 -Wall -DRADIUS=0.20 -DLEVEL=9 rising_bubble.c -o rising_bubble -lm

#include "grid/multigrid.h"
#include "navier-stokes/centered.h"
#include "two-phase.h"
#include "tension.h"

#ifndef LEVEL
# define LEVEL 8
#endif

#ifndef RADIUS
# define RADIUS 0.25
#endif

#ifndef SNAPSHOT_DT
# define SNAPSHOT_DT 0.02
#endif

#ifndef T_END
# define T_END 3.0
#endif

u.t[right] = dirichlet(0);
u.t[left] = dirichlet(0);

int main()
{
  // Basilisk's Hysing test uses a rotated half-domain:
  // x is the vertical coordinate in [0, 2], y is the horizontal coordinate in [0, 0.5].
  // The Python converter mirrors it into X in [-0.5, 0.5], Y in [0, 2].
  dimensions (nx = 4);
  size (2 [1]);
  DT = 1. [0,1];
  init_grid (1 << LEVEL);

  rho1 = 1000. [0];
  rho2 = 100.;
  mu1 = 10.;
  mu2 = 1.;
  f.sigma = 24.5;
  TOLERANCE = 1e-4 [*];

  run();
}

event init (t = 0)
{
  // Liquid is f = 1, bubble is f = 0.
  fraction (f, sq(x - 0.5) + sq(y) - sq(RADIUS));
}

event acceleration (i++)
{
  face vector av = a;
  foreach_face(x)
    av.x[] -= 0.98;
}

event logfile (i++)
{
  double xb = 0., vb = 0., sb = 0.;
  foreach(reduction(+:xb) reduction(+:vb) reduction(+:sb)) {
    double dvb = (1. - f[])*dv();
    xb += x*dvb;
    vb += u.x[]*dvb;
    sb += dvb;
  }
  static double sb0 = 0.;
  if (i == 0) {
    sb0 = sb;
    fprintf (stdout, "t volume_error center_y velocity_y dt\n");
  }
  fprintf (stdout, "%g %g %g %g %g\n", t, (sb - sb0)/sb0, xb/sb, vb/sb, dt);
  fflush (stdout);
}

event snapshots (t = 0.; t <= T_END + 1e-12; t += SNAPSHOT_DT)
{
  char name[128];
  snprintf (name, sizeof(name), "snapshot-%07.4f.tsv", t);
  FILE * fp = fopen (name, "w");
  fprintf (fp, "# t x y f ux uy p\n");
  foreach()
    fprintf (fp, "%.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
             t, x, y, f[], u.x[], u.y[], p[]);
  fclose (fp);
}

event interface (t = T_END)
{
  FILE * fp = fopen ("interface-final.dat", "w");
  output_facets (f, fp);
  fclose (fp);
}
