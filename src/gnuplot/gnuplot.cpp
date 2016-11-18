#include <stdint.h>
#include <iostream>
#include <math.h>

#include "gnuplot-iostream/gnuplot-iostream.h"

void gnuplotOmega(uint64_t x, uint64_t y, uint16_t c)
{
  Gnuplot gp;
  // calculate bounds
  uint64_t z = x/y;
  uint64_t qrtx = pow(x,0.25);
  float x_flt = x;

  gp << "set terminal wxt size 2880, 1620\n";
  gp << "set tics front\n";
  gp << "set grid nopolar\n";
  gp << "set samples 10000\n";
  gp << "set grid xtics nomxtics ytics nomytics noztics nomztics nox2tics nomx2tics noy2tics nomy2tics nocbtics nomcbtics\n";
  gp << "set grid front linetype 0 linewidth 1.000, linetype 0 linewidth 1.000\n";
  gp << "set xrange [" << c << ":" << qrtx << "]\n";
  gp << "set yrange [0:" << y*1.1 << "]\n";
  gp << "set title 'Omega'\n";
  gp << "plot x > " << std::sqrt(y) << " ? " << y << " : 0 with filledcurve x1 title 'omega_3' enhanced, ";
  gp << "x < " << std::sqrt(y) << " ? " << y << " : 0 with filledcurve x1 title 'omega_1' enhanced, ";
  gp << "x < " << std::sqrt(y) << " ? " << " 0 : (" << y << "<" << x_flt << "/x**3 ? " << y << " : " << x_flt << "/x**3) with filledcurve x1 title 'omega_2' enhanced, ";
  gp << y << "/x with filledcurve x1 title 'not counted', ";
  gp << "x with filledcurve x1 title 'not counted'\n";
  // gp << "set arrow front from " << std::cbrt(z) << ",0 to " << std::cbrt(z) << "," << y << " nohead lw 3\n";
  // gp << "set arrow front from " << std::sqrt(y) << ",0 to " << std::sqrt(y) << "," << y << " nohead lw 3\n";
  gp << "replot\n";
  gp << "pause -1\n";

 }
