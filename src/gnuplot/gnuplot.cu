#include <stdint.h>
#include <iostream>
#include <math.h>
#include <cuda_uint128.h>

#include "gnuplot-iostream/gnuplot-iostream.h"

void gnuplotOmega(uint64_t x, uint64_t y, uint16_t c)
{
  Gnuplot gp;
  // calculate bounds
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

 void gnuplotA_Omega(uint64_t x, uint64_t y, uint16_t c)
 {
   Gnuplot gp;
   // calculate bounds
   uint64_t qrtx = pow(x,0.25);
   uint64_t x38 = pow(x, 0.375);
   uint64_t cbrtx = std::cbrt(x);
   float x_flt = x;

   gp << "set terminal wxt size 1620, 1620\n";
   gp << "set tics front\n";
   gp << "set grid nopolar\n";
   gp << "set samples 10000\n";
   gp << "set grid xtics nomxtics ytics nomytics noztics nomztics nox2tics nomx2tics noy2tics nomy2tics nocbtics nomcbtics\n";
   gp << "set grid front linetype 0 linewidth 1.000, linetype 0 linewidth 1.000\n";
   gp << "set xrange [" << c << ":" << cbrtx << "]\n";
   gp << "set yrange [0:" << y*1.1 << "]\n";
   gp << "set title 'Omega'\n";
   gp << "plot x > " << std::sqrt(y) << " && x <= " << qrtx << " ? " << y << " : 0 with filledcurve x1 title 'omega_3' enhanced, ";
   gp << "x < " << std::sqrt(y) << " && x <= " << qrtx << " ? " << y << " : 0 with filledcurve x1 title 'omega_1' enhanced, ";
   gp << "x < " << std::sqrt(y) << " && x <= " << qrtx << " ? " << " 0 : (" << y << "<" << x_flt << "/x**3 ? " << y << " : " << x_flt << "/x**3) with filledcurve x1 title 'omega_2' enhanced, ";
   gp << "x < " << qrtx << " ? 0 : sqrt(" << x_flt << "/x) with filledcurve x1 title 'A' enhanced, ";
   gp << y << "/x with filledcurve x1 title 'not counted', ";
   gp << "x < " << cbrtx << " ? x : 0 with filledcurve x1 title 'not counted',";
   for(float a = std::sqrt(x); a > x38; a -= pow(10,9))
      gp << x/a << "/x with line lw 2 title 'pi_m_a_x = " << (uint64_t) a << "',";
   for(float a = x38; a > cbrtx; a -= pow(10,9))
      gp << x/a << "/x with line lw 2 title 'pi_m_a_x = " << (uint64_t)a << "',";
   gp << x/cbrtx << "/x with line lw 2 title 'pi_m_a_x = " << cbrtx << "'\n";
   gp << "pause -1\n";

  }

  void gnuplotA_Omega(uint128_t x, uint64_t y, uint16_t c)
  {
    Gnuplot gp;
    // calculate bounds
    double x_dbl = uint128_t::u128_to_double(x);
    double sqrtx = uint128_t::sqrt(x);
    double qrtx = pow(x_dbl,0.25);
    double x38 = pow(x_dbl, 0.375);
    double cbrtx = std::cbrt(x_dbl);

    gp << "set terminal wxt size 1620, 1620\n";
    gp << "set tics front\n";
    gp << "set grid nopolar\n";
    gp << "set samples 10000\n";
    gp << "set grid xtics nomxtics ytics nomytics noztics nomztics nox2tics nomx2tics noy2tics nomy2tics nocbtics nomcbtics\n";
    gp << "set grid front linetype 0 linewidth 1.000, linetype 0 linewidth 1.000\n";
    gp << "set xrange [" << c << ":" << cbrtx << "]\n";
    gp << "set yrange [0:" << y*1.1 << "]\n";
    gp << "set title 'Omega'\n";
    gp << "plot x > " << std::sqrt(y) << " && x <= " << qrtx << " ? " << y << " : 0 with filledcurve x1 title 'omega_3' enhanced, ";
    gp << "x < " << std::sqrt(y) << " && x <= " << qrtx << " ? " << y << " : 0 with filledcurve x1 title 'omega_1' enhanced, ";
    gp << "x < " << std::sqrt(y) << " && x <= " << qrtx << " ? " << " 0 : (" << y << "<" << x_dbl << "/x**3 ? " << y << " : " << x_dbl << "/x**3) with filledcurve x1 title 'omega_2' enhanced, ";
    gp << "x < " << qrtx << " ? 0 : sqrt(" << x_dbl << "/x) with filledcurve x1 title 'A' enhanced, ";
    gp << y << "/x with filledcurve x1 title 'not counted', ";
    gp << "x < " << cbrtx << " ? x : 0 with filledcurve x1 title 'not counted',";
    //for(float a = std::sqrt(x_dbl); a > x38; a -= pow(2,30))
    double a = sqrtx;
       gp << x_dbl/a << "/x with line lw 2 title 'pi_m_a_x = " << (uint64_t) a << "',";
    //for(float a = x38; a > cbrtx; a -= pow(2,30))
    a = x38;
       gp << x_dbl/a << "/x with line lw 2 title 'pi_m_a_x = " << (uint64_t)a << "',";
    gp << x_dbl/cbrtx << "/x with line lw 2 title 'pi_m_a_x = " << (uint64_t) cbrtx << "'\n";
    gp << "pause -1\n";

   }
