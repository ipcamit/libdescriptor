//from https://people.math.sc.edu/Burkardt/c_src/c_src.html
#ifndef GAMMA_HPP
#define GAMMA_HPP


//Allan Macleod, Algorithm AS 245,
double lgamma ( double xvalue);

double gamma_inc (double p,  double x);

#endif //GAMMA_HPP


// void gamma_log_values ( int *n_data, double *x, double *fx ){
//   double fx_vec[20] = {
//       0.1524063822430784E+01,
//       0.7966778177017837E+00,
//       0.3982338580692348E+00,
//       0.1520596783998375E+00,
//       0.0000000000000000E+00,
//      -0.4987244125983972E-01,
//      -0.8537409000331584E-01,
//      -0.1081748095078604E+00,
//      -0.1196129141723712E+00,
//      -0.1207822376352452E+00,
//      -0.1125917656967557E+00,
//      -0.9580769740706586E-01,
//      -0.7108387291437216E-01,
//      -0.3898427592308333E-01,
//      0.00000000000000000E+00,
//      0.69314718055994530E+00,
//      0.17917594692280550E+01,
//      0.12801827480081469E+02,
//      0.39339884187199494E+02,
//      0.71257038967168009E+02 };

//   double x_vec[20] = {
//       0.20E+00,
//       0.40E+00,
//       0.60E+00,
//       0.80E+00,
//       1.00E+00,
//       1.10E+00,
//       1.20E+00,
//       1.30E+00,
//       1.40E+00,
//       1.50E+00,
//       1.60E+00,
//       1.70E+00,
//       1.80E+00,
//       1.90E+00,
//       2.00E+00,
//       3.00E+00,
//       4.00E+00,
//      10.00E+00,
//      20.00E+00,
//      30.00E+00 };

//   if ( *n_data < 0 ){
//     *n_data = 0;
//   }

//   *n_data = *n_data + 1;

//   if ( 20 < *n_data ){
//     *n_data = 0;
//     *x = 0.0;
//     *fx = 0.0;
//   } else {
//     *x = x_vec[*n_data-1];
//     *fx = fx_vec[*n_data-1];
//   }

//   return;
// }


// double lanczos_lngamma ( double z, int *ier ){
//   double a[9] = {
//          0.9999999999995183,
//        676.5203681218835,
//     - 1259.139216722289,
//        771.3234287757674,
//      - 176.6150291498386,
//         12.50734324009056,
//        - 0.1385710331296526,
//          0.9934937113930748E-05,
//          0.1659470187408462E-06 };
//   int j;
//   double lnsqrt2pi = 0.9189385332046727;
//   double tmp;
//   double value;

//   if ( z <= 0.0 )
//   {
//     *ier = 1;
//     value = 0.0;
//     return value;
//   }

//   *ier = 0;

//   value = 0.0;
//   tmp = z + 7.0;
//   for ( j = 8; 1 <= j; j-- )
//   {
//     value = value + a[j] / tmp;
//     tmp = tmp - 1.0;
//   }

//   value = value + a[0];
//   value = log ( value ) + lnsqrt2pi - ( z + 6.5 )
//     + ( z - 0.5 ) * log ( z + 6.5 );

//   return value;
// }
