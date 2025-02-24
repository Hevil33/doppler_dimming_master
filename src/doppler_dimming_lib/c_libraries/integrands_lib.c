#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K_CGS (1.380649e-16)           // erg K-1 / cm2 g s-2 K-1
#define M_E_CGS (9.1093897e-28)        // grams
#define C_CGS (2.99792458e10)          // speed of light cm s-1
#define C_CGS_INV (3.33564095e-11)     // speed of light cm s-1
#define SIGMA_E_CGS (6.6524587158e-25) // electron cross section [cm2]
#define KM_TO_CM (1.e5)                // Km to cm conversion
#define CM_TO_A (1.e8)                 // cm to angstrom conversion
#define A_TO_CM (1.e-8)                // angstrom to cm conversion
#define K_SI (1.380649e-23)            // J K-1
#define M_E_SI (9.1093897e-31)         // Kg
#define C_SI (2.99792458e8)            // speed of light m s-1
#define SIGMA_E_SI (6.6524587158e-29)  // electron cross section [m2]
#define M_TO_A (1.e10)                 // m to angstrom conversion
// #define EARTH_DISTANCE_SR (1.495979e8 / 6.9566e5) // [km/km] sun to earth distance in solar radii, 215.03 in theory
#define EARTH_DISTANCE_SR (215.03)
#define R_SUN_CM (6.957e10)
#define PI (3.14159265359)
#define SQRT2 (1.4142135623730951)
#define DOUBLE_SQRTPI (3.5449077018110318)

#define debug(x) printf("value of \"%s\" is: %8.4e\n", #x, x);

/* ----------------------------------------- */

double N_e_from_function(double);
double q_lambda(double);
double I_lambda_mu(double, double, double);
double I_dlambda_domega(int, double *, void *);
double I_dlambda_domega_dx(int, double *, void *);
double Q_T(double, double, double);
double Q_R(double, double, double);
double interpolate_Ne(double, double *, double *, int);

/*
double integrate_simpson(double (*Func)(double), double, double, int);
double integrate_simpson(double (*Func)(double x1), double xa, double xb, int n)
{
    double sum = 0.0, dx, x;
    dx = (xb - xa) / (n * 1.0);

    for (int i = 1; i < n; i++)
    {
        x = xa + i * dx; // calcola il punto iniziale

        if (i % 2 == 0)
            sum += 2. * Func(x); // fpari
        else if (i % 2 == 1)
            sum += 4. * Func(x); // fdispari
    }
    return ((sum + Func(xa) + Func(xb)) * dx / 3.);
}
double int_x(double a, double b)
{
    printf("second %f", integrate_simpson(x, a, b, 6));
    return integrate_simpson(x, a, b, 6);
}
*/

/* ----------------------------------------- */

double N_e_from_function(double r)
{
    // return 1.67 * pow(10, 4 + 4.04 / r);

    return 1.0e8 * (0.036 * pow(r, -1.5) + 1.55 * pow(r, -6) + 2.99 * pow(r, -16)); // Baumbach 1937, cm-3

    double z = 1. / r;
    double z2 = z * z;
    return 2.6e-3 * exp(5.5986 * z + 5.4155 * z2) * z2 * (1. + 0.82902 * z - 5.6654 * z2 + 3.9784 * z2 * z); // Guhathakurta 2006, cm-3
}

double q_lambda(double lambda)
{
    double lambda_angstrom = lambda * CM_TO_A;

    if (lambda_angstrom < 3700.)
    {
        return 0.922 - (lambda_angstrom - 3220.) * 2.5E-4; // 0.120 / 480 = 0.0003
    }
    else
    {
        return 0.862 - (lambda_angstrom - 3700.) * 1.630769231E-4; // 0.212 / 1300 = 1.6...x10^-4
    }
}

double I_lambda_mu(double mu, double lambda, double F_lambda)
{
    double q = q_lambda(lambda);

    // F_lambda already irradiance, multiply by limb darkening
    return F_lambda * ((1. - q + (q * mu)) / (1. - (q / 3.0)));

    // return EARTH_DISTANCE_SR * EARTH_DISTANCE_SR * (1. / PI) * ((1. - q + (q * mu)) / (1. - (q / 3.0))) * F_lambda;
}

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

struct input_data_struct
{
    int n_points;
    double *c_wls;
    double *c_Flambdas;
    int los_n_points;
    double *los_positions;
    double *los_densities;
};

double dlambda_prime_integral(double lambda, double cos_omega, double wind_speed, double T_corona, double b, double mu, struct input_data_struct table)
{
    /*THIS INTEGRAL IS DONE IN CM*/

    double nsigma = 4.; // difference between lambdas to take. Computing times with spectrum 1: 3sigma 115s, 4sigma 120s, 5sigma 130s
    int points_n = table.n_points;
    double *wls = table.c_wls;
    double *F_lambdas = table.c_Flambdas;

    /*
    printf("here\n");
    printf("%d\n", table.n_points);
    printf("%d\n", points_n);
    debug(table.c_wls[0]);
    debug(table.c_Flambdas[0]);
    printf("%d\n", table.los_n_points);
    // debug(los_n_points);
    debug(table.los_positions[0]);
    debug(table.los_densities[0]);
    printf("\n\n");
    /**/

    double wind_factor = 1.0 + 2.0 * b * b * cos_omega * wind_speed * C_CGS_INV; // W in cm/s

    double I_lambda_sum = 0;
    double lambda_nm1, lambda_n, lambda_np1;
    double F_lambda_nm1, F_lambda_n, F_lambda_np1;
    double delta_nm1, delta_n, delta_np1;
    double exponent_nm1, exponent_n, exponent_np1;
    double I_lambda_nm1, I_lambda_n, I_lambda_np1, q_c;
    int counter = 0;

    double nsigma_sqrt2_b = nsigma * SQRT2 * b;

    // q_c = 5508. * sqrt(T_corona * 1.e-6) * KM_TO_CM * C_CGS_INV; // [cm/s]/c, Cram 1976 pag 7
    // q_c = 5505.694902726358 * sqrt(T_corona * 1.e-6) * KM_TO_CM * C_CGS_INV; // more precise from scipy constants
    // debug(q_c);
    q_c = sqrt(2. * K_CGS * T_corona / M_E_CGS) * C_CGS_INV; // complete
    // debug(q_c);
    // debug(sqrt(2. * K_CGS / M_E_CGS / 1.e-6));

    // Trapezoidal rule
    for (int i = 0; i < points_n - 1; i++)
    {
        lambda_nm1 = wls[i];
        F_lambda_nm1 = F_lambdas[i];

        delta_nm1 = lambda_nm1 * q_c; //* C_CGS_INV;0
        if (fabs(lambda_nm1 - lambda) > nsigma_sqrt2_b * delta_nm1)
            continue; // skip if two lambdas are too far from each other
        counter++;

        lambda_n = wls[i + 1];
        F_lambda_n = F_lambdas[i + 1];

        exponent_nm1 = (lambda - lambda_nm1 * wind_factor) / (2 * delta_nm1 * b);

        delta_n = lambda_n * q_c; // * C_CGS_INV;
        exponent_n = (lambda - lambda_n * wind_factor) / (2 * delta_n * b);

        I_lambda_nm1 = (I_lambda_mu(mu, lambda_nm1, F_lambda_nm1) * exp((-(exponent_nm1 * exponent_nm1))) / (DOUBLE_SQRTPI * delta_nm1 * b));
        I_lambda_n = (I_lambda_mu(mu, lambda_n, F_lambda_n) * exp((-(exponent_n * exponent_n))) / (DOUBLE_SQRTPI * delta_n * b));

        I_lambda_sum += fabs(lambda_n - lambda_nm1) * (I_lambda_nm1 + I_lambda_n);
    }

    /*
    printf("\n");
    printf("value of counter is: %d\n", counter);
    debug(mu);
    debug(I_lambda_mu(mu, lambda_n, F_lambda_n));
    debug(lambda);
    debug(F_lambda_nm1);
    debug(lambda_nm1);
    debug(fabs(lambda_nm1 - lambda));
    debug(nsigma * SQRT2 * delta_nm1 * b);
    debug(delta_nm1);
    debug(I_lambda_n);
    debug(I_lambda_nm1);
    debug(q_c);
    debug(b); /**/

    I_lambda_sum *= 0.5; // trapezoid area brought out of integral

    return I_lambda_sum;
};

double I_dlambda_domega(int n, double *xx, void *userdata)
{
    double phi = xx[0];
    double cos_omega = xx[1];
    double x = (xx[2]);

    double rho = xx[3];
    double lambda = xx[4];     // cm
    double T_corona = xx[5];   // Kelvin
    double wind_speed = xx[6]; // cm/s
    double component = xx[7];  // 0=radial, 1=tangential, 3=sum of them

    struct input_data_struct table = *(struct input_data_struct *)userdata;

    double r = sqrt(x * x + rho * rho);
    // double x = sqrt(r * r - rho * rho);
    double sin_omega = sqrt(1.0 - cos_omega * cos_omega);
    // double omega = acos(cos_omega);
    double sin_phi = sin(phi);

    double chi = acos(fabs(x) / r);
    double Theta = PI - acos(cos_omega * cos(chi) + sin_omega * sin(chi) * sin_phi);

    double gamma = (PI - Theta) * 0.5;
    double b = cos(gamma);

    double theta = asin(r * sin_omega);
    double mu = cos(theta);

    double alpha_inv = sin_phi * sin_omega / sin(PI - Theta);
    if (alpha_inv > 1)
        alpha_inv = 1.;
    else if (alpha_inv < -1.)
        alpha_inv = -1.;
    double alpha = asin(alpha_inv);
    double cos_alpha = cos(alpha), sin_alpha = sin(alpha);
    double cos_Theta = cos(Theta);

    // double N_e_r = r * N_e_from_function(r) / x; // dx = r/sqrt(r^2 - rho^2) dr

    double I_lambda = dlambda_prime_integral(lambda, cos_omega, wind_speed, T_corona, b, mu, table); // I[erg cm-2 sr-1 A-1]

    double cos_omega_star = cos(asin(1. / sqrt(x * x + rho * rho)));
    if (cos_omega < cos_omega_star)
    {
        printf("ERROR: integrating over cos_omega boundary\n\n");
        printf("\n");
        debug(chi);
        debug(mu);
        debug(theta);
        debug(sin_omega);
        debug(cos_omega);
        debug(cos_omega_star);
        debug(r * sin_omega);
    }

    /*
    debug(r);
    debug(x);
    debug(rho);
    debug(phi);
    debug(cos_omega);
    debug(x);
    debug(rho);
    debug(lambda);
    debug(T_corona);
    debug(W);
    debug(component);
    */

    double Q_normalization = 3.0 * SIGMA_E_CGS / (16. * PI);

    if (component == 0)
    {
        return Q_normalization * Q_R(cos_alpha, sin_alpha, cos_Theta) * I_lambda;
    }
    else if (component == 1)
    {
        return Q_normalization * Q_T(cos_alpha, sin_alpha, cos_Theta) * I_lambda;
    }
    else if (component == 3)
    {
        return Q_normalization * (Q_R(cos_alpha, sin_alpha, cos_Theta) + Q_T(cos_alpha, sin_alpha, cos_Theta)) * I_lambda; // original
    }
    else
    {
        printf("ERROR: invalid component (0, 1 allowed.)\n");
    }
}

double Q_R(double cos_alpha, double sin_alpha, double cos_Theta)
{
    return cos_alpha * cos_alpha * cos_Theta * cos_Theta + sin_alpha * sin_alpha;
}

double Q_T(double cos_alpha, double sin_alpha, double cos_Theta)
{
    return cos_alpha * cos_alpha + sin_alpha * sin_alpha * cos_Theta * cos_Theta;
}

double interpolate_Ne(double x, double *xs, double *Nes, int len_xs)
{
    double m;
    double result;

    // this shows why its so slow
    /*static int i = 0;
    i++;
    printf("done %d times\n", i);
    */

    if (1 == 1) // bisection
    {
        int index;
        int min_i = 0, max_i = len_xs - 1;

        int counter = 0;
        while (min_i < max_i)
        {
            index = (min_i + max_i) / 2;

            if (fabs(x - xs[min_i]) < fabs(x - xs[max_i]))
                max_i = index;
            else
                min_i = index;

            if (max_i == (min_i + 1))
                break;

            counter++;
            if (counter > 100)
            {
                exit(0);
            }
        }
        /*
        printf("index %d\n", index);
        printf("min %d\n", min_i);
        printf("max %d\n", max_i);
        printf("\n");
        printf("index 1 = %d\n", min_i);
        /**/

        if (xs[index] > x)
            index -= 1;

        m = (Nes[index + 1] - Nes[index]) / (xs[index + 1] - xs[index]);
        result = m * (x - xs[index]) + Nes[index];

        debug(result);
        return result;
    }
    else // loop array
    {
        for (int i = 0; i < (len_xs - 1); i++)
        {
            // printf("searching interp\n");
            if ((xs[i] <= x) && (xs[i + 1] > x))
            {
                m = (Nes[i + 1] - Nes[i]) / (xs[i + 1] - xs[i]);
                /*
                printf("found interpolated %d\n", i);
                debug(x);
                debug(Nes[i]);
                debug(Nes[i + 1]);
                debug(xs[i]);
                debug(xs[i + 1]);
                debug(m * (x - xs[i]) + Nes[i]);
                /**/
                printf("index 2 = %d\n", i);
                return m * (x - xs[i]) + Nes[i];
            }
        }
    }
}

double I_dlambda_domega_dx(int n, double *xx, void *userdata)
{
    // double phi = xx[0];
    // double cos_omega = xx[1];
    double x = xx[2];

    // double rho = xx[3];
    // double lambda = xx[4];    // cm
    // double T_corona = xx[5];  // Kelvin
    // double W = xx[6];         // cm/s
    // double component = xx[7]; // 0=radial, 1=tangential, 3=sum of them

    struct input_data_struct table = *(struct input_data_struct *)userdata;
    double *xs = table.los_positions;
    double *densities = table.los_densities;
    int los_n_points = table.los_n_points;

    // printf("%d\n", n_points);
    // printf("%d\n", los_n_points);
    for (int i = 0; i < 5; i++)
    {
        /*debug(table.n_points);
        debug(n_points);
        debug(table.c_wls[i]);
        debug(table.c_Flambdas[i]);
        debug(table.los_n_points);
        debug(los_n_points);
        debug(table.los_positions[i]);
        debug(table.los_densities[i]);
        printf("\n\n");*/
    }

    double density = interpolate_Ne(x, xs, densities, los_n_points);

    return density * I_dlambda_domega(n, xx, userdata);
}