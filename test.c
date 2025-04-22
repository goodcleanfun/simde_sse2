#include <stdio.h>
#include <math.h>
#include <float.h>
#include "aligned/aligned.h"
#include "greatest/greatest.h"
#include "sse2.h"

#define CONST_128D(name, val) static const alignas(16) double name[2] = {(val), (val)}

static inline void remez9_0_log2_sse(double *values, size_t num)
{
    size_t i;
    CONST_128D(one, 1.);
    CONST_128D(log2e, 1.4426950408889634073599);
    CONST_128D(maxlog, 7.09782712893383996843e2);   // log(2**1024)
    CONST_128D(minlog, -7.08396418532264106224e2);  // log(2**-1022)
    CONST_128D(c1, 6.93145751953125E-1);
    CONST_128D(c2, 1.42860682030941723212E-6);
    CONST_128D(w9, 3.9099787920346160288874633639268318097077213911751e-6);
    CONST_128D(w8, 2.299608440919942766555719515783308016700833740918e-5);
    CONST_128D(w7, 1.99930498409474044486498978862963995247838069436646e-4);
    CONST_128D(w6, 1.38812674551586429265054343505879910146775323730237e-3);
    CONST_128D(w5, 8.3335688409829575034112982839739473866857586300664e-3);
    CONST_128D(w4, 4.1666622504201078708502686068113075402683415962893e-2);
    CONST_128D(w3, 0.166666671414320541875332123507829990378055646330574);
    CONST_128D(w2, 0.49999999974109940909767965915362308135415179642286);
    CONST_128D(w1, 1.0000000000054730504284163017295863259125942049362);
    CONST_128D(w0, 0.99999999999998091336479463057053516986466888462081);
    const simde__m128i offset = simde_mm_setr_epi32(1023, 1023, 0, 0);

    for (i = 0;i < num;i += 4) {
        simde__m128i k1, k2;
        simde__m128d p1, p2;
        simde__m128d a1, a2;
        simde__m128d xmm0, xmm1;
        simde__m128d x1, x2;

        /* Load four double values. */
        xmm0 = simde_mm_load_pd(maxlog);
        xmm1 = simde_mm_load_pd(minlog);
        x1 = simde_mm_load_pd(values+i);
        x2 = simde_mm_load_pd(values+i+2);
        x1 = simde_mm_min_pd(x1, xmm0);
        x2 = simde_mm_min_pd(x2, xmm0);
        x1 = simde_mm_max_pd(x1, xmm1);
        x2 = simde_mm_max_pd(x2, xmm1);

        /* a = x / log2; */
        xmm0 = simde_mm_load_pd(log2e);
        xmm1 = simde_mm_setzero_pd();
        a1 = simde_mm_mul_pd(x1, xmm0);
        a2 = simde_mm_mul_pd(x2, xmm0);

        /* k = (int)floor(a); p = (float)k; */
        p1 = simde_mm_cmplt_pd(a1, xmm1);
        p2 = simde_mm_cmplt_pd(a2, xmm1);
        xmm0 = simde_mm_load_pd(one);
        p1 = simde_mm_and_pd(p1, xmm0);
        p2 = simde_mm_and_pd(p2, xmm0);
        a1 = simde_mm_sub_pd(a1, p1);
        a2 = simde_mm_sub_pd(a2, p2);
        k1 = simde_mm_cvttpd_epi32(a1);
        k2 = simde_mm_cvttpd_epi32(a2);
        p1 = simde_mm_cvtepi32_pd(k1);
        p2 = simde_mm_cvtepi32_pd(k2);

        /* x -= p * log2; */
        xmm0 = simde_mm_load_pd(c1);
        xmm1 = simde_mm_load_pd(c2);
        a1 = simde_mm_mul_pd(p1, xmm0);
        a2 = simde_mm_mul_pd(p2, xmm0);
        x1 = simde_mm_sub_pd(x1, a1);
        x2 = simde_mm_sub_pd(x2, a2);
        a1 = simde_mm_mul_pd(p1, xmm1);
        a2 = simde_mm_mul_pd(p2, xmm1);
        x1 = simde_mm_sub_pd(x1, a1);
        x2 = simde_mm_sub_pd(x2, a2);

        /* Compute e^x using a polynomial approximation. */
        xmm0 = simde_mm_load_pd(w9);
        xmm1 = simde_mm_load_pd(w8);
        a1 = simde_mm_mul_pd(x1, xmm0);
        a2 = simde_mm_mul_pd(x2, xmm0);
        a1 = simde_mm_add_pd(a1, xmm1);
        a2 = simde_mm_add_pd(a2, xmm1);

        xmm0 = simde_mm_load_pd(w7);
        xmm1 = simde_mm_load_pd(w6);
        a1 = simde_mm_mul_pd(a1, x1);
        a2 = simde_mm_mul_pd(a2, x2);
        a1 = simde_mm_add_pd(a1, xmm0);
        a2 = simde_mm_add_pd(a2, xmm0);
        a1 = simde_mm_mul_pd(a1, x1);
        a2 = simde_mm_mul_pd(a2, x2);
        a1 = simde_mm_add_pd(a1, xmm1);
        a2 = simde_mm_add_pd(a2, xmm1);

        xmm0 = simde_mm_load_pd(w5);
        xmm1 = simde_mm_load_pd(w4);
        a1 = simde_mm_mul_pd(a1, x1);
        a2 = simde_mm_mul_pd(a2, x2);
        a1 = simde_mm_add_pd(a1, xmm0);
        a2 = simde_mm_add_pd(a2, xmm0);
        a1 = simde_mm_mul_pd(a1, x1);
        a2 = simde_mm_mul_pd(a2, x2);
        a1 = simde_mm_add_pd(a1, xmm1);
        a2 = simde_mm_add_pd(a2, xmm1);

        xmm0 = simde_mm_load_pd(w3);
        xmm1 = simde_mm_load_pd(w2);
        a1 = simde_mm_mul_pd(a1, x1);
        a2 = simde_mm_mul_pd(a2, x2);
        a1 = simde_mm_add_pd(a1, xmm0);
        a2 = simde_mm_add_pd(a2, xmm0);
        a1 = simde_mm_mul_pd(a1, x1);
        a2 = simde_mm_mul_pd(a2, x2);
        a1 = simde_mm_add_pd(a1, xmm1);
        a2 = simde_mm_add_pd(a2, xmm1);

        xmm0 = simde_mm_load_pd(w1);
        xmm1 = simde_mm_load_pd(w0);
        a1 = simde_mm_mul_pd(a1, x1);
        a2 = simde_mm_mul_pd(a2, x2);
        a1 = simde_mm_add_pd(a1, xmm0);
        a2 = simde_mm_add_pd(a2, xmm0);
        a1 = simde_mm_mul_pd(a1, x1);
        a2 = simde_mm_mul_pd(a2, x2);
        a1 = simde_mm_add_pd(a1, xmm1);
        a2 = simde_mm_add_pd(a2, xmm1);

        /* p = 2^k; */
        k1 = simde_mm_add_epi32(k1, offset);
        k2 = simde_mm_add_epi32(k2, offset);
        k1 = simde_mm_slli_epi32(k1, 20);
        k2 = simde_mm_slli_epi32(k2, 20);
        k1 = simde_mm_shuffle_epi32(k1, _MM_SHUFFLE(1,3,0,2));
        k2 = simde_mm_shuffle_epi32(k2, _MM_SHUFFLE(1,3,0,2));
        p1 = simde_mm_castsi128_pd(k1);
        p2 = simde_mm_castsi128_pd(k2);

        /* a *= 2^k. */
        a1 = simde_mm_mul_pd(a1, p1);
        a2 = simde_mm_mul_pd(a2, p2);

        /* Store the results. */
        simde_mm_store_pd(values+i, a1);
        simde_mm_store_pd(values+i+2, a2);
    }
}

TEST test_sse2(void) {
    double values[4] = {1.0, 2.0, 3.0, 4.0};
    remez9_0_log2_sse(values, 4);

    ASSERT(fabs(values[0] - exp(1.0)) < FLT_EPSILON);
    ASSERT(fabs(values[1] - exp(2.0)) < FLT_EPSILON);
    ASSERT(fabs(values[2] - exp(3.0)) < FLT_EPSILON);
    ASSERT(fabs(values[3] - exp(4.0)) < FLT_EPSILON);
    PASS();
}

SUITE(sse2_suite) {
    RUN_TEST(test_sse2);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
    GREATEST_MAIN_BEGIN();
    RUN_SUITE(sse2_suite);
    GREATEST_MAIN_END();
}

